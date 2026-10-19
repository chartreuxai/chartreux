"""Waits and lifecycle behavior tests for subagent registry state.

Covers the registry lifecycle rethink: close-on-completion for unretained
launches, retained residency, retention precedence, reuse policy inheritance,
final unretained continuation, admission rollback, busy-replacement retention
conflicts, both replacement-unwind timings, tombstone TTL + insertion cap,
ephemeral-only reaper scheduling, no read-based deadline extension, expiry
during transcript reads, ``/clear`` with only tombstones, metadata expiry
leaving results retrievable, deterministic summary ordering, and the §3.1
reuse matrix. Also pins the WP2 checkpoint fixes: zero tombstone TTL stores no
terminal metadata, unwind closures publish their terminal transition, public
emission filters expired tombstones, the reaper publishes after metadata-only
pruning, eviction insertion rearms the reaper, and Active ordering uses the
stable first-admission timestamp. The round-2 re-review fixes are pinned too:
the reaper publishes whenever the inventory changed (pruning plus ineligible
victims), metadata-deadline scheduling survives stalled eviction cleanup,
zero-TTL eviction notifications are release-invalidatable, and reaper
finalizer rearm failures surface while a loop is running. The round-4
re-review fixes are pinned too: pending-eviction authorities settle on every
path (successful notifications of both TTL variants, and insertion-cap
tombstone removal), and drain owns displaced reapers so no inventory payload
is delivered after drain's final empty payload. The round-4 drain guard's
regression is pinned as well: a disappearance publication suppressed or
cancelled at the drain boundary is still delivered by drain's final empty
inventory. The round-5 re-review fix is pinned too: an ordinary rearm that
replaces a live reaper keeps the displaced task drain-owned, so no stale
snapshot is delivered after drain's final empty payload. The pre-merge
deep-review fixes are pinned too: /clear retiring only resident agents
publishes the reduced inventory, metadata-expired (unpruned) tombstones
behave as unknown at both reuse admission sites, and eviction notifications
carry the latest run id and outcome (single and batched).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.app_server._file_revision_registry import FileRevisionRegistry
from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntimeRegistry,
    _AgentState,
    _AgentTombstone,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.models import AgentProfile
from chartreux.core.file_revision import FileRevision, full_revision
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.subagents import (
    AgentAvailability,
    AgentEviction,
    LaunchConfigError,
    RetentionPolicy,
    TaskArgs,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _config,
    _GatedWakeup,
    _ManualClock,
    _reaper_registry,
    _rearm,
    _record,
    _retention_registry,
    _root_registry,
    _run,
    _stored_result,
    _tombstone,
    _zero_retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _matches_revision(known: FileRevision | None, expected: FileRevision) -> bool:
    return known is not None and known.matches(expected)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


async def _settle(registry: SessionRuntimeRegistry, record: AgentRecord) -> None:
    completion = record.current_run.completion_task if record.current_run else None
    if completion is not None:
        await asyncio.gather(completion, return_exceptions=True)
    await asyncio.gather(*tuple(registry._teardown_tasks))


# ---------------------------------------------------------------------------
# 1. Default close-on-completion (publication before close)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2. Retained residency + both-zero-retained-unlimited
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retained_agent_stays_resident_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")]), zero_limits=False
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", agent_type="worker", background=True, retain=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.retained is True
        await _settle(registry, record)
        # Retained: returns to idle, stays resident, no tombstone, runtime open.
        assert launch.agent_id in registry._agent_records
        assert record.state is _AgentState.IDLE
        assert launch.agent_id not in registry._evicted_agents
        assert record.runtime._closed is False
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_both_zero_retained_snapshot_keeps_agent_resident_unlimited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # (0, 0, True): retained via the snapshot with no TTL and no idle cap.
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")]), zero_limits=True
    )
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0, max_idle_agents=0, retain_agents=True
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", agent_type="worker", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.retained is True
        await _settle(registry, record)
        assert launch.agent_id in registry._agent_records
        assert record.state is _AgentState.IDLE
        # No TTL deadline and no idle cap: nothing for the reaper to schedule.
        assert registry._reaper_task is None
        assert launch.agent_id not in registry._evicted_agents
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# 3. Retention precedence: explicit task arg -> profile -> snapshot default
# ---------------------------------------------------------------------------


def _persistent_profile(*, retain: bool | None) -> AgentProfile:
    return AgentProfile(
        name="persistent",
        display_name="Persistent",
        description="precedence profile",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.SUBAGENT,
        retain=retain,
    )


async def _launch_retained(
    monkeypatch: pytest.MonkeyPatch, args: TaskArgs, profile: AgentProfile | None
) -> tuple[SessionRuntimeRegistry, Any, AgentRecord]:
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")]), zero_limits=False
    )
    if profile is not None:
        parent.agent_manager._discovered[profile.name] = profile
    launch = await _background_result(
        registry,
        args,
        InvokeContext(tool_call_id="launch", session_id=parent.session_id),
    )
    assert launch.agent_id is not None
    record = registry._agent_records[launch.agent_id]
    return registry, parent, record


@pytest.mark.asyncio
async def test_explicit_task_arg_beats_profile_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Profile opts in, but an explicit retain=False wins.
    registry, parent, record = await _launch_retained(
        monkeypatch,
        TaskArgs(task="work", agent_type="persistent", background=True, retain=False),
        _persistent_profile(retain=True),
    )
    try:
        assert record.retained is False
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_explicit_retain_true_beats_unretained_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Snapshot defaults to unretained; an explicit retain=True wins.
    registry, parent, record = await _launch_retained(
        monkeypatch,
        TaskArgs(task="work", agent_type="worker", background=True, retain=True),
        None,
    )
    try:
        assert record.retained is True
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_profile_retain_beats_unretained_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No explicit arg: profile retain=True beats snapshot retain_agents=False.
    registry, parent, record = await _launch_retained(
        monkeypatch,
        TaskArgs(task="work", agent_type="persistent", background=True),
        _persistent_profile(retain=True),
    )
    try:
        assert record.retained is True
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_snapshot_retain_agents_is_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No explicit arg, profile retain=None: the snapshot default decides.
    registry, parent, record = await _launch_retained(
        monkeypatch,
        TaskArgs(task="work", agent_type="worker", background=True),
        _persistent_profile(retain=None),
    )
    try:
        # Production default snapshot has retain_agents=False.
        assert record.retained is False
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# 4/5. Reuse policy inheritance + final unretained continuation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reuse_inherits_committed_retention_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One stream per run so the reuse below completes: this test pins the
    # retention-policy inheritance, not failure-state semantics (a failed
    # run suspends warm under WP2 — pinned in tests/recovery/).
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch,
        FakeBackend([
            [mock_llm_chunk(content="done")],
            [mock_llm_chunk(content="done")],
        ]),
        zero_limits=False,
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="first", agent_type="worker", background=True, retain=True),
            InvokeContext(tool_call_id="first", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        await _settle(registry, record)
        assert record.state is _AgentState.IDLE
        # Reuse with no retain argument inherits the committed retained policy.
        again = await _background_result(
            registry,
            TaskArgs(
                task="second",
                agent_type="worker",
                agent_id=launch.agent_id,
                background=True,
            ),
            InvokeContext(tool_call_id="second", session_id=parent.session_id),
        )
        assert again.agent_id == launch.agent_id
        assert record.retained is True
        await _settle(registry, record)
        assert launch.agent_id in registry._agent_records
        assert record.state is _AgentState.IDLE
        assert launch.agent_id not in registry._evicted_agents
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_explicit_retain_false_reuse_is_final_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One stream per run so the final continuation below completes:
    # retain=False makes the next completion final (a failed run would
    # suspend warm under WP2 — pinned in tests/recovery/).
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch,
        FakeBackend([
            [mock_llm_chunk(content="done")],
            [mock_llm_chunk(content="done")],
        ]),
        zero_limits=False,
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="first", agent_type="worker", background=True, retain=True),
            InvokeContext(tool_call_id="first", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        await _settle(registry, record)
        assert record.state is _AgentState.IDLE
        # Explicit retain=False: one final continuation, then close.
        final = await _background_result(
            registry,
            TaskArgs(
                task="final",
                agent_type="worker",
                agent_id=launch.agent_id,
                background=True,
                retain=False,
            ),
            InvokeContext(tool_call_id="final", session_id=parent.session_id),
        )
        assert final.agent_id == launch.agent_id
        assert record.retained is False
        await _settle(registry, record)
        assert launch.agent_id not in registry._agent_records
        assert launch.agent_id in registry._evicted_agents
        tombstone = registry._evicted_agents[launch.agent_id]
        assert tombstone.reason == "closed"
        assert tombstone.summary.availability is AgentAvailability.CLOSED
        result = await registry.get_agent_result(launch.agent_id, final.run_id)
        assert result is not None and result.run_id == final.run_id
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# 6. Admission rollback restores the prior committed policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_admission_restores_prior_retention_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")]), zero_limits=False
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="first", agent_type="worker", background=True, retain=True),
            InvokeContext(tool_call_id="first", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        await _settle(registry, record)
        assert record.retained is True and record.state is _AgentState.IDLE

        # A turn-start failure after the retain=False commit rolls the policy
        # back to the prior retained=True and restores idle residency.
        def fail_start(*_args: object, **_kwargs: object) -> tuple[str, object]:
            raise RuntimeError("start failed")

        monkeypatch.setattr(SessionRuntimeRegistry, "_start_child_turn", fail_start)
        with pytest.raises(RuntimeError, match="start failed"):
            await _background_result(
                registry,
                TaskArgs(
                    task="final",
                    agent_type="worker",
                    agent_id=launch.agent_id,
                    background=True,
                    retain=False,
                ),
                InvokeContext(tool_call_id="final", session_id=parent.session_id),
            )
        assert record.retained is True
        assert record.state is _AgentState.IDLE
        assert record.current_run is None
        assert launch.agent_id in registry._agent_records
        assert launch.agent_id not in registry._evicted_agents
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# 7. Busy-replacement retention conflict rejection (before stopping anything)
# ---------------------------------------------------------------------------


async def _busy_record(
    monkeypatch: pytest.MonkeyPatch, *, retain: bool
) -> tuple[SessionRuntimeRegistry, Any, AgentRecord, BlockingBackend, Any]:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    launch = await _background_result(
        registry,
        TaskArgs(task="work", agent_type="worker", background=True, retain=retain),
        InvokeContext(tool_call_id="launch", session_id=parent.session_id),
    )
    assert launch.agent_id is not None
    record = registry._agent_records[launch.agent_id]
    await backend.started.wait()
    return registry, parent, record, backend, launch


@pytest.mark.asyncio
async def test_busy_replacement_rejects_conflicting_retain_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, backend, launch = await _busy_record(
        monkeypatch, retain=True
    )
    try:
        assert record.retained is True
        with pytest.raises(LaunchConfigError, match="retention policy"):
            await _background_result(
                registry,
                TaskArgs(
                    task="new",
                    agent_type="worker",
                    agent_id=launch.agent_id,
                    background=True,
                    replace_run=True,
                    retain=False,
                ),
                InvokeContext(tool_call_id="replace", session_id=parent.session_id),
            )
        # Rejected before stopping: the old run is still running and unreserved.
        assert record.state is _AgentState.RUNNING
        assert record.reserved_for_replacement is False
        assert record.current_run is not None
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_busy_replacement_rejects_retain_true_on_unretained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, backend, launch = await _busy_record(
        monkeypatch, retain=False
    )
    try:
        assert record.retained is False
        with pytest.raises(LaunchConfigError, match="retention policy"):
            await _background_result(
                registry,
                TaskArgs(
                    task="new",
                    agent_type="worker",
                    agent_id=launch.agent_id,
                    background=True,
                    replace_run=True,
                    retain=True,
                ),
                InvokeContext(tool_call_id="replace", session_id=parent.session_id),
            )
        assert record.state is _AgentState.RUNNING
        assert record.reserved_for_replacement is False
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# 8. Replacement-unwind rule: both timings re-apply the committed policy
# ---------------------------------------------------------------------------


def _unwind_record(
    registry: SessionRuntimeRegistry,
    clock: _ManualClock,
    *,
    retained: bool,
    completion_done: bool,
) -> tuple[AgentRecord, RunRecord]:
    record = _record(
        "agent-1", clock=clock, state=_AgentState.RUNNING, retained=retained
    )
    run = _run(record, "run-1", started_at=0, completed_at=10, done=completion_done)
    record.current_run = None
    record.run_history.append(run)
    record.latest_run_id = run.run_id
    record.reserved_for_replacement = True
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    return record, run


@pytest.mark.asyncio
async def test_unwind_direct_idle_closes_unretained_instance() -> None:
    # Timing A: reservation released with the old run already settled.
    clock = _ManualClock(0)
    registry, _root = _root_registry(clock)
    # No tombstone TTL / idle TTL: the close is under test, not the reaper.
    # A zero tombstone TTL retains no terminal metadata (design §7): the
    # record still detaches and closes, but no tombstone is stored and the
    # handle behaves as unknown.
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=0,
        max_tombstones=32,
    )
    record, _run_record = _unwind_record(
        registry, clock, retained=False, completion_done=True
    )
    async with registry._registry_lock:
        registry._clear_replacement_locked(record)
    await asyncio.gather(*tuple(registry._teardown_tasks))

    assert record.agent_id not in registry._agent_records
    assert record.agent_id not in registry._evicted_agents
    cast(AsyncMock, record.runtime.close).assert_awaited()


@pytest.mark.asyncio
async def test_unwind_pending_completion_closes_unretained_instance() -> None:
    # Timing A-pending: reservation released while the old run's completion is
    # still pending; the unwind helper closes once it settles.
    clock = _ManualClock(0)
    registry, _root = _root_registry(clock)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=0,
        max_tombstones=32,
    )
    record, run = _unwind_record(registry, clock, retained=False, completion_done=False)
    async with registry._registry_lock:
        registry._clear_replacement_locked(record)
    assert record.state is _AgentState.FINALIZING
    run.completion_task.set_result(None)
    await asyncio.gather(*tuple(registry._teardown_tasks))

    assert record.agent_id not in registry._agent_records
    # Zero tombstone TTL: no terminal metadata is retained (design §7).
    assert record.agent_id not in registry._evicted_agents
    cast(AsyncMock, record.runtime.close).assert_awaited()


@pytest.mark.asyncio
async def test_unwind_rollback_closes_unretained_after_failed_admission() -> None:
    # Timing B: a failed admission after the old run was stopped closes an
    # unretained instance via the reused-claim rollback (apply_policy=True).
    clock = _ManualClock(0)
    registry, _root = _root_registry(clock)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=0,
        max_tombstones=32,
    )
    record, _run_record = _unwind_record(
        registry, clock, retained=False, completion_done=True
    )
    record.reserved_for_replacement = False
    await registry._rollback_reused_claim(record, None, apply_policy=True)
    await asyncio.gather(*tuple(registry._teardown_tasks))

    assert record.agent_id not in registry._agent_records
    # Zero tombstone TTL: no terminal metadata is retained (design §7).
    assert record.agent_id not in registry._evicted_agents
    cast(AsyncMock, record.runtime.close).assert_awaited()


@pytest.mark.asyncio
async def test_unwind_keeps_retained_instance_resident() -> None:
    # A retained instance returning to idle on unwind stays resident.
    clock = _ManualClock(0)
    registry, _root = _root_registry(clock)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    record, _run_record = _unwind_record(
        registry, clock, retained=True, completion_done=True
    )
    async with registry._registry_lock:
        registry._clear_replacement_locked(record)
    await asyncio.gather(*tuple(registry._teardown_tasks))

    assert record.agent_id in registry._agent_records
    assert record.state is _AgentState.IDLE
    assert record.agent_id not in registry._evicted_agents


@pytest.mark.asyncio
async def test_plain_reuse_failure_keeps_unretained_fixture_resident() -> None:
    # A plain (non-replacement) reuse failure restores idle residency even for
    # an unretained fixture record; the unwind close is replacement-only.
    clock = _ManualClock(0)
    registry, _root = _root_registry(clock)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=0,
        max_tombstones=32,
    )
    record = _record("agent-1", clock=clock, state=_AgentState.RUNNING, retained=False)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    await registry._rollback_reused_claim(record, 0, apply_policy=False)
    assert record.agent_id in registry._agent_records
    assert record.state is _AgentState.IDLE
    assert record.agent_id not in registry._evicted_agents


# ---------------------------------------------------------------------------
# 9. Tombstone TTL (manual clock) + insertion-time cap overflow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tombstone_insertion_cap_evicts_oldest_first() -> None:
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock, ttl=10)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=2,
    )
    async with registry._registry_lock:
        for index in range(4):
            agent_id = f"agent-{index}"
            registry._insert_tombstone_locked(
                agent_id,
                _tombstone(
                    agent_id,
                    reason="closed",
                    closed_at=float(index),
                    expires_at=float(index) + 1000,
                ),
            )
    # Cap is 2: the two oldest (agent-0, agent-1) were evicted at insertion.
    assert set(registry._evicted_agents) == {"agent-2", "agent-3"}
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 10. Ephemeral-only reaper scheduling (no idle records)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_schedules_for_tombstone_deadline_with_no_idle_records() -> None:
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1", reason="closed", closed_at=0, expires_at=100
    )
    await _rearm(registry)
    # No idle records, but the earliest tombstone expiry arms the reaper.
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 100
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 11. No read-based deadline extension
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reads_do_not_extend_tombstone_deadline() -> None:
    clock = _ManualClock(0)
    registry = _retention_registry(clock)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1", reason="closed", closed_at=0, expires_at=100
    )
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")

    assert any(s.agent_id == "agent-1" for s in await registry.check_agents())
    result = await registry.get_agent_result("agent-1", "run-1")
    assert result is not None and result.run_id == "run-1"
    clock.advance(50)
    assert any(s.agent_id == "agent-1" for s in await registry.check_agents())
    result = await registry.get_agent_result("agent-1", "run-1")
    assert result is not None and result.run_id == "run-1"
    # Reads never moved the deadline.
    assert registry._evicted_agents["agent-1"].expires_at == 100
    clock.advance(50)
    assert not any(s.agent_id == "agent-1" for s in await registry.check_agents())
    assert registry._evicted_agents["agent-1"].expires_at == 100


# ---------------------------------------------------------------------------
# 12. Expiry during transcript reads
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_expired_tombstone_behaves_as_absent_for_transcript_reads(
    tmp_path: Any,
) -> None:
    clock = _ManualClock(0)
    registry = _retention_registry(clock)
    root = MagicMock()
    root.agent_loop.session_id = "parent"
    root.agent_loop._session_generation = 7
    root.agent_loop.session_logger.session_dir = tmp_path / "parent"
    root.agent_loop.session_logger.session_metadata = MagicMock(child_sessions=[])
    registry._root = root
    registry._generation_identity = ("parent", 7)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1",
        reason="closed",
        closed_at=0,
        expires_at=100,
        parent_identity=("parent", 7),
    )
    # Live tombstone: disk-read resolution falls through (returns a snapshot or
    # raises only on a missing link), never as unknown.
    live_snapshot = await registry.resolve_transcript_read("agent-1")
    assert live_snapshot is not None
    # Expired: behaves as absent everywhere.
    clock.advance(100)
    with pytest.raises(UnknownAgentError):
        await registry.resolve_transcript_read("agent-1")
    with pytest.raises(UnknownAgentError):
        await registry.resolve_resident_transcript_read("agent-1")
    assert not await registry.transcript_read_is_current(live_snapshot)
    assert not any(s.agent_id == "agent-1" for s in await registry.check_agents())


# ---------------------------------------------------------------------------
# 14. Deterministic summary ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_summaries_deterministic_grouped_ordering() -> None:
    clock = _ManualClock(0)
    registry = _retention_registry(clock)
    # Active: oldest launch first (agent-1 @1, agent-3 @3, agent-2 @5).
    for agent_id, started in (("agent-1", 1), ("agent-2", 5), ("agent-3", 3)):
        record = _record(agent_id, clock=clock, state=_AgentState.RUNNING)
        record.current_run = _run(
            record, f"{agent_id}-run", started_at=started, done=False
        )
        registry._agent_records[agent_id] = record
    # Retained idle: most recently completed first (agent-4 @20, agent-5 @10).
    for agent_id, idle_since in (("agent-4", 20), ("agent-5", 10)):
        record = _record(agent_id, clock=clock, state=_AgentState.IDLE)
        record.idle_since = idle_since
        registry._agent_records[agent_id] = record
    # Recent: newest closure first (agent-6 @200, agent-7 @100).
    registry._evicted_agents["agent-6"] = _tombstone(
        "agent-6", reason="closed", closed_at=200, expires_at=None
    )
    registry._evicted_agents["agent-7"] = _tombstone(
        "agent-7", reason="closed", closed_at=100, expires_at=None
    )

    order = [s.agent_id for s in registry._agent_summaries()]
    assert order == [
        "agent-1",
        "agent-3",
        "agent-2",
        "agent-4",
        "agent-5",
        "agent-6",
        "agent-7",
    ]


# ---------------------------------------------------------------------------
# 16. Unwind closures publish the terminal transition
# ---------------------------------------------------------------------------


def _unwind_publish_registry(
    clock: _ManualClock,
) -> tuple[SessionRuntimeRegistry, AsyncMock]:
    """A root-bound registry with a tombstone TTL and a snapshot notifier."""
    registry, _root = _root_registry(clock)
    snapshots = AsyncMock()
    registry._notify_agents = snapshots
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    return registry, snapshots


def _assert_unwind_published_closed_transition(
    snapshots: AsyncMock, agent_id: str
) -> None:
    """The client inventory reflects the unwind closure (closed snapshot)."""
    snapshots.assert_awaited()
    payload = snapshots.await_args_list[-1].args[0]
    assert [summary.agent_id for summary in payload] == [agent_id]
    assert payload[0].availability == "closed"


@pytest.mark.asyncio
async def test_unwind_direct_close_publishes_terminal_transition() -> None:
    # Timing A: reservation released with the old run already settled.
    clock = _ManualClock(0)
    registry, snapshots = _unwind_publish_registry(clock)
    record, _run_record = _unwind_record(
        registry, clock, retained=False, completion_done=True
    )
    async with registry._registry_lock:
        registry._clear_replacement_locked(record)
    await asyncio.gather(*tuple(registry._teardown_tasks))

    # The closed tombstone is stored (TTL configured) and published.
    assert registry._evicted_agents[record.agent_id].reason == "closed"
    _assert_unwind_published_closed_transition(snapshots, record.agent_id)
    await registry.drain_children()


@pytest.mark.asyncio
async def test_unwind_pending_close_publishes_terminal_transition() -> None:
    # Timing A-pending: the unwind helper closes once the old run settles.
    clock = _ManualClock(0)
    registry, snapshots = _unwind_publish_registry(clock)
    record, run = _unwind_record(registry, clock, retained=False, completion_done=False)
    async with registry._registry_lock:
        registry._clear_replacement_locked(record)
    run.completion_task.set_result(None)
    # The unwind task schedules the publication while it runs; a second
    # gather joins the scheduled publish.
    await asyncio.gather(*tuple(registry._teardown_tasks))
    await asyncio.gather(*tuple(registry._teardown_tasks))

    _assert_unwind_published_closed_transition(snapshots, record.agent_id)
    await registry.drain_children()


@pytest.mark.asyncio
async def test_unwind_rollback_close_publishes_terminal_transition() -> None:
    # Timing B: the reused-claim rollback closes and publishes.
    clock = _ManualClock(0)
    registry, snapshots = _unwind_publish_registry(clock)
    record, _run_record = _unwind_record(
        registry, clock, retained=False, completion_done=True
    )
    record.reserved_for_replacement = False
    await registry._rollback_reused_claim(record, None, apply_policy=True)
    await asyncio.gather(*tuple(registry._teardown_tasks))

    _assert_unwind_published_closed_transition(snapshots, record.agent_id)
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 17. Public emission filters expired tombstones
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_emitted_update_filters_expired_tombstones() -> None:
    clock = _ManualClock(0)
    registry = _retention_registry(clock)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    snapshots = AsyncMock()
    registry._notify_agents = snapshots
    # A live tombstone and an already-expired tombstone.
    registry._evicted_agents["agent-live"] = _tombstone(
        "agent-live", reason="closed", closed_at=0, expires_at=100
    )
    registry._evicted_agents["agent-stale"] = _tombstone(
        "agent-stale", reason="closed", closed_at=0, expires_at=50
    )
    clock.advance(60)

    await registry._emit_agents_update()

    # The published inventory never carries expired tombstones...
    assert snapshots.await_count == 1
    payload = snapshots.await_args_list[-1].args[0]
    assert {summary.agent_id for summary in payload} == {"agent-live"}
    assert {s.agent_id for s in await registry.check_agents()} == {"agent-live"}
    # ...while the internal unfiltered snapshot keeps the frozen tombstone
    # until the reaper prunes it.
    assert {s.agent_id for s in registry._agent_summaries()} == {
        "agent-live",
        "agent-stale",
    }


# ---------------------------------------------------------------------------
# 18. Reaper publishes after metadata-only pruning
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_publishes_after_metadata_only_pruning() -> None:
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    notify = cast(AsyncMock, registry._notify_agents)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1", reason="closed", closed_at=0, expires_at=100
    )
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 100
    assert notify.await_count == 0

    clock.advance(100)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await asyncio.wait_for(reaper, timeout=1)

    # The expired tombstone was pruned with no eviction victims, and the
    # reaper published the reduced (empty) inventory.
    assert "agent-1" not in registry._evicted_agents
    assert notify.await_count == 1
    assert notify.await_args_list[-1].args[0] == []
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 19. Active ordering uses the stable first-admission timestamp
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_active_ordering_uses_stable_first_admission_timestamp() -> None:
    clock = _ManualClock(0)
    registry = _retention_registry(clock)
    # agent-1 was admitted first (t=1) but is a reused instance whose earliest
    # surviving run start (t=500) is later than agent-2's: run-history pruning
    # dropped the original run, so the old min-run-start key misordered it.
    reused = _record("agent-1", clock=clock, state=_AgentState.RUNNING)
    reused.first_admitted_at = 1.0
    reused.run_history.append(_run(reused, "agent-1-old", started_at=500))
    reused.current_run = _run(reused, "agent-1-new", started_at=900, done=False)
    newer = _record("agent-2", clock=clock, state=_AgentState.RUNNING)
    newer.first_admitted_at = 100.0
    newer.current_run = _run(newer, "agent-2-run", started_at=100, done=False)
    # A record admitted before the timestamp was tracked falls back to the
    # earliest run start.
    legacy = _record("agent-3", clock=clock, state=_AgentState.RUNNING)
    legacy.current_run = _run(legacy, "agent-3-run", started_at=50, done=False)
    registry._agent_records.update({
        reused.agent_id: reused,
        newer.agent_id: newer,
        legacy.agent_id: legacy,
    })

    order = [s.agent_id for s in registry._agent_summaries()]
    assert order == ["agent-1", "agent-3", "agent-2"]


# ---------------------------------------------------------------------------
# 20. Reaper publishes whenever the inventory changed (pruning + victims
#     that lost eligibility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_publishes_when_pruning_lands_but_victims_lose_eligibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The re-review reproduction: the reaper schedules victims, pruning
    # removes expired tombstones, and the victims then lose eligibility —
    # no eviction notification fires, yet the inventory changed and must be
    # published regardless of eviction-task existence or outcomes.
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    notify = cast(AsyncMock, registry._notify_agents)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=100,
        max_tombstones=32,
    )
    # An expired tombstone (pruned by this run) and an idle record whose TTL
    # elapses at the same instant (scheduled as a victim).
    registry._evicted_agents["agent-old"] = _tombstone(
        "agent-old", reason="closed", closed_at=0, expires_at=10
    )
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    # The victim loses eligibility after the reaper selected it (it starts
    # running again before its eviction task executes).
    original_evict = registry._evict_agent

    async def ineligible_evict(
        agent_id: str,
        reason: str,
        notifications: list[
            tuple[AgentRecord, AgentEviction, _AgentTombstone | None, object]
        ]
        | None = None,
    ) -> bool:
        resident = registry._agent_records.get(agent_id)
        if resident is not None:
            resident.state = _AgentState.RUNNING
        return await original_evict(agent_id, reason, notifications)

    monkeypatch.setattr(registry, "_evict_agent", ineligible_evict)
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    clock.advance(10)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await asyncio.wait_for(reaper, timeout=1)

    # Pruning removed the tombstone and the victim stayed resident, but the
    # inventory changed: the reduced inventory was published anyway.
    assert "agent-old" not in registry._evicted_agents
    assert "agent-1" in registry._agent_records
    assert notify.await_count == 1
    call = notify.await_args_list[-1]
    assert [summary.agent_id for summary in call.args[0]] == ["agent-1"]
    assert call.args[1] == []
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 21. Metadata-deadline scheduling survives stalled eviction cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_prunes_tombstone_on_schedule_while_eviction_cleanup_stalls() -> (
    None
):
    # The re-review reproduction: with gated (stalled) eviction cleanup, a
    # tombstone expiring at clock 110 stays stored and the reaper stays
    # blocked — unless the schedule was handed to a fresh timer before the
    # cleanup await. Pruning must fire on schedule regardless.
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=32,
    )
    # A tombstone expiring at clock 110 (the metadata deadline).
    registry._evicted_agents["agent-old"] = _tombstone(
        "agent-old", reason="closed", closed_at=10, expires_at=110
    )
    # An idle record whose TTL elapses at clock 10 (wakes the reaper).
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def stalled_close() -> None:
        close_started.set()
        await close_release.wait()

    record.runtime.close = stalled_close
    await _rearm(registry)
    # Earliest deadline: the idle record at 10 (the tombstone is at 110).
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    clock.advance(10)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    # The eviction starts and its runtime close stalls.
    await asyncio.wait_for(close_started.wait(), timeout=1)
    # The schedule was handed to a fresh timer before the cleanup await: the
    # tombstone deadline (110) is armed even though the reaper is blocked.
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 100
    clock.advance(100)
    wakeup.gates[1].set()
    # The fresh timer fires at the tombstone deadline: pruning happens on
    # schedule while the eviction cleanup is still gated, the reduced
    # inventory is published, and the timer rearms for the remaining
    # tombstone deadline.
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 900
    assert "agent-old" not in registry._evicted_agents
    notify = cast(AsyncMock, registry._notify_agents)
    payloads = [call.args[0] for call in notify.await_args_list]
    assert payloads
    assert all(
        not any(summary.agent_id == "agent-old" for summary in payload)
        for payload in payloads
    )
    # Release the stalled cleanup; the blocked reaper finishes normally.
    close_release.set()
    await asyncio.wait_for(reaper, timeout=1)
    assert "agent-1" not in registry._agent_records
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 22. Zero-TTL eviction notifications are release-invalidatable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zero_ttl_eviction_notification_is_released_with_the_agent() -> None:
    # The re-review reproduction: a zero-TTL eviction stores no tombstone, so
    # its notification was authorized unconditionally — after release_agent
    # purged the record/result and completed, a pending eviction's cleanup
    # resumption still injected a parent message and published an update.
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock, ttl=10)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=0,
        max_tombstones=32,
    )
    root.turns.active_turn = MagicMock()
    root.agent_loop._pending_injected_messages = []
    record = _record("agent-1", clock=clock)
    record.parent_session_id = "root"
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def stalled_close() -> None:
        close_started.set()
        await close_release.wait()

    record.runtime.close = stalled_close
    notify = cast(AsyncMock, registry._notify_agents)
    eviction = asyncio.create_task(registry._evict_agent("agent-1", "ttl"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    # The record detached and no tombstone is stored (zero TTL)...
    assert "agent-1" not in registry._agent_records
    assert "agent-1" not in registry._evicted_agents
    # ...but the release-invalidatable eviction authority is registered.
    assert "agent-1" in registry._pending_evictions
    # Release completes while the eviction cleanup is still gated.
    await registry.release_agent("agent-1")
    assert "agent-1" not in registry._pending_evictions
    # The cleanup resumes: the stale eviction must not notify.
    close_release.set()
    assert await eviction
    assert notify.await_count == 0
    assert root.agent_loop._pending_injected_messages == []
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 23. Reaper finalizer rearm failures surface while a loop is running
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reaper_finally_rearm_failure_surfaces_with_running_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The re-review reproduction: the loop-less destruction guard in the
    # reaper finalizer silently swallowed a fault-injected rearm RuntimeError
    # while a loop was running — the task looked successful and the timer
    # was lost. With a running loop the failure must surface.
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._evicted_agents["agent-old"] = _tombstone(
        "agent-old", reason="closed", closed_at=0, expires_at=50
    )
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 50

    def faulty_rearm(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected rearm failure")

    monkeypatch.setattr(registry, "_rearm_reaper_locked", faulty_rearm)
    clock.advance(50)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    with pytest.raises(RuntimeError, match="injected rearm failure"):
        await asyncio.wait_for(reaper, timeout=1)
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 24. Pending-eviction authorities settle on every path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_capped_evictions_leave_no_pending_authorities() -> None:
    # The round-4 reproduction: with max_tombstones=1, five completed
    # positive-TTL evictions each stored a tombstone (authorized by
    # tombstone identity) but never consumed the pending-eviction
    # authority, and the insertion cap dropped tombstones without purging
    # their authorities — five completed evictions left five authorities.
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock, ttl=10)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=1,
    )
    for index in range(5):
        agent_id = f"agent-{index}"
        record = _record(agent_id, clock=clock)
        registry._agent_records[agent_id] = record
        registry._children[record.session_id] = record.runtime
        assert await registry._evict_agent(agent_id, "ttl")
        # The eviction notification settled: its authority is consumed and
        # the insertion cap holds exactly one tombstone.
        assert agent_id not in registry._pending_evictions
        assert len(registry._evicted_agents) == 1
    assert registry._pending_evictions == {}
    assert set(registry._evicted_agents) == {"agent-4"}
    await registry.drain_children()


@pytest.mark.asyncio
async def test_positive_ttl_eviction_consumes_its_authority() -> None:
    # A positive-TTL eviction is authorized by its stored tombstone; the
    # successful notification must also consume the release-invalidatable
    # authority registered at eviction commit.
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock, ttl=10)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=32,
    )
    record = _record("agent-1", clock=clock)
    registry._agent_records["agent-1"] = record
    registry._children[record.session_id] = record.runtime
    notify = cast(AsyncMock, registry._notify_agents)
    assert await registry._evict_agent("agent-1", "ttl")
    # The tombstone is stored and authorized the delivered notification...
    assert "agent-1" in registry._evicted_agents
    assert notify.await_count == 1
    await_args = notify.await_args
    assert await_args is not None
    evictions = await_args.args[1]
    assert [eviction.agent_id for eviction in evictions] == ["agent-1"]
    # ...and the settlement consumed the pending-eviction authority.
    assert "agent-1" not in registry._pending_evictions
    await registry.drain_children()


@pytest.mark.asyncio
async def test_tombstone_cap_removal_purges_pending_eviction_authority() -> None:
    # The round-4 reproduction (cap path): the insertion cap dropped the
    # oldest tombstone without purging its pending-eviction authority, so
    # a stalled eviction whose metadata was capped kept its authority.
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock, ttl=10)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=1,
    )
    record = _record("agent-1", clock=clock)
    registry._agent_records["agent-1"] = record
    registry._children[record.session_id] = record.runtime
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def stalled_close() -> None:
        close_started.set()
        await close_release.wait()

    record.runtime.close = stalled_close
    eviction = asyncio.create_task(registry._evict_agent("agent-1", "ttl"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    # The stalled eviction stored its tombstone and registered its
    # release-invalidatable authority (the notification is still pending).
    assert "agent-1" in registry._evicted_agents
    assert "agent-1" in registry._pending_evictions
    # Inserting a newer tombstone caps out the oldest (agent-1): the cap
    # removal must purge that agent's authority along with the tombstone.
    async with registry._registry_lock:
        registry._insert_tombstone_locked(
            "agent-2",
            _tombstone("agent-2", reason="closed", closed_at=1, expires_at=1001),
        )
    assert set(registry._evicted_agents) == {"agent-2"}
    assert "agent-1" not in registry._pending_evictions
    # The stalled eviction settles: its capped-out metadata no longer
    # authorizes a notification.
    close_release.set()
    assert await asyncio.wait_for(eviction, timeout=1)
    notify = cast(AsyncMock, registry._notify_agents)
    assert notify.await_count == 0
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 28. /clear with only resident agents publishes the reduced inventory
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_with_only_resident_agents_publishes_reduced_inventory() -> None:
    # The blocking-finding reproduction: retiring a generation that contains
    # only resident agents (a retained idle record plus a running one) removed
    # the records without publishing, so the browser kept ghost rows until the
    # next unrelated update. Anything removed — records as well as tombstones —
    # publishes the reduced inventory, and the runtime closes stay tracked.
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock, generation=1)
    notify = cast(AsyncMock, registry._notify_agents)
    idle = _record("agent-1", clock=clock, retained=True)
    idle.parent_session_id = "root"
    running = _record("agent-2", clock=clock, state=_AgentState.RUNNING)
    running.parent_session_id = "root"
    running.current_run = _run(running, "agent-2-run", started_at=0, done=False)
    registry._agent_records["agent-1"] = idle
    registry._children[idle.session_id] = idle.runtime
    registry._agent_records["agent-2"] = running
    registry._children[running.session_id] = running.runtime
    # A tombstone already belonging to the next generation survives retirement,
    # keeping the published reduced inventory non-empty.
    registry._evicted_agents["agent-3"] = _tombstone(
        "agent-3",
        reason="closed",
        closed_at=0,
        expires_at=None,
        parent_identity=("root", 2),
    )
    notify.reset_mock()

    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    await asyncio.gather(*tuple(registry._teardown_tasks))

    # Both resident records were removed and their runtime closes tracked.
    assert not registry._agent_records
    assert not registry._children
    cast(AsyncMock, idle.runtime.close).assert_awaited_once()
    cast(AsyncMock, running.runtime.close).assert_awaited_once()
    # The reduced inventory was published: only the surviving next-generation
    # tombstone remains, so the browser drops the ghost rows.
    notify.assert_awaited()
    await_args = notify.await_args
    assert await_args is not None
    payload = await_args.args[0]
    assert [summary.agent_id for summary in payload] == ["agent-3"]
    await registry.drain_children()


# ---------------------------------------------------------------------------
# Working-set spawn handoff (AM-D2): the dispatcher hands each child its named
# working-set files + current outlines, shares one revision registry across the
# session tree, and leaks no state across children.
# ---------------------------------------------------------------------------


async def _file_state_registry(
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeBackend,
    *,
    cwd: Path,
    seed: dict[str, bytes] | None = None,
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    """A retention registry whose parent owns a working set and the tree registry."""
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "read_file", "edit"]),
        backend=FakeBackend(),
        cwd=cwd,
        file_revision_registry=FileRevisionRegistry(),
    )
    for path, content in (seed or {}).items():
        parent.file_state_context.observe_edit(path, full_revision(content))
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


async def _spawned_child(
    registry: SessionRuntimeRegistry, parent: AgentLoop, task: str
) -> AgentLoop:
    """Spawn a background child through the registry and await its completion."""
    ctx = InvokeContext(tool_call_id=f"spawn-{task}", session_id=parent.session_id)
    launch = await _background_result(registry, TaskArgs(task=task), ctx)
    assert launch.agent_id is not None
    record = registry._agent_records[launch.agent_id]
    run = record.current_run
    assert run is not None
    await run.completion_task
    return record.runtime.agent_loop


def _handoff_blocks(loop: AgentLoop) -> list[LLMMessage]:
    return [
        message
        for message in loop.messages
        if message.injected
        and message.role == Role.user
        and "file_state_handoff" in str(message.content)
    ]


def _state_blocks(loop: AgentLoop) -> list[LLMMessage]:
    return [
        message
        for message in loop.messages
        if message.injected
        and message.role == Role.user
        and "<file_state " in str(message.content)
    ]


def _edit_call(target: Path, old: str, new: str, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="edit",
            arguments=json.dumps({
                "file_path": str(target),
                "old_string": old,
                "new_string": new,
            }),
        ),
    )


@pytest.mark.asyncio
async def test_spawn_handoff_gives_child_working_set_and_outlines(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "mod.py"
    target.write_text("class Foo:\n    pass\n\n\ndef bar():\n    return 1\n")
    backend = FakeBackend([[mock_llm_chunk(content="child done")]])
    registry, parent = await _file_state_registry(
        monkeypatch, backend, cwd=tmp_path, seed={str(target): target.read_bytes()}
    )
    try:
        child = await _spawned_child(registry, parent, "work")

        # The child receives the parent's named working set: one handoff
        # block ahead of the task brief, with the path, its last-seen
        # revision, and the current outline.
        handoffs = _handoff_blocks(child)
        assert len(handoffs) == 1
        handoff = handoffs[0]
        assert handoff.role == Role.user and handoff.injected is True
        content = str(handoff.content)
        assert str(target) in content
        seeded = parent.file_state.known_revision(str(target))
        assert seeded is not None
        assert seeded.revision in content
        assert "class Foo (lines 1-2)" in content
        assert "function bar (lines 5-6)" in content
        # The handoff precedes the task brief in the transcript.
        user_messages = [
            message for message in child.messages if message.role == Role.user
        ]
        assert user_messages[0] is handoff

        # The child's store is seeded (a deep copy) and the revision registry
        # is the one shared instance of the session tree.
        assert child.file_state.known_revision(str(target)) == seeded
        assert child.file_state.outline_for(str(target)) is not None
        assert child.file_revision_registry is parent.file_revision_registry
        assert child.file_state.entry(str(target)) is not parent.file_state.entry(
            str(target)
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_spawn_handoff_absent_when_parent_has_no_working_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    backend = FakeBackend([[mock_llm_chunk(content="child done")]])
    registry, parent = await _file_state_registry(monkeypatch, backend, cwd=tmp_path)
    try:
        child = await _spawned_child(registry, parent, "work")
        assert _handoff_blocks(child) == []
        assert len(child.file_state) == 0
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_child_state_never_leaks_to_parent_siblings_or_after_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "mod.py"
    target.write_text("a = 1\n")
    other = tmp_path / "other.py"
    other.write_text("b = 2\n")
    backend = FakeBackend([[mock_llm_chunk(content="child done")]])
    registry, parent = await _file_state_registry(
        monkeypatch, backend, cwd=tmp_path, seed={str(target): b"a = 1\n"}
    )
    try:
        child_a = await _spawned_child(registry, parent, "work")
        # The child observes another file (its own working-set extension).
        child_a.file_state_context.observe_edit(str(other), full_revision(b"b = 2\n"))
        child_a.file_state_context.flush_blocks()
        assert str(other) in child_a.file_state

        # A sibling spawned afterwards sees only the parent's working set.
        child_b = await _spawned_child(registry, parent, "more")
        assert str(target) in child_b.file_state
        assert str(other) not in child_b.file_state
        assert str(other) not in parent.file_state

        # A child history clear resets only that child's working set; the
        # parent's set and the tree registry survive.
        await child_a.clear_history()
        assert len(child_a.file_state) == 0
        assert str(target) in parent.file_state
        assert str(target) in child_b.file_state
        tree_registry = parent.file_revision_registry
        assert tree_registry is not None
        assert tree_registry.known_revision(str(target)) is not None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_external_bump_from_sibling_reported_in_next_block(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n")
    original = full_revision(b"def f():\n    return 1\n")
    edited = full_revision(b"def f():\n    return 2\n")
    backend = FakeBackend([
        # Worker A's turn: edit the file, then finish.
        [
            mock_llm_chunk(
                tool_calls=[_edit_call(target, "return 1", "return 2", "edit")]
            )
        ],
        [mock_llm_chunk(content="A done")],
        # Worker B's turn (driven manually below): one text reply.
        [mock_llm_chunk(content="B done")],
    ])
    registry, parent = await _file_state_registry(
        monkeypatch,
        backend,
        cwd=tmp_path,
        seed={str(target): b"def f():\n    return 1\n"},
    )
    child_b: AgentLoop | None = None
    try:
        # Worker B is created (and seeded with the parent's pre-edit
        # baseline) before worker A runs, so its baseline is deterministic.
        child_b = await registry._runtime_factory.create_child(parent, "worker")
        assert _matches_revision(
            child_b.file_state.known_revision(str(target)), original
        )
        assert child_b.file_revision_registry is parent.file_revision_registry

        # Worker A edits the file through its own loop: the edit result
        # publishes the post-edit revision to the shared registry.
        child_a = await _spawned_child(registry, parent, "edit")
        assert _matches_revision(child_a.file_state.known_revision(str(target)), edited)
        tree_registry = parent.file_revision_registry
        assert tree_registry is not None
        assert _matches_revision(tree_registry.known_revision(str(target)), edited)

        # Worker B's next turn reports the external bump: the registry moved
        # past B's baseline, which B never observed itself.
        async for _event in child_b.act("look"):
            pass
        blocks = _state_blocks(child_b)
        assert len(blocks) == 1
        content = str(blocks[0].content)
        assert "source='external'" in content
        assert str(target) in content
        # The fresh on-disk outline is included in the bump report.
        assert "function f (lines 1-2)" in content
        # The bumped revision is adopted: no repeated report on the next flush.
        assert _matches_revision(child_b.file_state.known_revision(str(target)), edited)
        assert child_b.file_state_context.flush_blocks() == []
    finally:
        if child_b is not None:
            await child_b.aclose()
        await registry.drain_children()
        await parent.aclose()
