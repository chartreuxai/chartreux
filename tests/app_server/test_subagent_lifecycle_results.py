"""Result availability and reuse-admission behavior tests for the subagent lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import AgentRecord, SessionRuntimeRegistry
from chartreux.core.subagents import (
    AgentClosedError,
    AgentEvictedError,
    RetentionPolicy,
    RunStatus,
    TaskArgs,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _background_result,
    _evicted_tombstone,
    _GatedWakeup,
    _ManualClock,
    _reaper_registry,
    _rearm,
    _record,
    _root_registry,
    _run,
    _stored_result,
    _tombstone,
)


def _reuse_args(agent_id: str, **extra: Any) -> TaskArgs:
    return TaskArgs(
        task="reuse", identity="coder", agent_id=agent_id, background=True, **extra
    )


def _reuse_ctx(agent_id: str) -> InvokeContext:
    return InvokeContext(tool_call_id=f"reuse-{agent_id}", session_id="root")


@pytest.mark.asyncio
async def test_tombstone_ttl_expires_metadata_but_keeps_result() -> None:
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
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")
    assert await registry._evict_agent("agent-1", "ttl")

    tombstone = registry._evicted_agents["agent-1"]
    assert tombstone.expires_at == 100
    # Live until the deadline.
    assert any(s.agent_id == "agent-1" for s in await registry.check_agents())
    # The eviction insertion armed the reaper for the tombstone deadline
    # (no idle records); no manual rearm needed.
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 100
    clock.advance(100)
    # Expired metadata behaves as absent in the public listing...
    assert not any(s.agent_id == "agent-1" for s in await registry.check_agents())
    # ...but the frozen internal snapshot remains until the reaper prunes it.
    assert registry._agent_summaries()[0].agent_id == "agent-1"
    # Reaper prunes the expired tombstone at its deadline.
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await asyncio.wait_for(reaper, timeout=1)
    assert "agent-1" not in registry._evicted_agents
    # Metadata expiry never consumed the result.
    result = await registry.get_agent_result("agent-1", "run-1")
    assert result is not None and result.run_id == "run-1"
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 13. /clear with only tombstones publishes and preserves results
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_clear_with_only_tombstones_publishes_and_preserves_results() -> None:
    clock = _ManualClock(0)
    registry, root = _reaper_registry(clock, ttl=10, generation=1)
    notify = cast(AsyncMock, registry._notify_agents)
    # An old-generation closed tombstone plus a retrievable result.
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1",
        reason="closed",
        closed_at=0,
        expires_at=None,
        parent_identity=("root", 1),
    )
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")
    registry._latest_run_ids["agent-1"] = "run-1"
    notify.reset_mock()

    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    await asyncio.gather(*tuple(registry._teardown_tasks))

    # Old-generation tombstone removed; the new inventory was published.
    assert "agent-1" not in registry._evicted_agents
    notify.assert_awaited()
    # Results and latest-run indexes survive /clear.
    result = await registry.get_agent_result("agent-1", "run-1")
    assert result is not None and result.run_id == "run-1"
    assert registry._latest_run_ids["agent-1"] == "run-1"
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 15. §3.1 reuse matrix: closed / evicted / unknown selection by reason
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reuse_matrix_selects_typed_error_by_tombstone_reason() -> None:
    clock = _ManualClock(0)
    registry, _root = _root_registry(clock)
    registry._evicted_agents["closed-agent"] = _tombstone(
        "closed-agent", reason="closed", closed_at=0, expires_at=None
    )
    registry._evicted_agents["evicted-agent"] = _evicted_tombstone(
        "evicted-agent", run_id=None, status=None
    )

    with pytest.raises(AgentClosedError, match="closed"):
        await _background_result(
            registry, _reuse_args("closed-agent"), _reuse_ctx("closed-agent")
        )
    # The evicted subtype is reserved for TTL/cap eviction and is a subtype of
    # the closed error.
    with pytest.raises(AgentEvictedError) as evicted:
        await _background_result(
            registry, _reuse_args("evicted-agent"), _reuse_ctx("evicted-agent")
        )
    assert isinstance(evicted.value, AgentClosedError)
    assert not type(evicted.value) is AgentClosedError
    with pytest.raises(UnknownAgentError):
        await _background_result(
            registry, _reuse_args("missing"), _reuse_ctx("missing")
        )


@pytest.mark.asyncio
async def test_reuse_after_release_and_after_metadata_prune_is_unknown() -> None:
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
    # Released tombstone: reuse behaves as unknown.
    registry._evicted_agents["released"] = _evicted_tombstone(
        "released", run_id="run-1", status=RunStatus.COMPLETED
    )
    registry._result_store[("released", "run-1")] = _stored_result("released", "run-1")
    await registry.release_agent("released")
    with pytest.raises(UnknownAgentError):
        await _background_result(
            registry, _reuse_args("released"), _reuse_ctx("released")
        )

    # Metadata-expired (pruned) tombstone: reuse behaves as unknown.
    record = _record("pruned", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    assert await registry._evict_agent("pruned", "ttl")
    with pytest.raises(AgentEvictedError):
        await _background_result(registry, _reuse_args("pruned"), _reuse_ctx("pruned"))
    # Prune the expired tombstone via the reaper (armed at eviction
    # insertion); it then behaves as unknown.
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 100
    clock.advance(100)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await asyncio.wait_for(reaper, timeout=1)
    assert "pruned" not in registry._evicted_agents
    with pytest.raises(UnknownAgentError):
        await _background_result(registry, _reuse_args("pruned"), _reuse_ctx("pruned"))
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 29. Reuse of a metadata-expired (unpruned) tombstone behaves as unknown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reuse_of_metadata_expired_unpruned_tombstone_is_unknown() -> None:
    # The warning reproduction: between the tombstone deadline and the reaper's
    # prune, both reuse admission sites read the raw tombstone and raised the
    # typed closed/evicted errors. Per the §3.1 matrix, metadata-expired
    # behaves as unknown for reuse — at BOTH admission sites — until the
    # reaper prunes the tombstone.
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
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    assert await registry._evict_agent("agent-1", "ttl")
    # The live tombstone still selects the typed evicted error.
    with pytest.raises(AgentEvictedError):
        await _background_result(
            registry, _reuse_args("agent-1"), _reuse_ctx("agent-1")
        )

    # Past the metadata deadline but before the reaper prunes: reuse behaves
    # as unknown at both admission sites, not as the closed/evicted errors.
    clock.advance(101)
    with pytest.raises(UnknownAgentError):
        await _background_result(
            registry, _reuse_args("agent-1"), _reuse_ctx("agent-1")
        )
    with pytest.raises(UnknownAgentError):
        await _background_result(
            registry,
            TaskArgs(
                task="replace the evicted run",
                agent_id="agent-1",
                background=True,
                replace_run=True,
            ),
            _reuse_ctx("agent-1"),
        )

    # After the reaper prunes the tombstone, reuse is still unknown.
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 100
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await asyncio.wait_for(reaper, timeout=1)
    assert "agent-1" not in registry._evicted_agents
    with pytest.raises(UnknownAgentError):
        await _background_result(
            registry, _reuse_args("agent-1"), _reuse_ctx("agent-1")
        )
    await registry.drain_children()


# ---------------------------------------------------------------------------
# 30. Eviction notifications carry the latest run id and outcome
# ---------------------------------------------------------------------------


def _evicted_record_with_run(
    registry: SessionRuntimeRegistry,
    agent_id: str,
    run_id: str,
    status: RunStatus,
    *,
    clock: _ManualClock,
) -> AgentRecord:
    """An idle resident record whose latest run settled with ``status``."""
    record = _record(agent_id, clock=clock)
    record.parent_session_id = "root"
    record.idle_since = 0
    run = _run(record, run_id, started_at=0, completed_at=10, status=status)
    record.run_history.append(run)
    record.latest_run_id = run_id
    record.last_run_status = status
    registry._agent_records[agent_id] = record
    registry._children[record.session_id] = record.runtime
    return record


@pytest.mark.asyncio
async def test_eviction_notification_carries_run_id_and_outcome() -> None:
    # Design §5: the single eviction notification names the agent alongside
    # the handle, run, and outcome.
    clock = _ManualClock(10)
    registry, root = _reaper_registry(clock, ttl=1)
    root.turns.active_turn = MagicMock()
    root.agent_loop._pending_injected_messages = []
    _evicted_record_with_run(
        registry, "agent-1", "run-1", RunStatus.COMPLETED, clock=clock
    )

    assert await registry._evict_agent("agent-1", "ttl")

    messages = root.agent_loop._pending_injected_messages
    assert len(messages) == 1
    assert messages[0].content == (
        "Background agent coder-1 (agent-1, run run-1) evicted "
        "(ttl, idle 10s, last run completed); "
        "launch a new agent for further work."
    )
    await registry.drain_children()


@pytest.mark.asyncio
async def test_batched_eviction_notification_identifies_each_run_and_outcome() -> None:
    # Design §5: each batched entry is individually identified with its own
    # run id and terminal outcome.
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, root = _reaper_registry(clock, cap=1, wakeup=wakeup)
    root.turns.active_turn = MagicMock()
    root.agent_loop._pending_injected_messages = []
    _evicted_record_with_run(
        registry, "agent-a", "run-a", RunStatus.COMPLETED, clock=clock
    )
    _evicted_record_with_run(
        registry, "agent-b", "run-b", RunStatus.FAILED, clock=clock
    )
    _evicted_record_with_run(
        registry, "agent-c", "run-c", RunStatus.CANCELLED, clock=clock
    )

    await _rearm(registry)
    await wakeup.arrivals.get()
    wakeup.gates[0].set()
    reaper_task = registry._reaper_task
    assert reaper_task is not None
    await asyncio.wait_for(reaper_task, timeout=1)

    messages = root.agent_loop._pending_injected_messages
    assert len(messages) == 1
    assert messages[0].content == (
        "Background agents evicted: "
        "coder-a (agent-a, run run-a, idle_cap, idle 10s, last run completed); "
        "coder-b (agent-b, run run-b, idle_cap, idle 10s, last run failed); "
        "launch a new agent for further work."
    )
    await registry.drain_children()
