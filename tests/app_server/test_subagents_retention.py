"""Retention behavior tests for subagents."""

from __future__ import annotations

import asyncio
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.app_server._sessions import (
    RunRecord,
    SessionRuntime,
    SessionRuntimeRegistry,
    StoredRunResult,
    _AgentState,
    _AgentTombstone,
)
from chartreux.app_server.models import PublicTurn, PublicTurnStatus
from chartreux.core.agents.models import AgentProfile
from chartreux.core.events import AssistantEvent
from chartreux.core.message_list import MessageList
from chartreux.core.subagents import (
    AgentAvailability,
    AgentClosedError,
    AgentEvictedError,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    TaskArgs,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    ImmediateFailureBackend,
    _background_result,
    _GatedWakeup,
    _idle_record,
    _ManualClock,
    _reaper_registry,
    _rearm,
    _retention_registry,
    _stored_result,
    _zero_retention_registry,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["ttl", "idle_cap"])
async def test_run_duration_clock_completion_and_eviction(reason: str) -> None:
    clock = _ManualClock(10)
    registry = _retention_registry(clock)
    record = _idle_record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    assert registry._agent_summaries()[0].run_elapsed_seconds is None
    run = RunRecord(
        run_id="run-1",
        agent_id=record.agent_id,
        profile="worker",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
        started_at=clock(),
    )
    record.current_run = run
    record.latest_run_id = run.run_id
    record.state = _AgentState.RUNNING
    clock.advance(252)
    assert registry._agent_summaries()[0].run_elapsed_seconds == 252
    run.completed_at = clock()
    run.status = RunStatus.COMPLETED
    record.state = _AgentState.FINALIZING
    clock.advance(50)
    assert registry._agent_summaries()[0].run_elapsed_seconds == 252
    record.current_run = None
    record.run_history.append(run)
    record.state = _AgentState.IDLE
    record.idle_since = run.completed_at
    summary = registry._agent_summaries()[0]
    assert summary.run_elapsed_seconds == 252
    assert summary.latest_run_id == "run-1"
    if reason == "idle_cap":
        registry._retention_policy = RetentionPolicy(
            idle_ttl_seconds=0, max_idle_agents=1, retain_agents=True
        )
        registry._agent_records["other"] = _idle_record("other", clock=clock)
    assert await registry._evict_agent(record.agent_id, reason)
    clock.advance(100)
    tombstone = registry._evicted_agents[record.agent_id].summary
    assert tombstone.run_elapsed_seconds == 252
    assert tombstone.idle_seconds == 50
    assert tombstone.latest_run_id == "run-1"
    await registry.drain_children()


@pytest.mark.asyncio
async def test_idle_summary_marks_the_latest_expired_result() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    record.latest_run_id = "run-1"
    registry._agent_records[record.agent_id] = record
    registry._expired_results.add((record.agent_id, record.latest_run_id))

    summary = (await registry.check_agents())[0]

    assert summary.result_expired


@pytest.mark.asyncio
async def test_dispatch_and_eviction_race_has_a_deterministic_winner() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    root = MagicMock()
    root.agent_loop.config.subagents.max_running_subagents = 16
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = 0
    root.turns._projector = None
    registry._root = root
    registry._generation_identity = ("root", root.agent_loop._session_generation)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=60, max_idle_agents=16, retain_agents=True
    )
    record = _idle_record("agent-1", clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None
    cast(MagicMock, record.runtime.turns.start).return_value = (
        MagicMock(id="child-turn"),
        lambda: None,
    )
    record.runtime.turns.wait_for_operation = AsyncMock(
        return_value=MagicMock(
            error=None, status=PublicTurnStatus.COMPLETED, stop_reason=None
        )
    )
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    args = TaskArgs(
        task="reuse",
        agent_type="worker",
        agent_id=record.agent_id,
        background=True,
        retain=True,
    )
    context = InvokeContext(tool_call_id="reuse", session_id="root")

    async with registry._registry_lock:
        dispatch = asyncio.create_task(_background_result(registry, args, context))
        eviction = asyncio.create_task(registry._evict_agent(record.agent_id, "ttl"))
        await asyncio.sleep(0)
    acknowledgement = await dispatch
    assert acknowledgement.status == "launched"
    assert acknowledgement.completed
    assert acknowledgement.turns_used == 0
    assert not await eviction
    assert record.agent_id in registry._agent_records

    run = record.current_run or record.run_history[-1]
    monitor = run.completion_task
    assert isinstance(monitor, asyncio.Task)
    await monitor
    assert run.status is RunStatus.FAILED
    assert run.stop_reason is RunStopReason.ERROR
    result = await registry.get_agent_result(record.agent_id, run.run_id)
    assert result is not None
    assert not result.completed
    assert "Completed subagent produced an empty response" in result.response
    assert await registry._evict_agent(record.agent_id, "ttl")
    with pytest.raises(AgentEvictedError):
        await _background_result(registry, args, context)
    await registry.drain_children()


@pytest.mark.asyncio
async def test_observation_does_not_refresh_idle_age_or_idle_task_summary() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    registry._root = MagicMock()
    registry._root.agent_loop.config.subagents.idle_ttl_seconds = 10
    record = _idle_record("agent-1", clock=clock, summary="first task")
    record.last_task_summary = "latest task"
    registry._agent_records[record.agent_id] = record

    clock.advance(7)
    summary = (await registry.check_agents())[0]
    assert summary.idle_seconds == 7
    assert summary.ttl_remaining_seconds == 3
    assert summary.current_task_summary == "latest task"
    with pytest.raises(UnknownAgentError):
        await registry.get_agent_result("agent-1", "unknown")
    clock.advance(2)
    assert (await registry.check_agents())[0].idle_seconds == 9
    assert await registry._evict_agent("agent-1", "ttl")
    await registry.drain_children()


@pytest.mark.asyncio
async def test_eviction_tombstone_contains_no_runtime_reference() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    registry._notify_agents = AsyncMock()
    notifier = cast(AsyncMock, registry._notify_agents)
    record = _idle_record("agent-1", clock=clock)
    record.effective_model = "strong"
    record.effective_thinking = "high"
    record.last_run_status = RunStatus.COMPLETED
    record.context_tokens = 12000
    record.context_window = 45000
    record.stop_reason = RunStopReason.BUDGET_EXCEEDED
    result = _stored_result("agent-1", "run-1")
    record.run_history.append(
        RunRecord(
            run_id=result.run_id,
            agent_id=result.agent_id,
            profile="worker",
            status=RunStatus.COMPLETED,
            completion_task=asyncio.get_running_loop().create_future(),
            result=result.result,
        )
    )
    registry._agent_records[record.agent_id] = record
    registry._result_store[(result.agent_id, result.run_id)] = result

    assert await registry._evict_agent(record.agent_id, "ttl")
    tombstone = registry._evicted_agents[record.agent_id].summary
    assert (tombstone.effective_model, tombstone.effective_thinking) == (
        "strong",
        "high",
    )
    assert tombstone.last_run_status is RunStatus.COMPLETED
    assert (tombstone.context_tokens, tombstone.context_window) == (12000, 45000)
    assert not tombstone.compacting
    assert tombstone.stop_reason is RunStopReason.BUDGET_EXCEEDED
    assert not tombstone.result_expired
    await registry._emit_agents_update()
    notification = notifier.await_args
    assert notification is not None
    preserved_model = notification.args[0][0]
    assert "child_session_id" not in preserved_model.model_dump()
    assert "parent_identity" not in preserved_model.model_dump()
    assert not preserved_model.result_expired

    for index in range(32):
        stored_result = _stored_result(
            f"agent-{index:02}", "run", completed_at=index + 1
        )
        registry._result_store[(stored_result.agent_id, stored_result.run_id)] = (
            stored_result
        )
    registry._expire_results_locked()

    summary = (await registry.check_agents())[0]
    assert summary.result_expired
    await registry._emit_agents_update()
    notification = notifier.await_args
    assert notification is not None
    expired_model = notification.args[0][0]
    assert expired_model.result_expired
    stored = result
    assert set(StoredRunResult.__dataclass_fields__) == {
        "agent_id",
        "run_id",
        "result",
        "completed_at",
        "root_generation",
        "terminal_identity",
    }
    assert all(
        getattr(stored, field_name) is not record.runtime
        for field_name in StoredRunResult.__dataclass_fields__
    )
    await registry.drain_children()


@pytest.mark.asyncio
async def test_reaper_ttl_ages_from_completion_not_observation() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    record = _idle_record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record

    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    assert wakeup.calls == [10]
    clock.advance(7)
    assert (await registry.check_agents())[0].idle_seconds == 7
    with pytest.raises(UnknownAgentError):
        await registry.get_agent_result(record.agent_id, "unknown")
    clock.advance(3)
    wakeup.gates[0].set()
    reaper_task = registry._reaper_task
    assert reaper_task is not None
    await asyncio.wait_for(reaper_task, timeout=1)

    assert record.agent_id not in registry._agent_records
    assert registry._evicted_agents[record.agent_id].summary.idle_seconds == 10
    await registry.drain_children()


@pytest.mark.asyncio
async def test_reaper_uses_profile_ttl_before_global_ttl() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    profile = _idle_record("profile", clock=clock, idle_ttl_seconds=5)
    global_ttl = _idle_record("global", clock=clock)
    registry._agent_records = {
        profile.agent_id: profile,
        global_ttl.agent_id: global_ttl,
    }

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 5
    clock.advance(5)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper
    assert set(registry._agent_records) == {"global"}
    assert await wakeup.arrivals.get() == 5
    await registry.drain_children()


@pytest.mark.asyncio
async def test_zero_profile_ttl_disables_ttl_eviction_but_not_cap_eviction() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, cap=1, wakeup=wakeup)
    exempt = _idle_record("exempt", clock=clock, idle_ttl_seconds=0)
    other = _idle_record("other", clock=clock)
    other.idle_since = 1
    registry._agent_records = {exempt.agent_id: exempt, other.agent_id: other}

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 0
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper
    assert set(registry._agent_records) == {"other"}
    assert (
        registry._evicted_agents["exempt"].summary.availability
        is AgentAvailability.EVICTED
    )
    await registry.drain_children()


@pytest.mark.asyncio
async def test_reaper_arms_for_earliest_mixed_ttl_and_not_all_exempt_agents() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    slow = _idle_record("slow", clock=clock, idle_ttl_seconds=20)
    fast = _idle_record("fast", clock=clock, idle_ttl_seconds=5)
    registry._agent_records = {slow.agent_id: slow, fast.agent_id: fast}

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 5
    slow.idle_ttl_seconds = 0
    fast.idle_ttl_seconds = 0
    await _rearm(registry)
    assert registry._reaper_task is None


@pytest.mark.asyncio
async def test_idle_summaries_use_each_record_ttl() -> None:
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=10)
    exempt = _idle_record("exempt", clock=clock, idle_ttl_seconds=0)
    overridden = _idle_record("overridden", clock=clock, idle_ttl_seconds=5)
    inherited = _idle_record("inherited", clock=clock)
    registry._agent_records = {
        record.agent_id: record for record in (exempt, overridden, inherited)
    }

    clock.advance(3)
    summaries = {summary.agent_id: summary for summary in await registry.check_agents()}
    assert summaries["exempt"].ttl_remaining_seconds is None
    assert summaries["overridden"].ttl_remaining_seconds == 2
    assert summaries["inherited"].ttl_remaining_seconds == 7


@pytest.mark.asyncio
async def test_reaper_evicts_oldest_eligible_agents_with_agent_id_tie_break() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, cap=1, wakeup=wakeup)
    for agent_id, idle_since in (("agent-b", 2), ("agent-a", 2), ("agent-c", 3)):
        record = _idle_record(agent_id, clock=clock)
        record.idle_since = idle_since
        registry._agent_records[agent_id] = record

    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 0
    assert wakeup.calls == [0]
    wakeup.gates[0].set()
    reaper_task = registry._reaper_task
    assert reaper_task is not None
    await asyncio.wait_for(reaper_task, timeout=1)

    assert set(registry._agent_records) == {"agent-c"}
    assert list(registry._evicted_agents) == ["agent-a", "agent-b"]
    await registry.drain_children()


@pytest.mark.asyncio
async def test_eviction_injects_parent_notification() -> None:
    clock = _ManualClock(10)
    registry, root = _reaper_registry(clock, ttl=1)
    root.turns.active_turn = MagicMock()
    root.agent_loop._pending_injected_messages = []
    record = _idle_record("agent-1", clock=clock)
    record.parent_session_id = "root"
    record.idle_since = 0
    registry._agent_records[record.agent_id] = record

    assert await registry._evict_agent(record.agent_id, "ttl")

    messages = root.agent_loop._pending_injected_messages
    assert len(messages) == 1
    assert messages[0].content == (
        "Background agent worker-1 (agent-1) evicted (ttl, idle 10s); "
        "launch a new agent for further work."
    )
    await registry.drain_children()


@pytest.mark.asyncio
async def test_reaper_batches_eviction_parent_notification() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, root = _reaper_registry(clock, cap=1, wakeup=wakeup)
    root.turns.active_turn = MagicMock()
    root.agent_loop._pending_injected_messages = []
    for agent_id in ("agent-a", "agent-b", "agent-c"):
        record = _idle_record(agent_id, clock=clock)
        record.parent_session_id = "root"
        record.idle_since = 0
        registry._agent_records[agent_id] = record

    await _rearm(registry)
    await wakeup.arrivals.get()
    wakeup.gates[0].set()
    reaper_task = registry._reaper_task
    assert reaper_task is not None
    await asyncio.wait_for(reaper_task, timeout=1)

    messages = root.agent_loop._pending_injected_messages
    assert len(messages) == 1
    assert messages[0].content == (
        "Background agents evicted: worker-a (agent-a, idle_cap, idle 10s); "
        "worker-b (agent-b, idle_cap, idle 10s); "
        "launch a new agent for further work."
    )
    await registry.drain_children()


@pytest.mark.asyncio
async def test_idle_cap_eviction_revalidates_eligibility_at_commit() -> None:
    clock = _ManualClock(10)
    registry, _root = _reaper_registry(clock, cap=1)
    first = _idle_record("agent-a", clock=clock)
    second = _idle_record("agent-b", clock=clock)
    registry._agent_records.update({first.agent_id: first, second.agent_id: second})
    registry._draining_children = True
    await registry.release_agent(first.agent_id)
    registry._draining_children = False

    assert not await registry._evict_agent(second.agent_id, "idle_cap")
    assert second.agent_id in registry._agent_records


@pytest.mark.asyncio
async def test_reaper_never_evicts_running_permission_waiting_or_finalizing_agents() -> (
    None
):
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=1, cap=1)
    running = _idle_record("running", clock=clock)
    permission_waiting = _idle_record("permission-waiting", clock=clock)
    finalizing = _idle_record("finalizing", clock=clock)
    running.state = _AgentState.RUNNING
    permission_waiting.state = _AgentState.RUNNING
    finalizing.state = _AgentState.FINALIZING
    registry._agent_records.update({
        record.agent_id: record for record in (running, permission_waiting, finalizing)
    })

    await _rearm(registry)
    assert registry._reaper_task is None
    for record in (running, permission_waiting, finalizing):
        assert not await registry._evict_agent(record.agent_id, "ttl")
    assert set(registry._agent_records) == {
        "running",
        "permission-waiting",
        "finalizing",
    }


@pytest.mark.asyncio
async def test_reaper_has_one_timer_and_rearms_or_stops_with_eligibility() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    later = _idle_record("later", clock=clock)
    later.idle_since = 5
    registry._agent_records[later.agent_id] = later
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 5
    first = registry._reaper_task
    assert first is not None and wakeup.calls == [5]

    sooner = _idle_record("sooner", clock=clock)
    sooner.idle_since = 3
    registry._agent_records[sooner.agent_id] = sooner
    await _rearm(registry)
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=1)
    assert registry._reaper_task is not first
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 3
    assert wakeup.calls == [5, 3]

    sooner.state = _AgentState.RUNNING
    later.state = _AgentState.RUNNING
    await _rearm(registry)
    assert registry._reaper_task is None
    await registry.release_agent("later")
    await registry.release_agent("sooner")
    assert registry._reaper_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ttl,cap,expect_timer", [(0, 1, True), (1, 0, True), (0, 0, False)]
)
async def test_reaper_zero_limit_combinations_disable_the_corresponding_policy(
    ttl: float, cap: int, expect_timer: bool
) -> None:
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=ttl, cap=cap)
    for agent_id in range(2 if ttl == 0 else 1):
        record = _idle_record(f"agent-{agent_id}", clock=clock)
        registry._agent_records[record.agent_id] = record

    await _rearm(registry)
    assert (registry._reaper_task is not None) is expect_timer
    await registry.drain_children()


@pytest.mark.asyncio
async def test_generation_fences_pending_eviction_notification_and_reports_generation() -> (
    None
):
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=1, generation=4)
    notify_agents = cast(AsyncMock, registry._notify_agents)
    record = _idle_record("agent-1", clock=clock)
    record.root_generation = 4
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def close() -> None:
        close_started.set()
        await close_release.wait()

    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    eviction = asyncio.create_task(registry._evict_agent(record.agent_id, "ttl"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    root.agent_loop._session_generation = 5
    registry.begin_root_generation()
    close_release.set()
    assert await eviction
    notify_agents.assert_not_awaited()

    root.agent_loop._session_generation = 6
    registry.begin_root_generation()
    current = _idle_record("agent-2", clock=clock)
    current.root_generation = 6
    registry._agent_records[current.agent_id] = current
    assert await registry._evict_agent(current.agent_id, "ttl")
    await_args = notify_agents.await_args
    assert await_args is not None
    evictions = await_args.args[1]
    assert evictions[0].root_generation == 6
    await registry.drain_children()


@pytest.mark.asyncio
async def test_retention_policy_is_snapshotted_per_root_generation() -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=10, cap=2, generation=1)
    assert registry._retention_policy == RetentionPolicy(
        idle_ttl_seconds=10, max_idle_agents=2, retain_agents=False
    )
    root.agent_loop.config.subagents.idle_ttl_seconds = 3
    root.agent_loop.config.subagents.max_idle_agents = 1
    assert registry._retention_policy == RetentionPolicy(
        idle_ttl_seconds=10, max_idle_agents=2, retain_agents=False
    )

    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    assert registry._retention_policy == RetentionPolicy(
        idle_ttl_seconds=3, max_idle_agents=1, retain_agents=False
    )


@pytest.mark.asyncio
async def test_notification_failure_still_idles_agent_and_cleans_pending_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=10)
    root.turns._projector = None
    root.turns.active_turn = None
    root.turns._active_task = None
    root.turns.queue_state.items = []
    root.agent_loop._pending_injected_messages = []
    record = _idle_record("agent-1", clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None

    async def wait_for_operation(initial_turn_id: str) -> PublicTurn:
        assert record.runtime.turns._event_sink is not None
        await record.runtime.turns._event_sink(
            AssistantEvent(content="Child completed")
        )
        return MagicMock(
            error=None, status=PublicTurnStatus.COMPLETED, stop_reason=None
        )

    record.runtime.turns.wait_for_operation = wait_for_operation
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    monkeypatch.setattr(
        SessionRuntimeRegistry,
        "_start_child_turn",
        lambda *_args: ("turn-1", lambda: None),
    )
    monkeypatch.setattr(
        registry,
        "_emit_agents_update",
        AsyncMock(side_effect=[None, RuntimeError(), None, None]),
    )

    await _background_result(
        registry,
        # The fabricated record is reused and must stay resident after the
        # run completes, so the reuse opts into retention explicitly.
        TaskArgs(
            task="work",
            agent_type="worker",
            agent_id="agent-1",
            background=True,
            retain=True,
        ),
        InvokeContext(tool_call_id="work", session_id="root"),
    )
    monitor = next(iter(registry._monitor_tasks))
    await monitor

    assert record.state is _AgentState.IDLE
    assert registry._pending_notifications == set()
    assert (
        record.agent_id,
        cast(str, record.latest_run_id),
    ) not in registry._pending_notifications
    await registry.drain_children()


@pytest.mark.asyncio
async def test_handoff_generation_allows_new_current_agent_to_be_evicted() -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=1, generation=4)
    stale = _idle_record("stale", clock=clock)
    stale.root_generation = 4
    registry._agent_records[stale.agent_id] = stale
    registry._children[stale.session_id] = stale.runtime
    root.agent_loop._session_generation = 5
    registry.handoff_root("old-root", "root")
    assert stale.agent_id not in registry._agent_records
    await asyncio.sleep(0)
    cast(AsyncMock, stale.runtime.close).assert_awaited_once()
    record = _idle_record("current", clock=clock)
    record.root_generation = root.agent_loop._session_generation
    registry._agent_records[record.agent_id] = record

    assert await registry._evict_agent(record.agent_id, "ttl")
    assert record.agent_id in registry._evicted_agents
    await registry.drain_children()


@pytest.mark.asyncio
async def test_eviction_close_failure_does_not_skip_other_victims_or_reaper() -> None:
    clock = _ManualClock(1)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, cap=1, wakeup=wakeup)
    first = _idle_record("first", clock=clock)
    second = _idle_record("second", clock=clock)
    first.runtime.close = AsyncMock(side_effect=RuntimeError("close failed"))
    registry._agent_records.update({first.agent_id: first, second.agent_id: second})

    await _rearm(registry)
    await wakeup.arrivals.get()
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper

    assert set(registry._evicted_agents) == {"first"}
    assert set(registry._agent_records) == {"second"}
    assert registry._reaper_task is not None
    await registry.drain_children()


@pytest.mark.asyncio
async def test_committed_eviction_notification_survives_reaper_rearm() -> None:
    clock = _ManualClock(1)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, cap=1, wakeup=wakeup)
    first = _idle_record("first", clock=clock)
    second = _idle_record("second", clock=clock)
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def close() -> None:
        close_started.set()
        await close_release.wait()

    first.runtime.close = close
    registry._agent_records.update({first.agent_id: first, second.agent_id: second})
    await _rearm(registry)
    await wakeup.arrivals.get()
    wakeup.gates[0].set()
    await close_started.wait()
    await _rearm(registry)
    close_release.set()
    evictions = [task for task in registry._eviction_tasks]
    if evictions:
        await asyncio.gather(*evictions)
    await asyncio.sleep(0)

    assert "first" in registry._evicted_agents
    notify_agents = cast(AsyncMock, registry._notify_agents)
    assert any(call.args[1] for call in notify_agents.await_args_list)
    await registry.drain_children()


@pytest.mark.asyncio
async def test_reaper_batch_evicts_all_required_victims_despite_close_failure() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=20, cap=1, wakeup=wakeup)
    records = [
        _idle_record(agent_id, clock=clock)
        for agent_id in ("agent-a", "agent-b", "agent-c")
    ]
    for offset, record in enumerate(records):
        record.idle_since = float(offset)
        run = RunRecord(
            run_id=f"{record.agent_id}-run",
            agent_id=record.agent_id,
            profile=record.profile,
            status=RunStatus.COMPLETED,
            completion_task=asyncio.get_running_loop().create_future(),
            result=_stored_result(record.agent_id, f"{record.agent_id}-run").result,
        )
        record.run_history.append(run)
        record.latest_run_id = run.run_id
        registry._agent_records[record.agent_id] = record
        registry._children[record.session_id] = record.runtime
        registry._result_store[(record.agent_id, run.run_id)] = _stored_result(
            record.agent_id, run.run_id
        )
    records[0].runtime.close = AsyncMock(side_effect=RuntimeError("close failed"))

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 0
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper

    assert set(registry._evicted_agents) == {"agent-a", "agent-b"}
    assert set(registry._agent_records) == {"agent-c"}
    cast(AsyncMock, records[0].runtime.close).assert_awaited_once()
    cast(AsyncMock, records[1].runtime.close).assert_awaited_once()
    cast(AsyncMock, records[2].runtime.close).assert_not_awaited()
    notifications = cast(AsyncMock, registry._notify_agents).await_args_list
    assert {
        eviction.agent_id for call in notifications for eviction in call.args[1]
    } == {"agent-a", "agent-b"}
    evicted_result = await registry.get_agent_result("agent-a", "agent-a-run")
    assert evicted_result is not None and evicted_result.completed
    assert wakeup.calls[-1] == 12
    await registry.drain_children()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_ttl_seconds", [300, 0])
async def test_profile_ttl_overrides_zero_global_retention_at_completion(
    monkeypatch: pytest.MonkeyPatch, idle_ttl_seconds: int
) -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, parent, _notifications = await _zero_retention_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    parent.agent_manager._discovered["persistent"] = AgentProfile(
        name="persistent",
        display_name="Persistent",
        description="Retained under zero global retention.",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.SUBAGENT,
        idle_ttl_seconds=idle_ttl_seconds,
        # A profile TTL alone no longer implies retention; the profile opts
        # in explicitly so the launch stays resident for the reaper checks.
        retain=True,
    )
    registry._clock = clock
    registry._wakeup = wakeup
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="persist", agent_type="persistent", background=True),
            InvokeContext(tool_call_id="persistent", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.idle_ttl_seconds == idle_ttl_seconds
        completion = record.current_run.completion_task if record.current_run else None
        assert isinstance(completion, asyncio.Task)
        await completion
        assert record.state is _AgentState.IDLE

        if idle_ttl_seconds == 0:
            assert registry._reaper_task is None
            assert launch.agent_id in registry._agent_records
        else:
            assert await wakeup.arrivals.get() == 300
            clock.advance(300)
            wakeup.gates[0].set()
            reaper = registry._reaper_task
            assert reaper is not None
            await reaper
            assert launch.agent_id not in registry._agent_records
    finally:
        await registry.drain_children()
        await parent.aclose()


async def _assert_default_ephemeral_outcome(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, *, cancelled: bool = False
) -> None:
    """An unretained launch under the default policy closes on completion.

    The registry keeps the production-default subagent config, so the launch
    below carries no ``retain`` argument and resolves to the default
    ephemeral policy. The result is published before the runtime detaches and
    closes, and the closed handle then rejects reuse with the typed error.
    """
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend, zero_limits=False
    )
    runtime: SessionRuntime | None = None
    clock = _ManualClock(100)
    registry._clock = clock
    snapshots = AsyncMock()
    registry._notify_agents = snapshots
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="default ephemeral", agent_type="worker", background=True),
            InvokeContext(
                tool_call_id="default-ephemeral", session_id=parent.session_id
            ),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        agent_id = launch.agent_id
        run_id = launch.run_id
        # Publication-before-close observation: the closed tombstone is
        # inserted after the record detaches, so the result must already be
        # in the store at that moment. Asserting inside the insertion
        # observation pins the ordering itself, not just the end state.
        result_store = registry._result_store
        published_before_close: list[bool] = []
        original_insert = registry._insert_tombstone_locked

        def observe_tombstone_insert(
            inserted_id: str, tombstone: _AgentTombstone
        ) -> None:
            if inserted_id == agent_id:
                published_before_close.append((agent_id, run_id) in result_store)
            original_insert(inserted_id, tombstone)

        monkeypatch.setattr(
            registry, "_insert_tombstone_locked", observe_tombstone_insert
        )
        record = registry._agent_records[launch.agent_id]
        assert record.current_run is not None
        assert record.current_run.started_at == 100
        assert registry._agent_summaries()[0].run_elapsed_seconds == 0
        clock.advance(252)
        runtime = record.runtime
        record.context_tokens = 321
        record.context_window = 6543
        completion = record.current_run.completion_task if record.current_run else None
        assert isinstance(completion, asyncio.Task)
        if cancelled:
            assert isinstance(backend, BlockingBackend)
            await asyncio.wait_for(backend.started.wait(), timeout=1)
            completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)
        await asyncio.gather(*tuple(registry._teardown_tasks))

        # The result was published before the record detached and closed.
        assert published_before_close == [True]
        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        tombstone = registry._evicted_agents[launch.agent_id].summary
        assert (tombstone.context_tokens, tombstone.context_window) == (
            record.context_tokens,
            record.context_window,
        )
        assert tombstone.stop_reason is record.stop_reason
        assert tombstone.run_elapsed_seconds == 252
        assert tombstone.latest_run_id == launch.run_id
        assert result is not None and result.run_id == launch.run_id
        assert any("Background agent" in message for message in notifications)
        # Reuse of the closed handle raises the typed closed error; the
        # evicted-tombstone subtype stays reserved for TTL/cap eviction. The
        # pin applies within the metadata window: past the tombstone deadline
        # (the clock advance below), reuse behaves as unknown per §3.1 —
        # pinned in tests/app_server/test_subagent_lifecycle.py.
        with pytest.raises(AgentClosedError):
            await _background_result(
                registry,
                TaskArgs(
                    task="reuse",
                    agent_type="worker",
                    agent_id=launch.agent_id,
                    background=True,
                ),
                InvokeContext(tool_call_id="reuse", session_id=parent.session_id),
            )
        clock.advance(1000)
        assert registry._agent_summaries()[0].run_elapsed_seconds == 252
        finishing = [
            call.args[0][0]
            for call in snapshots.await_args_list
            if call.args[0] and call.args[0][0].availability == "finalizing"
        ]
        assert finishing
        expected_reason = (
            "orchestrator_cancelled"
            if cancelled
            else "error"
            if isinstance(backend, ImmediateFailureBackend)
            else None
        )
        assert finishing[0].stop_reason == expected_reason
        assert finishing[0].run_elapsed_seconds == 252
        assert finishing[0].last_run_status == record.last_run_status
        assert not finishing[0].compacting
        assert not tombstone.compacting
        assert runtime._closed
    finally:
        if isinstance(backend, BlockingBackend):
            backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_launch_reuse_resets_duration_with_registry_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    clock = _ManualClock(10)
    registry._clock = clock
    try:
        args = TaskArgs(task="first", agent_type="worker", background=True)
        context = InvokeContext(tool_call_id="first", session_id=parent.session_id)
        launch = await _background_result(registry, args, context)
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        first = record.current_run
        assert first is not None and first.started_at == 10
        await backend.started.wait()
        clock.advance(252)
        assert registry._agent_summaries()[0].run_elapsed_seconds == 252
        backend.release.set()
        await first.completion_task
        clock.advance(62)
        assert registry._agent_summaries()[0].run_elapsed_seconds == 252
        backend.started.clear()
        backend.release.clear()
        await _background_result(
            registry,
            TaskArgs(
                task="second",
                agent_type="worker",
                agent_id=launch.agent_id,
                background=True,
            ),
            context,
        )
        second = record.current_run
        assert second is not None and second.started_at == 324
        assert second.run_id != first.run_id
        summary = registry._agent_summaries()[0]
        assert summary.run_elapsed_seconds == 0
        assert summary.idle_seconds is None
        await backend.started.wait()
        clock.advance(2)
        backend.release.set()
        await second.completion_task
        assert registry._agent_summaries()[0].run_elapsed_seconds == 2
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_default_ephemeral_success_publishes_result_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_default_ephemeral_outcome(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )


@pytest.mark.asyncio
async def test_default_ephemeral_failure_publishes_result_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_default_ephemeral_outcome(
        monkeypatch, ImmediateFailureBackend(RuntimeError("failed"))
    )


@pytest.mark.asyncio
async def test_default_ephemeral_cancellation_publishes_result_then_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_default_ephemeral_outcome(
        monkeypatch, BlockingBackend(), cancelled=True
    )
