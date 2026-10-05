from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from chartreux.app_server._projection import project_stats
from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server.models import (
    JsonPatchOperation,
    PublicCheckpointEntry,
    PublicEntryGenerationStatus,
    PublicTurn,
    PublicTurnStatus,
    PublicTurnStopReason,
)
from chartreux.app_server.protocol import (
    HistoryEntryAddedParams,
    HistoryEntryUpdatedParams,
    SessionCompactedParams,
    StatsUpdatedParams,
    TurnCompletedParams,
)
from chartreux.core.subagents import RunStatus, RunStopReason
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.stubs.fake_backend import FakeBackend


def _capture_registry():
    forwarded = AsyncMock()
    updates = AsyncMock()
    registry = SessionRuntimeRegistry(
        forwarded, AsyncMock(), lambda _: 0, notify_agents=updates
    )
    child = build_test_agent_loop(
        config=build_test_vibe_config(auto_compact_threshold=12345),
        backend=FakeBackend(),
    )
    runtime = registry._build_child_runtime(child)
    registry._root = runtime
    registry._generation_identity = (child.session_id, child._session_generation)
    registry._children[child.session_id] = runtime
    run = RunRecord(
        "run",
        "agent",
        "worker",
        RunStatus.RUNNING,
        asyncio.get_running_loop().create_future(),
    )
    record = AgentRecord(
        "agent",
        "worker",
        child.session_id,
        runtime,
        child._session_generation,
        state=_AgentState.RUNNING,
        current_run=run,
    )
    registry._agent_records[record.agent_id] = record
    return registry, runtime, record, forwarded, updates


def _stats(runtime, tokens: int) -> StatsUpdatedParams:
    stats = project_stats(runtime.agent_loop).model_copy(
        update={"context_tokens": tokens}
    )
    return StatsUpdatedParams(
        event_id=0,
        emitted_at=0,
        session_id=runtime.agent_loop.session_id,
        stats=stats,
        context_window=12345,
    )


def _start(record) -> HistoryEntryAddedParams:
    return HistoryEntryAddedParams(
        event_id=0,
        emitted_at=0,
        session_id=record.session_id,
        entry=PublicCheckpointEntry(
            id="compact",
            session_id=record.session_id,
            kind="compaction",
            created_at=0,
            updated_at=0,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        ),
    )


def _finish(record) -> HistoryEntryUpdatedParams:
    return HistoryEntryUpdatedParams(
        event_id=0,
        emitted_at=0,
        session_id=record.session_id,
        entry_id="compact",
        patch=[
            JsonPatchOperation(
                op="replace", path="/generationStatus", value="completed"
            )
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [PublicTurnStatus.COMPLETED, PublicTurnStatus.FAILED, PublicTurnStatus.INTERRUPTED],
)
@pytest.mark.parametrize(
    "reason",
    [
        RunStopReason.USER_CANCELLED,
        RunStopReason.ORCHESTRATOR_CANCELLED,
        RunStopReason.RETASKED,
    ],
)
async def test_requested_reason_shared_resolver_agrees_with_capture(
    status, reason
) -> None:
    registry, runtime, record, _, _ = _capture_registry()
    run = record.current_run
    assert run is not None
    run.requested_stop_reason = reason
    turn = PublicTurn(
        id="turn", session_id=record.session_id, started_at=0, status=status
    )
    try:
        resolved_status, resolved_reason = registry._resolve_run_outcome(run, turn)
        await runtime.turns._notify(
            "turn/completed",
            TurnCompletedParams(
                event_id=0, emitted_at=0, session_id=record.session_id, turn=turn
            ),
        )
        assert run.stop_reason == resolved_reason == record.stop_reason
        assert resolved_reason is (
            reason
            if status is PublicTurnStatus.INTERRUPTED
            else RunStopReason.ERROR
            if status is PublicTurnStatus.FAILED
            else None
        )
        if status is PublicTurnStatus.INTERRUPTED:
            assert registry._resolve_run_outcome(run) == (RunStatus.CANCELLED, reason)
        else:
            run.status = resolved_status
            assert registry._resolve_run_outcome(run)[0] is resolved_status
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["generation", "persistence"])
async def test_actual_compaction_phase_cancel_clears_activity_and_preserves_reason(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from chartreux.core.middleware import AutoCompactMiddleware
    from chartreux.core.subagents import CancelOutcome, TaskArgs
    from chartreux.core.tools.base import InvokeContext
    from tests.app_server.test_subagents import (
        _background_result,
        _zero_retention_registry,
    )
    from tests.mock.utils import mock_llm_chunk

    backend = FakeBackend([
        [mock_llm_chunk(content="initial")],
        [mock_llm_chunk(content="<summary>compacted</summary>")],
    ])
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = (1000, 10)
    entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    ctx = InvokeContext(tool_call_id="compact", session_id=parent.session_id)
    try:
        launch = await _background_result(registry, TaskArgs(task="initial"), ctx)
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        first = record.current_run
        assert first is not None
        await first.completion_task
        child = record.runtime.agent_loop
        before = AutoCompactMiddleware.before_turn

        async def force(middleware, context):
            if context.stats is child.stats:
                context.stats.context_tokens = 500_000
            return await before(middleware, context)

        monkeypatch.setattr(AutoCompactMiddleware, "before_turn", force)
        manager = child.compaction_manager
        original = manager._complete if phase == "generation" else manager._save

        async def gated(*args, **kwargs):
            entered.set()
            try:
                await release.wait()
                return await original(*args, **kwargs)
            finally:
                stopped.set()

        monkeypatch.setattr(
            manager, "_complete" if phase == "generation" else "_save", gated
        )
        await _background_result(
            registry, TaskArgs(task="compact", agent_id=record.agent_id), ctx
        )
        run = record.current_run
        assert run is not None
        await asyncio.wait_for(entered.wait(), 3)
        assert record.compacting
        outcome = await registry.cancel_run(
            record.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        result = await asyncio.wait_for(registry.wait_for_agent(record.agent_id), 3)
        assert (
            result.stop_reason is RunStopReason.USER_CANCELLED and not result.completed
        )
        assert run.status is RunStatus.CANCELLED and stopped.is_set()
        assert not record.compacting and record.compaction_entry_id is None
        assert record.stop_reason is RunStopReason.USER_CANCELLED
        assert any(
            message.context_boundary == "compaction" for message in child.messages
        ) == (phase == "persistence")
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_compaction_handoff_is_atomic_and_duplicate_stats_are_silent() -> None:
    registry, runtime, record, forwarded, updates = _capture_registry()
    try:
        notifications = [
            _stats(runtime, 8000),
            _stats(runtime, 8000),
            _start(record),
            SessionCompactedParams.model_construct(session_id=record.session_id),
            _stats(runtime, -1),
            _finish(record),
            _stats(runtime, 1000),
        ]
        for params in notifications:
            await runtime.turns._notify(params.NOTIFICATION_METHOD, params)
        snapshots = [
            (
                call.args[0][0].context_tokens,
                call.args[0][0].context_window,
                call.args[0][0].compacting,
            )
            for call in updates.await_args_list
        ]
        assert snapshots == [
            (8000, 12345, False),
            (8000, 12345, True),
            (None, 12345, False),
            (1000, 12345, False),
        ]
        assert forwarded.await_count == len(notifications)
        for call, params in zip(forwarded.await_args_list, notifications, strict=True):
            assert call.args == (params.NOTIFICATION_METHOD, params)
        assert (await registry.check_agents())[0].context_tokens == 1000
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status", [PublicTurnStatus.FAILED, PublicTurnStatus.INTERRUPTED]
)
async def test_failed_compaction_keeps_usage_and_clears_activity(status) -> None:
    _, runtime, record, _, updates = _capture_registry()
    try:
        await runtime.turns._notify("session/statsUpdated", _stats(runtime, 8000))
        await runtime.turns._notify("history/entryAdded", _start(record))
        await runtime.turns._notify("history/entryUpdated", _finish(record))
        assert record.compacting  # Finalizer checkpoint completion is not success.
        turn = PublicTurn(
            started_at=0, id="turn", session_id=record.session_id, status=status
        )
        await runtime.turns._notify(
            "turn/completed",
            TurnCompletedParams(
                event_id=0, emitted_at=0, session_id=record.session_id, turn=turn
            ),
        )
        assert updates.await_args is not None
        snapshot = updates.await_args.args[0][0]
        assert snapshot.context_tokens == 8000
        assert not snapshot.compacting
        assert snapshot.stop_reason is not None
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "reuse",
        "record",
        "runtime",
        "eviction",
        "release",
        "generation",
        "drain",
        "child_generation",
        "suppression",
    ],
)
async def test_transport_race_cannot_publish_stale_child_stats(change) -> None:
    registry, runtime, record, forwarded, updates = _capture_registry()
    entered = asyncio.Event()
    resume = asyncio.Event()

    async def block(*_args):
        entered.set()
        await resume.wait()

    async def send() -> None:
        await runtime.turns._notify("session/statsUpdated", _stats(runtime, 42))

    forwarded.side_effect = block
    task = asyncio.create_task(send())
    try:
        await entered.wait()
        if change == "reuse":
            record.current_run = None
        elif change == "record":
            registry._agent_records.clear()
        elif change == "runtime":
            registry._children.clear()
        elif change in {"eviction", "release"}:
            record.state = _AgentState.EVICTING
        elif change == "generation":
            registry._generation_identity = ("different-root", 99)
        elif change == "child_generation":
            runtime.agent_loop._session_generation += 1
        elif change == "drain":
            registry._draining_children = True
        else:
            registry._suppressed_notifications.add(record.agent_id)
        resume.set()
        await task
        updates.assert_not_awaited()
        assert record.context_tokens is None
        forwarded.assert_awaited_once()
    finally:
        resume.set()
        await task
        await runtime.close()


@pytest.mark.asyncio
async def test_model_resolution_failure_publishes_unknown_context_window() -> None:
    registry, runtime, record, forwarded, _ = _capture_registry()
    try:
        with patch.object(
            type(runtime.agent_loop.config),
            "get_active_model",
            side_effect=ValueError("unresolved model"),
        ):
            await runtime.turns._emit_stats()
        assert forwarded.await_args is not None
        params = forwarded.await_args.args[1]
        assert isinstance(params, StatsUpdatedParams)
        assert params.context_window is None
        assert record.context_window is None
        assert (await registry.check_agents())[0].context_window is None
    finally:
        await runtime.close()


@pytest.mark.parametrize(
    "reason", [PublicTurnStopReason.LIMIT, PublicTurnStopReason.BUDGET_UNVERIFIABLE]
)
def test_structured_budget_stop_mapping(reason) -> None:
    turn = PublicTurn(
        started_at=0,
        id="turn",
        session_id="child",
        status=PublicTurnStatus.COMPLETED,
        stop_reason=reason,
    )
    expected = (
        RunStopReason.BUDGET_EXCEEDED
        if reason is PublicTurnStopReason.LIMIT
        else RunStopReason.BUDGET_UNVERIFIABLE
    )
    assert SessionRuntimeRegistry._run_stop_reason(turn) is expected
