"""Release behavior tests for subagents."""

from __future__ import annotations

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import RunRecord
from chartreux.core.subagents import (
    AgentResultExpiredError,
    RunStatus,
    UnknownAgentError,
)
from tests.app_server._subagent_test_support import (
    GatedSequenceBackend,
    _background_result,
    _evicted_tombstone,
    _idle_record,
    _ManualClock,
    _real_reused_background,
    _retention_registry,
    _stored_result,
)
from tests.mock.utils import mock_llm_chunk


@pytest.mark.asyncio
async def test_release_running_agent_invalidates_waiters_without_leaking_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="done")]])
    started, release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()

    try:
        launched = await _background_result(registry, args, context)
        assert launched.run_id is not None
        run = record.current_run
        assert run is not None and isinstance(run.completion_task, asyncio.Task)
        await started.wait()
        waiter = asyncio.create_task(
            registry.wait_for_agent(record.agent_id, run.run_id)
        )
        cancelled_waiter = asyncio.create_task(
            registry.wait_for_agent(record.agent_id, run.run_id)
        )
        await asyncio.sleep(0)
        assert registry._wait_leases[(record.agent_id, run.run_id)] == 2

        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter

        await registry.release_agent(record.agent_id)
        with pytest.raises((UnknownAgentError, AgentResultExpiredError)):
            await waiter
        assert (record.agent_id, run.run_id) not in registry._result_write_tokens
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_release_agent_cleans_evicted_results_and_expiry_markers() -> None:
    registry = _retention_registry()
    agent_id = "agent-1"
    registry._evicted_agents[agent_id] = _evicted_tombstone(
        agent_id, run_id="run-1", status=RunStatus.COMPLETED
    )
    registry._result_store[(agent_id, "run-1")] = _stored_result(agent_id, "run-1")
    registry._expired_results.add((agent_id, "run-0"))

    registry._emit_agents_update = AsyncMock()
    outcome = await registry.release_agent(agent_id)

    assert outcome.value == "evicted"
    assert agent_id not in registry._evicted_agents
    assert not [key for key in registry._result_store if key[0] == agent_id]
    assert not [key for key in registry._expired_results if key[0] == agent_id]
    cast(AsyncMock, registry._emit_agents_update).assert_awaited_once()


@pytest.mark.asyncio
async def test_release_removes_pending_publication_tokens_for_all_agent_runs() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._result_write_tokens.update({
        (record.agent_id, "old"): object(),
        (record.agent_id, "current"): object(),
        ("other", "run"): object(),
    })
    registry._latest_run_ids[record.agent_id] = "current"

    await registry.release_agent(record.agent_id)

    assert not [
        key for key in registry._result_write_tokens if key[0] == record.agent_id
    ]
    assert record.agent_id not in registry._latest_run_ids
    assert ("other", "run") in registry._result_write_tokens
    cast(AsyncMock, record.runtime.close).assert_awaited_once()


@pytest.mark.asyncio
async def test_release_cancellation_waits_for_cancelled_monitor_cleanup() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    cancelled = asyncio.Event()
    settle = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def monitor() -> None:
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cancelled.set()
            await settle.wait()
            raise

    async def close() -> None:
        close_started.set()
        await close_release.wait()

    task = asyncio.create_task(monitor())
    run = RunRecord("run-1", "agent-1", "worker", RunStatus.RUNNING, task)
    record.current_run = run
    record.runtime.close = close
    cast(Any, record.runtime.turns).active_turn = None
    record.runtime.turns.close = AsyncMock()
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._monitor_tasks.add(task)
    release = asyncio.create_task(registry.release_agent(record.agent_id))
    await cancelled.wait()
    release.cancel()
    assert registry._teardown_tasks
    assert record.agent_id in registry._suppressed_notifications
    assert not close_started.is_set()
    settle.set()
    await close_started.wait()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await release
    await registry.drain_children()
    assert not registry._suppressed_notifications
    assert registry._children == {}


@pytest.mark.asyncio
async def test_repeated_release_cancellation_still_closes_once() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_calls = 0

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1
        close_started.set()
        await close_release.wait()

    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    release = asyncio.create_task(registry.release_agent(record.agent_id))
    await close_started.wait()
    release.cancel()
    await asyncio.sleep(0)
    release.cancel()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await release
    await registry.drain_children()
    assert close_calls == 1
    assert registry._teardown_tasks == set()
