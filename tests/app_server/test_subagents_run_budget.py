"""B2c: the ``[subagents]`` run-budget config seam and its exhaustion surface.

Covers the config-default transport (``[subagents].run_max_price`` /
``run_max_session_tokens`` -> subagent ``AgentRuntimePolicy`` ->
``RunBudgetMiddleware``), the absent-default no-op, explicit-session-option
precedence, and the end-to-end exhaustion mapping through the existing
budget-stop chain (``RunStopReason.BUDGET_EXCEEDED`` /
``RunStopReason.BUDGET_UNVERIFIABLE`` plus the parent-facing
"budget-stopped" wording). The empty-replay exhaustion tests pin the
parent-facing TERMINAL outcome of B1's ``EmptyReplayBudgetStop``: an
empty response whose in-call replay the run budget denied settles the
run through the middleware STOP path — budget stop reason, notification,
exactly one attempted response, and no recovery or resume scheduling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.middleware import RunBudgetMiddleware
from chartreux.core.session_types import AgentStats
from chartreux.core.subagents import RunStatus, RunStopReason, TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _zero_retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _unresolved_tool_call() -> ToolCall:
    # A tool call no loop can resolve: the failed-call path appends a tool
    # result, so the child runs a second turn where the run budget evaluates
    # against the first response's consumption.
    return ToolCall(
        id="call-1", index=0, function=FunctionCall(name="no_such_tool", arguments="{}")
    )


class _TurnPairBackend(FakeBackend):
    """Two-turn child drive: an unresolved tool call, then an unused response."""

    def __init__(self, hook: Callable[[], None] | None = None) -> None:
        super().__init__([
            [mock_llm_chunk(content="", tool_calls=[_unresolved_tool_call()])],
            [mock_llm_chunk(content="unreachable after the budget stop")],
        ])
        self._hook = hook

    async def complete(self, **kwargs):
        if self._hook is not None:
            self._hook()
        return await super().complete(**kwargs)


async def _drain_and_close(
    registry: Any, parent: AgentLoop
) -> None:  # pragma: no cover - cleanup helper
    await registry.drain_children()
    await parent.aclose()


@pytest.mark.asyncio
async def test_run_budget_config_default_absent_no_middleware() -> None:
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    child = await AgentRuntimeFactory().create_child(parent, "coder")
    try:
        child_types = [type(mw) for mw in child.middleware_pipeline.middlewares]
        assert RunBudgetMiddleware not in child_types
        assert child._run_max_price is None
        assert child._run_max_session_tokens is None
        assert child.runtime_policy.run_max_price is None
        assert child.runtime_policy.run_max_session_tokens is None
    finally:
        await child.aclose()
        await parent.aclose()


@pytest.mark.asyncio
async def test_run_budget_config_applies_to_subagent_loops() -> None:
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    # Live-config mutation (the shared support pattern): the run-budget
    # defaults arm subagents created after the mutation, never the root.
    parent.config.subagents.run_max_price = 0.25
    parent.config.subagents.run_max_session_tokens = 2000
    child = await AgentRuntimeFactory().create_child(parent, "coder")
    try:
        run_budgets = [
            mw
            for mw in child.middleware_pipeline.middlewares
            if isinstance(mw, RunBudgetMiddleware)
        ]
        assert len(run_budgets) == 1
        assert run_budgets[0].run_max_price == 0.25
        assert run_budgets[0].run_max_session_tokens == 2000
        assert child._run_max_price == 0.25
        assert child._run_max_session_tokens == 2000

        root_types = [type(mw) for mw in parent.middleware_pipeline.middlewares]
        assert RunBudgetMiddleware not in root_types
        assert parent._run_max_price is None
        assert parent._run_max_session_tokens is None
    finally:
        await child.aclose()
        await parent.aclose()

    # An explicit session-options value rides the policy and wins over the
    # config default.
    explicit = build_test_agent_loop(
        config=_config(), backend=FakeBackend(), run_max_price=0.5
    )
    explicit.config.subagents.run_max_price = 0.25
    child = await AgentRuntimeFactory().create_child(explicit, "coder")
    try:
        assert child._run_max_price == 0.5
    finally:
        await child.aclose()
        await explicit.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_run_budget_exhaustion_surfaces_budget_exceeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, _TurnPairBackend()
    )
    parent.config.subagents.run_max_price = 0.00001
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="budget", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await run.completion_task
        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert result is not None and not result.completed
        summary = (await registry.check_agents())[0]
        assert summary.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert any("budget-stopped (budget exceeded)" in text for text in notifications)
    finally:
        await _drain_and_close(registry, parent)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_run_budget_unverifiable_surfaces_budget_unverifiable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The run middleware's fresh-latch branch: the unknown-cost latch must
    # flip AFTER run admission (the admission anchor captured has_unknown_cost
    # as False), which the child's first backend call does here.
    child_stats: list[AgentStats] = []
    create_child = AgentRuntimeFactory.create_child

    async def capturing_child(factory, loop, candidate, **kwargs):
        child = await create_child(factory, loop, candidate, **kwargs)
        child_stats.append(child.stats)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", capturing_child)
    backend = _TurnPairBackend(
        lambda: (
            setattr(child_stats[0], "has_unknown_cost", True) if child_stats else None
        )
    )
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    parent.config.subagents.run_max_price = 100.0
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="budget", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await run.completion_task
        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert result is not None and not result.completed
        summary = (await registry.check_agents())[0]
        assert summary.stop_reason is RunStopReason.BUDGET_UNVERIFIABLE
        assert any(
            "budget-stopped (budget unverifiable)" in text for text in notifications
        )
    finally:
        await _drain_and_close(registry, parent)


async def _assert_terminal_without_recovery(
    registry: Any, record: Any, run: Any, backend: FakeBackend
) -> None:
    """Pin the terminal, non-recovering settle shared by the replay tests.

    Exactly ONE attempted response (no in-call replay, no failover request —
    the unreachable second stream stays unconsumed); the run settles
    terminally through the middleware STOP path, not as a failure: no typed
    failure envelope (nothing for the recovery controller to schedule from),
    no recovery episode or pending resume, no successor run admitted after
    the STOP, and no reactive self-heal on the child loop.
    """
    assert len(backend.requests_messages) == 1
    assert run.status is RunStatus.COMPLETED
    assert run.failure is None
    assert registry._recovery_controller._episodes == {}
    assert record.pending_resume is None
    assert len(record.run_history) == 1
    assert record.runtime.agent_loop._reactive_recovery_used is False


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_empty_replay_exhaustion_surfaces_budget_exceeded_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The child's first response is EMPTY with usage that pushes the run
    # price over the cap, so the gateway denies the bounded in-call empty
    # replay and the loop terminates through the middleware STOP path
    # re-evaluated at the EmptyReplayBudgetStop catch boundary.
    backend = FakeBackend([
        [mock_llm_chunk(content=" \n")],
        [mock_llm_chunk(content="must not be called")],
    ])
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    parent.config.subagents.run_max_price = 0.00001
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="budget", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await run.completion_task
        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert result is not None and not result.completed
        summary = (await registry.check_agents())[0]
        assert summary.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert any("budget-stopped (budget exceeded)" in text for text in notifications)
        await _assert_terminal_without_recovery(registry, record, run, backend)
    finally:
        await _drain_and_close(registry, parent)


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_empty_replay_unknown_cost_surfaces_budget_unverifiable_without_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The replay boundary's fresh-latch branch: the child's first (empty)
    # response carries NO usage, so has_unknown_cost flips AFTER run
    # admission (the anchor captured has_unknown_cost as False), the replay
    # admission denies, and the re-evaluated STOP keeps the
    # budget-unverifiable metadata the run mapping projects from.
    backend = FakeBackend([
        [mock_llm_chunk(content=" \n").model_copy(update={"usage": None})],
        [mock_llm_chunk(content="must not be called")],
    ])
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    parent.config.subagents.run_max_price = 100.0
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="budget", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await run.completion_task
        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert result is not None and not result.completed
        summary = (await registry.check_agents())[0]
        assert summary.stop_reason is RunStopReason.BUDGET_UNVERIFIABLE
        assert any(
            "budget-stopped (budget unverifiable)" in text for text in notifications
        )
        await _assert_terminal_without_recovery(registry, record, run, backend)
    finally:
        await _drain_and_close(registry, parent)
