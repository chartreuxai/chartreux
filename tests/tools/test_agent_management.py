from __future__ import annotations

import asyncio
from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest

from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.events import ToolResultEvent
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.subagents import (
    AgentAvailability,
    AgentResultExpiredError,
    AgentSummary,
    CancelOutcome,
    CancelResult,
    EffectVerification,
    ReconcileOutcome,
    ReconcileResult as ReconcileResultModel,
    ReleaseAgentOutcome,
    ResumeAgentResult,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    WaitForAgentResult,
    WaitStatus,
    derive_agent_name,
    still_running_guidance,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.cancel_agent import (
    CancelAgent,
    CancelAgentArgs,
    CancelAgentConfig,
)
from chartreux.core.tools.builtins.check_agents import (
    CheckAgents,
    CheckAgentsArgs,
    CheckAgentsConfig,
)
from chartreux.core.tools.builtins.get_agent_result import (
    GetAgentResult,
    GetAgentResultArgs,
    GetAgentResultConfig,
)
from chartreux.core.tools.builtins.reconcile_result import (
    ReconcileResult,
    ReconcileResultArgs,
    ReconcileResultConfig,
)
from chartreux.core.tools.builtins.release_agent import (
    ReleaseAgent,
    ReleaseAgentArgs,
    ReleaseAgentConfig,
)
from chartreux.core.tools.builtins.resume_agent import (
    ResumeAgent,
    ResumeAgentArgs,
    ResumeAgentConfig,
)
from chartreux.core.tools.builtins.task import Task, TaskToolConfig
from chartreux.core.tools.builtins.wait_for_agent import (
    WaitForAgent,
    WaitForAgentArgs,
    WaitForAgentConfig,
)
from tests.conftest import ConfigBuilder, OrchestratorLoader, build_test_vibe_config
from tests.mock.utils import collect_result


class FakeSubagentManager:
    def __init__(self, agents: list[AgentSummary] | None = None) -> None:
        self.agents = agents or []
        self.results: dict[tuple[str, str | None], TaskResult] = {}
        self.completed: dict[tuple[str, str | None], asyncio.Event] = {}
        self.released_agent_ids: list[str] = []
        self.get_error: ValueError | None = None
        self.wait_error: ValueError | None = None
        self.release_error: ValueError | None = None
        self.resume_error: ValueError | None = None
        self.reconcile_error: ValueError | None = None
        self.resume_calls: list[tuple[str, str, str, str]] = []
        self.reconcile_calls: list[tuple[str, str, str, list, str]] = []

    async def check_agents(self) -> list[AgentSummary]:
        return self.agents

    async def get_agent_result(
        self, agent_id: str, run_id: str | None = None
    ) -> TaskResult | None:
        if self.get_error is not None:
            raise self.get_error
        return self.results.get((agent_id, run_id))

    async def wait_for_agent(
        self, agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> WaitForAgentResult:
        if self.wait_error is not None:
            raise self.wait_error
        key = (agent_id, run_id)
        if not any(agent.agent_id == agent_id for agent in self.agents):
            raise ValueError(f"Unknown agent: {agent_id}")
        event = self.completed.setdefault(key, asyncio.Event())
        try:
            if timeout is None:
                await event.wait()
            else:
                await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            summary = next(agent for agent in self.agents if agent.agent_id == agent_id)
            assert timeout is not None
            elapsed = float(timeout)
            turns_used = summary.turns_used or 0
            return WaitForAgentResult(
                status=WaitStatus.STILL_RUNNING,
                agent_id=agent_id,
                run_id=run_id,
                elapsed_seconds=elapsed,
                run_status=RunStatus.RUNNING,
                turns_used=turns_used,
                guidance=still_running_guidance(
                    agent_id=agent_id,
                    run_id=run_id or "",
                    elapsed_seconds=elapsed,
                    run_status=RunStatus.RUNNING,
                    turns_used=turns_used,
                ),
            )
        return WaitForAgentResult(status=WaitStatus.COMPLETED, result=self.results[key])

    async def cancel_run(
        self,
        agent_id: str,
        run_id: str | None = None,
        *,
        reason: RunStopReason,
        requester_session_id: str,
    ) -> CancelResult:
        return CancelResult(outcome=CancelOutcome.NOT_RUNNING, run_id=run_id)

    async def release_agent(self, agent_id: str) -> ReleaseAgentOutcome:
        if self.release_error is not None:
            raise self.release_error
        agent = next(
            (agent for agent in self.agents if agent.agent_id == agent_id), None
        )
        if agent is None:
            raise ValueError(f"Unknown agent: {agent_id}")
        self.released_agent_ids.append(agent_id)
        return (
            ReleaseAgentOutcome.EVICTED
            if agent.availability is AgentAvailability.EVICTED
            else ReleaseAgentOutcome.RELEASED
        )

    async def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> ResumeAgentResult:
        if self.resume_error is not None:
            raise self.resume_error
        self.resume_calls.append((
            agent_id,
            expected_run_id,
            idempotency_key,
            requester_session_id,
        ))
        return ResumeAgentResult(
            outcome=ResumeOutcome.NOT_RESUMABLE,
            agent_id=agent_id,
            run_id=expected_run_id,
            reason="Agent is idle, not suspended",
            reason_codes=["not_suspended"],
        )

    async def reconcile_result(
        self,
        agent_id: str,
        expected_run_id: str,
        transcript_boundary: str,
        verified_effects: Sequence[EffectVerification],
        *,
        requester_session_id: str,
    ) -> ReconcileResultModel:
        if self.reconcile_error is not None:
            raise self.reconcile_error
        self.reconcile_calls.append((
            agent_id,
            expected_run_id,
            transcript_boundary,
            list(verified_effects),
            requester_session_id,
        ))
        return ReconcileResultModel(
            outcome=ReconcileOutcome.NOT_PENDING,
            agent_id=agent_id,
            run_id=expected_run_id,
            transcript_boundary=transcript_boundary,
            reason="No reconcile barrier is pending",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", list(CancelOutcome))
@pytest.mark.parametrize("run_id", [None, "pinned-run"])
async def test_cancel_agent_forwards_requester_and_displays_disposition(
    outcome: CancelOutcome, run_id: str | None
) -> None:
    tool = CancelAgent(config_getter=lambda: CancelAgentConfig(), state=BaseToolState())
    manager = FakeSubagentManager()
    expected = CancelResult(
        outcome=outcome,
        run_id="resolved",
        stop_reason=RunStopReason.ORCHESTRATOR_CANCELLED,
    )
    manager.cancel_run = AsyncMock(return_value=expected)
    ctx = InvokeContext(
        tool_call_id="stop", session_id="parent", subagent_manager=manager
    )
    result = await collect_result(
        tool.run(CancelAgentArgs(agent_id="agent-1", run_id=run_id), ctx)
    )
    assert result == expected
    manager.cancel_run.assert_awaited_once_with(
        "agent-1",
        run_id,
        reason=RunStopReason.ORCHESTRATOR_CANCELLED,
        requester_session_id="parent",
    )
    display = tool.get_result_display(
        ToolResultEvent(
            tool_name="cancel_agent",
            tool_class=CancelAgent,
            result=result,
            tool_call_id="stop",
        )
    )
    assert display.verb == outcome.value.replace("_", " ").capitalize()
    assert "resolved" in display.message and display.verb != "Cancelled"


@pytest.mark.asyncio
async def test_cancel_agent_surfaces_foreground_rejection() -> None:
    tool = CancelAgent(config_getter=lambda: CancelAgentConfig(), state=BaseToolState())
    manager = FakeSubagentManager()
    manager.cancel_run = AsyncMock(
        side_effect=ValueError("Foreground agents cannot be stopped with cancel_agent")
    )
    ctx = InvokeContext(
        tool_call_id="stop", session_id="parent", subagent_manager=manager
    )
    with pytest.raises(ToolError, match="Foreground agents cannot be stopped"):
        await collect_result(tool.run(CancelAgentArgs(agent_id="child-session"), ctx))


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["missing", "manager", "requester", "subagent"])
async def test_cancel_agent_context_guards(context: str) -> None:
    tool = CancelAgent(config_getter=lambda: CancelAgentConfig(), state=BaseToolState())
    manager = FakeSubagentManager()
    manager.cancel_run = AsyncMock()
    ctx = InvokeContext(
        tool_call_id="stop", session_id="parent", subagent_manager=manager
    )
    if context == "missing":
        ctx = None
    elif context == "manager":
        ctx.subagent_manager = None
    elif context == "requester":
        ctx.session_id = None
    else:
        ctx.is_subagent = True
    with pytest.raises(ToolError):
        await collect_result(tool.run(CancelAgentArgs(agent_id="agent-1"), ctx))
    manager.cancel_run.assert_not_awaited()


@pytest.mark.asyncio
async def test_wait_tool_returns_cancelled_partial_result() -> None:
    manager = FakeSubagentManager([
        AgentSummary(
            agent_id="agent-1",
            identity="coder",
            availability=AgentAvailability.IDLE,
            current_run_id=None,
            current_run_status=None,
            name=derive_agent_name("agent-1", "coder", None, None, None),
        )
    ])
    expected = TaskResult(
        response="partial",
        turns_used=1,
        completed=False,
        stop_reason=RunStopReason.ORCHESTRATOR_CANCELLED,
    )
    manager.results[("agent-1", "run-1")] = expected
    manager.completed[("agent-1", "run-1")] = asyncio.Event()
    manager.completed[("agent-1", "run-1")].set()
    result = await collect_result(
        _wait_for_agent_tool().run(
            WaitForAgentArgs(agent_id="agent-1", run_id="run-1"), _context(manager)
        )
    )
    assert result.status is WaitStatus.COMPLETED
    assert result.result == expected


def _context(manager: FakeSubagentManager | None = None) -> InvokeContext:
    return InvokeContext(tool_call_id="test-call-id", subagent_manager=manager)


def _check_agents_tool() -> CheckAgents:
    return CheckAgents(config_getter=lambda: CheckAgentsConfig(), state=BaseToolState())


def _get_agent_result_tool() -> GetAgentResult:
    return GetAgentResult(
        config_getter=lambda: GetAgentResultConfig(), state=BaseToolState()
    )


def _wait_for_agent_tool() -> WaitForAgent:
    return WaitForAgent(
        config_getter=lambda: WaitForAgentConfig(), state=BaseToolState()
    )


def _release_agent_tool() -> ReleaseAgent:
    return ReleaseAgent(
        config_getter=lambda: ReleaseAgentConfig(), state=BaseToolState()
    )


@pytest.mark.asyncio
async def test_check_agents_returns_empty_list_when_no_agents() -> None:
    result = await collect_result(
        _check_agents_tool().run(CheckAgentsArgs(), _context(FakeSubagentManager()))
    )

    assert result.agents == []
    assert "task(agent_id=..." in result.model_dump(mode="json")["reuse_guidance"]


@pytest.mark.asyncio
async def test_check_agents_returns_agent_summaries() -> None:
    summary = AgentSummary(
        agent_id="agent-1",
        identity="coder",
        availability=AgentAvailability.EVICTED,
        current_run_id=None,
        current_run_status=None,
        initial_task_summary="Inspect agent management",
        current_task_summary=None,
        idle_seconds=3.5,
        ttl_remaining_seconds=None,
        effective_model="strong",
        base_model="base",
        active_provider="test-provider",
        effective_thinking="high",
        context_tokens=12_500,
        context_window=200_000,
        compacting=False,
        stop_reason=RunStopReason.BUDGET_EXCEEDED,
        name=derive_agent_name("agent-1", "coder", None, "base", "strong"),
    )
    result = await collect_result(
        _check_agents_tool().run(
            CheckAgentsArgs(), _context(FakeSubagentManager([summary]))
        )
    )

    serialized = result.model_dump(mode="json")["agents"][0]
    assert serialized == {
        "agent_id": "agent-1",
        "availability": "evicted",
        "current_run_id": None,
        "current_run_status": None,
        "turns_used": None,
        "initial_task_summary": "Inspect agent management",
        "current_task_summary": None,
        "idle_seconds": 3.5,
        "run_elapsed_seconds": None,
        "latest_run_id": None,
        "ttl_remaining_seconds": None,
        "effective_model": "strong",
        "base_model": "base",
        "active_provider": "test-provider",
        "effective_thinking": "high",
        "result_expired": False,
        "last_run_status": None,
        "context_tokens": 12_500,
        "context_window": 200_000,
        "compacting": False,
        "stop_reason": "budget_exceeded",
        "last_failure": None,
        "recovery_chain_id": None,
        "same_agent_resumes_used": None,
        "resumed_from_run_id": None,
        "recovery_eligibility": None,
        # WP5: the predecessor/package lineage pair defaults to None.
        "work_package_id": None,
        "predecessor_agent_id": None,
        "name": "coder-base-1",
        # The identity pair carries the committed launch identity.
        "identity": "coder",
        "identity_revision": None,
    }


@pytest.mark.asyncio
async def test_check_agents_carries_last_failure_envelope() -> None:
    """WP1: the tool surface is additive — a summary's last_failure envelope
    round-trips through check_agents; a summary without one keeps it null.
    """
    from chartreux.core.recovery_models import build_failure_envelope

    envelope = build_failure_envelope(
        observation=None, agent_id="agent-1", run_id="run-1"
    )
    summary = AgentSummary(
        agent_id="agent-1",
        identity="coder",
        availability=AgentAvailability.EVICTED,
        current_run_id=None,
        current_run_status=None,
        last_failure=envelope,
        name=derive_agent_name("agent-1", "coder", None, "base", "strong"),
    )
    result = await collect_result(
        _check_agents_tool().run(
            CheckAgentsArgs(), _context(FakeSubagentManager([summary]))
        )
    )
    serialized = result.model_dump(mode="json")["agents"][0]
    assert serialized["last_failure"]["event_id"] == envelope.event_id
    assert serialized["last_failure"]["recovery"]["eligibility"] == "prohibited"

    plain = AgentSummary(
        agent_id="agent-2",
        identity="coder",
        availability=AgentAvailability.IDLE,
        current_run_id=None,
        current_run_status=None,
    )
    result = await collect_result(
        _check_agents_tool().run(
            CheckAgentsArgs(), _context(FakeSubagentManager([plain]))
        )
    )
    assert result.model_dump(mode="json")["agents"][0]["last_failure"] is None


@pytest.mark.asyncio
async def test_get_agent_result_returns_pending_for_running_agent() -> None:
    result = await collect_result(
        _get_agent_result_tool().run(
            GetAgentResultArgs(agent_id="unknown"), _context(FakeSubagentManager())
        )
    )

    assert result.result is None
    assert (
        result.message
        == "Agent is still running; use wait_for_agent or try again later."
    )


@pytest.mark.asyncio
async def test_get_agent_result_returns_completed_result() -> None:
    manager = FakeSubagentManager()
    expected = TaskResult(response="done", turns_used=1, completed=True)
    manager.results[("agent-1", "run-1")] = expected

    result = await collect_result(
        _get_agent_result_tool().run(
            GetAgentResultArgs(agent_id="agent-1", run_id="run-1"), _context(manager)
        )
    )

    assert result.result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args", "error_attribute", "error", "message"),
    [
        (
            _get_agent_result_tool,
            GetAgentResultArgs(agent_id="agent-1"),
            "get_error",
            UnknownAgentError(),
            "No such agent or run",
        ),
        (
            _wait_for_agent_tool,
            WaitForAgentArgs(agent_id="agent-1"),
            "wait_error",
            AgentResultExpiredError(),
            "result expired",
        ),
        (
            _wait_for_agent_tool,
            WaitForAgentArgs(agent_id="agent-1"),
            "wait_error",
            UnknownAgentError(),
            "No such agent or run",
        ),
        (
            _release_agent_tool,
            ReleaseAgentArgs(agent_id="agent-1"),
            "release_error",
            UnknownAgentError(),
            "No such agent",
        ),
    ],
)
async def test_management_tools_map_lifecycle_errors(
    tool, args, error_attribute: str, error: ValueError, message: str
) -> None:
    manager = FakeSubagentManager()
    setattr(manager, error_attribute, error)

    with pytest.raises(ToolError, match=message):
        await collect_result(tool().run(args, _context(manager)))


@pytest.mark.asyncio
async def test_wait_for_evicted_agent_returns_retained_result() -> None:
    summary = AgentSummary(
        agent_id="agent-1",
        identity="coder",
        availability=AgentAvailability.EVICTED,
        current_run_id="run-1",
        current_run_status=RunStatus.COMPLETED,
        name=derive_agent_name("agent-1", "coder", None, None, None),
    )
    manager = FakeSubagentManager([summary])
    expected = TaskResult(response="done", turns_used=1, completed=True)
    manager.results[("agent-1", "run-1")] = expected
    manager.completed[("agent-1", "run-1")] = asyncio.Event()
    manager.completed[("agent-1", "run-1")].set()

    result = await collect_result(
        _wait_for_agent_tool().run(
            WaitForAgentArgs(agent_id="agent-1", run_id="run-1"), _context(manager)
        )
    )

    assert result.status is WaitStatus.COMPLETED
    assert result.result == expected
    summary = AgentSummary(
        agent_id="agent-1",
        identity="coder",
        availability=AgentAvailability.RUNNING,
        current_run_id="run-1",
        current_run_status=RunStatus.RUNNING,
        name=derive_agent_name("agent-1", "coder", None, None, None),
    )
    manager = FakeSubagentManager([summary])
    wait = asyncio.create_task(
        collect_result(
            _wait_for_agent_tool().run(
                WaitForAgentArgs(agent_id="agent-1", run_id="run-1"), _context(manager)
            )
        )
    )
    await asyncio.sleep(0)
    assert not wait.done()

    expected = TaskResult(response="done", turns_used=1, completed=True)
    manager.results[("agent-1", "run-1")] = expected
    manager.completed[("agent-1", "run-1")].set()

    wait_result = await wait
    assert wait_result.status is WaitStatus.COMPLETED
    assert wait_result.result == expected


@pytest.mark.asyncio
async def test_wait_for_agent_timeout_returns_still_running_without_cancelling() -> (
    None
):
    summary = AgentSummary(
        agent_id="agent-1",
        identity="coder",
        availability=AgentAvailability.RUNNING,
        current_run_id="run-1",
        current_run_status=RunStatus.RUNNING,
        name=derive_agent_name("agent-1", "coder", None, None, None),
    )
    manager = FakeSubagentManager([summary])

    result = await collect_result(
        _wait_for_agent_tool().run(
            WaitForAgentArgs(agent_id="agent-1", run_id="run-1", timeout=0.01),
            _context(manager),
        )
    )

    # A timed-out wait is a normal still_running result, not a tool failure.
    assert result.status is WaitStatus.STILL_RUNNING
    assert result.agent_id == "agent-1"
    assert result.run_id == "run-1"
    assert result.run_status is RunStatus.RUNNING
    assert result.elapsed_seconds == 0.01
    assert result.result is None
    assert result.guidance is not None
    assert "End your turn" in result.guidance
    assert "later turn" in result.guidance
    assert "check_agents" in result.guidance

    # The run was never cancelled: the completion event stays unset.
    assert not manager.completed[("agent-1", "run-1")].is_set()

    display = WaitForAgent.get_result_display(
        ToolResultEvent(
            tool_name="wait_for_agent",
            tool_class=WaitForAgent,
            result=result,
            tool_call_id="call",
        )
    )
    assert display.success is True
    assert display.verb == "Waiting"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "args"),
    [
        (_check_agents_tool, CheckAgentsArgs()),
        (_get_agent_result_tool, GetAgentResultArgs(agent_id="agent-1")),
        (_wait_for_agent_tool, WaitForAgentArgs(agent_id="agent-1")),
        (_release_agent_tool, ReleaseAgentArgs(agent_id="agent-1")),
    ],
)
async def test_management_tools_require_subagent_manager(tool, args) -> None:
    with pytest.raises(ToolError, match="subagent manager"):
        await collect_result(tool().run(args, _context()))


@pytest.mark.asyncio
async def test_wait_for_unknown_agent_is_a_tool_error() -> None:
    with pytest.raises(ToolError, match="Unknown agent"):
        await collect_result(
            _wait_for_agent_tool().run(
                WaitForAgentArgs(agent_id="unknown"), _context(FakeSubagentManager())
            )
        )


@pytest.mark.asyncio
async def test_release_agent_calls_manager() -> None:
    summary = AgentSummary(
        agent_id="agent-1",
        identity="coder",
        availability=AgentAvailability.IDLE,
        current_run_id=None,
        current_run_status=None,
        name=derive_agent_name("agent-1", "coder", None, None, None),
    )
    manager = FakeSubagentManager([summary])

    result = await collect_result(
        _release_agent_tool().run(
            ReleaseAgentArgs(agent_id="agent-1"), _context(manager)
        )
    )

    assert manager.released_agent_ids == ["agent-1"]
    assert result.agent_id == "agent-1"
    assert result.message == "Agent released"


@pytest.mark.asyncio
async def test_release_evicted_agent_succeeds() -> None:
    manager = FakeSubagentManager([
        AgentSummary(
            agent_id="agent-1",
            identity="coder",
            availability=AgentAvailability.EVICTED,
            current_run_id=None,
            current_run_status=None,
            name=derive_agent_name("agent-1", "coder", None, None, None),
        )
    ])

    result = await collect_result(
        _release_agent_tool().run(
            ReleaseAgentArgs(agent_id="agent-1"), _context(manager)
        )
    )

    assert manager.released_agent_ids == ["agent-1"]
    assert result.message == "Evicted agent tombstone removed"

    with pytest.raises(ToolError, match="Unknown agent"):
        await collect_result(
            _release_agent_tool().run(
                ReleaseAgentArgs(agent_id="unknown"), _context(FakeSubagentManager())
            )
        )


@pytest.mark.asyncio
async def test_task_validates_background_agent_id_usage(
    build_config: ConfigBuilder,
    load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
) -> None:
    task = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
    ctx = InvokeContext(
        tool_call_id="test-call-id",
        agent_manager=AgentManager(load_orchestrator(build_config())),
    )

    with pytest.raises(ToolError, match="agent_id can only"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work", identity="coder", background=False, agent_id="agent-1"
                ),
                ctx,
            )
        )

    ctx.is_subagent = True
    with pytest.raises(ToolError, match="Background agents cannot"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work", identity="coder", background=True, agent_id="agent-1"
                ),
                ctx,
            )
        )


def test_management_tools_are_discoverable_when_explicitly_enabled() -> None:
    from chartreux.core.tools.manager import ToolManager

    manager = ToolManager(
        lambda: build_test_vibe_config(
            enabled_tools=[
                "check_agents",
                "get_agent_result",
                "wait_for_agent",
                "release_agent",
            ]
        )
    )

    assert set(manager.available_tools) == {
        "check_agents",
        "get_agent_result",
        "wait_for_agent",
        "release_agent",
    }


def test_resume_and_reconcile_tools_are_discoverable_when_explicitly_enabled() -> None:
    """WP2: the explicit resume/reconcile tools load from the default tool
    directory when explicitly enabled.
    """
    from chartreux.core.tools.manager import ToolManager

    manager = ToolManager(
        lambda: build_test_vibe_config(
            enabled_tools=["resume_agent", "reconcile_result"]
        )
    )

    assert set(manager.available_tools) == {"resume_agent", "reconcile_result"}


def _resume_agent_tool() -> ResumeAgent:
    return ResumeAgent(config_getter=lambda: ResumeAgentConfig(), state=BaseToolState())


def _reconcile_result_tool() -> ReconcileResult:
    return ReconcileResult(
        config_getter=lambda: ReconcileResultConfig(), state=BaseToolState()
    )


@pytest.mark.asyncio
async def test_resume_agent_forwards_requester_and_displays_disposition() -> None:
    """WP2: resume_agent forwards the run pin and the idempotency key
    (defaulting to '<agent_id>:<expected_run_id>') and displays the typed
    disposition.
    """
    tool = _resume_agent_tool()
    manager = FakeSubagentManager()
    expected = ResumeAgentResult(
        outcome=ResumeOutcome.SCHEDULED,
        agent_id="agent-1",
        run_id="run-1",
        successor_run_id="run-2",
        recovery_operation_id="resume-op:agent-1:run-2",
        eligibility=RecoveryEligibility.ELIGIBLE,
        reason="Resume admitted; the successor run started immediately",
        same_agent_resumes_used=1,
    )
    manager.resume_agent = AsyncMock(return_value=expected)
    ctx = InvokeContext(
        tool_call_id="resume", session_id="parent", subagent_manager=manager
    )
    result = await collect_result(
        tool.run(ResumeAgentArgs(agent_id="agent-1", expected_run_id="run-1"), ctx)
    )
    assert result == expected
    manager.resume_agent.assert_awaited_once_with(
        "agent-1", "run-1", "agent-1:run-1", requester_session_id="parent"
    )
    display = tool.get_result_display(
        ToolResultEvent(
            tool_name="resume_agent",
            tool_class=ResumeAgent,
            result=result,
            tool_call_id="resume",
        )
    )
    assert display.success is True
    assert display.verb == "Scheduled"
    assert "run-1 -> run-2" in display.message

    # An explicit idempotency key is forwarded verbatim.
    manager.resume_agent.reset_mock()
    await collect_result(
        tool.run(
            ResumeAgentArgs(
                agent_id="agent-1", expected_run_id="run-1", idempotency_key="key-1"
            ),
            ctx,
        )
    )
    manager.resume_agent.assert_awaited_once_with(
        "agent-1", "run-1", "key-1", requester_session_id="parent"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["missing", "manager", "requester", "subagent"])
async def test_resume_agent_context_guards(context: str) -> None:
    """WP2: resume_agent requires a subagent manager, a requester session,
    and a non-subagent (parent orchestrator) context.
    """
    tool = _resume_agent_tool()
    manager = FakeSubagentManager()
    manager.resume_agent = AsyncMock()
    ctx = InvokeContext(
        tool_call_id="resume", session_id="parent", subagent_manager=manager
    )
    if context == "missing":
        ctx = None
    elif context == "manager":
        ctx.subagent_manager = None
    elif context == "requester":
        ctx.session_id = None
    else:
        ctx.is_subagent = True
    with pytest.raises(ToolError):
        await collect_result(
            tool.run(ResumeAgentArgs(agent_id="agent-1", expected_run_id="run-1"), ctx)
        )
    manager.resume_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_agent_surfaces_manager_error_as_tool_error() -> None:
    tool = _resume_agent_tool()
    manager = FakeSubagentManager()
    manager.resume_agent = AsyncMock(side_effect=ValueError("boom"))
    ctx = InvokeContext(
        tool_call_id="resume", session_id="parent", subagent_manager=manager
    )
    with pytest.raises(ToolError, match="boom"):
        await collect_result(
            tool.run(ResumeAgentArgs(agent_id="agent-1", expected_run_id="run-1"), ctx)
        )


@pytest.mark.asyncio
async def test_reconcile_result_forwards_pinned_evidence_and_displays_disposition() -> (
    None
):
    """WP2: reconcile_result forwards the run/boundary pin and the verified
    effects, and displays the typed disposition.
    """
    tool = _reconcile_result_tool()
    manager = FakeSubagentManager()
    effects = [
        EffectVerification(tool_call_id="call-1", disposition="settled"),
        EffectVerification(
            tool_call_id="call-2", disposition="unrepeatable_damage", note="gone"
        ),
    ]
    expected = ReconcileResultModel(
        outcome=ReconcileOutcome.PROHIBITED,
        agent_id="agent-1",
        run_id="run-1",
        transcript_boundary="turn-1",
        eligibility=RecoveryEligibility.PROHIBITED,
        reason_codes=["verified_unrepeatable_damage", "call-2"],
        reason="Verification showed unrepeatable damage; recovery is prohibited.",
    )
    manager.reconcile_result = AsyncMock(return_value=expected)
    ctx = InvokeContext(
        tool_call_id="reconcile", session_id="parent", subagent_manager=manager
    )
    result = await collect_result(
        tool.run(
            ReconcileResultArgs(
                agent_id="agent-1",
                expected_run_id="run-1",
                transcript_boundary="turn-1",
                verified_effects=effects,
            ),
            ctx,
        )
    )
    assert result == expected
    manager.reconcile_result.assert_awaited_once_with(
        "agent-1", "run-1", "turn-1", effects, requester_session_id="parent"
    )
    display = tool.get_result_display(
        ToolResultEvent(
            tool_name="reconcile_result",
            tool_class=ReconcileResult,
            result=result,
            tool_call_id="reconcile",
        )
    )
    assert display.success is True
    assert display.verb == "Prohibited"
    assert "run-1" in display.message and "prohibited" in display.message


@pytest.mark.asyncio
async def test_reconcile_result_accepts_the_zero_entry_submission() -> None:
    """Round-2 V4: the public tool port accepts the complete zero-entry
    accounting (one verified_effects entry per unknown tool call — a run
    that left no tool call outstanding submits an EMPTY list) and forwards
    it verbatim to the barrier.

    FAILS pre-fix: ReconcileResultArgs enforced min_length=1, so the
    zero-entry submission raised a ValidationError at the public port and
    the no-calls reconcile barrier was publicly unclearable (the only
    exits were retask replace_run=True or a damage report).
    """
    tool = _reconcile_result_tool()
    manager = FakeSubagentManager()
    ctx = InvokeContext(
        tool_call_id="reconcile", session_id="parent", subagent_manager=manager
    )
    result = await collect_result(
        tool.run(
            ReconcileResultArgs(
                agent_id="agent-1",
                expected_run_id="run-1",
                transcript_boundary="turn-1",
                verified_effects=[],
            ),
            ctx,
        )
    )
    assert result.outcome is ReconcileOutcome.NOT_PENDING
    assert manager.reconcile_calls == [("agent-1", "run-1", "turn-1", [], "parent")]


@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["missing", "manager", "requester", "subagent"])
async def test_reconcile_result_context_guards(context: str) -> None:
    """WP2: reconcile_result requires a subagent manager, a requester
    session, and a non-subagent (parent orchestrator) context.
    """
    tool = _reconcile_result_tool()
    manager = FakeSubagentManager()
    manager.reconcile_result = AsyncMock()
    ctx = InvokeContext(
        tool_call_id="reconcile", session_id="parent", subagent_manager=manager
    )
    if context == "missing":
        ctx = None
    elif context == "manager":
        ctx.subagent_manager = None
    elif context == "requester":
        ctx.session_id = None
    else:
        ctx.is_subagent = True
    with pytest.raises(ToolError):
        await collect_result(
            tool.run(
                ReconcileResultArgs(
                    agent_id="agent-1",
                    expected_run_id="run-1",
                    transcript_boundary="turn-1",
                    verified_effects=[
                        EffectVerification(tool_call_id="call-1", disposition="settled")
                    ],
                ),
                ctx,
            )
        )
    manager.reconcile_result.assert_not_awaited()


@pytest.mark.asyncio
async def test_reconcile_result_surfaces_manager_error_as_tool_error() -> None:
    tool = _reconcile_result_tool()
    manager = FakeSubagentManager()
    manager.reconcile_result = AsyncMock(side_effect=ValueError("boom"))
    ctx = InvokeContext(
        tool_call_id="reconcile", session_id="parent", subagent_manager=manager
    )
    with pytest.raises(ToolError, match="boom"):
        await collect_result(
            tool.run(
                ReconcileResultArgs(
                    agent_id="agent-1",
                    expected_run_id="run-1",
                    transcript_boundary="turn-1",
                    verified_effects=[
                        EffectVerification(tool_call_id="call-1", disposition="settled")
                    ],
                ),
                ctx,
            )
        )
