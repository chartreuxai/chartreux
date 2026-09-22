from __future__ import annotations

import asyncio

import pytest

from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.subagents import (
    AgentAvailability,
    AgentResultExpiredError,
    AgentSummary,
    ReleaseAgentOutcome,
    RunStatus,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
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
from chartreux.core.tools.builtins.release_agent import (
    ReleaseAgent,
    ReleaseAgentArgs,
    ReleaseAgentConfig,
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
    ) -> TaskResult:
        if self.wait_error is not None:
            raise self.wait_error
        key = (agent_id, run_id)
        if not any(agent.agent_id == agent_id for agent in self.agents):
            raise ValueError(f"Unknown agent: {agent_id}")
        event = self.completed.setdefault(key, asyncio.Event())
        if timeout is None:
            await event.wait()
        else:
            await asyncio.wait_for(event.wait(), timeout)
        return self.results[key]

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


@pytest.mark.asyncio
async def test_check_agents_returns_agent_summaries() -> None:
    summary = AgentSummary(
        agent_id="agent-1",
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id=None,
        current_run_status=None,
        initial_task_summary="Inspect agent management",
        current_task_summary=None,
        idle_seconds=3.5,
        ttl_remaining_seconds=None,
        effective_model="strong",
        base_model="base",
        active_provider="test/provider",
        effective_thinking="high",
    )
    result = await collect_result(
        _check_agents_tool().run(
            CheckAgentsArgs(), _context(FakeSubagentManager([summary]))
        )
    )

    serialized = result.model_dump(mode="json")["agents"][0]
    assert serialized == {
        "agent_id": "agent-1",
        "profile": "worker",
        "availability": "evicted",
        "current_run_id": None,
        "current_run_status": None,
        "turns_used": None,
        "initial_task_summary": "Inspect agent management",
        "current_task_summary": None,
        "idle_seconds": 3.5,
        "ttl_remaining_seconds": None,
        "effective_model": "strong",
        "base_model": "base",
        "active_provider": "test/provider",
        "effective_thinking": "high",
        "result_expired": False,
        "last_run_status": None,
    }


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
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id="run-1",
        current_run_status=RunStatus.COMPLETED,
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

    assert result.result == expected
    summary = AgentSummary(
        agent_id="agent-1",
        profile="worker",
        availability=AgentAvailability.RUNNING,
        current_run_id="run-1",
        current_run_status=RunStatus.RUNNING,
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

    assert (await wait).result == expected


@pytest.mark.asyncio
async def test_wait_for_agent_timeout_does_not_cancel_run() -> None:
    summary = AgentSummary(
        agent_id="agent-1",
        profile="worker",
        availability=AgentAvailability.RUNNING,
        current_run_id="run-1",
        current_run_status=RunStatus.RUNNING,
    )
    manager = FakeSubagentManager([summary])

    with pytest.raises(
        ToolError, match="Timed out waiting for agent agent-1 run run-1"
    ):
        await collect_result(
            _wait_for_agent_tool().run(
                WaitForAgentArgs(agent_id="agent-1", run_id="run-1", timeout=0.01),
                _context(manager),
            )
        )

    assert not manager.completed[("agent-1", "run-1")].is_set()


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
        profile="worker",
        availability=AgentAvailability.IDLE,
        current_run_id=None,
        current_run_status=None,
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
            profile="worker",
            availability=AgentAvailability.EVICTED,
            current_run_id=None,
            current_run_status=None,
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
                    task="work", agent="worker", background=False, agent_id="agent-1"
                ),
                ctx,
            )
        )

    ctx.is_subagent = True
    with pytest.raises(ToolError, match="Background agents cannot"):
        await collect_result(
            task.run(
                TaskArgs(
                    task="work", agent="worker", background=True, agent_id="agent-1"
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
