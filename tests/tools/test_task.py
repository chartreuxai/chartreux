from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.agents.manager import AgentManager
from chartreux.core.agents.models import BUILTIN_SUBAGENTS, AgentType
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.events import ToolResultEvent, ToolStreamEvent
from chartreux.core.llm.format import (
    APIToolFormatHandler,
    ParsedMessage,
    ParsedToolCall,
)
from chartreux.core.subagents import LaunchOutcome
from chartreux.core.tools.base import (
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.builtins.task import (
    Task,
    TaskArgs,
    TaskResult,
    TaskToolConfig,
)
from chartreux.core.tools.permissions import PermissionContext
from tests.conftest import ConfigBuilder, OrchestratorLoader
from tests.mock.utils import collect_result
from tests.stubs.fake_interaction_requests import FakeInteractionRequests


@pytest.fixture
def task_tool() -> Task:
    return Task(
        config_getter=lambda: TaskToolConfig(allowlist=["worker", "worker"]),
        state=BaseToolState(),
    )


class TestTaskArgs:
    def test_task_prompt_documents_single_model_expression_contract(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        assert "single model expression: canonical name or `@role`" in prompt
        assert "fallback array" not in prompt
        assert "tag or fallback" not in prompt

    def test_task_prompt_documents_background_launch_acknowledgment(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        assert 'status: "launched"' in prompt
        assert "not the subagent's terminal result" in prompt

    def test_task_prompt_documents_provider_failover_visibility(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        assert "metadata.switch_notices" in prompt
        assert "providers_used" in prompt
        assert "committed base model" in prompt

    def test_task_prompt_documents_single_preset_semantics(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()
        prompt = " ".join(prompt.split())

        assert '"model": "@large"' in prompt
        assert '"model": "strong"' not in prompt
        assert "A role such as `@large`" in prompt
        assert "@reviewers" not in prompt
        assert "one default model and thinking level" in prompt
        assert (
            "Roles are single presets. Launch separate tasks with explicit "
            "presets/models for multiple agents."
        ) in prompt

    def test_default_subagent_is_worker(self) -> None:
        args = TaskArgs(task="do something")
        assert args.agent_type == "worker"

    def test_custom_values(self) -> None:
        args = TaskArgs(
            task="do something",
            task_summary="Implement task tool validation",
            agent_type="worker",
        )
        assert args.task == "do something"
        assert args.task_summary == "Implement task tool validation"
        assert args.agent_type == "worker"


class TestTaskToolValidation:
    @pytest.fixture
    def ctx(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> InvokeContext:
        config = build_config()
        manager = AgentManager(load_orchestrator(config))
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            launch_context=dict[str, object](
                agent_entrypoint="cli",
                agent_version="1.0.0",
                client_name="vibe_cli",
                client_version="1.0.0",
                terminal_emulator="vscode",
            ),
        )

    @pytest.mark.asyncio
    async def test_rejects_nonexistent_agent(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="do something", agent_type="nonexistent")

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "Unknown agent_type profile: nonexistent" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_requires_agent_manager_in_context(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent_type="worker")
        ctx = InvokeContext(tool_call_id="test-call-id")  # No agent_manager

        with pytest.raises(ToolError) as exc_info:
            await collect_result(task_tool.run(args, ctx))

        assert "agent_manager" in str(exc_info.value).lower()

    def test_explore_agent_is_valid_subagent(self) -> None:
        agent = BUILTIN_SUBAGENTS["worker"]
        assert agent.agent_type == AgentType.SUBAGENT


class TestTaskToolResolvePermission:
    def test_worker_allowed_by_default(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent_type="worker")
        result = task_tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_unknown_agent_returns_none(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", agent_type="custom_agent")
        result = task_tool.resolve_permission(args)
        assert result is None

    def test_denylist_takes_precedence(self) -> None:
        config = TaskToolConfig(allowlist=["worker"], denylist=["worker"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent_type="worker")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_glob_pattern_in_allowlist(self) -> None:
        config = TaskToolConfig(allowlist=["work*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent_type="worker")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_glob_pattern_in_denylist(self) -> None:
        config = TaskToolConfig(denylist=["danger*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent_type="dangerous_agent")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_empty_lists_returns_none(self) -> None:
        config = TaskToolConfig(allowlist=[], denylist=[])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent_type="worker")
        result = tool.resolve_permission(args)
        assert result is None

    def test_default_config_has_worker_in_allowlist(self) -> None:
        config = TaskToolConfig()
        assert "worker" in config.allowlist


@pytest.mark.parametrize("outcome", list(LaunchOutcome))
def test_launch_outcome_controls_display(outcome: LaunchOutcome) -> None:
    result = TaskResult(
        response=outcome.value,
        turns_used=0,
        completed=outcome is LaunchOutcome.LAUNCHED,
        launch_outcome=outcome,
        agent_id="agent-1",
        run_id="run-1",
    )
    event = ToolResultEvent(
        tool_name="task", tool_class=Task, result=result, tool_call_id="call"
    )
    display = Task.get_result_display(event)
    assert (display.verb == "Launched") == (outcome is LaunchOutcome.LAUNCHED)
    assert display.success == (outcome is LaunchOutcome.LAUNCHED)
    assert (
        TaskResult.model_validate_json(result.model_dump_json()).launch_outcome
        is outcome
    )


class TestTaskToolExecution:
    @pytest.mark.asyncio
    async def test_instance_handle_as_type_fails_before_dispatch(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        runner = ctx.subagent_runner
        assert isinstance(runner, FakeSubagentRunner)
        with pytest.raises(ValidationError, match="agent_type takes a profile name"):
            await collect_result(
                task_tool.run(TaskArgs(task="work", agent_type="agent-1"), ctx)
            )
        assert runner.calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "values", [{}, {"agent_id": "agent-1", "background": False}]
    )
    async def test_replace_guards_before_delegation(
        self, task_tool: Task, ctx: InvokeContext, values
    ) -> None:
        runner = ctx.subagent_runner
        assert isinstance(runner, FakeSubagentRunner)
        with pytest.raises(ToolError, match="replace_run requires"):
            await collect_result(
                task_tool.run(TaskArgs(task="work", replace_run=True, **values), ctx)
            )
        assert not runner.calls

    @pytest.fixture
    def ctx(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> InvokeContext:
        config = build_config()
        manager = AgentManager(load_orchestrator(config))
        runner = FakeSubagentRunner([
            ToolStreamEvent(
                tool_name="task",
                message="read_file: completed",
                tool_call_id="test-call-id",
            ),
            TaskResult(response="done", turns_used=1, completed=True),
        ])
        return InvokeContext(
            tool_call_id="test-call-id",
            agent_manager=manager,
            interaction_requests=FakeInteractionRequests(),
            subagent_runner=runner,
            launch_context=dict[str, object](
                agent_entrypoint="cli",
                agent_version="1.0.0",
                client_name="vibe_cli",
                client_version="1.0.0",
                terminal_emulator="vscode",
            ),
        )

    @pytest.mark.asyncio
    async def test_happy_path_returns_subagent_response(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="explore the codebase", agent_type="worker")
        events = [event async for event in task_tool.run(args, ctx)]

        assert isinstance(events[0], ToolStreamEvent)
        assert events[0].message == "read_file: completed"
        assert events[1] == TaskResult(response="done", turns_used=1, completed=True)
        runner = ctx.subagent_runner
        assert isinstance(runner, FakeSubagentRunner)
        assert runner.calls == [(args, ctx)]

    @pytest.mark.asyncio
    async def test_requires_subagent_runner(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        ctx.subagent_runner = None

        with pytest.raises(ToolError, match="subagent runner"):
            await collect_result(
                task_tool.run(TaskArgs(task="do something", agent_type="worker"), ctx)
            )


class FakeSubagentRunner:
    def __init__(self, events: list[ToolStreamEvent | TaskResult]) -> None:
        self.events = events
        self.calls: list[tuple[TaskArgs, InvokeContext]] = []

    async def run(self, args: TaskArgs, ctx: InvokeContext):
        self.calls.append((args, ctx))
        for event in self.events:
            yield event


def _resolve_task_call(raw_args: dict[str, Any]):
    resolved = APIToolFormatHandler().resolve_tool_calls(
        ParsedMessage(
            tool_calls=[
                ParsedToolCall(tool_name="task", raw_args=raw_args, call_id="task-1")
            ]
        ),
        cast(Any, type("TaskOnlyManager", (), {"available_tools": {"task": Task}})()),
    )

    assert not resolved.failed_calls
    assert len(resolved.tool_calls) == 1
    return resolved.tool_calls[0]


class TestTaskToolFormatterHandoff:
    @pytest.fixture
    def formatter_ctx(
        self,
        build_config: ConfigBuilder,
        load_orchestrator: OrchestratorLoader[ChartreuxConfigSchema],
    ) -> InvokeContext:
        manager = AgentManager(load_orchestrator(build_config()))
        runner = FakeSubagentRunner([
            ToolStreamEvent(
                tool_name="task", message="read_file: completed", tool_call_id="task-1"
            ),
            TaskResult(response="done", turns_used=1, completed=True),
        ])
        return InvokeContext(
            tool_call_id="task-1", agent_manager=manager, subagent_runner=runner
        )

    @pytest.mark.asyncio
    async def test_json_string_config_decodes_at_invocation(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        call = _resolve_task_call({
            "task": "do something",
            "config": '{"model": "strong"}',
        })

        assert call.args_dict == {"task": "do something", "config": {"model": "strong"}}
        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert args.config is not None
        assert args.config.model == "strong"

    @pytest.mark.asyncio
    async def test_sparse_config_preserves_omitted_fields_at_invocation(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        call = _resolve_task_call({
            "task": "do something",
            "config": {"model": "strong"},
        })

        assert call.args_dict == {"task": "do something", "config": {"model": "strong"}}
        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert args.config is not None
        assert args.config.model == "strong"
        assert args.config.model_fields_set == {"model"}

    @pytest.mark.asyncio
    async def test_omitted_agent_stays_omitted_for_retained_agent(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        call = _resolve_task_call({
            "task": "do something",
            "agent_id": "retained-agent",
            "config": {},
        })

        assert "agent_type" not in call.args_dict
        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert args.agent_type == "worker"
        assert "agent_type" not in args.model_fields_set
        assert args.config is not None
        assert args.config.model_fields_set == set()

    @pytest.mark.asyncio
    async def test_explicit_agent_stays_explicit_at_invocation(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        call = _resolve_task_call({
            "task": "do something",
            "agent_type": "worker",
            "agent_id": "retained-agent",
        })

        assert call.args_dict["agent_type"] == "worker"
        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert args.agent_type == "worker"
        assert "agent_type" in args.model_fields_set

    def test_explicit_null_config_is_accepted_as_omitted(self) -> None:
        null_config = _resolve_task_call({"task": "do something", "config": None})
        omitted_config = _resolve_task_call({"task": "do something"})

        assert "config" not in null_config.args_dict
        assert "config" not in omitted_config.args_dict

    @pytest.mark.asyncio
    async def test_direct_invocation_treats_explicit_null_config_as_omitted(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        events = [
            event
            async for event in task_tool.invoke(
                formatter_ctx, task="do something", config=None
            )
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        assert runner.calls[-1][0].config is None

    @pytest.mark.asyncio
    async def test_full_config_still_validates_at_invocation(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        config = {
            "model": "strong",
            "instructions": "Use the strongest model.",
            "system_prompt_id": "default",
            "thinking": "medium",
            "enabled_tools": ["read_file"],
            "disabled_tools": ["bash"],
            "tools": {"bash": {"permission": "always"}},
        }
        call = _resolve_task_call({"task": "do something", "config": config})

        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert args.config is not None
        assert args.config.model_dump(exclude_unset=True) == config


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
async def test_nested_task_is_rejected_by_tool_and_direct_runner(
    task_tool: Task, background: bool
) -> None:
    args = TaskArgs(task="nested", agent_type="worker", background=background)
    nested_context = InvokeContext(
        tool_call_id="nested",
        is_subagent=True,
        agent_manager=cast(AgentManager, object()),
    )

    with pytest.raises(ToolError, match="depth limit|cannot be spawned"):
        await collect_result(task_tool.run(args, nested_context))

    async def ignore(*_: object) -> None:
        pass

    registry = SessionRuntimeRegistry(ignore, ignore, lambda _: 0)
    with pytest.raises(RuntimeError, match="depth limit"):
        await anext(registry.run(args, nested_context))


def test_background_task_result_display_uses_launch_handle() -> None:
    result = TaskResult(
        response="",
        turns_used=0,
        completed=True,
        status="launched",
        agent_id="agent-1",
        run_id="run-1",
        metadata={"base_model": "base", "active_provider": "provider"},
    )
    event = ToolResultEvent(
        tool_name="task", tool_class=Task, tool_call_id="task-1", result=result
    )

    display = Task.get_result_display(event)

    assert display.verb == "Launched"
    assert display.message == "agent agent-1 (run run-1): provider/base"
    assert Task.project_result(result) == {"models": ["provider/base"]}
