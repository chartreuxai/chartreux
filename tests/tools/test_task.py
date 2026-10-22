from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError
import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.agents.identities import builtin_identity_names
from chartreux.core.agents.identity_launch import ModelShorthand, parse_model_shorthand
from chartreux.core.agents.manager import AgentManager
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.dispatch.renderer import (
    render_task_description_for_composition,
    task_skeleton,
)
from chartreux.core.events import ToolCallEvent, ToolResultEvent, ToolStreamEvent
from chartreux.core.llm.format import (
    APIToolFormatHandler,
    ParsedMessage,
    ParsedToolCall,
)
from chartreux.core.model_catalog.defaults import SHIPPED_CATALOG
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
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
from tests.conftest import (
    ConfigBuilder,
    OrchestratorLoader,
    build_test_vibe_config,
    multi_model_catalog,
    single_model_catalog,
)
from tests.mock.utils import collect_result
from tests.stubs.fake_interaction_requests import FakeInteractionRequests


@pytest.fixture
def task_tool() -> Task:
    return Task(
        config_getter=lambda: TaskToolConfig(allowlist=["coder"]), state=BaseToolState()
    )


def _rendered_task(catalog: ModelCatalog) -> tuple[str, str]:
    """The composed task tool description for a catalog, and its session model."""
    config = build_test_vibe_config(
        include_model_info=False,
        include_prompt_detail=False,
        include_project_context=False,
        include_commit_signature=False,
    ).attach_catalog_snapshot(CatalogSnapshot(catalog, "task-prompt"))
    alias = config.get_active_model().alias
    return (
        render_task_description_for_composition(
            routes=dict(config.purpose_routes.routes),
            session_model=alias,
            skeleton=task_skeleton(),
        ),
        alias,
    )


class TestTaskArgs:
    def test_task_prompt_documents_canonical_launch_fields(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        assert "`model`, `identity`, `purpose`, and `seat`" in prompt
        assert "`@<model>[:<identity>]`" in prompt
        assert "`:researcher` for a bare built-in identity" in prompt
        assert "`:user:foo` for a `user:`-qualified custom id" in prompt
        assert "`builtin:`- or `user:`-qualified id" in prompt
        assert "default child identity is `coder`" in prompt
        assert "supplying both `model` and `config.model`" in prompt
        # The retired per-launch prompt-id selection no longer rides in config.
        assert "system_prompt_id" not in prompt
        assert "fallback array" not in prompt
        assert "tag or fallback" not in prompt

    def test_task_prompt_documents_identity_keyed_denylist(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        assert "the task denylist keys on the resolved identity" in prompt
        assert "`Task denied for launch identity`" in prompt
        assert "Task denied for agent profile" not in prompt

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

    @pytest.mark.parametrize(
        "catalog", [multi_model_catalog(), single_model_catalog(), SHIPPED_CATALOG]
    )
    def test_task_prompt_documents_single_preset_semantics(
        self, catalog: ModelCatalog
    ) -> None:
        prompt, alias = _rendered_task(catalog)
        prompt = " ".join(prompt.split())

        # One configured model and ten produce identical routing: the
        # launch-target table binds the session model for every purpose.
        assert f"| `implementation` | `coder` | substantive | — | `{alias}` |" in prompt
        assert "A purpose route binds the launch defaults for one purpose" in prompt
        assert (
            "Explicit `model` and `identity` fields override the route's "
            "corresponding fields"
        ) in prompt
        assert "@reviewers" not in prompt
        assert (
            "Roles are single presets. Launch separate tasks with explicit "
            "presets/models for multiple agents."
        ) in prompt

    @pytest.mark.parametrize(
        "catalog", [multi_model_catalog(), single_model_catalog(), SHIPPED_CATALOG]
    )
    def test_task_prompt_documents_mechanical_tier_routes(
        self, catalog: ModelCatalog
    ) -> None:
        prompt, alias = _rendered_task(catalog)
        prompt = " ".join(prompt.split())

        # The mechanical-tier purposes render with their tier in the
        # launch-target table, bound to the session model.
        assert f"| `mechanical-edit` | `coder` | mechanical | — | `{alias}` |" in prompt
        assert f"| `search` | `researcher` | mechanical | — | `{alias}` |" in prompt
        assert f"| `verification` | `coder` | mechanical | — | `{alias}` |" in prompt
        # The reuse prose re-states the purpose on every retask.
        assert "re-state the purpose explicitly on every reuse" in prompt
        # The retired slot-prose examples never render.
        assert "previously `@scout`" not in prompt
        assert 'config={"model": "@scout"}' not in prompt
        assert '"model": "@heavy"' not in prompt

    def test_task_prompt_opener_covers_routine_bounded_delegation(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        assert "bounded delegated work of any size" in prompt
        assert "do not treat delegation as limited to exceptional work" in prompt
        # The retired opener and escalation example are gone.
        assert "complex multi-step work" not in prompt
        assert "To escalate an idle agent" not in prompt
        assert "stronger `model`" not in prompt
        assert "for corrective work, including" not in prompt

    def test_task_prompt_carries_worker_report_disclosure_region(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        # The disclosure region is rendered scaffolding: the orchestrator reads
        # that every brief must carry the denial record and touched-path set.
        assert "$dispatch_disclosure" in prompt

    @pytest.mark.parametrize(
        "catalog", [multi_model_catalog(), single_model_catalog(), SHIPPED_CATALOG]
    )
    def test_rendered_task_description_carries_worker_report_disclosure(
        self, catalog: ModelCatalog
    ) -> None:
        rendered, _alias = _rendered_task(catalog)

        assert "Every task brief must carry the worker disclosure requirement" in (
            rendered
        )
        assert "including edits later reverted" in rendered

    def test_default_launch_target_is_unset_identity(self) -> None:
        args = TaskArgs(task="do something")
        assert "identity" not in args.model_fields_set
        # No canonical target: the default child identity is resolved at launch.
        assert args.identity is None

    def test_custom_values(self) -> None:
        args = TaskArgs(
            task="do something",
            task_summary="Implement task tool validation",
            identity="coder",
        )
        assert args.task == "do something"
        assert args.task_summary == "Implement task tool validation"
        assert args.identity == "coder"

    @pytest.mark.parametrize(
        "profile", ["worker", "advisor", "reviewer", "custom-agent"]
    )
    def test_legacy_agent_type_bridge_is_retired(self, profile: str) -> None:
        # The package-11b bridge is gone with the profile axis (16a): the
        # legacy field fails with the successor guidance instead of mapping.
        with pytest.raises(ValidationError, match="agent_type is retired") as error:
            TaskArgs.model_validate({"task": "work", "agent_type": profile})
        assert "identity field" in str(error.value)

    def test_canonical_fields_ride_on_the_task_args_surface(self) -> None:
        args = TaskArgs.model_validate({
            "task": "research the thing",
            "model": "@fast:researcher",
            "purpose": "research",
        })
        assert (args.model, args.identity, args.purpose) == (
            "@fast:researcher",
            None,
            "research",
        )
        # The identity component rides in the shorthand until resolution.
        assert args.model is not None
        assert parse_model_shorthand(args.model) == ModelShorthand(
            "@fast", "researcher"
        )

    def test_shorthand_identity_component_conflicts_with_structured_identity(
        self,
    ) -> None:
        with pytest.raises(ValidationError, match="carries an identity component"):
            TaskArgs.model_validate({
                "task": "work",
                "model": "@fast:researcher",
                "identity": "coder",
            })

    def test_dual_model_supply_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="supply exactly one"):
            TaskArgs.model_validate({
                "task": "work",
                "model": "@fast",
                "config": {"model": "fast"},
            })

    def test_seat_without_purpose_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="seat is supplied without purpose"):
            TaskArgs.model_validate({"task": "work", "seat": "analytical"})

    def test_task_args_round_trips_through_serialized_fields(self) -> None:
        for original in (
            TaskArgs(task="work"),
            TaskArgs.model_validate({"task": "work", "identity": "reviewer"}),
            TaskArgs.model_validate({"task": "work", "identity": "researcher"}),
            TaskArgs.model_validate({"task": "work", "model": "@fast:researcher"}),
        ):
            revalidated = TaskArgs.model_validate(original.model_dump())
            assert revalidated == original
            assert revalidated.identity == original.identity


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
    async def test_tool_runs_without_an_agent_manager(self, task_tool: Task) -> None:
        # Identity resolution moved to the launch boundary (package 16a):
        # the task tool itself never consults an agent manager.
        runner = FakeSubagentRunner([
            TaskResult(response="done", turns_used=1, completed=True)
        ])
        ctx = InvokeContext(tool_call_id="test-call-id", subagent_runner=runner)
        args = TaskArgs(task="do something", identity="coder")

        result = await collect_result(task_tool.run(args, ctx))

        assert result.completed
        assert runner.calls == [(args, ctx)]

    def test_coder_is_a_builtin_child_identity(self) -> None:
        # The shipped profile registry is gone; the launch target is a
        # built-in identity (the registry's "worker" profile maps to coder).
        assert "coder" in builtin_identity_names()


class TestTaskToolResolvePermission:
    def test_explicit_coder_identity_is_allowed_by_default(
        self, task_tool: Task
    ) -> None:
        args = TaskArgs(task="do something", identity="coder")
        result = task_tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_default_launch_matches_default_identity(self, task_tool: Task) -> None:
        result = task_tool.resolve_permission(TaskArgs(task="do something"))
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_other_identity_returns_none_under_default_allowlist(
        self, task_tool: Task
    ) -> None:
        args = TaskArgs(task="do something", identity="researcher")
        result = task_tool.resolve_permission(args)
        assert result is None

    def test_unknown_identity_returns_none(self, task_tool: Task) -> None:
        args = TaskArgs(task="do something", identity="custom_agent")
        result = task_tool.resolve_permission(args)
        assert result is None

    def test_denylist_takes_precedence(self) -> None:
        config = TaskToolConfig(allowlist=["coder"], denylist=["coder"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", identity="coder")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_denylist_keys_on_the_resolved_identity(self) -> None:
        config = TaskToolConfig(allowlist=["coder"], denylist=["research*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", identity="researcher")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_shorthand_identity_component_keys_the_denylist(self) -> None:
        config = TaskToolConfig(denylist=["research*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", model="@fast:researcher")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_glob_pattern_in_allowlist(self) -> None:
        config = TaskToolConfig(allowlist=["co*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", identity="coder")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.ALWAYS

    def test_glob_pattern_in_denylist(self) -> None:
        config = TaskToolConfig(denylist=["danger*"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", identity="dangerous_agent")
        result = tool.resolve_permission(args)
        assert isinstance(result, PermissionContext)
        assert result.permission is ToolPermission.NEVER

    def test_empty_lists_returns_none(self) -> None:
        config = TaskToolConfig(allowlist=[], denylist=[])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", identity="coder")
        result = tool.resolve_permission(args)
        assert result is None

    def test_retained_reuse_without_target_skips_the_denylist(self) -> None:
        config = TaskToolConfig(denylist=["coder"])
        tool = Task(config_getter=lambda: config, state=BaseToolState())
        args = TaskArgs(task="do something", agent_id="agent-1", background=True)
        result = tool.resolve_permission(args)
        assert result is None

    def test_default_config_has_coder_in_allowlist(self) -> None:
        config = TaskToolConfig()
        assert "coder" in config.allowlist


def test_call_display_keys_on_the_launch_identity() -> None:
    event = ToolCallEvent(
        tool_call_id="call-1",
        tool_name="task",
        tool_class=Task,
        args=TaskArgs(task="map the surface", identity="researcher"),
    )
    assert (
        Task.get_call_display(event).message
        == "researcher agent (background): map the surface"
    )

    default = ToolCallEvent(
        tool_call_id="call-2",
        tool_name="task",
        tool_class=Task,
        args=TaskArgs(task="do something"),
    )
    assert (
        Task.get_call_display(default).message
        == "coder agent (background): do something"
    )

    foreground = ToolCallEvent(
        tool_call_id="call-4",
        tool_name="task",
        tool_class=Task,
        args=TaskArgs(task="map the surface", identity="researcher", background=False),
    )
    assert (
        Task.get_call_display(foreground).message == "researcher agent: map the surface"
    )


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
    async def test_legacy_agent_type_fails_before_dispatch(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        runner = ctx.subagent_runner
        assert isinstance(runner, FakeSubagentRunner)
        # The retired bridge fails at argument validation — before dispatch.
        result = await collect_result(
            task_tool.run(TaskArgs(task="work", identity="coder"), ctx)
        )
        assert result.response == "done"
        assert len(runner.calls) == 1

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
    async def test_run_denylist_keys_on_launch_identity(
        self, ctx: InvokeContext
    ) -> None:
        denied = Task(
            config_getter=lambda: TaskToolConfig(denylist=["research*"]),
            state=BaseToolState(),
        )
        args = TaskArgs(task="do something", identity="researcher")

        with pytest.raises(
            ToolError, match="Task denied for launch identity: researcher"
        ):
            await collect_result(denied.run(args, ctx))

        runner = ctx.subagent_runner
        assert isinstance(runner, FakeSubagentRunner)
        assert not runner.calls

    @pytest.mark.asyncio
    async def test_happy_path_returns_subagent_response(
        self, task_tool: Task, ctx: InvokeContext
    ) -> None:
        args = TaskArgs(task="explore the codebase", identity="coder")
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
                task_tool.run(TaskArgs(task="do something", identity="coder"), ctx)
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

        assert "identity" not in call.args_dict
        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert "identity" not in args.model_fields_set
        assert args.config is not None
        assert args.config.model_fields_set == set()

    @pytest.mark.asyncio
    async def test_explicit_identity_stays_explicit_at_invocation(
        self, task_tool: Task, formatter_ctx: InvokeContext
    ) -> None:
        call = _resolve_task_call({
            "task": "do something",
            "identity": "coder",
            "agent_id": "retained-agent",
        })

        assert call.args_dict["identity"] == "coder"
        events = [
            event async for event in task_tool.invoke(formatter_ctx, **call.args_dict)
        ]

        assert len(events) == 2
        runner = cast(FakeSubagentRunner, formatter_ctx.subagent_runner)
        args, _ = runner.calls[-1]
        assert args.identity == "coder"
        assert "identity" in args.model_fields_set

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
    args = TaskArgs(task="nested", identity="coder", background=background)
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

    def test_task_prompt_documents_working_set_handoff(self) -> None:
        prompt = (
            Path(__file__).parents[2] / "chartreux/core/tools/builtins/prompts/task.md"
        ).read_text()

        # The dispatch documentation surfaces the spawn handoff: the working-set
        # files with revisions and outlines, and the append-only state blocks.
        assert "file_state_handoff" in prompt
        assert "working-set files" in prompt
        assert "<file_state>" in prompt
        assert "never rewritten" in prompt
