from __future__ import annotations

import json

from pydantic import ValidationError
import pytest

from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.subagents import (
    AgentAvailability,
    AgentBusyError,
    AgentEvictedError,
    AgentEviction,
    AgentProfileMismatchError,
    AgentResultExpiredError,
    AgentSummary,
    CancelOutcome,
    CancelResult,
    EmptySubagentResponseError,
    LaunchConfigError,
    MissingAgentProfileError,
    RunStatus,
    RunStopReason,
    SubagentManagementError,
    SubagentRunAccumulator,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    normalize_task_summary,
)
from chartreux.core.tools.builtins.bash import Bash, CapturedShellResult


@pytest.mark.parametrize("outcome", list(CancelOutcome))
def test_cancel_result_contract_roundtrips(outcome: CancelOutcome) -> None:
    result = CancelResult(
        outcome=outcome, run_id="run", stop_reason=RunStopReason.USER_CANCELLED
    )
    assert CancelResult.model_validate_json(result.model_dump_json()) == result
    assert result.model_dump(mode="json") == {
        "outcome": outcome.value,
        "run_id": "run",
        "stop_reason": "user_cancelled",
    }
    assert CancelResult(outcome=outcome).run_id is None
    task = TaskResult(response="partial", turns_used=1, completed=False)
    assert "stop_reason" not in task.model_dump()
    task.stop_reason = result.stop_reason
    assert json.loads(task.model_dump_json())["stop_reason"] == "user_cancelled"


def test_task_profile_name_schema_has_no_agent_alias() -> None:
    properties = TaskArgs.model_json_schema()["properties"]
    assert "agent" not in properties
    assert properties["agent_type"]["type"] == "string"
    assert "agent_id" in properties["agent_type"]["description"]
    assert "agent_type" in properties["agent_id"]["description"]
    with pytest.raises(ValidationError) as error:
        TaskArgs.model_validate({"task": "work", "agent": "worker"})
    assert error.value.errors()[0]["loc"] == ("agent",)
    assert error.value.errors()[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("handle", ["agent-0", "agent-1", "agent-123"])
def test_task_profile_rejects_instance_handles(handle: str) -> None:
    with pytest.raises(
        ValidationError, match="agent_type takes a profile name"
    ) as error:
        TaskArgs(task="work", agent_type=handle)
    assert "agent_id (with background: true)" in str(error.value)


@pytest.mark.parametrize(
    "profile", ["worker", "custom-worker", "agent-helper", "agent-1-extra"]
)
def test_task_profile_accepts_non_handle_names(profile: str) -> None:
    assert TaskArgs(task="work", agent_type=profile).agent_type == profile


def test_replace_run_default_and_schema() -> None:
    args = TaskArgs(task="work")
    assert args.replace_run is False
    field = TaskArgs.model_json_schema()["properties"]["replace_run"]
    assert field["type"] == "boolean" and field["default"] is False
    assert (
        "launch_outcome"
        not in TaskResult(response="done", turns_used=1, completed=True).model_dump()
    )


def test_summary_keeps_last_recorded_context_and_terminal_outcome() -> None:
    summary = AgentSummary(
        agent_id="agent",
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id="run",
        current_run_status=RunStatus.COMPLETED,
        context_tokens=123,
        context_window=456,
        stop_reason=RunStopReason.BUDGET_UNVERIFIABLE,
    )
    assert (summary.context_tokens, summary.context_window, summary.compacting) == (
        123,
        456,
        False,
    )
    assert summary.stop_reason is RunStopReason.BUDGET_UNVERIFIABLE
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        AssistantEvent(content="Budget exceeded", stopped_by_middleware=True),
        tool_call_id="task",
    )
    assert not accumulator.build_result(turns_used=1).completed
    # TaskResult's completion flag and prose do not encode a structured reason.
    assert "stop_reason" not in accumulator.build_result(turns_used=1).model_dump()


def test_subagent_run_accumulates_response_and_tool_progress() -> None:
    accumulator = SubagentRunAccumulator()

    assert (
        accumulator.observe(
            AssistantEvent(content="Found the issue"), tool_call_id="task-1"
        )
        is None
    )
    progress = accumulator.observe(
        ToolResultEvent(
            tool_name="bash",
            tool_class=Bash,
            result=CapturedShellResult(command="pwd", stdout="/repo", stderr=""),
            tool_call_id="bash-1",
        ),
        tool_call_id="task-1",
    )

    assert progress is not None
    assert progress.tool_call_id == "task-1"
    assert progress.message == "bash: Ran pwd"
    assert accumulator.build_result(turns_used=1) == TaskResult(
        response="Found the issue", turns_used=1, completed=True
    )


def test_subagent_run_combines_observed_and_runtime_failures() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        AssistantEvent(content="Partial", stopped_by_middleware=True),
        tool_call_id="task-1",
    )
    accumulator.record_error("child failed")

    assert accumulator.build_result(turns_used=2, completed=False) == TaskResult(
        response="Partial\n[Subagent error: child failed]",
        turns_used=2,
        completed=False,
    )


def test_task_args_supports_background_launch_and_agent_reuse() -> None:
    args = TaskArgs(task="inspect the repository", background=True, agent_id="agent-1")

    assert args.background is True
    assert args.agent_id == "agent-1"


def test_task_args_defaults_to_background() -> None:
    args = TaskArgs.model_validate({
        "task": "inspect the repository",
        "agent_type": "worker",
    })

    assert args.background is True
    assert args.agent_id is None


def test_task_args_launch_config_json_string_decodes_before_validation() -> None:
    args = TaskArgs.model_validate({"task": "inspect", "config": '{"model": "strong"}'})

    assert args.config is not None
    assert args.config.model == "strong"
    assert args.config.model_fields_set == {"model"}


@pytest.mark.parametrize(
    ("config", "error"),
    [
        ("not JSON", "config must be a valid JSON object"),
        ("[]", "config JSON must decode to an object"),
        ("42", "config JSON must decode to an object"),
    ],
)
def test_task_args_rejects_invalid_json_string_launch_config(
    config: str, error: str
) -> None:
    with pytest.raises(ValidationError, match=error):
        TaskArgs.model_validate({"task": "inspect", "config": config})


def test_task_args_launch_config_preserves_supplied_fields() -> None:
    args = TaskArgs.model_validate({
        "task": "inspect",
        "config": {
            "instructions": "",
            "enabled_tools": [],
            "tools": {"bash": {"permission": "always"}},
        },
    })

    assert args.config is not None
    config = args.config
    assert config.model_fields_set == {"instructions", "enabled_tools", "tools"}
    assert config.tools is not None
    assert config.tools["bash"].model_fields_set == {"permission"}
    assert config.model_dump(exclude_unset=True) == {
        "instructions": "",
        "enabled_tools": [],
        "tools": {"bash": {"permission": "always"}},
    }
    assert "agent_type" not in args.model_fields_set
    assert "config" in args.model_fields_set


@pytest.mark.parametrize(
    ("data", "location"),
    [
        ({"task": "inspect", "unknown": True}, ("unknown",)),
        ({"task": "inspect", "config": {"hooks": {}}}, ("config", "hooks")),
        ({"task": "inspect", "config": '{"hooks": {}}'}, ("config", "hooks")),
        (
            {"task": "inspect", "config": {"tools": {"bash": {"denylist": []}}}},
            ("config", "tools", "bash", "denylist"),
        ),
        ({"task": "inspect", "config": {"mcp_servers": []}}, ("config", "mcp_servers")),
        ({"task": "inspect", "config": {"tool_paths": []}}, ("config", "tool_paths")),
    ],
)
def test_task_args_rejects_unknown_launch_config_fields(
    data: dict[str, object], location: tuple[str, ...]
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        TaskArgs.model_validate(data)

    assert exc_info.value.errors()[0]["loc"] == location


@pytest.mark.parametrize(
    ("data", "expected_config_fields", "config_supplied"),
    [
        ({"task": "inspect", "config": None}, None, False),
        ({"task": "inspect", "config": "null"}, None, False),
        ({"task": "inspect", "config": {"model": None}}, set(), True),
    ],
)
def test_task_args_treats_explicit_launch_config_nulls_as_omitted(
    data: dict[str, object],
    expected_config_fields: set[str] | None,
    config_supplied: bool,
) -> None:
    args = TaskArgs.model_validate(data)

    assert (args.config is not None) is config_supplied
    if args.config is not None:
        assert args.config.model_fields_set == expected_config_fields
    assert ("config" in args.model_fields_set) is config_supplied


def test_launch_config_errors_expose_safe_field_paths() -> None:
    error = MissingAgentProfileError("config.model")

    assert isinstance(error, LaunchConfigError)
    assert error.field == error.field_path == "config.model"
    assert str(error) == "Invalid launch configuration at config.model"


def test_task_summary_normalization_and_fallback_preview() -> None:
    args = TaskArgs(task="inspect", task_summary="  inspect\n  the   repository  ")
    task = "  inspect\n  the   repository  "

    assert args.task_summary == "inspect the repository"
    assert normalize_task_summary(None, fallback=task) == "inspect the repository"
    assert normalize_task_summary("word " * 100) == ("word " * 100)[:240]


def test_task_result_handles_default_to_none() -> None:
    foreground_result = TaskResult(response="done", turns_used=1, completed=True)
    background_result = TaskResult(
        response="", turns_used=0, completed=True, agent_id="agent-1", run_id="run-1"
    )

    assert foreground_result.agent_id is None
    assert foreground_result.run_id is None
    assert background_result.agent_id == "agent-1"
    assert background_result.run_id == "run-1"


def test_singular_task_result_serialization_omits_members() -> None:
    result = TaskResult(response="done", turns_used=1, completed=True)

    assert "members" not in result.model_dump()
    assert "members" not in result.model_dump_json()


def test_singular_task_result_members_omission_survives_nested_json_serialization() -> (
    None
):
    result = TaskResult(response="done", turns_used=1, completed=True)

    assert json.loads(json.dumps({"result": result.model_dump(mode="json")})) == {
        "result": {
            "response": "done",
            "turns_used": 1,
            "completed": True,
            "agent_id": None,
            "run_id": None,
            "metadata": None,
        }
    }


def test_fan_out_is_rejected_for_single_preset_roles() -> None:
    message = (
        "Roles are single presets. Launch separate tasks with explicit presets/models "
        "for multiple agents."
    )

    with pytest.raises(ValidationError, match=message):
        TaskArgs.model_validate({"task": "inspect", "fan_out": True})

    assert "fan_out" not in TaskArgs.model_json_schema()["properties"]
    assert TaskArgs.model_validate({"task": "inspect", "fan_out": False}) == TaskArgs(
        task="inspect"
    )


def test_background_subagent_handle_types() -> None:
    assert [status.value for status in RunStatus] == [
        "running",
        "completed",
        "failed",
        "cancelled",
    ]
    assert [availability.value for availability in AgentAvailability] == [
        "running",
        "finalizing",
        "idle",
        "evicted",
    ]
    summary = AgentSummary(
        agent_id="agent-1",
        profile="worker",
        availability=AgentAvailability.RUNNING,
        current_run_id="run-1",
        current_run_status=RunStatus.RUNNING,
    )
    assert summary == AgentSummary(
        agent_id="agent-1",
        profile="worker",
        availability=AgentAvailability.RUNNING,
        current_run_id="run-1",
        current_run_status=RunStatus.RUNNING,
    )
    assert summary.initial_task_summary is None
    assert summary.current_task_summary is None
    assert summary.idle_seconds is None
    assert summary.ttl_remaining_seconds is None
    assert AgentAvailability.EVICTED.value == "evicted"


def test_subagent_accumulator_rejects_empty_successful_result() -> None:
    with pytest.raises(EmptySubagentResponseError, match="empty response"):
        SubagentRunAccumulator().build_result(turns_used=0)


@pytest.mark.parametrize("response", ["", " \t\n"])
@pytest.mark.parametrize("completed", [False, True])
def test_subagent_accumulator_gates_only_prospective_completion(
    response: str, completed: bool
) -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content=response), tool_call_id="task")

    if completed:
        with pytest.raises(EmptySubagentResponseError, match="empty response"):
            accumulator.build_result(turns_used=1, completed=completed)
    else:
        assert accumulator.build_result(
            turns_used=1, completed=completed
        ) == TaskResult(response=response, turns_used=1, completed=False)


@pytest.mark.parametrize("completed", [False, True])
def test_subagent_accumulator_allows_empty_middleware_stopped_result(
    completed: bool,
) -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(
        AssistantEvent(content="", stopped_by_middleware=True), tool_call_id="task"
    )

    assert accumulator.build_result(turns_used=1, completed=completed) == TaskResult(
        response="", turns_used=1, completed=False
    )


def test_subagent_accumulator_nonempty_result_has_no_handles() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator.observe(AssistantEvent(content="  done\n"), tool_call_id="task")
    result = accumulator.build_result(turns_used=1)

    assert result.response == "  done\n"
    assert result.completed
    assert result.agent_id is None
    assert result.run_id is None


def test_lifecycle_errors_and_eviction_metadata_are_distinguishable() -> None:
    errors = [
        AgentEvictedError(),
        AgentBusyError(),
        AgentProfileMismatchError(),
        AgentResultExpiredError(),
        UnknownAgentError(),
    ]

    assert all(isinstance(error, SubagentManagementError) for error in errors)
    assert len({type(error) for error in errors}) == len(errors)
    assert (
        AgentEviction(
            agent_id="agent-1",
            run_id="run-1",
            reason="ttl",
            idle_duration_seconds=1.5,
            root_generation=2,
        ).reason
        == "ttl"
    )
