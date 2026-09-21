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
    LaunchConfigError,
    MissingAgentProfileError,
    RunStatus,
    SubagentManagementError,
    SubagentRunAccumulator,
    TaskArgs,
    TaskMemberResult,
    TaskResult,
    UnknownAgentError,
    normalize_task_summary,
)
from chartreux.core.tools.builtins.bash import Bash, CapturedShellResult


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
        "agent": "worker",
    })

    assert args.background is True
    assert args.agent_id is None


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
    assert "agent" not in args.model_fields_set
    assert "config" in args.model_fields_set


@pytest.mark.parametrize(
    ("data", "location"),
    [
        ({"task": "inspect", "unknown": True}, ("unknown",)),
        ({"task": "inspect", "config": {"hooks": {}}}, ("config", "hooks")),
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


def test_fan_out_collection_serializes_exactly_in_dump_and_json() -> None:
    result = TaskResult(
        response="",
        turns_used=3,
        completed=False,
        members=[
            TaskMemberResult(
                index=0,
                base_model="small",
                provider="test/first",
                display_name="test/first/small-wire",
                status="completed",
                agent_id="agent-1",
                run_id="run-1",
                result="done",
            ),
            TaskMemberResult(
                index=1,
                base_model="large",
                provider="test/second",
                display_name="test/second/large-wire",
                status="failed",
                error={"code": "runtime_failed", "message": "boom"},
            ),
        ],
    )
    expected = {
        "response": "",
        "turns_used": 3,
        "completed": False,
        "agent_id": None,
        "run_id": None,
        "metadata": None,
        "members": [
            {
                "index": 0,
                "base_model": "small",
                "provider": "test/first",
                "display_name": "test/first/small-wire",
                "status": "completed",
                "agent_id": "agent-1",
                "run_id": "run-1",
                "result": "done",
                "error": None,
            },
            {
                "index": 1,
                "base_model": "large",
                "provider": "test/second",
                "display_name": "test/second/large-wire",
                "status": "failed",
                "agent_id": None,
                "run_id": None,
                "result": None,
                "error": {"code": "runtime_failed", "message": "boom"},
            },
        ],
    }

    assert result.model_dump() == expected
    assert json.loads(result.model_dump_json()) == expected


def test_fan_out_member_optional_fields_are_omitted_when_requested() -> None:
    member = TaskMemberResult(
        index=0,
        base_model="small",
        provider="test/first",
        display_name="test/first/small-wire",
        status="running",
    )

    assert member.model_dump(exclude_none=True) == {
        "index": 0,
        "base_model": "small",
        "provider": "test/first",
        "display_name": "test/first/small-wire",
        "status": "running",
    }


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


def test_subagent_accumulator_result_has_no_handles() -> None:
    result = SubagentRunAccumulator().build_result(turns_used=0)

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
