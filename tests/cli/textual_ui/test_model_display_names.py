from __future__ import annotations

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.agent_sidebar import AgentSidebar
from chartreux.core.events import ToolResultEvent
from chartreux.core.subagents import TaskMemberResult, TaskResult
from chartreux.core.tools.builtins.task import Task
from chartreux.model_display import format_model_display_name


def _agent(
    *,
    agent_id: str = "agent-1",
    base_model: str | None = "base",
    active_provider: str | None = "provider",
    effective_model: str | None = None,
) -> AgentSummaryModel:
    return AgentSummaryModel(
        agent_id=agent_id,
        profile="worker",
        availability="idle",
        effective_model=effective_model,
        base_model=base_model,
        active_provider=active_provider,
    )


def _result_event(result: TaskResult) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="task", tool_class=Task, tool_call_id="call-1", result=result
    )


def test_shared_display_name_helper_formats_provider_and_missing_values() -> None:
    assert format_model_display_name("mistral", "large") == "mistral/large"
    assert format_model_display_name("mistral", "mistral/large") == "mistral/large"
    assert format_model_display_name(None, "large") == "large"
    assert format_model_display_name("mistral", None) == "mistral"
    assert format_model_display_name(None, None) == "unknown"


def test_agent_summary_renders_effective_model_as_provider_model() -> None:
    sidebar = AgentSidebar()
    sidebar.update_agents((_agent(),))

    assert "Model: provider/base" in str(sidebar.render())


def test_agent_bar_and_sidebar_entries_render_provider_model() -> None:
    agent = _agent()
    bar = AgentBar()
    bar.update_agents((agent,))
    sidebar = AgentSidebar()
    sidebar.update_agents((agent,))

    assert "provider/base" in str(bar.render())
    assert "provider/base" in str(sidebar.render())


def test_launch_results_render_singular_and_fan_out_display_names() -> None:
    singular = TaskResult(
        response="",
        turns_used=0,
        completed=True,
        agent_id="agent-1",
        run_id="run-1",
        metadata={"base_model": "base", "active_provider": "provider"},
    )
    fan_out = TaskResult(
        response="",
        turns_used=0,
        completed=True,
        members=[
            TaskMemberResult(
                index=0,
                base_model="base-a",
                provider="provider-a",
                display_name="provider-a/wire-a",
                status="running",
            ),
            TaskMemberResult(
                index=1,
                base_model="base-b",
                provider="provider-b",
                display_name="provider-b/wire-b",
                status="running",
            ),
        ],
    )

    assert "provider/base" in Task.get_result_display(_result_event(singular)).message
    fan_out_display = Task.get_result_display(_result_event(fan_out)).message
    assert "provider-a/wire-a" in fan_out_display
    assert "provider-b/wire-b" in fan_out_display


def test_switch_notices_render_base_providers_and_reason() -> None:
    result = TaskResult(
        response="",
        turns_used=1,
        completed=True,
        metadata={
            "switch_notices": [
                {
                    "base_model": "base",
                    "old_provider": "first",
                    "new_provider": "second",
                    "reason": "timeout",
                }
            ]
        },
    )

    rendered = Task.get_result_display(_result_event(result)).message
    assert "Switched base: first → second (timeout)" in rendered
    assert Task.project_result(result) == {
        "switches": ["Switched base: first → second (timeout)"]
    }


def test_switched_agent_entry_uses_active_provider() -> None:
    agent = _agent(active_provider="second", effective_model="first/base")
    bar = AgentBar()
    bar.update_agents((agent,))
    sidebar = AgentSidebar()
    sidebar.update_agents((agent,))

    assert "second/base" in str(bar.render())
    assert "second/base" in str(sidebar.render())
    assert "first/base" not in str(sidebar.render())


def test_fan_out_batch_of_six_renders_each_member_identity() -> None:
    agents = tuple(
        _agent(
            agent_id=f"agent-{index}",
            base_model=f"base-{index}",
            active_provider=f"provider-{index}",
        )
        for index in range(6)
    )
    bar = AgentBar()
    bar.update_agents(agents)
    sidebar = AgentSidebar()
    sidebar.update_agents(agents)

    bar_rendered = str(bar.render())
    sidebar_rendered = str(sidebar.render())
    for index in range(6):
        identity = f"provider-{index}/base-{index}"
        assert identity in bar_rendered
        assert identity in sidebar_rendered
