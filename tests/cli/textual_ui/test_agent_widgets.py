from __future__ import annotations

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.events import AgentsUpdate
from chartreux.app_server.protocol import AgentEvictionModel, AgentSummaryModel
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar, pack_agent_chips
from chartreux.cli.textual_ui.widgets.agent_sidebar import AgentSidebar


def _agent(
    *,
    agent_id: str = "agent-1",
    profile: str = "worker",
    availability: str = "idle",
    run_id: str | None = None,
    run_status: str | None = None,
    last_run_status: str | None = None,
    initial_task_summary: str | None = None,
    current_task_summary: str | None = None,
    idle_seconds: float | None = None,
    ttl_remaining_seconds: float | None = None,
    effective_model: str | None = None,
    base_model: str | None = None,
    active_provider: str | None = None,
    effective_thinking: str | None = None,
    result_expired: bool = False,
) -> AgentSummaryModel:
    return AgentSummaryModel(
        agent_id=agent_id,
        profile=profile,
        availability=availability,
        current_run_id=run_id,
        current_run_status=run_status,
        last_run_status=last_run_status,
        initial_task_summary=initial_task_summary,
        current_task_summary=current_task_summary,
        idle_seconds=idle_seconds,
        ttl_remaining_seconds=ttl_remaining_seconds,
        effective_model=effective_model,
        base_model=base_model,
        active_provider=active_provider,
        effective_thinking=effective_thinking,
        result_expired=result_expired,
    )


def test_agent_bar_renders_agent_states_and_hides_when_empty() -> None:
    widget = AgentBar()
    widget.update_agents((
        _agent(availability="running", run_status="running"),
        _agent(agent_id="agent-2"),
        _agent(agent_id="agent-3", run_status="failed"),
    ))

    rendered = str(widget.render())

    assert "⠋ agent-1·worker" in rendered
    assert "✓ agent-2·worker" in rendered
    assert "✗ agent-3·worker" in rendered
    widget.update_agents(())
    assert not widget.display
    assert str(widget.render()) == ""


def test_agent_bar_renders_failed_last_run_for_idle_agent() -> None:
    widget = AgentBar()
    widget.update_agents((_agent(last_run_status="failed"),))

    rendered = str(widget.render())

    assert "✗ agent-1·worker" in rendered
    assert "✓ agent-1·worker" not in rendered


def test_agent_bar_renders_evicted_and_unknown_states_safely() -> None:
    widget = AgentBar()
    widget.update_agents((
        _agent(availability="evicted", run_status="running"),
        _agent(agent_id="agent-2", availability="unexpected"),
    ))

    rendered = str(widget.render())

    assert "⊘ agent-1·worker evicted" in rendered
    assert "? agent-2·worker" in rendered
    assert "✓ agent-1·worker" not in rendered
    assert "⠋ agent-1·worker" not in rendered


def test_agent_bar_packs_few_agents_on_one_line() -> None:
    chips = [f"✓ agent-{index}·worker" for index in range(1, 5)]

    assert pack_agent_chips(chips, width=100) == ["  ".join(chips)]


def test_agent_bar_wraps_whole_chips() -> None:
    chips = [f"✓ agent-{index}·worker" for index in range(1, 7)]

    lines = pack_agent_chips(chips, width=40)

    assert len(lines) == 3
    assert all(chip in "\n".join(lines) for chip in chips)
    assert all(len(line) <= 40 for line in lines)


def test_agent_bar_caps_overflow_at_three_lines() -> None:
    chips = [f"✓ agent-{index}·worker" for index in range(1, 16)]

    lines = pack_agent_chips(chips, width=40, max_lines=3)

    assert len(lines) == 3
    assert lines[-1].endswith("+10 more")
    assert all(chip in "\n".join(lines) for chip in chips[:5])
    assert all(chip not in "\n".join(lines) for chip in chips[5:])


def test_agent_bar_reflows_for_narrower_width() -> None:
    chips = [f"✓ agent-{index}·worker" for index in range(1, 7)]

    wide_lines = pack_agent_chips(chips, width=60)
    narrow_lines = pack_agent_chips(chips, width=40)

    assert len(wide_lines) < len(narrow_lines)


def test_agent_sidebar_renders_details_and_empty_state() -> None:
    widget = AgentSidebar()
    assert "No background agents" in str(widget.render())

    widget.update_agents(
        (
            _agent(
                availability="evicted",
                run_id="run-1",
                run_status="completed",
                initial_task_summary="Initial task",
                current_task_summary="Current task",
                idle_seconds=3.5,
                ttl_remaining_seconds=42,
            ),
        ),
        (
            AgentEvictionModel(
                agent_id="agent-1",
                run_id="run-1",
                reason="idle_cap",
                idle_duration_seconds=3.5,
                root_generation=1,
            ),
        ),
    )

    rendered = str(widget.render())
    assert "Background Agents" in rendered
    assert "agent-1" in rendered
    assert "Profile: worker" in rendered
    assert "Availability: evicted" in rendered
    assert "Run: run-1" in rendered
    assert "Status: completed" in rendered
    assert "Initial task: Initial task" in rendered
    assert "Current task: Current task" in rendered
    assert "Idle: 3.5s" in rendered
    assert "TTL remaining (advisory): 42s" in rendered
    assert "Evicted: result preserved" in rendered
    widget.update_agents((_agent(availability="evicted", result_expired=True),))
    assert "Evicted: result expired" in str(widget.render())
    assert "Eviction reason: idle_cap (3.5s idle)" in rendered


def test_agent_widgets_render_effective_configuration_and_truncate_long_models() -> (
    None
):
    model = "very-long-model-alias-for-a-narrow-agent-bar"
    agent = _agent(effective_model=model, effective_thinking="high")
    bar = AgentBar()
    bar.update_agents((agent,))
    sidebar = AgentSidebar()
    sidebar.update_agents((agent,))

    assert f"[high; {model}]" in str(bar.render())
    assert f"Model: {model}" in str(sidebar.render())
    assert "Thinking: high" in str(sidebar.render())
    assert pack_agent_chips([f"✓ agent-1·worker [high; {model}]"], width=20)[
        0
    ].endswith("…")


def test_agent_widgets_omit_legacy_effective_configuration_cleanly() -> None:
    agent = _agent()
    bar = AgentBar()
    bar.update_agents((agent,))
    sidebar = AgentSidebar()
    sidebar.update_agents((agent,))

    assert "[None]" not in str(bar.render())
    assert "Model: None" not in str(sidebar.render())
    assert "Thinking: None" not in str(sidebar.render())


@pytest.mark.asyncio
async def test_mounted_agent_bar_shows_effective_thinking_after_escalation(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(32, 20)) as pilot:
        await chartreux_app._handle_turn_event(
            AgentsUpdate([_agent(effective_model="small", effective_thinking="low")])
        )
        await pilot.pause()
        bar = chartreux_app.query_one(AgentBar)
        assert "low; small" in str(bar.render())

        await chartreux_app._handle_turn_event(
            AgentsUpdate([
                _agent(
                    effective_model="very-long-model-alias-for-a-narrow-agent-bar",
                    effective_thinking="high",
                )
            ])
        )
        await pilot.pause()

        rendered = str(bar.render())
        assert "high" in rendered
        assert rendered.endswith("…")


@pytest.mark.asyncio
async def test_mounted_agent_bar_reports_overflow_at_three_lines(
    chartreux_app: ChartreuxApp,
) -> None:
    agents = [
        _agent(
            agent_id=f"agent-{index}",
            profile="background-worker-profile",
            effective_model=f"wire-model-name-{index}-with-extra-detail",
            active_provider="provider/default",
        )
        for index in range(1, 7)
    ]

    async with chartreux_app.run_test(size=(80, 24)) as pilot:
        await chartreux_app._handle_turn_event(AgentsUpdate(agents))
        await pilot.pause()

        rendered = str(chartreux_app.query_one(AgentBar).render())
        assert len(rendered.splitlines()) == 3
        assert "+4 more" in rendered
        assert "agent-6" not in rendered


@pytest.mark.asyncio
async def test_mounted_agent_widgets_show_cancelled_and_wire_model_name(
    chartreux_app: ChartreuxApp,
) -> None:
    agent = _agent(
        last_run_status="cancelled",
        base_model="glm-5-3",
        active_provider="mistral/default",
        effective_model="org/zai-glm-5-3",
    )

    async with chartreux_app.run_test() as pilot:
        await chartreux_app._handle_turn_event(AgentsUpdate([agent]))
        await pilot.pause()

        bar = chartreux_app.query_one(AgentBar)
        assert "⊘ agent-1·worker" in str(bar.render())
        assert "✓ agent-1·worker" not in str(bar.render())
        assert "mistral/default/org/zai-glm-5-3" in str(bar.render())

        await pilot.press("ctrl+shift+a")
        sidebar = chartreux_app.query_one(AgentSidebar)
        rendered = str(sidebar.render())
        assert "Status: cancelled" in rendered
        assert "Model: mistral/default/org/zai-glm-5-3" in rendered
        assert "mistral/default/glm-5-3" not in rendered


@pytest.mark.asyncio
async def test_agents_update_routes_to_widgets_and_sidebar_toggle(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        agent = _agent(availability="evicted", run_id="run-1", run_status="completed")
        eviction = AgentEvictionModel(
            agent_id="agent-1",
            run_id="run-1",
            reason="ttl",
            idle_duration_seconds=5,
            root_generation=1,
        )
        await chartreux_app._handle_turn_event(AgentsUpdate([agent], [eviction]))
        await pilot.pause()

        bar = chartreux_app.query_one(AgentBar)
        assert "agent-1·worker" in str(bar.render())

        await pilot.press("ctrl+shift+a")
        sidebar = chartreux_app.query_one(AgentSidebar)
        assert "Run: run-1" in str(sidebar.render())
        assert "Eviction reason: ttl (5s idle)" in str(sidebar.render())

        failed = _agent(run_id="run-2", run_status="failed")
        await chartreux_app._handle_turn_event(AgentsUpdate([failed]))
        assert "Status: failed" in str(sidebar.render())

        await pilot.press("ctrl+shift+a")
        assert len(chartreux_app.query(AgentSidebar)) == 0


@pytest.mark.asyncio
async def test_agents_slash_command_toggles_sidebar(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        assert await chartreux_app._handle_command("/agents")
        await pilot.pause()
        assert len(chartreux_app.query(AgentSidebar)) == 1

        assert await chartreux_app._handle_command("/agents")
        await pilot.pause()
        assert len(chartreux_app.query(AgentSidebar)) == 0


class _AgentSidebarApp(App[None]):
    def __init__(self) -> None:
        super().__init__()
        self.sidebar = AgentSidebar()
        self.opened_agent_ids: list[str] = []

    def compose(self) -> ComposeResult:
        yield self.sidebar

    def on_agent_sidebar_transcript_open(
        self, message: AgentSidebar.TranscriptOpen
    ) -> None:
        self.opened_agent_ids.append(message.agent_id)


@pytest.mark.asyncio
async def test_agent_sidebar_navigation_and_transcript_open() -> None:
    app = _AgentSidebarApp()

    async with app.run_test() as pilot:
        sidebar = app.sidebar
        assert sidebar.selected_agent_id is None
        await pilot.press("enter")
        assert app.opened_agent_ids == []

        sidebar.update_agents((_agent(),))
        await pilot.pause()
        assert sidebar.selected_agent_id == "agent-1"
        assert app.screen.focused is sidebar
        app.set_focus(None)
        sidebar.focus_selection()
        await pilot.pause()
        assert app.screen.focused is sidebar
        await pilot.press("enter")
        assert app.opened_agent_ids == ["agent-1"]

        sidebar.update_agents((_agent(), _agent(agent_id="agent-2")))
        await pilot.press("down", "enter", "up", "enter")
        assert app.opened_agent_ids == ["agent-1", "agent-2", "agent-1"]


@pytest.mark.asyncio
async def test_agent_sidebar_selection_survives_reorder_and_status_updates() -> None:
    app = _AgentSidebarApp()

    async with app.run_test() as pilot:
        sidebar = app.sidebar
        sidebar.update_agents((
            _agent(agent_id="running", availability="running"),
            _agent(agent_id="idle"),
            _agent(agent_id="evicted", availability="evicted"),
        ))
        await pilot.press("down")
        assert sidebar.selected_agent_id == "idle"

        await pilot.press("down", "enter", "up")
        assert app.opened_agent_ids == ["evicted"]
        assert sidebar.selected_agent_id == "idle"

        sidebar.update_agents((
            _agent(agent_id="evicted", availability="evicted"),
            _agent(agent_id="idle", availability="running"),
            _agent(agent_id="running"),
        ))
        await pilot.pause()
        assert sidebar.selected_agent_id == "idle"

        await pilot.press("down", "enter")
        assert app.opened_agent_ids == ["evicted", "running"]
        sidebar.update_agents((
            _agent(agent_id="evicted", availability="evicted"),
            _agent(agent_id="idle", availability="evicted"),
            _agent(agent_id="running", availability="running"),
        ))
        await pilot.pause()
        assert sidebar.selected_agent_id == "running"
        await pilot.press("up", "enter")
        assert app.opened_agent_ids == ["evicted", "running", "idle"]


@pytest.mark.asyncio
async def test_agent_sidebar_drops_released_selection_with_adjacent_fallback() -> None:
    app = _AgentSidebarApp()

    async with app.run_test() as pilot:
        sidebar = app.sidebar
        sidebar.update_agents((
            _agent(agent_id="before"),
            _agent(agent_id="released"),
            _agent(agent_id="after"),
        ))
        await pilot.press("down")
        assert sidebar.selected_agent_id == "released"

        sidebar.update_agents((
            _agent(agent_id="before"),
            _agent(agent_id="released", availability="released"),
            _agent(agent_id="after"),
        ))
        await pilot.pause()
        assert sidebar.selected_agent_id == "after"
        assert "released" not in str(sidebar.render())
        await pilot.press("enter")
        assert app.opened_agent_ids == ["after"]

        sidebar.update_agents(())
        await pilot.pause()
        assert sidebar.selected_agent_id is None
        await pilot.press("enter")
        assert app.opened_agent_ids == ["after"]
