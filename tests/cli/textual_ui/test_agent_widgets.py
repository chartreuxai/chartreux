from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chartreux.app_server.protocol import AgentEvictionModel, AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar


def _agent(agent_id: str = "agent-1", **kwargs: Any) -> AgentSummaryModel:
    values: dict[str, Any] = {"profile": "worker", "availability": "idle"}
    values.update(kwargs)
    return AgentSummaryModel(agent_id=agent_id, **values)


def _content(bar: AgentBar) -> str:
    return str(bar.query_one("#agent-bar-content", Static).render())


@pytest.mark.asyncio
async def test_statusline_aggregates_states_and_expanded_rows_include_turns() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((
            _agent(
                "run",
                availability="running",
                current_run_status="running",
                turns_used=2,
            ),
            _agent("idle"),
        ))
        await pilot.pause()
        assert "2 agents: 1 running · 1 idle" in _content(app.bar)
        assert app.bar._spinner_timer is not None
        app.bar.open_browser()
        await pilot.pause()
        rendered = _content(app.bar)
        assert "Main agent" in rendered
        assert "turns 2" in rendered
        assert "run · worker" in rendered


@pytest.mark.asyncio
async def test_expanded_list_filters_released_and_keeps_evicted_metadata() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents(
            (
                _agent("gone", availability="released"),
                _agent("old", availability="evicted", result_expired=True),
            ),
            (
                AgentEvictionModel(
                    agent_id="old",
                    run_id="r",
                    reason="ttl",
                    idle_duration_seconds=5,
                    root_generation=1,
                ),
            ),
        )
        app.bar.open_browser()
        await pilot.pause()
        rendered = _content(app.bar)
        assert "gone" not in rendered
        assert (
            "old" in rendered
            and "result expired" in rendered
            and "evicted: ttl" in rendered
        )


class _BrowserApp(App[None]):
    CSS = """
    AgentBar { max-height: 10; overflow-y: auto; }
    AgentBar.-expanded { height: auto; }
    #agent-bar-content { width: 100%; height: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Static("Above the browser")
        self.bar = AgentBar()
        yield self.bar


@pytest.mark.asyncio
async def test_browser_keyboard_selection_is_stable_and_scrollable() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 10)) as pilot:
        app.bar.update_agents(tuple(_agent(f"agent-{index}") for index in range(16)))
        app.bar.open_browser()
        await pilot.pause()
        assert app.bar.styles.max_height is not None
        await pilot.press(*("down",) * 16)
        await pilot.pause()
        assert app.bar.selected_agent_id == "agent-15"
        assert app.bar.scroll_y > 0
        selected_row = 17
        assert app.bar.scroll_y <= selected_row < app.bar.scroll_y + app.bar.size.height
        app.bar.update_agents(
            tuple(_agent(f"agent-{index}") for index in reversed(range(16)))
        )
        assert app.bar.selected_agent_id == "agent-15"
        await pilot.press("escape")
        assert not app.bar.expanded


@pytest.mark.asyncio
async def test_browser_click_uses_widget_relative_row_below_screen_origin() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent("one"), _agent("two")))
        app.bar.open_browser()
        await pilot.pause()
        assert app.bar.region.y > 0
        await pilot.click(app.bar, offset=(1, 2))
        await pilot.pause()
        assert app.bar.selected_agent_id == "one"


@pytest.mark.asyncio
async def test_browser_click_uses_content_relative_row_when_scrolled() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 10)) as pilot:
        app.bar.update_agents(tuple(_agent(f"agent-{index}") for index in range(16)))
        app.bar.open_browser()
        await pilot.pause()
        app.bar.scroll_to(y=8, animate=False, force=True, immediate=True)
        await pilot.pause()

        await pilot.click(app.bar, offset=(1, 1))
        await pilot.pause()

        assert app.bar.selected_agent_id == "agent-7"
