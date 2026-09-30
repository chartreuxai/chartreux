from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.content import Content
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
        assert "run · worker" in rendered
        assert "turns 2" not in rendered
        app.bar.action_cursor_down()
        await pilot.pause()
        assert "turns 2" in str(app.bar.query_one("#agent-bar-detail", Static).render())
        content = app.bar.query_one("#agent-bar-content", Static).render()
        assert isinstance(content, Content)
        assert any("reverse" in str(span.style) for span in content.spans)
        app.query_one("#other-focus", Static).focus()
        await pilot.pause()
        content = app.bar.query_one("#agent-bar-content", Static).render()
        assert isinstance(content, Content)
        assert all("reverse" not in str(span.style) for span in content.spans)
        assert "▸" in content.plain


@pytest.mark.asyncio
async def test_collapsed_agent_bar_skips_unchanged_content_updates() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent("one", idle_seconds=1),))
        await pilot.pause()
        content_widget = app.bar._content
        assert content_widget is not None

        with patch.object(
            content_widget, "update", wraps=content_widget.update
        ) as update:
            app.bar.update_agents((_agent("one", idle_seconds=2),))
            await pilot.pause()
            assert update.call_count == 0

            app.bar.update_agents((_agent("one", availability="failed"),))
            await pilot.pause()

        assert update.call_count == 1
        assert update.call_args.args == ("1 agents: 1 failed",)


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
        assert "old" in rendered
        assert "result expired" not in rendered
        app.bar.action_cursor_down()
        await pilot.pause()
        selected = str(app.bar.query_one("#agent-bar-detail", Static).render())
        assert "result expired" in selected and "evicted: ttl" in selected
        assert selected.index("result expired") < selected.index("run ")
        assert selected.index("evicted: ttl") < selected.index("turns ")
        metadata = app.bar.full_metadata(app.bar.agents[0])
        assert "Eviction reason: ttl" in metadata
        assert "Eviction root generation: 1" in metadata
        assert "Result expired: True" in metadata


@pytest.mark.asyncio
async def test_browser_click_markers_distinguish_expand_and_selection(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = _BrowserApp()
    with caplog.at_level(logging.DEBUG, logger="vibe"):
        async with app.run_test() as pilot:
            app.bar.update_agents((_agent("one"),))
            await pilot.pause()
            await pilot.click(app.bar, offset=(1, 0))
            await pilot.pause()
            await pilot.click(app.bar, offset=(1, 2))
            await pilot.pause()
    messages = [record.getMessage() for record in caplog.records]
    assert any("phase=expand-start" in text for text in messages)
    assert any("phase=expand-done" in text for text in messages)
    assert any(
        "phase=row-select-start" in text and "target=one" in text for text in messages
    )
    assert any(
        "phase=row-select-posted" in text and "target=one" in text for text in messages
    )


@pytest.mark.asyncio
async def test_agent_rows_clip_by_cells_and_ascii_markers() -> None:
    app = _BrowserApp()
    app.config = SimpleNamespace(ascii_chrome=True)
    async with app.run_test(size=(25, 12)) as pilot:
        app.bar.update_agents((_agent("世界" * 20, availability="evicted"),))
        app.bar.open_browser()
        await pilot.pause()
        app.bar.action_cursor_down()
        row = _content(app.bar).splitlines()[1]
        assert row.startswith("> ")
        assert "... · evicted" in row
        assert row.endswith("evicted")
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        detail = str(app.bar.query_one("#agent-bar-detail", Static).render())
        assert "! evicted" in detail
        await pilot.press("d")
        assert app.bar._show_full_details
        full = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert "Identity: " + "世界" * 20 in full
        assert "Availability: evicted" in full
        await pilot.press("escape")
        assert app.bar.expanded and not app.bar._show_full_details


class _BrowserApp(App[None]):
    config: SimpleNamespace

    CSS = """
    AgentBar { height: 1; max-height: 10; }
    AgentBar.-expanded { height: auto; }
    #agent-bar-title, #agent-bar-detail, #agent-bar-footer { height: 1; }
    #agent-bar-rows { height: 1fr; overflow-y: auto; }
    #agent-bar-content { width: 100%; height: auto; }
    """

    def compose(self) -> ComposeResult:
        other = Static("Above the browser", id="other-focus")
        other.can_focus = True
        yield other
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
        rows = app.bar.query_one("#agent-bar-rows")
        assert rows.scroll_y > 0
        selected_row = 16
        assert rows.scroll_y <= selected_row < rows.scroll_y + rows.size.height
        assert app.bar.query_one("#agent-bar-footer").display
        assert app.bar.query_one("#agent-bar-detail").display
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
        app.bar.query_one("#agent-bar-rows").scroll_to(
            y=8, animate=False, force=True, immediate=True
        )
        await pilot.pause()

        await pilot.click(app.bar, offset=(1, 1))
        await pilot.pause()

        assert app.bar.selected_agent_id == "agent-7"
