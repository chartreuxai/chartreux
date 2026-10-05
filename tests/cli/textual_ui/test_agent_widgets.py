from __future__ import annotations

from itertools import permutations
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.content import Content
from textual.widgets import Button, Static

from chartreux.app_server.protocol import (
    AgentEvictionModel,
    AgentsCancelResponse,
    AgentSummaryModel,
    CancelOutcome,
)
from chartreux.cli.textual_ui.widgets.agent_bar import (
    AgentBar,
    agent_is_active,
    agent_state,
)


def _agent(agent_id: str = "agent-1", **kwargs: Any) -> AgentSummaryModel:
    values: dict[str, Any] = {"profile": "worker", "availability": "idle"}
    values.update(kwargs)
    return AgentSummaryModel(agent_id=agent_id, **values)


def _content(bar: AgentBar) -> str:
    return str(bar.query_one("#agent-bar-content", Static).render())


@pytest.mark.asyncio
async def test_snapshot_receipt_precedes_delayed_close_and_reactive_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.app_server.events import AgentsUpdate
    from chartreux.cli.textual_ui.app import ChartreuxApp

    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr("chartreux.cli.textual_ui.app.time", clock)
    monkeypatch.setattr("chartreux.cli.textual_ui.widgets.agent_bar.time", clock)
    app = _BrowserApp()
    async with app.run_test() as pilot:

        async def close(**_: Any) -> None:
            now[0] += 10

        target = SimpleNamespace(
            _track_turn_outcome=Mock(),
            _refresh_status_for_event=Mock(),
            _agent_transcript_viewer=None,
            _agent_selection_target="removed",
            _request_agent_transcript_close=AsyncMock(side_effect=close),
            _turn_outcome_notice=SimpleNamespace(display=False),
            _agent_evictions={},
            _agent_bar=app.bar,
        )
        agent = _agent(
            availability="running", run_elapsed_seconds=252, current_run_id="r1"
        )
        observed: list[float] = []
        render = app.bar._render_agents

        def check_render() -> None:
            if app.bar.agents:
                observed.append(app.bar._duration_samples[agent.agent_id][1])
            render()

        monkeypatch.setattr(app.bar, "_render_agents", check_render)
        await ChartreuxApp._handle_turn_event(cast(Any, target), AgentsUpdate([agent]))
        app.bar.open_browser()
        await pilot.pause()
        assert observed and all(anchor == 100 for anchor in observed)
        assert "Run 4m22s" in app.bar._agent_details(agent)


@pytest.mark.asyncio
@pytest.mark.parametrize("ascii_mode", [False, True])
async def test_duration_receipt_refresh_freeze_and_retask(
    monkeypatch: pytest.MonkeyPatch, ascii_mode: bool
) -> None:
    app = _BrowserApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    now = [100.0]
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.agent_bar.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    async with app.run_test(size=(100, 24)) as pilot:
        with patch.object(
            app.bar, "set_interval", side_effect=AssertionError("no ticking")
        ):
            agent = _agent(
                availability="running", current_run_id="r1", run_elapsed_seconds=252
            )
            app.bar.update_agents((agent,), received_at=90)
            app.bar.open_browser()
            app.bar.action_cursor_down()
            await pilot.pause()
            assert "Run 4m22s" in app.bar._agent_details(agent)
            anchor = app.bar._duration_samples[agent.agent_id][1]
            await pilot.pause()
            now[0] += 60
            await pilot.pause()
            # Passage of time alone does not repaint the detail row.
            assert app.bar._detail is not None
            assert "Run 4m22s" in str(app.bar._detail.render())
            app.bar._render_agents()
            assert "Run 5m22s" in app.bar._agent_details(agent)
            app.bar._render_agents()
            assert "Run 5m22s" in app.bar._agent_details(agent)
            assert app.bar._duration_samples[agent.agent_id][1] == anchor
            compact = agent.model_copy(update={"compacting": True})
            app.bar.update_agents((compact,), received_at=90)
            assert "Run 5m22s" in app.bar._agent_details(compact)
            finishing = _agent(
                availability="finalizing", run_elapsed_seconds=252, latest_run_id="r1"
            )
            app.bar.update_agents((finishing,))
            now[0] += 60
            assert "Finishing" in app.bar._agent_details(finishing)
            assert "Run 4m12s" in app.bar._agent_details(finishing)
            assert "Idle" not in app.bar._agent_details(finishing)
            idle = _agent(idle_seconds=2, run_elapsed_seconds=252, latest_run_id="r1")
            app.bar.update_agents((idle,))
            now[0] += 60
            text = app.bar._agent_details(idle)
            assert "Last run 4m12s" in text and "Idle 1m02s" in text
            retask = _agent(
                availability="running", current_run_id="r2", run_elapsed_seconds=0
            )
            app.bar.update_agents((retask,))
            assert "Run <0.1s" in app.bar._agent_details(retask)
            assert "Idle" not in app.bar._agent_details(retask)
            evicted = _agent(
                availability="evicted", idle_seconds=62, run_elapsed_seconds=252
            )
            app.bar.update_agents((evicted,))
            now[0] += 500
            assert "Idle at eviction 1m02s" in app.bar._agent_details(evicted)
            assert "Last run 4m12s" in app.bar._agent_details(evicted)
            assert "Run" not in _content(app.bar)


@pytest.mark.asyncio
@pytest.mark.parametrize("availability,label", [("running", "Run"), ("idle", "Idle")])
async def test_equal_snapshot_refreshes_duration_receipt(
    monkeypatch: pytest.MonkeyPatch, availability: str, label: str
) -> None:
    now = [100.0]
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.agent_bar.time",
        SimpleNamespace(monotonic=lambda: now[0]),
    )
    app = _BrowserApp()
    async with app.run_test(size=(100, 24)) as pilot:
        agent = _agent(
            availability=availability,
            current_run_id="r1" if availability == "running" else None,
            latest_run_id="r1",
            run_elapsed_seconds=10,
            idle_seconds=10 if availability == "idle" else None,
        )
        app.bar.update_agents((agent,))
        app.bar.open_browser()
        app.bar.action_cursor_down()
        await pilot.pause()
        now[0] += 10
        app.bar._render_agents()
        assert app.bar._detail is not None
        assert f"{label} 20.0s" in str(app.bar._detail.render())

        replacement = agent.model_copy()
        assert replacement == agent and replacement is not agent
        app.bar.update_agents((replacement,))
        # Textual keeps the original object, but this receipt replaces its anchor.
        assert app.bar.agents[0] is agent
        assert f"{label} 10.0s" in str(app.bar._detail.render())
        now[0] += 5
        app.bar._render_agents()
        assert f"{label} 15.0s" in str(app.bar._detail.render())
        app.bar._render_agents()
        assert f"{label} 15.0s" in str(app.bar._detail.render())
        assert app.bar._duration_samples[agent.agent_id][1] == 110


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seconds,expected", [(None, "unknown"), (0, "<0.1s"), (7389, "2h03m09s")]
)
@pytest.mark.parametrize(
    "reason,word",
    [
        ("error", "Failed"),
        ("user_cancelled", "Cancelled"),
        ("budget_exceeded", "Budget stopped"),
    ],
)
async def test_duration_unknown_zero_long_and_outcome_precedence(
    seconds: float | None, expected: str, reason: str, word: str
) -> None:
    app = _BrowserApp()
    async with app.run_test(size=(100, 24)) as pilot:
        agent = _agent(run_elapsed_seconds=seconds, stop_reason=reason)
        app.bar.update_agents((agent,))
        app.bar.open_browser()
        await pilot.pause()
        text = app.bar._agent_details(agent)
        assert f"Last run {expected}" in text
        assert text.index(word) < text.index("Last run")


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
        assert not hasattr(app.bar, "_spinner_timer")
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
        assert "result expired" in selected and "Idle at eviction" in selected
        assert len(selected.splitlines()) == 2
        assert all(
            cell_len(line) <= app.bar.query_one("#agent-bar-detail").size.width
            for line in selected.splitlines()
        )
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
        assert "... | Evicted" in row
        assert row.rstrip().endswith("Evicted")
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        detail = str(app.bar.query_one("#agent-bar-detail", Static).render())
        assert "Evicted" in detail
        await pilot.press("d")
        assert app.bar._show_full_details
        full = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert "Identity: " + "世界" * 20 in full
        assert "Availability: evicted" in full
        await pilot.press("escape")
        assert app.bar.expanded and not app.bar._show_full_details


class _BrowserApp(App[None]):
    config: SimpleNamespace

    selections: list[str | None]
    closed: int = 0

    def on_agent_bar_selection_requested(
        self, event: AgentBar.SelectionRequested
    ) -> None:
        self.selections.append(event.agent_id)

    def on_agent_bar_stop_requested(self, event: AgentBar.StopRequested) -> None:
        self.stops.append((event.agent_id, event.run_id))

    def on_agent_bar_closed(self) -> None:
        self.closed += 1

    def compose(self) -> ComposeResult:
        other = Static("Above the browser", id="other-focus")
        other.can_focus = True
        yield other
        self.selections = []
        self.stops: list[tuple[str, str]] = []
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
        assert app.focused is app.bar
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.bar.selected_agent_id == "agent-8"
        assert app.selections[-1] == "agent-8"


@pytest.mark.parametrize(
    ("values", "state", "active"),
    [
        ({"availability": "running", "current_run_status": "running"}, "running", True),
        ({"availability": "running", "compacting": True}, "compacting", True),
        (
            {"availability": "finalizing", "current_run_status": "running"},
            "finishing",
            False,
        ),
        (
            {"availability": "finalizing", "current_run_status": "failed"},
            "finishing",
            False,
        ),
        (
            {
                "availability": "finalizing",
                "stop_reason": "budget_exceeded",
                "compacting": True,
            },
            "finishing",
            False,
        ),
        ({}, "idle", False),
        ({"last_run_status": "failed"}, "failed", False),
        ({"stop_reason": "error"}, "failed", False),
        ({"current_run_status": "cancelled"}, "cancelled", False),
        ({"stop_reason": "user_cancelled"}, "cancelled", False),
        ({"stop_reason": "orchestrator_cancelled"}, "cancelled", False),
        ({"stop_reason": "retasked"}, "cancelled", False),
        (
            {"stop_reason": "budget_exceeded", "last_run_status": "cancelled"},
            "budget-stopped",
            False,
        ),
        (
            {"stop_reason": "budget_unverifiable", "last_run_status": "failed"},
            "budget-stopped",
            False,
        ),
        (
            {"availability": "evicted", "stop_reason": "error", "compacting": True},
            "evicted",
            False,
        ),
    ],
)
def test_agent_presentation_state_and_shared_active_predicate(
    values: dict[str, Any], state: str, active: bool
) -> None:
    agent = _agent(**values)
    assert agent_state(agent) == state
    assert agent_is_active(agent) is active


@pytest.mark.asyncio
async def test_snapshot_order_selection_append_release_and_fallback() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        agents = (_agent("a"), _agent("b"), _agent("c"))
        app.bar.update_agents(agents)
        app.bar.open_browser()
        await pilot.press("down", "down")
        assert app.bar.selected_agent_id == "b"
        for snapshot in permutations(agents):
            app.bar.update_agents(snapshot)
            assert tuple(agent.agent_id for agent in app.bar.agents) == ("a", "b", "c")
            assert app.bar.selected_agent_id == "b"
        app.bar.update_agents((
            _agent("new"),
            _agent("c", availability="running"),
            _agent("b", availability="failed"),
            _agent("a"),
        ))
        assert tuple(agent.agent_id for agent in app.bar.agents) == (
            "a",
            "b",
            "c",
            "new",
        )
        assert app.bar.selected_agent_id == "b"
        app.bar.update_agents((
            _agent("new"),
            _agent("b", availability="released"),
            _agent("c"),
            _agent("a"),
        ))
        assert tuple(agent.agent_id for agent in app.bar.agents) == ("a", "c", "new")
        assert app.bar.selected_agent_id == "c"
        app.bar.update_agents((_agent("b"), _agent("new"), _agent("a"), _agent("c")))
        assert tuple(agent.agent_id for agent in app.bar.agents) == (
            "a",
            "c",
            "new",
            "b",
        )
        assert app.bar.selected_agent_id == "c"
        app.bar.update_agents((_agent("a"),))
        assert app.bar.selected_agent_id == "a"
        app.bar.update_agents(())
        assert app.bar.selected_agent_id is None
        assert not app.bar.expanded


@pytest.mark.asyncio
async def test_main_details_local_help_escape_and_output_activation() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent(),))
        app.bar.update_main_details(
            {
                "Session ID": "root-public-id",
                "Model": "provider/model",
                "Task": "root task",
            },
            context_tokens=135_000,
            auto_compact_threshold=400_000,
        )
        app.bar.open_browser()
        await pilot.pause()
        assert app.bar.selected_agent_id is None
        await pilot.press("d")
        full = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert "Identity: Main agent" in full
        assert "Session ID: root-public-id" in full
        assert "Model: provider/model" in full
        assert "\n135k/400k (34%)\n" in full
        assert "Compacting: False" in full
        await pilot.press("f1")
        assert app.bar._show_help and app.bar._show_full_details
        help_body = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert help_body.startswith("Agent browser help")
        assert "single-click" in help_body
        await pilot.press("enter", "down", "escape")
        assert app.bar._show_full_details and not app.bar._show_help
        assert app.bar.selected_agent_id is None
        await pilot.press("escape")
        assert app.bar.expanded and not app.bar._show_full_details
        await pilot.press("escape")
        assert not app.bar.expanded
        assert app.selections == []
        assert app.closed == 1
        app.bar.open_browser()
        await pilot.press("enter")
        assert app.selections == [None]
        app.bar.update_main_details(
            {"Session ID": "root-public-id"},
            context_tokens=135_000,
            auto_compact_threshold=400_000,
            compacting=True,
        )
        await pilot.press("d")
        full = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert "Compacting: True" in full and "—/400k" in full
        assert "135k/400k" not in full


@pytest.mark.asyncio
async def test_single_click_activates_and_hover_never_moves_selection() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent("one"), _agent("two")))
        app.bar.open_browser()
        await pilot.pause()
        await pilot.hover(app.bar, offset=(1, 3))
        assert app.bar.selected_agent_id is None
        assert app.selections == []
        await pilot.click(app.bar, offset=(1, 2))
        assert app.bar.selected_agent_id == "one"
        assert app.selections == ["one"]
        await pilot.hover(app.bar, offset=(1, 1))
        assert app.bar.selected_agent_id == "one"
        await pilot.press("down", "enter")
        assert app.selections == ["one", "two"]
        await pilot.click(app.bar, offset=(1, 1))
        assert app.selections == ["one", "two", None]
        await pilot.click(app.bar, offset=(0, 2))
        await pilot.click(app.bar, offset=(1, 5))
        assert app.selections == ["one", "two", None]


@pytest.mark.asyncio
async def test_no_color_footer_uses_shared_shortcut_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent(),))
        app.bar.open_browser()
        await pilot.pause()
        footer = app.bar.query_one("#agent-bar-footer", Static).render()
        assert isinstance(footer, Content)
        assert all("$foreground" in str(span.style) for span in footer.spans)
        assert "F1 Help" in footer.plain


@pytest.mark.asyncio
async def test_collapsed_counts_deterministic_and_protected_at_narrow_widths() -> None:
    app = _BrowserApp()
    async with app.run_test():
        agents = (
            _agent("evicted", availability="evicted"),
            _agent("idle"),
            _agent("cancelled", stop_reason="retasked"),
            _agent("budget", stop_reason="budget_unverifiable"),
            _agent("failed", last_run_status="failed"),
            _agent("finishing", availability="finalizing", last_run_status="failed"),
            _agent("compacting", availability="running", compacting=True),
            _agent("running", availability="running"),
        )
        app.bar.update_agents(agents)
        wide = "8 agents: 2 running · 1 finishing · 2 failed · 1 budget-stopped · 1 cancelled · 1 idle · 1 evicted"
        assert app.bar.collapsed_summary(200) == wide
        app.bar.update_agents(tuple(reversed(agents)))
        assert app.bar.collapsed_summary(200) == wide
        for width in range(5, 120):
            summary = app.bar.collapsed_summary(width)
            assert "2 failed" in summary or "2F" in summary
            assert "1 budget-stopped" in summary or "1B" in summary
            assert cell_len(summary) <= width
        assert app.bar.collapsed_summary(5) == "2F 1B"
        app.bar.update_agents((_agent("one", availability="running"),))
        assert app.bar.collapsed_summary(12) == "1 running"


@pytest.mark.parametrize("ascii_mode", [False, True])
@pytest.mark.asyncio
async def test_widget_local_sheet_budget_footer_and_cell_fitting(
    ascii_mode: bool,
) -> None:
    app = _BrowserApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((
            _agent("agent-x", profile="世界" * 80, availability="failed"),
        ))
        assert app.bar.collapsed_summary(80) == "1 agents: 1 failed"
        assert app.bar.border_title == "Background agents"
        app.bar.open_browser()
        await pilot.press("down")
        assert app.bar.region.height == 10
        assert app.bar.query_one("#agent-bar-rows").size.height == 5
        assert app.bar.query_one("#agent-bar-detail").size.height == 2
        assert not app.bar.query_one("#agent-bar-title").display
        row = _content(app.bar).splitlines()[1]
        assert "agent-x" in row
        assert "Failed" in row
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        footer = app.bar.query_one("#agent-bar-footer", Static).render()
        assert isinstance(footer, Content) and footer.spans
        assert (
            footer.plain
            == ("Up/Down" if ascii_mode else "↑↓")
            + " Move  Enter Open output  D Details  F1 Help  Esc Close"
        )
        assert "C Cancel" not in footer.plain and "T Retask" not in footer.plain
        await pilot.resize_terminal(25, 24)
        await pilot.pause()
        row = _content(app.bar).splitlines()[1]
        assert "agent-x" in row and "Failed" in row
        assert "世界" not in row
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        footer = app.bar.query_one("#agent-bar-footer", Static).render()
        assert isinstance(footer, Content) and "Esc" in footer.plain
        assert footer.cell_length <= app.bar.query_one("#agent-bar-footer").size.width
        detail = str(app.bar.query_one("#agent-bar-detail", Static).render())
        assert len(detail.splitlines()) == 2
        assert all(
            cell_len(line) <= app.bar.query_one("#agent-bar-detail").size.width
            for line in detail.splitlines()
        )
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.bar.region.height == 10
        assert app.bar.selected_agent_id == "agent-x"


@pytest.mark.parametrize(
    ("values", "label"),
    [
        ({"availability": "running"}, "Running"),
        ({"availability": "running", "compacting": True}, "… Compacting"),
        ({"availability": "finalizing"}, "… Finishing"),
        (
            {"availability": "finalizing", "last_run_status": "failed"},
            "… Finishing · Failed",
        ),
        (
            {"availability": "finalizing", "stop_reason": "retasked"},
            "… Finishing · Cancelled",
        ),
        (
            {"availability": "finalizing", "stop_reason": "budget_exceeded"},
            "… Finishing · Budget stopped",
        ),
        ({}, "Idle"),
        ({"last_run_status": "failed"}, "✗ Failed"),
        ({"stop_reason": "user_cancelled"}, "! Cancelled"),
        ({"stop_reason": "budget_exceeded"}, "! Budget stopped"),
        ({"availability": "evicted"}, "Evicted"),
    ],
)
@pytest.mark.asyncio
async def test_all_state_labels_and_full_context_metadata(
    values: dict[str, Any], label: str
) -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        agent = _agent(
            "identity", context_tokens=135_000, context_window=400_000, **values
        )
        with patch.object(
            app.bar, "set_interval", wraps=app.bar.set_interval
        ) as interval:
            app.bar.update_agents((agent,))
            app.bar.open_browser()
            await pilot.press("down")
            assert not interval.called
        assert label in _content(app.bar)
        assert label in str(app.bar.query_one("#agent-bar-detail", Static).render())
        await pilot.press("d")
        metadata = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert f"State: {label}" in metadata
        assert f"Compacting: {agent.compacting}" in metadata
        expected = "—/400k" if agent.compacting else "135k/400k (34%)"
        if agent.availability == "evicted":
            expected += " (last recorded)"
        assert "\n" + expected in metadata
        if agent.stop_reason:
            assert f"Stop reason: {agent.stop_reason}" in metadata


@pytest.mark.asyncio
async def test_eight_cell_transition_markers_do_not_collide_with_outcomes() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((_agent("agent"),))
        app.bar.open_browser()
        app.bar.query_one("#agent-bar-content").styles.width = 8
        await pilot.pause()
        for state in ("… Finishing", "Compacting", "Running"):
            row = app.bar._row(None, "agent", state=state)
            assert row.rstrip().endswith("…")
            assert cell_len(row) == 8
        for state, marker in (("Failed", "F"), ("Cancelled", "C"), ("Budget", "B")):
            assert app.bar._row(None, "agent", state=state).rstrip().endswith(marker)


@pytest.mark.parametrize("ascii_mode", [False, True])
@pytest.mark.asyncio
async def test_narrow_finishing_outcome_and_ascii_chrome(ascii_mode: bool) -> None:
    app = _BrowserApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    async with app.run_test(size=(25, 24)) as pilot:
        app.bar.update_agents((
            _agent(
                "agent-identifier",
                profile="profile" * 20,
                availability="finalizing",
                stop_reason="budget_exceeded",
            ),
        ))
        app.bar.open_browser()
        await pilot.press("down")
        row = _content(app.bar).splitlines()[1]
        assert "Fin:B" in row and "agent-" in row
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        if ascii_mode:
            assert row.isascii()
            detail = str(app.bar.query_one("#agent-bar-detail", Static).render())
            assert detail.isascii()
            assert app.bar.collapsed_summary(80).isascii()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "values,eligible",
    [
        ({"availability": "running", "current_run_id": "r1"}, True),
        ({"availability": "running", "current_run_id": "r1", "compacting": True}, True),
        ({"availability": "running"}, False),
        ({"current_run_id": "r1"}, False),
        ({"availability": "finalizing", "current_run_id": "r1"}, False),
    ],
)
async def test_stop_shortcut_eligibility(
    values: dict[str, Any], eligible: bool
) -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((_agent(**values),))
        app.bar.open_browser()
        await pilot.press("c")
        assert app.bar._stop_confirmation is None  # root is never a target
        await pilot.press("down")
        footer = str(app.bar.query_one("#agent-bar-footer", Static).render())
        assert ("C Stop run" in footer) is eligible
        await pilot.press("c")
        assert (app.bar._stop_confirmation is not None) is eligible
        if eligible:
            assert app.focused is app.bar.query_one("#agent-stop-cancel", Button)
            await pilot.press("enter")
            assert app.stops == [] and app.selections == []
            assert app.focused is app.bar and app.bar.expanded


@pytest.mark.asyncio
@pytest.mark.parametrize("ascii_mode", [False, True])
@pytest.mark.parametrize("theme", ["textual-dark", "ansi-dark", "ansi-light"])
async def test_stop_confirmation_controls_scope_focus_resize(
    ascii_mode: bool, theme: str
) -> None:
    app = _BrowserApp()
    app.theme = theme
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    identity, run = "worker-" * 12, "run-" * 15
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((
            _agent(identity, availability="running", current_run_id=run),
        ))
        app.bar.open_browser()
        await pilot.press("down", "c")
        cancel = app.bar.query_one("#agent-stop-cancel", Button)
        submit = app.bar.query_one("#agent-stop-submit", Button)
        assert str(cancel.label) == "[Cancel]" and str(submit.label) == "[Stop run]"

        def assert_controls_rendered() -> None:
            confirmation = app.bar.query_one("#agent-stop-confirmation")
            footer = app.bar.query_one("#agent-bar-footer")
            assert confirmation.region.height == 2
            for button in (submit, cancel):
                assert button.content_size.height == 1
                assert button.content_size.width >= cell_len(str(button.label))
                assert button.region in confirmation.region
                assert button.region.bottom <= footer.region.y
                assert str(button.label) in button.render_line(0).text

        assert_controls_rendered()
        # Theme-specific hover borders must not consume the single label row either.
        await pilot.hover("#agent-stop-submit")
        await pilot.pause()
        assert_controls_rendered()
        assert app.focused is cancel
        prompt = str(app.bar.query_one("#agent-stop-prompt", Static).render())
        assert identity in prompt and run in prompt
        assert "transcript and partial output preserved" in prompt
        assert "retention/release policy" in prompt
        assert app.bar.region.height == 10
        assert app.bar.query_one("#agent-bar-rows").size.height == 5
        await pilot.press("tab")
        assert app.focused is submit
        await pilot.press("shift+tab")
        assert app.focused is cancel
        await pilot.press("down")
        assert app.focused is submit
        await pilot.press("up")
        assert app.focused is cancel
        await pilot.press("pagedown")
        assert app.bar.query_one("#agent-stop-scroll").scroll_y > 0
        await pilot.resize_terminal(100, 30)
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert app.focused is cancel
        assert app.bar._stop_confirmation == (identity, run)
        assert app.bar.region.height == 10
        assert_controls_rendered()
        if ascii_mode:
            assert _content(app.bar).isascii()
        await pilot.press("escape")
        assert app.bar.expanded and app.focused is app.bar and app.closed == 0
        await pilot.press("c")
        await pilot.click("#agent-stop-cancel")
        assert app.stops == [] and app.selections == []
        assert app.focused is app.bar
        await pilot.press("c")
        await pilot.click("#agent-stop-submit")
        assert app.stops == [(identity, run)] and app.selections == []
        assert app.focused is app.bar
        await pilot.press("c")
        assert app.bar._stop_confirmation is None
        assert "Stopping" in _content(app.bar)
        assert "Stopping" in app.bar._agent_details(app.bar.agents[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["new_run", "terminal", "finishing", "removed"])
async def test_stop_confirmation_invalidated_by_update(change: str) -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        agent = _agent(availability="running", current_run_id="r1")
        app.bar.update_agents((agent, _agent("sibling")))
        app.bar.open_browser()
        await pilot.press("down", "c", "left")
        replacement = {
            "new_run": _agent(availability="running", current_run_id="r2"),
            "terminal": _agent(latest_run_id="r1", stop_reason="user_cancelled"),
            "finishing": _agent(availability="finalizing", latest_run_id="r1"),
        }.get(change)
        app.bar.update_agents((
            _agent("sibling"),
            *([replacement] if replacement else []),
        ))
        await pilot.pause()
        assert app.bar._stop_confirmation is None
        assert app.focused is app.bar
        assert app.stops == []
        assert [item.agent_id for item in app.bar.agents] == (
            ["agent-1", "sibling"] if replacement else ["sibling"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", [*CancelOutcome, None])
async def test_stop_settlement_matrix(outcome: CancelOutcome | None) -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        agent = _agent(availability="running", current_run_id="r1")
        app.bar.update_agents((agent, _agent("sibling")))
        app.bar.open_browser()
        await pilot.press("down", "c", "left", "enter", "c")
        assert app.stops == [("agent-1", "r1")]
        assert app.bar._stop_confirmation is None
        assert app.bar.stop_is_pending("agent-1", "r1")
        response = (
            AgentsCancelResponse(outcome=outcome, run_id="r1") if outcome else None
        )
        app.bar.settle_stop("agent-1", "r1", response)
        keep = outcome in {CancelOutcome.STOP_REQUESTED, CancelOutcome.ALREADY_STOPPING}
        assert app.bar.stop_is_pending("agent-1", "r1") is keep
        assert ("Stopping" in _content(app.bar)) is keep
        assert app.focused is app.bar
        if keep:
            finishing = _agent(availability="finalizing", latest_run_id="r1")
            app.bar.update_agents((_agent("sibling"), finishing))
            assert "Stopping" in _content(app.bar)
            terminal = _agent(latest_run_id="r1", stop_reason="user_cancelled")
            app.bar.update_agents((_agent("sibling"), terminal))
            assert "Cancelled" in _content(app.bar) and "Stopping" not in _content(
                app.bar
            )
            assert not app.bar.stop_is_pending("agent-1", "r1")
        assert tuple(item.agent_id for item in app.bar.agents) == ("agent-1", "sibling")


@pytest.mark.asyncio
@pytest.mark.parametrize("new_run", [False, True])
@pytest.mark.parametrize("outcome", list(CancelOutcome))
async def test_stop_notification_before_response_fences_stale_reply(
    new_run: bool, outcome: CancelOutcome
) -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent(availability="running", current_run_id="r1"),))
        app.bar.open_browser()
        await pilot.press("down", "c", "left", "enter")
        update = (
            _agent(availability="running", current_run_id="r2")
            if new_run
            else _agent(latest_run_id="r1", stop_reason="user_cancelled")
        )
        app.bar.update_agents((update,))
        app.bar.settle_stop(
            "agent-1", "r1", AgentsCancelResponse(outcome=outcome, run_id="r1")
        )
        assert not app.bar.stop_is_pending("agent-1", "r1")
        assert "Stopping" not in _content(app.bar)
        assert ("Running" if new_run else "Cancelled") in _content(app.bar)
        if new_run:
            await pilot.press("c", "left", "enter")
            app.bar.settle_stop(
                "agent-1", "r1", AgentsCancelResponse(outcome=outcome, run_id="r1")
            )
            assert app.bar.stop_is_pending("agent-1", "r2")
            assert app.stops == [("agent-1", "r1"), ("agent-1", "r2")]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["removed", "released", "evicted", "finishing"])
async def test_pending_stop_update_and_finishing_reconciliation(change: str) -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((
            _agent(availability="running", current_run_id="r1"),
            _agent("sibling"),
        ))
        app.bar.open_browser()
        await pilot.press("down", "c", "left", "enter")
        update = {
            "released": _agent(availability="released", latest_run_id="r1"),
            "evicted": _agent(
                availability="evicted", latest_run_id="r1", stop_reason="user_cancelled"
            ),
            "finishing": _agent(availability="finalizing", latest_run_id="r1"),
        }.get(change)
        app.bar.update_agents((_agent("sibling"), *([update] if update else [])))
        app.bar.settle_stop(
            "agent-1",
            "r1",
            AgentsCancelResponse(
                outcome=CancelOutcome.ALREADY_FINISHING
                if change == "finishing"
                else CancelOutcome.STOP_REQUESTED,
                run_id="r1",
            ),
        )
        assert not app.bar.stop_is_pending("agent-1", "r1")
        assert "Stopping" not in _content(app.bar)
        if change == "finishing":
            assert "Finishing" in _content(app.bar)
        assert app.focused is app.bar


@pytest.mark.asyncio
async def test_stop_response_wrong_run_does_not_overlay_replacement() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((_agent(availability="running", current_run_id="r1"),))
        app.bar.open_browser()
        await pilot.press("down", "c", "left", "enter")
        app.bar.settle_stop(
            "agent-1",
            "r1",
            AgentsCancelResponse(outcome=CancelOutcome.STOP_REQUESTED, run_id="r2"),
        )
        assert not app.bar.stop_is_pending("agent-1", "r1")
        assert "Running" in _content(app.bar) and "Stopping" not in _content(app.bar)
