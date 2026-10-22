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
    agent_group,
    agent_is_active,
    agent_state,
)
from chartreux.core.subagents import derive_agent_name
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def _agent(agent_id: str = "agent-1", **kwargs: Any) -> AgentSummaryModel:
    values: dict[str, Any] = {"identity": "coder", "availability": "idle"}
    values.update(kwargs)
    values.setdefault(
        "name",
        derive_agent_name(
            agent_id=agent_id,
            identity=values["identity"],
            identity_display_label=None,
            base_model=values.get("base_model"),
            effective_model=values.get("effective_model"),
        ),
    )
    return AgentSummaryModel(agent_id=agent_id, **values)


def _content(bar: AgentBar) -> str:
    return str(bar.query_one("#agent-bar-content", Static).render())


@pytest.mark.asyncio
async def test_snapshot_receipt_precedes_delayed_close_and_reactive_render(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.app_server.events import AgentsUpdate
    from chartreux.cli.textual_ui.agent_browser import AgentBrowserController
    from chartreux.cli.textual_ui.app import ChartreuxApp

    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr("chartreux.cli.textual_ui.app.time", clock)
    monkeypatch.setattr("chartreux.cli.textual_ui.widgets.agent_bar.time", clock)
    app = _BrowserApp()
    async with app.run_test() as pilot:

        async def close(**_: Any) -> None:
            now[0] += 10

        # Controller stand-in: top-level controller state plus a nested _app
        # namespace for the late-bound host calls. _refresh_status_for_event
        # stays top-level for _handle_turn_event, which remains on
        # ChartreuxApp; _track_turn_outcome moved to the turn supervisor.
        target = SimpleNamespace(
            _turn_supervisor=SimpleNamespace(_track_turn_outcome=Mock()),
            _refresh_status_for_event=Mock(),
            agent_transcript_viewer=None,
            selection_target="removed",
            agent_summaries=[],
            _agent_evictions={},
            _app=SimpleNamespace(
                _request_agent_transcript_close=AsyncMock(side_effect=close),
                _mount_and_scroll=AsyncMock(),
                _turn_outcome_notice=SimpleNamespace(display=False),
                _agent_bar=app.bar,
                _show_no_output_outcome=Mock(),
                _agent_is_active=Mock(return_value=True),
            ),
        )
        target._handle_agents_update = lambda event, received_at: (  # type: ignore[attr-defined]
            AgentBrowserController.handle_agents_update(
                cast(Any, target), event, received_at
            )
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
async def test_statusline_counts_active_only_and_rows_show_names() -> None:
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
        # Active-only counts: the idle instance is not advertised.
        assert "1 running" in _content(app.bar)
        assert "idle" not in _content(app.bar)
        assert not hasattr(app.bar, "_spinner_timer")
        app.bar.open_browser()
        await pilot.pause()
        rendered = _content(app.bar)
        assert "Main agent" in rendered
        # Rows show the derived name primary with the handle as secondary.
        assert "coder-run" in rendered
        assert "run" in rendered
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

            app.bar.update_agents((_agent("one", availability="running"),))
            await pilot.pause()

        assert update.call_count == 1
        assert update.call_args.args == ("1 running",)


@pytest.mark.asyncio
async def test_browser_consumes_the_authoritative_inventory() -> None:
    """Released agents are absent because the server omits them (design §8).

    The client keeps no released filter: the published list is authoritative.
    Closed and evicted tombstones stay in the Recent group with their metadata.
    """
    app = _BrowserApp()
    eviction = AgentEvictionModel(
        agent_id="old",
        run_id="r",
        reason="ttl",
        idle_duration_seconds=5,
        root_generation=1,
    )
    async with app.run_test() as pilot:
        app.bar.update_agents(
            (
                _agent("gone", availability="idle"),
                _agent("old", availability="evicted", result_expired=True),
                _agent("closed", availability="closed"),
            ),
            (eviction,),
        )
        app.bar.open_browser()
        await pilot.pause()
        rendered = _content(app.bar)
        assert "gone" in rendered
        # The Resident default view hides tombstones while it has members.
        assert "Recent" not in rendered
        # The next update omits the released handle, exactly as the server does.
        app.bar.update_agents(
            (
                _agent("old", availability="evicted", result_expired=True),
                _agent("closed", availability="closed"),
            ),
            (eviction,),
        )
        await pilot.pause()
        rendered = _content(app.bar)
        assert "gone" not in rendered
        # Resident is now empty: the Recent group shows with a note.
        assert "No resident agents" in rendered
        assert "Recent" in rendered
        assert "old" in rendered and "closed" in rendered
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
        assert "Name: coder-old" in metadata
        assert "Agent ID: old" in metadata
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
            await pilot.click(app.bar, offset=(1, 4))
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
        # Resident is empty: the Recent group shows with a note (never blank).
        assert "No resident agents" in _content(app.bar)
        await pilot.press("right", "right", "right")  # recent filter
        await pilot.pause()
        app.bar.action_cursor_down()
        row = _content(app.bar).splitlines()[2]
        assert row.startswith("> ")
        assert "... | Evicted" in row
        assert row.rstrip().endswith("Evicted")
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        detail = str(app.bar.query_one("#agent-bar-detail", Static).render())
        assert "Evicted" in detail
        await pilot.press("d")
        assert app.bar._show_full_details
        full = str(app.bar.query_one("#agent-bar-full-content", Static).render())
        assert "Agent ID: " + "世界" * 20 in full
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

    def on_agent_bar_release_requested(self, event: AgentBar.ReleaseRequested) -> None:
        self.releases.append((event.agent_id, event.expected_run_id))

    def on_agent_bar_closed(self) -> None:
        self.closed += 1

    def compose(self) -> ComposeResult:
        other = Static("Above the browser", id="other-focus")
        other.can_focus = True
        yield other
        self.selections = []
        self.stops: list[tuple[str, str]] = []
        self.releases: list[tuple[str, str | None]] = []
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
        # Main row, group header, then agent-15: the shared mapping counts both.
        selected_row = 17
        assert rows.scroll_y <= selected_row < rows.scroll_y + rows.size.height
        assert app.bar.query_one("#agent-bar-footer").display
        assert app.bar.query_one("#agent-bar-detail").display
        app.bar.update_agents(
            tuple(_agent(f"agent-{index}") for index in reversed(range(16)))
        )
        # Server ordering is consumed verbatim; selection stays handle-based.
        assert tuple(agent.agent_id for agent in app.bar.agents) == tuple(
            f"agent-{index}" for index in reversed(range(16))
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
        await pilot.click(app.bar, offset=(1, 4))
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

        await pilot.click(app.bar, offset=(1, 3))
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
        ({"availability": "closed"}, "closed", False),
        ({"availability": "closed", "stop_reason": "error"}, "failed", False),
        (
            {"availability": "closed", "last_run_status": "cancelled"},
            "cancelled",
            False,
        ),
        (
            {"availability": "closed", "stop_reason": "budget_exceeded"},
            "budget-stopped",
            False,
        ),
        # WP2: suspension/restoring render truthfully instead of falling back
        # to the run-outcome label, and restoring counts as active work.
        (
            {"availability": "suspended", "last_run_status": "failed"},
            "suspended",
            False,
        ),
        ({"availability": "suspended", "stop_reason": "error"}, "suspended", False),
        (
            {"availability": "restoring", "current_run_status": "running"},
            "restoring",
            True,
        ),
        ({"availability": "restoring", "last_run_status": "failed"}, "restoring", True),
    ],
)
def test_agent_presentation_state_and_shared_active_predicate(
    values: dict[str, Any], state: str, active: bool
) -> None:
    agent = _agent(**values)
    assert agent_state(agent) == state
    assert agent_is_active(agent) is active


@pytest.mark.parametrize(
    ("values", "group"),
    [
        ({"availability": "running"}, "active"),
        ({"availability": "running", "compacting": True}, "active"),
        ({"availability": "finalizing"}, "active"),
        # WP2: suspended (warm-preserved) and restoring agents group as active.
        ({"availability": "suspended"}, "active"),
        ({"availability": "restoring"}, "active"),
        ({}, "idle"),
        ({"availability": "failed"}, "idle"),
        ({"availability": "idle"}, "idle"),
        ({"availability": "closed"}, "recent"),
        ({"availability": "evicted"}, "recent"),
    ],
)
def test_agent_group_keys_on_lifecycle_availability(
    values: dict[str, Any], group: str
) -> None:
    # Grouping uses availability, never agent_state: a retasked running agent
    # with a historical cancel outcome still groups as active work.
    agent = _agent(**values)
    assert agent_group(agent) == group
    if values.get("availability") == "running":
        agent = agent.model_copy(update={"stop_reason": "retasked"})
        assert agent_group(agent) == "active"
        assert agent_state(agent) == "cancelled"


@pytest.mark.asyncio
async def test_server_order_is_consumed_and_selection_follows_handles() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        agents = (_agent("a"), _agent("b"), _agent("c"))
        app.bar.update_agents(agents)
        app.bar.open_browser()
        await pilot.press("down", "down")
        assert app.bar.selected_agent_id == "b"
        for snapshot in permutations(agents):
            app.bar.update_agents(snapshot)
            # The server owns ordering; the client consumes it verbatim.
            assert tuple(agent.agent_id for agent in app.bar.agents) == tuple(
                item.agent_id for item in snapshot
            )
            assert app.bar.selected_agent_id == "b"
        app.bar.update_agents((
            _agent("new"),
            _agent("c", availability="running"),
            _agent("b", availability="failed"),
            _agent("a"),
        ))
        assert tuple(agent.agent_id for agent in app.bar.agents) == (
            "new",
            "c",
            "b",
            "a",
        )
        assert app.bar.selected_agent_id == "b"
        # The server omits released handles; the client keeps no filter.
        app.bar.update_agents((_agent("new"), _agent("c"), _agent("a")))
        assert tuple(agent.agent_id for agent in app.bar.agents) == ("new", "c", "a")
        assert app.bar.selected_agent_id == "a"
        app.bar.update_agents((_agent("b"), _agent("new"), _agent("a"), _agent("c")))
        assert tuple(agent.agent_id for agent in app.bar.agents) == (
            "b",
            "new",
            "a",
            "c",
        )
        assert app.bar.selected_agent_id == "a"
        app.bar.update_agents((_agent("a"),))
        assert app.bar.selected_agent_id == "a"
        # The expanded browser stays open on an empty inventory (design §8).
        app.bar.update_agents(())
        assert app.bar.selected_agent_id is None
        assert app.bar.expanded
        assert "No background agents" in _content(app.bar)


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
        # Rows: Main (offset y=2), group header (y=3), one (y=4), two (y=5).
        await pilot.hover(app.bar, offset=(1, 4))
        assert app.bar.selected_agent_id is None
        assert app.selections == []
        await pilot.click(app.bar, offset=(1, 2))
        assert app.bar.selected_agent_id is None
        assert app.selections == [None]
        # Group headers are labels: clicking one never selects.
        await pilot.click(app.bar, offset=(1, 3))
        assert app.bar.selected_agent_id is None
        assert app.selections == [None]
        await pilot.click(app.bar, offset=(1, 4))
        assert app.bar.selected_agent_id == "one"
        assert app.selections == [None, "one"]
        await pilot.hover(app.bar, offset=(1, 2))
        assert app.bar.selected_agent_id == "one"
        await pilot.press("down", "enter")
        assert app.selections == [None, "one", "two"]
        await pilot.click(app.bar, offset=(1, 2))
        assert app.selections == [None, "one", "two", None]
        await pilot.click(app.bar, offset=(0, 2))
        await pilot.click(app.bar, offset=(1, 6))
        assert app.selections == [None, "one", "two", None]


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
async def test_collapsed_counts_active_only_and_dim_notes_at_narrow_widths() -> None:
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
        # Active-only counts keyed on availability; historical outcomes and
        # idle/closed/evicted inventories are never advertised as work.
        wide = "2 running · 1 finishing"
        assert app.bar.collapsed_summary(200) == wide
        assert "idle" not in wide and "evicted" not in wide and "failed" not in wide
        app.bar.update_agents(tuple(reversed(agents)))
        assert app.bar.collapsed_summary(200) == wide
        for width in range(5, 120):
            summary = app.bar.collapsed_summary(width)
            assert cell_len(summary) <= width
        app.bar.update_agents((_agent("one", availability="running"),))
        assert app.bar.collapsed_summary(12) == "1 running"
        # Zero active with Recent entries: the dim recent note keeps the bar
        # non-empty while inventory exists.
        app.bar.update_agents((
            _agent("closed", availability="closed"),
            _agent("evicted", availability="evicted"),
        ))
        assert app.bar.collapsed_summary(80) == "2 recent"
        for width in range(5, 80):
            assert cell_len(app.bar.collapsed_summary(width)) <= width
        # Idle-only inventory: a dim idle note, never a bare failure count.
        app.bar.update_agents((_agent("one"), _agent("two")))
        assert app.bar.collapsed_summary(80) == "2 idle"


@pytest.mark.asyncio
async def test_collapsed_note_is_dimmed_and_counts_are_not() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((
            _agent("closed", availability="closed"),
            _agent("running", availability="running"),
        ))
        await pilot.pause()
        title = app.bar.query_one("#agent-bar-title", NoMarkupStatic).render()
        assert isinstance(title, Content)
        assert title.plain == "1 running"
        assert not title.spans
        app.bar.update_agents((_agent("closed", availability="closed"),))
        await pilot.pause()
        title = app.bar.query_one("#agent-bar-title", NoMarkupStatic).render()
        assert isinstance(title, Content)
        assert title.plain == "1 recent"
        assert [str(span.style) for span in title.spans] == ["dim"]


@pytest.mark.parametrize("ascii_mode", [False, True])
@pytest.mark.asyncio
async def test_widget_local_sheet_budget_footer_and_cell_fitting(
    ascii_mode: bool,
) -> None:
    app = _BrowserApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((
            _agent(
                "agent-x",
                identity="reviewer",
                availability="failed",
                base_model="glm-5.3",
            ),
        ))
        assert app.bar.collapsed_summary(80) == "1 idle"
        assert app.bar.border_title == "Background agents"
        app.bar.open_browser()
        await pilot.press("down")
        assert app.bar.region.height == 11
        assert app.bar.query_one("#agent-bar-rows").size.height == 5
        assert app.bar.query_one("#agent-bar-detail").size.height == 2
        # The title row carries the filter tabs while expanded.
        title = app.bar.query_one("#agent-bar-title")
        assert title.display
        tabs = str(title.render())
        assert "Resident 1" in tabs and "Idle 1" in tabs and "All 1" in tabs
        rows = _content(app.bar).splitlines()
        assert rows[1] == "Retained Idle"
        row = rows[2]
        assert "reviewer-glm-5.3-x" in row
        assert "agent-x" in row
        assert "Failed" in row
        assert cell_len(row) <= app.bar.query_one("#agent-bar-content").size.width
        footer = app.bar.query_one("#agent-bar-footer", Static).render()
        assert isinstance(footer, Content) and footer.spans
        if ascii_mode:
            assert "Esc" in footer.plain
            assert (
                footer.cell_length <= app.bar.query_one("#agent-bar-footer").size.width
            )
        else:
            assert (
                footer.plain == "↑↓ Move  ←→ Filter  Enter Open  R Release  D Details"
                "  F1 Help  Esc Close"
            )
        assert "C Stop run" not in footer.plain
        await pilot.resize_terminal(25, 24)
        await pilot.pause()
        row = _content(app.bar).splitlines()[2]
        assert "Failed" in row
        # The dim handle is clipped away before the name at narrow widths.
        assert "agent-x" not in row
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
        assert app.bar.region.height == 11
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
        # WP2: suspension shows with the warning glyph; restoring with the
        # running glyph — neither defaults to the idle label.
        ({"availability": "suspended", "last_run_status": "failed"}, "! Suspended"),
        ({"availability": "restoring", "current_run_status": "running"}, "… Restoring"),
        ({"availability": "evicted"}, "Evicted"),
        ({"availability": "closed"}, "Closed"),
        ({"availability": "closed", "last_run_status": "failed"}, "✗ Failed"),
        ({"availability": "closed", "stop_reason": "user_cancelled"}, "! Cancelled"),
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
        if agent.availability in {"evicted", "closed"}:
            expected += " (last recorded)"
        assert "\n" + expected in metadata
        if agent.stop_reason:
            assert f"Stop reason: {agent.stop_reason}" in metadata


@pytest.mark.asyncio
async def test_closed_tombstone_details_and_duration() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        agent = _agent(
            "closed-one",
            availability="closed",
            latest_run_id="r1",
            run_elapsed_seconds=252,
            idle_seconds=62,
            context_tokens=135_000,
            context_window=400_000,
        )
        app.bar.update_agents((agent,))
        app.bar.open_browser()
        await pilot.press("down")
        detail = str(app.bar.query_one("#agent-bar-detail", Static).render())
        # A closed tombstone is terminal: result state shows, no idle segment.
        assert "Closed" in detail
        assert "Last run 4m12s" in detail
        assert "Idle" not in detail
        assert "result preserved" in detail
        assert "135k/400k (34%) (last recorded)" in detail
        expired = agent.model_copy(update={"result_expired": True})
        assert "result expired" in app.bar._agent_details(expired)
        metadata = app.bar.full_metadata(agent)
        assert "Name: coder-closed-one" in metadata
        assert "Agent ID: closed-one" in metadata
        assert "Availability: closed" in metadata
        assert "State: Closed" in metadata


@pytest.mark.asyncio
async def test_full_metadata_presents_identity_keyed_record() -> None:
    app = _BrowserApp()
    async with app.run_test():
        agent = _agent(
            "researcher-1",
            identity="researcher",
            identity_revision="rev-7",
            availability="idle",
        )
        metadata = app.bar.full_metadata(agent)
        assert "Identity: researcher" in metadata
        assert "Identity revision: rev-7" in metadata
        assert "Profile:" not in metadata


@pytest.mark.asyncio
async def test_full_metadata_omits_identity_rows_without_binding() -> None:
    app = _BrowserApp()
    async with app.run_test():
        # A record carries no identity binding: neither the identity rows
        # nor the retired legacy profile fallback are presented.
        agent = _agent("legacy-1", identity=None, name="legacy-1")
        assert agent.identity is None
        metadata = app.bar.full_metadata(agent)
        assert "Name: legacy-1" in metadata
        assert "Identity:" not in metadata
        assert "Identity revision:" not in metadata
        assert "Profile:" not in metadata


@pytest.mark.asyncio
async def test_eight_cell_transition_markers_do_not_collide_with_outcomes() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((_agent("agent"),))
        app.bar.open_browser()
        app.bar.query_one("#agent-bar-content").styles.width = 8
        await pilot.pause()
        for state in ("… Finishing", "Compacting", "Running", "Closed"):
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
                identity="profile" * 20,
                availability="finalizing",
                stop_reason="budget_exceeded",
            ),
        ))
        app.bar.open_browser()
        await pilot.press("down")
        row = _content(app.bar).splitlines()[2]
        assert "Fin:B" in row and "profile" in row
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
        assert app.bar.region.height == 11
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
        assert app.bar.region.height == 11
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
        # The published order is authoritative: the update arrives as
        # (sibling, replacement) and the client consumes it verbatim.
        assert [item.agent_id for item in app.bar.agents] == (
            ["sibling", "agent-1"] if replacement else ["sibling"]
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
            # The published order is authoritative: (sibling, agent-1) verbatim.
            assert tuple(item.agent_id for item in app.bar.agents) == (
                "sibling",
                "agent-1",
            )
        else:
            assert tuple(item.agent_id for item in app.bar.agents) == (
                "agent-1",
                "sibling",
            )


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
@pytest.mark.parametrize("change", ["removed", "evicted", "finishing"])
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
            # The server omits released handles; "removed" already covers the
            # released-absence path (the client keeps no released filter).
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


@pytest.mark.asyncio
async def test_group_headers_are_labels_never_selections() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((
            _agent("run", availability="running", current_run_id="r1"),
            _agent("idle"),
            _agent("closed", availability="closed"),
        ))
        app.bar.open_browser()
        await pilot.pause()
        rows = app.bar._visible_rows()
        assert [row.kind for row in rows] == [
            "main",
            "header",
            "agent",
            "header",
            "agent",
        ]
        # Only Main and agent rows are selectable targets.
        assert app.bar._visible_ids() == (None, "run", "idle")
        # Keyboard movement skips headers through the shared mapping.
        await pilot.press("down")
        assert app.bar.selected_agent_id == "run"
        await pilot.press("down")
        assert app.bar.selected_agent_id == "idle"
        await pilot.press("up")
        assert app.bar.selected_agent_id == "run"
        # Mouse hit-testing resolves through the same mapping: the header row
        # at offset y=5 is a label, the agent row at y=6 selects.
        await pilot.click(app.bar, offset=(1, 5))
        assert app.bar.selected_agent_id == "run"
        assert app.selections == []
        await pilot.click(app.bar, offset=(1, 6))
        assert app.bar.selected_agent_id == "idle"
        assert app.selections == ["idle"]


@pytest.mark.asyncio
async def test_filter_tabs_show_truthful_counts_and_switch_groups() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((
            _agent("run", availability="running", current_run_id="r1"),
            _agent("idle"),
            _agent("closed", availability="closed"),
            _agent("gone", availability="evicted"),
        ))
        app.bar.open_browser()
        await pilot.pause()
        # Resident is the default view: Active + Retained Idle.
        tabs = str(app.bar.query_one("#agent-bar-title").render())
        assert "Resident 2" in tabs
        assert "Active 1" in tabs and "Idle 1" in tabs
        assert "Recent 2" in tabs and "All 4" in tabs
        rendered = _content(app.bar)
        assert "Active" in rendered and "Retained Idle" in rendered
        assert "Recent" not in rendered
        # Left/Right cycle the filters; the counts stay truthful.
        await pilot.press("right")
        assert app.bar._filter == "active"
        rendered = _content(app.bar)
        assert "run" in rendered and "idle" not in rendered
        await pilot.press("right")
        assert app.bar._filter == "idle"
        rendered = _content(app.bar)
        assert "idle" in rendered and "run" not in rendered
        await pilot.press("right")
        assert app.bar._filter == "recent"
        rendered = _content(app.bar)
        assert "closed" in rendered and "gone" in rendered
        assert "run" not in rendered and "idle" not in rendered
        await pilot.press("right")
        assert app.bar._filter == "all"
        rendered = _content(app.bar)
        assert (
            "Active" in rendered
            and "Retained Idle" in rendered
            and "Recent" in rendered
        )
        # Cycling wraps around back to the Resident default.
        await pilot.press("right")
        assert app.bar._filter == "resident"
        await pilot.press("left")
        assert app.bar._filter == "all"


@pytest.mark.asyncio
async def test_filter_switch_resolves_selection_to_visible_rows() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        app.bar.update_agents((
            _agent("run", availability="running", current_run_id="r1"),
            _agent("idle"),
        ))
        app.bar.open_browser()
        await pilot.press("down")
        assert app.bar.selected_agent_id == "run"
        await pilot.press("right")  # Active filter keeps the running agent
        assert app.bar.selected_agent_id == "run"
        await pilot.press("right")  # Idle filter hides it
        assert app.bar.selected_agent_id == "idle"
        assert app.bar.check_action("stop_run", ()) is False
        await pilot.press("left")  # back to the Active filter
        assert app.bar.selected_agent_id == "run"


@pytest.mark.asyncio
async def test_empty_state_semantics() -> None:
    app = _BrowserApp()
    async with app.run_test() as pilot:
        # The expanded browser is reachable with an empty inventory.
        app.bar.open_browser()
        await pilot.pause()
        assert app.bar.expanded
        assert app.bar.display
        rendered = _content(app.bar)
        assert "No background agents" in rendered
        assert app.bar.selected_agent_id is None
        # A Resident-empty inventory with Recent entries shows Recent + note.
        app.bar.update_agents((
            _agent("closed", availability="closed"),
            _agent("gone", availability="evicted"),
        ))
        await pilot.pause()
        rendered = _content(app.bar)
        assert "No resident agents" in rendered
        assert "Recent" in rendered
        assert "closed" in rendered and "gone" in rendered
        assert app.bar.selected_agent_id is None
        # Explicit empty filters are truthful, never a blank screen.
        await pilot.press("right")  # active
        assert "No active agents" in _content(app.bar)
        await pilot.press("right")  # idle
        assert "No retained idle agents" in _content(app.bar)
        await pilot.press("right")  # recent
        assert "closed" in _content(app.bar)
        # Esc still closes the browser from the empty state.
        await pilot.press("escape")
        assert not app.bar.expanded


@pytest.mark.asyncio
async def test_release_flow_confirmation_and_run_pinned_message() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((
            _agent("idle", latest_run_id="r1"),
            _agent("run", availability="running", current_run_id="r2"),
            _agent("closed", availability="closed", latest_run_id="r3"),
        ))
        app.bar.open_browser()
        await pilot.pause()
        await pilot.press("down", "down")
        assert app.bar.selected_agent_id == "idle"
        # Release is offered for Idle rows; Stop stays for active entries.
        footer = str(app.bar.query_one("#agent-bar-footer", Static).render())
        assert "R Release" in footer and "C Stop run" not in footer
        await pilot.press("r")
        assert app.bar._release_confirmation == ("idle", "r1")
        # Cancel is the default action of the confirmation.
        assert app.focused is app.bar.query_one("#agent-release-cancel", Button)
        prompt = str(app.bar.query_one("#agent-release-prompt", Static).render())
        assert "coder-idle (idle)" in prompt
        assert "Retained results and browser access are discarded" in prompt
        assert app.releases == []
        # Confirming posts the run-pinned release request.
        await pilot.press("left", "enter")
        assert app.releases == [("idle", "r1")]
        assert app.bar._release_confirmation is None
        assert app.focused is app.bar
        # Recent (closed) entries are release targets pinned to the latest run.
        await pilot.press("right", "right", "right")  # recent filter
        await pilot.press("down")
        assert app.bar.selected_agent_id == "closed"
        await pilot.press("r")
        assert app.bar._release_confirmation == ("closed", "r3")
        await pilot.press("left", "enter")
        assert app.releases == [("idle", "r1"), ("closed", "r3")]


@pytest.mark.asyncio
async def test_release_confirmation_cancel_and_escape_post_nothing() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((_agent("idle", latest_run_id="r1"),))
        app.bar.open_browser()
        await pilot.press("down", "r")
        assert app.bar._release_confirmation == ("idle", "r1")
        # Escape dismisses the confirmation without releasing.
        await pilot.press("escape")
        assert app.bar._release_confirmation is None
        assert app.bar.expanded
        assert app.releases == []
        # The Cancel button is the mouse path to the same dismissal.
        await pilot.press("r")
        await pilot.click("#agent-release-cancel")
        assert app.releases == []
        assert app.bar._release_confirmation is None
        assert app.focused is app.bar


@pytest.mark.asyncio
async def test_release_confirmation_fences_stale_run_identity() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        app.bar.update_agents((_agent("idle", latest_run_id="r1"),))
        app.bar.open_browser()
        await pilot.press("down", "r")
        assert app.bar._release_confirmation == ("idle", "r1")
        # The agent re-tasks while the confirmation is open: the captured
        # identity no longer matches, so the confirmation is dismissed and a
        # late submit cannot release the replacement run.
        app.bar.update_agents((_agent("idle", latest_run_id="r2"),))
        await pilot.pause()
        assert app.bar._release_confirmation is None
        await pilot.press("left", "enter")
        assert app.releases == []


@pytest.mark.asyncio
async def test_agent_row_shows_name_primary_with_dim_handle() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        agent = _agent("agent-7", identity="reviewer", base_model="glm-5.3")
        app.bar.update_agents((agent,))
        app.bar.open_browser()
        await pilot.pause()
        rows = app.bar._visible_rows()
        agent_row = rows[2]
        assert agent_row.kind == "agent" and agent_row.agent_id == "agent-7"
        assert agent_row.text.startswith("  reviewer-glm-5.3-7 · agent-7")
        content = app.bar.query_one("#agent-bar-content", Static).render()
        assert isinstance(content, Content)
        covered = {
            content.plain[span.start : span.end]
            for span in content.spans
            if str(span.style) == "dim"
        }
        # The handle renders dim; the group header renders dim as a label.
        assert "agent-7" in covered
        assert any(text.startswith("Retained Idle") for text in covered)


@pytest.mark.asyncio
async def test_markup_containing_name_renders_literally() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        agent = _agent("agent-9").model_copy(update={"name": "worker-[bold]x[/bold]-9"})
        app.bar.update_agents((agent,))
        app.bar.open_browser()
        await pilot.pause()
        rendered = _content(app.bar)
        assert "worker-[bold]x[/bold]-9" in rendered
        content = app.bar.query_one("#agent-bar-content", Static).render()
        assert isinstance(content, Content)
        assert "[bold]" in content.plain
        assert "[/bold]" in content.plain


@pytest.mark.asyncio
async def test_dim_handle_span_is_character_precise_with_wide_glyph_name() -> None:
    app = _BrowserApp()
    async with app.run_test(size=(80, 24)) as pilot:
        agent = _agent("agent-7", name="worker-分析器-7")
        app.bar.update_agents((agent,))
        app.bar.open_browser()
        await pilot.pause()
        text, dim = app.bar._agent_row(agent)
        assert dim is not None
        # Spans index characters, not cells: the wide-glyph name must not
        # shift the dim span off the handle.
        assert text[dim[0] : dim[1]] == "agent-7"
        rows = app.bar._visible_rows()
        content = app.bar._rows_content(rows)
        line_index = next(
            index for index, row in enumerate(rows) if row.agent_id == "agent-7"
        )
        base = sum(len(row.text) + 1 for row in rows[:line_index])
        handle_spans = [
            span
            for span in content.spans
            if str(span.style) == "dim"
            and content.plain[span.start : span.end] == "agent-7"
        ]
        assert [(span.start, span.end) for span in handle_spans] == [
            (base + dim[0], base + dim[1])
        ]


@pytest.mark.asyncio
async def test_disappearance_explanation_renders_markup_name_literally() -> None:
    from markdown_it import MarkdownIt

    from chartreux.app_server.events import AgentsUpdate
    from chartreux.cli.textual_ui.agent_browser import AgentBrowserController
    from chartreux.cli.textual_ui.widgets.messages import UserCommandMessage

    marked = _agent("one", name="worker-[bold]*evil*[/bold]-1")
    # Controller stand-in: top-level controller state plus a nested _app
    # namespace for the late-bound host calls.
    target = SimpleNamespace(
        agent_transcript_viewer=None,
        selection_target="one",
        agent_summaries=(marked,),
        _agent_evictions={},
        _app=SimpleNamespace(
            _request_agent_transcript_close=AsyncMock(),
            _mount_and_scroll=AsyncMock(),
            _turn_outcome_notice=SimpleNamespace(display=False),
            _agent_bar=None,
            _show_no_output_outcome=Mock(),
            _agent_is_active=Mock(return_value=True),
        ),
    )
    await AgentBrowserController.handle_agents_update(
        cast(Any, target), AgentsUpdate([]), 0.0
    )
    target._app._request_agent_transcript_close.assert_awaited_once_with(
        restore_focus=True
    )
    message = target._app._mount_and_scroll.await_args.args[0]
    assert isinstance(message, UserCommandMessage)
    # UserCommandMessage renders through a Markdown widget: the derived name
    # must survive rendering literally, never as markup.
    rendered = MarkdownIt("gfm-like").render(message._content)
    assert "worker-[bold]*evil*[/bold]-1" in rendered
    assert "<em>" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [80, 60, 40, 25])
async def test_browser_is_keyboard_complete_at_and_below_80(width: int) -> None:
    app = _BrowserApp()
    async with app.run_test(size=(width, 24)) as pilot:
        app.bar.update_agents((
            _agent("run", availability="running", current_run_id="r1"),
            _agent("idle", latest_run_id="r2"),
            _agent("closed", availability="closed"),
        ))
        app.bar.open_browser()
        await pilot.pause()
        assert app.bar.region.height == 11
        # Tabs, rows and the footer all stay cell-width aware below 80.
        tabs = str(app.bar.query_one("#agent-bar-title").render())
        assert cell_len(tabs) <= width - 2
        for row in _content(app.bar).splitlines():
            assert cell_len(row) <= width - 2
        footer = app.bar.query_one("#agent-bar-footer", Static).render()
        assert isinstance(footer, Content)
        assert footer.cell_length <= width - 2
        # The full keyboard workflow works at every width: move, open,
        # filter, release, and close.
        await pilot.press("down")
        assert app.bar.selected_agent_id == "run"
        await pilot.press("enter")
        assert app.selections == ["run"]
        await pilot.press("right", "right")  # idle filter
        await pilot.press("down")
        assert app.bar.selected_agent_id == "idle"
        await pilot.press("r", "escape")
        assert app.bar._release_confirmation is None
        assert app.releases == []
        await pilot.press("right")  # recent filter
        await pilot.press("down")
        assert app.bar.selected_agent_id == "closed"
        await pilot.press("enter")
        assert app.selections == ["run", "closed"]
        await pilot.press("escape")
        assert not app.bar.expanded
