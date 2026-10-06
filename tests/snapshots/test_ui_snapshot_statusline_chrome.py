from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from rich.cells import cell_len
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.pilot import Pilot
from textual.widgets import Static

from chartreux.app_server.config import StatusLineConfigView
from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.message_header import MessageHeader
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusLine,
    SessionStatusState,
)
from tests.snapshots.snap_compare import SnapCompare
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


class ChromeSnapshotApp(App[None]):
    """Bounded docked-sheet/status fixture, without backend timing or resources."""

    config: SimpleNamespace = SimpleNamespace(ascii_chrome=False)

    CSS = """
    #transcript { height: 1fr; }
    #chrome { dock: bottom; height: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Conversation remains above the docked agent sheet.", id="transcript"
        )
        with Vertical(id="chrome"):
            yield AgentBar()
            yield SessionStatusLine(
                SessionStatusState(
                    cwd="/test/workdir",
                    pid=0,
                    context_tokens=135_000,
                    auto_compact_threshold=400_000,
                    model_identity="ZAI / glm-5.3",
                    branch="feat/chrome",
                    branch_status="branch",
                ),
                StatusLineConfigView(
                    segments=[
                        "directory",
                        "pid",
                        "context",
                        "model",
                        "git-branch",
                        "spend-today",
                    ]
                ),
            )

    async def on_load(self) -> None:
        install_snapshot_wake()

    def on_mount(self) -> None:
        bar = self.query_one(AgentBar)
        bar.update_main_details(
            {"Model": "ZAI / glm-5.3", "State": "Idle", "Session": "snapshot-session"},
            context_tokens=135_000,
            auto_compact_threshold=400_000,
        )
        bar.update_agents(
            tuple(
                AgentSummaryModel.model_validate({
                    "agent_id": agent_id,
                    "profile": "reviewer",
                    "availability": availability,
                    "compacting": compacting,
                    "stop_reason": stop_reason,
                    "context_tokens": 135_000,
                    "context_window": 400_000,
                    "turns_used": 2,
                })
                for agent_id, availability, compacting, stop_reason in (
                    ("agent-running", "running", False, None),
                    ("agent-compacting", "running", True, None),
                    ("agent-failed", "failed", False, "error"),
                    ("agent-budget", "idle", False, "budget_exceeded"),
                    ("agent-finishing", "finalizing", False, None),
                    ("agent-cancelled", "idle", False, "user_cancelled"),
                    ("agent-evicted", "evicted", False, None),
                )
            )
        )


@pytest.mark.parametrize("state", ["list", "details", "help", "main-details"])
def test_agent_sheet(snap_compare: SnapCompare, state: str) -> None:
    async def before(pilot: Pilot) -> None:
        bar = pilot.app.query_one(AgentBar)
        bar.open_browser()
        await pilot.pause()
        if state == "details":
            await pilot.press("down", "d")
        elif state == "help":
            await pilot.press("f1")
        elif state == "main-details":
            await pilot.press("d")
        await pilot.pause()
        assert bar.region.height == 10
        assert bar.region.bottom <= pilot.app.query_one(SessionStatusLine).region.y
        assert bar.selected_agent_id == (
            "agent-running" if state == "details" else None
        )
        assert not bar.check_action("stop_run", ())
        assert "Stop run" not in str(
            bar.query_one("#agent-bar-footer", Static).render()
        )
        if state == "help":
            help_text = str(bar.query_one("#agent-bar-full-content", Static).render())
            assert (
                "C: confirm stopping the highlighted Running/Compacting run."
                in help_text
            )
            assert "Cancel preserves the running task" in help_text

    assert snap_compare(
        "test_ui_snapshot_statusline_chrome.py:ChromeSnapshotApp",
        terminal_size=(80, 24),
        run_before=before,
    )


@pytest.mark.parametrize("availability", ["running", "idle", "evicted"])
@pytest.mark.parametrize(
    "width,ascii_mode", [(80, False), (40, False), (80, True), (40, True)]
)
def test_agent_duration_details(
    snap_compare: SnapCompare,
    monkeypatch: pytest.MonkeyPatch,
    availability: str,
    width: int,
    ascii_mode: bool,
) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.widgets.agent_bar.time",
        SimpleNamespace(monotonic=lambda: 100.0),
    )
    app = ChromeSnapshotApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)

    async def before(pilot: Pilot) -> None:
        bar = pilot.app.query_one(AgentBar)
        bar.update_agents((
            AgentSummaryModel(
                agent_id="agent-review",
                profile="reviewer",
                availability=availability,
                current_run_id="run-1" if availability == "running" else None,
                latest_run_id="run-1",
                run_elapsed_seconds=252,
                idle_seconds=62 if availability != "running" else None,
                context_tokens=135_000,
                context_window=400_000,
                active_provider="ZAI",
                base_model="glm-5.3",
                turns_used=2,
            ),
        ))
        bar.open_browser()
        await pilot.press("down")
        await pilot.pause()
        assert bar.region.height == 10
        assert bar.region.bottom <= pilot.app.query_one(SessionStatusLine).region.y
        footer = bar.query_one("#agent-bar-footer", Static).render()
        assert cell_len(str(footer)) <= width - 2
        assert "Enter" in str(footer) and "Esc" in str(footer)
        assert bar.check_action("stop_run", ()) == (availability == "running")
        if availability == "running":
            assert "C" in str(footer)
            assert "Run 4m12s" in str(
                bar.query_one("#agent-bar-detail", Static).render()
            )
            if width == 80:
                assert "C Stop run" in str(footer)
        else:
            assert "Stop run" not in str(footer)

    assert snap_compare(app, terminal_size=(width, 24), run_before=before)


@pytest.mark.parametrize("width", [100, 80, 68, 60, 40, 24, 12])
def test_status_line_degradation(snap_compare: SnapCompare, width: int) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.pause()
        status = pilot.app.query_one(SessionStatusLine)
        assert status.size.height == 1
        rendered = status.render().plain
        if width <= 68:
            assert "pid" not in rendered
        if width >= 68:
            assert "Today —" in rendered
        if width == 68:
            # The shorter label retains spend after pid yields at this width.
            assert rendered == (
                "workdir | 135k/400k (34%) | ZAI / glm-5.3 | feat/chrome | Today —"
            )
        if width < 68:
            assert "Today" not in rendered
        assert "135k/400k" in rendered or width == 12

    assert snap_compare(
        "test_ui_snapshot_statusline_chrome.py:ChromeSnapshotApp",
        terminal_size=(width, 12),
        run_before=before,
    )


@pytest.mark.parametrize("count,width", [(0, 80), (2, 80), (2, 40), (2, 24)])
def test_background_jobs_status_line(
    snap_compare: SnapCompare, count: int, width: int
) -> None:
    async def before(pilot: Pilot) -> None:
        status = pilot.app.query_one(SessionStatusLine)
        status.set_state(replace(status.state, active_background_job_count=count))
        status.set_config(
            StatusLineConfigView(
                segments=["directory", "pid", "context", "background-jobs", "model"]
            )
        )
        await pilot.pause()
        rendered = status.render().plain
        assert status.state.active_background_job_count == count
        assert status.size.height == 1
        assert cell_len(rendered) <= width
        if width >= 40:
            assert f"Jobs {count}" in rendered
        else:
            assert "Jobs" not in rendered
        if width <= 40:
            assert "pid" not in rendered and "ZAI" not in rendered

    assert snap_compare(
        "test_ui_snapshot_statusline_chrome.py:ChromeSnapshotApp",
        terminal_size=(width, 12),
        run_before=before,
    )


class TimestampHeadersSnapshotApp(App[None]):
    CSS = """
    Static { height: 1; }
    #narrow-header { width: 9; }
    """

    def compose(self) -> ComposeResult:
        today = datetime(2026, 6, 15, 12, 34, tzinfo=UTC)
        yield Static("Today")
        yield MessageHeader("You", posted_at=today)
        yield Static("Earlier day")
        yield MessageHeader(
            "Assistant", posted_at=datetime(2026, 6, 14, 9, 5, tzinfo=UTC)
        )
        yield Static("Legacy history (no backfill)")
        yield MessageHeader("You")
        yield Static("Preference disabled")
        yield MessageHeader("Assistant", posted_at=today, show_message_timestamps=False)
        yield Static("Narrow: total survives, timestamp yields")
        header = MessageHeader("Assistant", posted_at=today, turn_duration_ms=372000)
        header.id = "narrow-header"
        yield header

    async def on_load(self) -> None:
        install_snapshot_wake()


def test_timestamp_headers(snap_compare: SnapCompare) -> None:
    async def before(pilot: Pilot) -> None:
        await pilot.pause()
        headers = list(pilot.app.query(MessageHeader))
        assert [
            header.timestamp_for_width(header.size.width) for header in headers
        ] == ["12:34", "2026-06-14 09:05", "", "", "6m12s"]

    assert snap_compare(
        "test_ui_snapshot_statusline_chrome.py:TimestampHeadersSnapshotApp",
        terminal_size=(50, 12),
        run_before=before,
    )
