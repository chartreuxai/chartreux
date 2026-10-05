from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from chartreux.app_server.events import TurnCompleted
from chartreux.app_server.models import (
    ConfigIssue,
    DebugLogEntry,
    DebugLogPage,
    PublicTurn,
    PublicTurnStatus,
)
from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import agent_is_active
from chartreux.cli.textual_ui.widgets.debug_console import DebugConsole
from chartreux.cli.textual_ui.widgets.log_level_picker import LogLevelPickerApp
from tests.conftest import build_test_chartreux_app


@pytest.mark.asyncio
async def test_config_and_mcp_recovery_indications_survive_navigation() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await pilot.pause(0.2)
        app._show_config_issue(ConfigIssue(file="broken.toml", message="invalid value"))
        app._set_recovery_issue("mcp:broken", "MCP broken; run /mcp")
        await app._show_theme()
        await pilot.pause(0.1)
        await app._switch_to_input_app()
        assert app.query_one("#recovery-notice").display
        assert "config:broken.toml:invalid value" in app._recovery_issues
        assert "mcp:broken" in app._recovery_issues
        app._show_config_issues()
        app._show_mcp_discovery_failures()
        assert not app._recovery_issues
        assert not app.query_one("#recovery-notice").display


@pytest.mark.asyncio
async def test_log_level_failed_save_reports_session_and_retains_recovery() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        with patch.object(
            app,
            "_persist_log_level_config",
            new=AsyncMock(side_effect=OSError("disk full")),
        ):
            await app.on_log_level_picker_app_applied(
                LogLevelPickerApp.Applied("DEBUG", "ERROR", config_cleared=False)
            )
        text = app._recovery_issues["log-level-save"][1]
        assert "disk full" in text
        assert "session override DEBUG applied" in text
        assert "not saved" in text and "/log-level" in text
        await app._show_theme()
        await pilot.pause(0.1)
        await app._switch_to_input_app()
        assert app.query_one("#recovery-notice").display
        with patch.object(app, "_persist_log_level_config", new=AsyncMock()):
            await app.on_log_level_picker_app_applied(
                LogLevelPickerApp.Applied("DEBUG", "ERROR", config_cleared=False)
            )
        assert "log-level-save" not in app._recovery_issues


@pytest.mark.asyncio
async def test_debug_console_loading_empty_failed_and_recovers() -> None:
    app = build_test_chartreux_app()
    gate = asyncio.Event()
    outcomes = [
        RuntimeError("initial read denied"),
        DebugLogPage(entries=[], cursor=None, has_more=False),
        RuntimeError("read denied"),
        DebugLogPage(
            entries=[
                DebugLogEntry(
                    id="one",
                    timestamp=datetime.now(UTC),
                    level="INFO",
                    message="hello",
                    pid=1,
                    ppid=0,
                    raw_line="hello",
                )
            ],
            cursor=None,
            has_more=False,
        ),
    ]

    async def read_logs(*, limit: int = 100, offset: int = 0) -> DebugLogPage:
        await gate.wait()
        result = outcomes.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async with app.run_test() as pilot:
        console = DebugConsole(app.app_server.resources.runtime)
        with patch.object(
            app.app_server.resources.runtime, "read_logs", side_effect=read_logs
        ):
            await app.mount(console)
            await pilot.pause(0.05)
            state = console.query_one("#debug-console-state")
            assert "Loading debug logs" in str(state.render())
            gate.set()
            await pilot.pause(0.1)
            assert "Could not load debug logs" in str(state.render())
            await console._load_page()
            assert "No debug logs yet" in str(state.render())
            await console._poll_latest()
            assert "Could not refresh debug logs" in str(state.render())
            await console._poll_latest()
            assert not state.display
            assert console._log_view is not None and console._log_view._lines


@pytest.mark.asyncio
@pytest.mark.parametrize("compacting", [False, True])
async def test_no_output_completion_distinguishes_active_agents(
    compacting: bool,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        # The agent list remains owned by the app rather than by turn completion.
        agent = AgentSummaryModel(
            agent_id="agent-1",
            profile="worker",
            availability="running",
            current_run_status="running",
            compacting=compacting,
        )
        assert agent_is_active(agent)
        with patch(
            "chartreux.cli.textual_ui.app.agent_is_active", wraps=agent_is_active
        ) as active:
            app._agent_summaries = [agent]
            await app._handle_turn_event(
                TurnCompleted(
                    PublicTurn(
                        id="turn",
                        session_id="test-session",
                        status=PublicTurnStatus.COMPLETED,
                        started_at=1,
                    )
                )
            )
            notice = app.query_one("#turn-outcome-notice")
            assert "agents are still running" in str(notice.render())
            assert notice.display
            active.assert_called_with(agent)
        app._agent_summaries = []
        app._show_no_output_outcome()
        assert "ready for another prompt" in str(notice.render())
        await pilot.pause()
