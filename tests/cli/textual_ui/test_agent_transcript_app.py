from __future__ import annotations

import asyncio
from contextlib import suppress
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.events import AgentsUpdate, TurnStarted
from chartreux.app_server.models import PublicTurn, PublicTurnStatus
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from tests.conftest import build_test_chartreux_app


def _agent(agent_id: str, availability: str = "idle") -> AgentSummaryModel:
    return AgentSummaryModel(
        agent_id=agent_id, profile="worker", availability=availability
    )


@pytest.mark.asyncio
async def test_agent_selection_hides_chat_and_escape_restores_it() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        assert app.query(AgentTranscriptViewer)
        assert not app.query_one("#chat").display
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)
        assert app.query_one("#chat").display


@pytest.mark.asyncio
async def test_eviction_keeps_viewer_open_and_release_closes_it() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one", "running")]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert viewer._live_timer is not None

        await app._handle_turn_event(AgentsUpdate([_agent("one", "evicted")]))
        await pilot.pause()
        assert app.query_one(AgentTranscriptViewer) is viewer
        assert viewer._live_timer is None

        await app._handle_turn_event(AgentsUpdate([]))
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)


@pytest.mark.asyncio
async def test_interrupt_action_collapses_expanded_agent_bar_before_interrupting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    interrupt = MagicMock()
    monkeypatch.setattr(app, "_try_interrupt", interrupt)
    async with app.run_test() as pilot:
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a")
        await pilot.pause()
        assert app._agent_bar is not None and app._agent_bar.expanded

        await pilot.press("escape")
        await pilot.pause()

        assert not app._agent_bar.expanded
        interrupt.assert_not_called()


@pytest.mark.asyncio
async def test_escape_interrupts_after_agents_are_released_from_expanded_bar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    interrupt = MagicMock()
    monkeypatch.setattr(app, "_try_interrupt", interrupt)
    async with app.run_test() as pilot:
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a")
        await pilot.pause()
        assert app._agent_bar is not None and app._agent_bar.expanded

        await app._handle_turn_event(AgentsUpdate([]))
        await pilot.pause()
        assert app._agent_bar is not None and not app._agent_bar.display
        assert not app._agent_bar.expanded

        await pilot.press("escape")
        await pilot.pause()
        interrupt.assert_called_once()


@pytest.mark.asyncio
async def test_empty_agent_bar_toggle_keeps_focus_on_chat_input() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        assert app._chat_input_container is not None
        app._chat_input_container.focus_input()
        await pilot.pause()
        input_widget = app._chat_input_container.input_widget
        assert input_widget is not None
        assert app.screen.focused is input_widget

        await pilot.press("ctrl+shift+a")
        await pilot.pause()

        assert app._agent_bar is not None and not app._agent_bar.expanded
        assert app.screen.focused is input_widget


@pytest.mark.asyncio
async def test_turn_started_event_closes_the_agent_transcript_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        assert app.query(AgentTranscriptViewer)

        async def events():
            yield TurnStarted(
                PublicTurn(
                    id="turn",
                    session_id=app.app_server.session_id,
                    status=PublicTurnStatus.IN_PROGRESS,
                    started_at=1,
                )
            )
            await asyncio.Event().wait()

        monkeypatch.setattr(app.app_server, "events", events)
        listener = asyncio.create_task(app._listen_app_server_events())
        try:
            await pilot.pause()
            assert not app.query(AgentTranscriptViewer)
        finally:
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener
