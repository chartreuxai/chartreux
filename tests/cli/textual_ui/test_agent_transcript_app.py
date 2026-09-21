from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from textual import events

from chartreux.app_server.events import AgentsUpdate
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.agent_sidebar import AgentSidebar
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.chat_input import ChatInputContainer
from chartreux.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from tests.conftest import build_test_chartreux_app


def _agent(agent_id: str, availability: str = "idle") -> AgentSummaryModel:
    return AgentSummaryModel(
        agent_id=agent_id, profile="worker", availability=availability
    )


def _response(*texts: str, has_more: bool = False) -> AgentTranscriptGetResponse:
    return AgentTranscriptGetResponse(
        state=AgentTranscriptState.AVAILABLE,
        entries=[
            AgentTranscriptEntry(
                entry_id=f"entry-{index}",
                kind=AgentTranscriptEntryKind.USER_TEXT,
                display_text=text,
            )
            for index, text in enumerate(texts)
        ],
        oldest_cursor="older" if has_more else None,
        has_more=has_more,
    )


class _DelayedTranscriptReader:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None, int]] = []
        self.responses: list[asyncio.Future[AgentTranscriptGetResponse]] = []

    async def __call__(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        self.calls.append((agent_id, before, limit))
        response: asyncio.Future[AgentTranscriptGetResponse] = asyncio.Future()
        self.responses.append(response)
        return await asyncio.shield(response)

    def respond(self, index: int, *texts: str, has_more: bool = False) -> None:
        self.responses[index].set_result(_response(*texts, has_more=has_more))


def _viewer_content(viewer: AgentTranscriptViewer) -> str:
    return str(viewer.query_one("#agent-transcript-content").render())


async def _open_viewer(
    app: ChartreuxApp, pilot: object, agent_id: str = "one"
) -> AgentTranscriptViewer:
    del agent_id
    if not app.query(AgentSidebar):
        await pilot.press("ctrl+shift+a")  # type: ignore[attr-defined]
        await pilot.pause()  # type: ignore[attr-defined]
    await pilot.press("enter")  # type: ignore[attr-defined]
    await pilot.pause()  # type: ignore[attr-defined]
    return app.query_one(AgentTranscriptViewer)


@pytest.mark.asyncio
async def test_release_closes_viewer_while_eviction_keeps_it_open() -> None:
    app = build_test_chartreux_app()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=_response("saved")
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a", "enter")
        await pilot.pause()
        assert app.query(AgentTranscriptViewer)

        await app._handle_turn_event(AgentsUpdate([_agent("one", "evicted")]))
        await pilot.pause()
        assert app.query(AgentTranscriptViewer)
        assert app.app_server.resources.sessions.read_agent_transcript.await_count == 1

        await app._handle_turn_event(AgentsUpdate([]))
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)
        assert app.query_one(AgentSidebar).selected_agent_id is None
        assert app.app_server.resources.sessions.read_agent_transcript.await_count == 1


@pytest.mark.asyncio
async def test_keyboard_journey_opens_pages_refreshes_and_esc_restores_sidebar_focus() -> (
    None
):
    app = build_test_chartreux_app()
    read = AsyncMock(
        side_effect=[
            _response("latest", has_more=True),
            _response("older"),
            _response("refreshed"),
        ]
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.wait_for(app._initial_history_loaded.wait(), timeout=1)
        app.app_server.resources.sessions.read_agent_transcript = read
        await app._handle_turn_event(AgentsUpdate([_agent("one"), _agent("two")]))

        await pilot.press("ctrl+shift+a")
        await pilot.pause()
        sidebar = app.query_one(AgentSidebar)
        assert sidebar.selected_agent_id == "one"
        assert app.screen.focused is sidebar

        await pilot.press("down", "up", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert viewer._agent_id == "two"
        assert app.screen.focused is viewer
        assert read.await_count == 1

        await pilot.press("pageup")
        await pilot.pause()
        await pilot.press("r")
        await pilot.pause()
        assert read.await_count == 3
        assert [call.kwargs["before"] for call in read.await_args_list] == [
            None,
            "older",
            None,
        ]

        assert app.check_action("interrupt", ()) is True
        with patch.object(app, "_try_interrupt") as interrupt:
            await pilot.press("escape")
        await asyncio.sleep(0.1)
        assert not app.query(AgentTranscriptViewer)
        interrupt.assert_not_called()
        assert app.screen.focused is sidebar
        assert app.check_action("interrupt", ()) is True


@pytest.mark.asyncio
async def test_terminal_refocus_preserves_agent_browser_focus() -> None:
    app = build_test_chartreux_app()

    async with app.run_test() as pilot:
        await pilot.pause()
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a")
        await pilot.pause()
        sidebar = app.query_one(AgentSidebar)
        assert app.screen.focused is sidebar

        app.post_message(events.AppFocus())
        await pilot.pause()
        assert app.screen.focused is sidebar

        viewer = await _open_viewer(app, pilot)
        assert app.screen.focused is viewer
        app.post_message(events.AppFocus())
        await pilot.pause()
        assert app.screen.focused is viewer

        await pilot.press("escape", "ctrl+shift+a")
        await pilot.pause()
        text_area = app.query_one(ChatInputContainer).query_one(ChatTextArea)
        assert app.screen.focused is text_area


@pytest.mark.asyncio
@pytest.mark.parametrize("scenario", ["initial", "older", "refresh"])
async def test_parent_reset_closes_viewer_and_discards_delayed_response(
    scenario: str,
) -> None:
    app = build_test_chartreux_app()
    read = _DelayedTranscriptReader()

    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.wait_for(app._initial_history_loaded.wait(), timeout=1)
        app.app_server.resources.sessions.read_agent_transcript = read
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await _open_viewer(app, pilot)

        if scenario == "older":
            read.respond(0, "latest", has_more=True)
            await pilot.pause()
            await pilot.press("pageup")
            await pilot.pause()
        elif scenario == "refresh":
            read.respond(0, "latest")
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()

        delayed_index = len(read.responses) - 1
        old_parent_id = app.app_server.session_id
        await app._clear_history()
        await pilot.pause()

        assert app.app_server.session_id != old_parent_id
        assert not app.query(AgentTranscriptViewer)
        read.respond(delayed_index, f"late {scenario}")
        await pilot.pause()

        assert not app.query(AgentTranscriptViewer)
        assert read.responses[delayed_index].done()


@pytest.mark.asyncio
async def test_parent_resets_reject_old_viewer_response_after_reopen() -> None:
    app = build_test_chartreux_app()
    read = _DelayedTranscriptReader()

    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.wait_for(app._initial_history_loaded.wait(), timeout=1)
        app.app_server.resources.sessions.read_agent_transcript = read
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await _open_viewer(app, pilot)
        first_parent_id = app.app_server.session_id

        await app._clear_history()
        second_parent_id = app.app_server.session_id
        await app._clear_history()
        assert second_parent_id != first_parent_id
        assert app.app_server.session_id != second_parent_id
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        if app.query(AgentSidebar):
            await app.action_toggle_agent_sidebar()
            await pilot.pause()

        reopened = await _open_viewer(app, pilot)
        read.respond(0, "old instance")
        await pilot.pause()

        assert app.query_one(AgentTranscriptViewer) is reopened
        assert "old instance" not in _viewer_content(reopened)
        read.respond(1, "reopened instance")
        await pilot.pause()
        assert "reopened instance" in _viewer_content(reopened)


@pytest.mark.asyncio
async def test_failed_session_switch_closes_viewer_without_restoring_stale_data() -> (
    None
):
    app = build_test_chartreux_app()

    async with app.run_test() as pilot:
        await pilot.pause()
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=_response("old parent transcript")
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await _open_viewer(app, pilot)
        await pilot.pause()
        assert "old parent transcript" in _viewer_content(
            app.query_one(AgentTranscriptViewer)
        )

        app.app_server.resume = AsyncMock(side_effect=RuntimeError("resume failed"))
        with pytest.raises(RuntimeError, match="resume failed"):
            await app._resume_local_session("other-parent")
        await pilot.pause()

        assert not app.query(AgentTranscriptViewer)


@pytest.mark.asyncio
async def test_viewer_lifecycle_never_calls_agent_actions() -> None:
    app = build_test_chartreux_app()
    read = AsyncMock(return_value=_response("saved"))

    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.wait_for(app._initial_history_loaded.wait(), timeout=1)
        app.app_server.resources.sessions.read_agent_transcript = read
        interrupt = AsyncMock()
        app.app_server.interrupt = interrupt
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await _open_viewer(app, pilot)

        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)
        interrupt.assert_not_awaited()
        assert read.await_count == 1
