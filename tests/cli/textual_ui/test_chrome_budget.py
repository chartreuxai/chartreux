from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from rich.cells import cell_len

from chartreux.app_server.events import AgentsUpdate
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine
from tests.cli.textual_ui.test_history_grouping import _message
from tests.conftest import build_test_chartreux_app


@pytest.mark.asyncio
async def test_ordinary_chat_80x24_region_budget() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()

        def height(selector: str) -> int:
            node = app.query_one(selector)
            return node.size.height if node.display else 0

        assert height("#chat") >= 14
        assert 24 - height("#chat") <= 10
        assert height("#bottom-app-container") == 3
        assert height("#agent-bar") <= 1
        assert height("#bottom-bar") == 1
        assert app.query_one(SessionStatusLine).size.height == 1
        assert height("#loading-area") <= 2
        input_area = app.query_one(ChatTextArea)
        input_area.load_text("\n".join(f"line {i}" for i in range(20)))
        await pilot.pause()
        assert height("#bottom-app-container") == 6
        assert height("#chat") >= 11


@pytest.mark.asyncio
async def test_status_overflow_reserves_known_usage() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(32, 24)) as pilot:
        status = app.query_one(SessionStatusLine)
        status.set_state(
            replace(
                status.state,
                # The default directory style uses only the basename. Make that
                # overflow too, independent of the process ID's digit count.
                cwd="/long/workspace/identity/that/cannot/fit-without-truncation",
                pid=12345,
                context_tokens=45_000,
                auto_compact_threshold=200_000,
            )
        )
        await pilot.pause()
        assert status.size.height == 1
        assert cell_len(status.render().plain) <= 32
        assert "pid" not in status.render().plain
        assert "45k/200k (22%)" in status.render().plain
        status.set_state(replace(status.state, context_tokens=0))
        await pilot.pause()
        assert "0/200k (0%)" in status.render().plain
        status.set_state(replace(status.state, context_tokens=None))
        assert "—/200k" in status.render().plain
        assert isinstance(app.query_one(AgentBar), AgentBar)


@pytest.mark.asyncio
async def test_docked_agent_sheet_has_fixed_ten_row_geometry() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        agents = [
            AgentSummaryModel(agent_id=str(i), profile="worker", availability="running")
            for i in range(8)
        ]
        await app._handle_turn_event(AgentsUpdate(agents))
        await pilot.press("ctrl+shift+a")
        await pilot.pause()
        bar = app.query_one(AgentBar)
        for _ in range(7):
            assert bar.region.height == 10
            assert bar.border_title == "Background agents"
            assert not app.query_one("#agent-bar-title").display
            assert app.query_one("#agent-bar-rows").size.height == 5
            assert app.query_one("#agent-bar-detail").size.height == 2
            footer = app.query_one("#agent-bar-footer")
            assert footer.size.height == 1
            assert footer.region.bottom <= bar.region.bottom - 1
            assert "Esc" in str(footer.render())
            await pilot.press("down")
        await app._handle_turn_event(
            AgentsUpdate([
                agent.model_copy(update={"compacting": True}) for agent in agents
            ])
        )
        await pilot.pause()
        assert bar.region.height == 10
        await pilot.press("d")
        await pilot.pause()
        assert bar.region.height == 10
        assert app.query_one("#agent-bar-full-details").size.height == 7
        await pilot.press("escape", "f1")
        await pilot.pause()
        assert bar.region.height == 10
        await pilot.resize_terminal(100, 30)
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        assert bar.region.height == 10


@pytest.mark.asyncio
async def test_session_preview_cancel_restores_draft_focus_and_scroll() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        composer = app.query_one(ChatTextArea)
        composer.load_text("unfinished draft")
        history = app.app_server._state.projection.state.history
        assert history is not None
        history.extend(_message(i) for i in range(45))
        await app._resume_history_from_messages()
        await app._mount_history_batch(
            history[15:25], app._messages_area, start_index=15, before=0
        )
        await pilot.pause()
        unit = app._transcript.units["message-18"]
        app._chat_widget.scroll_to(
            y=app._chat_widget.scroll_offset.y
            + unit.mounted_roots[0].region.y
            - app._chat_widget.region.y,
            animate=False,
            force=True,
            immediate=True,
        )
        await pilot.pause()
        origin = app._transcript.capture_anchor(app._messages_area, following=False)
        assert origin is not None and origin.entry_id == "message-18"
        admitted = app._transcript.admitted_start_index
        await app._show_session_picker()
        app._picker.preview_session_id = "other-session"
        await app._apply_picker_preview("other-session", [])
        await app.on_session_picker_app_cancelled(SessionPickerApp.Cancelled())
        await pilot.pause()
        restored = app._transcript.capture_anchor(app._messages_area, following=False)
        assert restored is not None
        assert restored.entry_id == origin.entry_id
        assert restored.row_offset == origin.row_offset
        assert app._transcript.admitted_start_index == admitted
        assert not app._transcript_following
        assert composer.text == "unfinished draft"
        assert app.screen.focused is composer


@pytest.mark.asyncio
async def test_resize_reflow_falls_back_when_viewport_cannot_match_screen() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        history = app.app_server._state.projection.state.history
        assert history is not None
        history.extend(
            _message(i).model_copy(
                update={
                    "content": [TextContentBlock(text=f"entry {i}\n\nsecond paragraph")]
                }
            )
            for i in range(12)
        )
        await app._resume_history_from_messages()
        app._chat_widget.styles.max_width = 50
        await pilot.pause()
        unit = app._transcript.units["message-4"]
        app._chat_widget.scroll_to(
            y=app._chat_widget.scroll_offset.y
            + unit.mounted_roots[0].region.y
            - app._chat_widget.region.y,
            animate=False,
            force=True,
            immediate=True,
        )
        await pilot.pause()
        before = app._transcript.capture_anchor(app._messages_area, following=False)
        assert before is not None and before.entry_id == "message-4"
        await pilot.resize_terminal(60, 24)
        task = app._resize_anchor_task
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), timeout=3)
        assert app._chat_widget.region.width != app.size.width
        after = app._transcript.capture_anchor(app._messages_area, following=False)
        assert after is not None
        assert after.entry_id == before.entry_id
        assert after.row_offset == before.row_offset
        assert not app._transcript_following


@pytest.mark.asyncio
@pytest.mark.parametrize("following", [False, True])
async def test_resize_restores_wrapped_reading_anchor(following: bool) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        history = app.app_server._state.projection.state.history
        assert history is not None
        history.extend(
            _message(i).model_copy(
                update={"content": [TextContentBlock(text=(f"entry-{i} " * 90))]}
            )
            for i in range(12)
        )
        await app._resume_history_from_messages()
        await pilot.pause()
        if following:
            app._chat_widget.scroll_end(animate=False, immediate=True)
            app._chat_widget.anchor()
        else:
            unit = app._transcript.units["message-4"]
            app._chat_widget.scroll_to(
                y=app._chat_widget.scroll_offset.y
                + unit.mounted_roots[0].region.y
                - app._chat_widget.region.y
                + 3,
                animate=False,
                force=True,
                immediate=True,
            )
        await pilot.pause()
        before = app._transcript.capture_anchor(app._messages_area, following=following)
        assert before is not None
        if not following:
            assert before.entry_id == "message-4"
            assert before.row_offset >= 2
            assert app._last_transcript_anchor is not None
            assert app._last_transcript_anchor.entry_id == before.entry_id
            assert not app._last_transcript_anchor.following
        prior_resize_task = app._resize_anchor_task
        await pilot.resize_terminal(60, 24)
        assert app._resize_anchor_task is not prior_resize_task
        assert app._resize_anchor_task is not None
        await app._resize_anchor_task
        await pilot.pause()
        after = app._transcript.capture_anchor(app._messages_area, following=following)
        assert after is not None
        assert app._transcript_following == following
        if following:
            assert app._chat_widget.is_at_bottom
        else:
            assert after.entry_id == before.entry_id
            assert after.row_offset == before.row_offset
