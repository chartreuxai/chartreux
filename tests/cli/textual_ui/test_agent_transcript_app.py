from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import suppress
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.widgets import Button

from chartreux.app_server.events import AgentsUpdate, TurnStarted
from chartreux.app_server.models import PublicTurn, PublicTurnStatus
from chartreux.app_server.protocol import (
    AgentsCancelResponse,
    AgentSummaryModel,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
    CancelOutcome,
)
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.observability.logging import (
    get_effective_log_level,
    init_file_logging,
    logger,
    set_log_level,
)
from tests.cli.textual_ui.test_agent_transcript import _available, _entry
from tests.cli.textual_ui.test_history_grouping import _message
from tests.conftest import build_test_chartreux_app


def _agent(agent_id: str, availability: str = "idle") -> AgentSummaryModel:
    return AgentSummaryModel(
        agent_id=agent_id, profile="worker", availability=availability
    )


@pytest.fixture
def product_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Generator[Path, None, None]:
    previous_level = logger.level
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setenv("DEBUG_MODE", "false")
    path = tmp_path / "chartreux.log"
    try:
        yield path
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.setLevel(previous_level)


@pytest.mark.asyncio
async def test_debug_startup_routes_viewer_and_app_records_once(
    product_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    init_file_logging(product_log)
    init_file_logging(product_log)
    assert get_effective_log_level() == "DEBUG"
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
        assert app.query_one(AgentTranscriptViewer)._heartbeat_timer is not None
        assert app._diagnostic_heartbeat_timer is not None
        app._diagnostic_last_heartbeat -= 1.0
        app._record_diagnostic_heartbeat()
    lines = product_log.read_text().splitlines()
    assert sum("App heartbeat drift:" in line for line in lines) == 1
    assert (
        sum("Agent transcript status remove_children:" in line for line in lines) == 1
    )
    assert sum("Agent transcript fetch:" in line for line in lines) == 1
    assert sum("Agent selection phase=submitted" in line for line in lines) == 1


@pytest.mark.asyncio
async def test_warning_skips_diagnostic_heartbeats_and_level_change_waits_for_mount(
    product_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    init_file_logging(product_log)
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        assert app._diagnostic_heartbeat_timer is None
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        assert app.query_one(AgentTranscriptViewer)._heartbeat_timer is None
        set_log_level("DEBUG")
        assert app._diagnostic_heartbeat_timer is None
        assert app.query_one(AgentTranscriptViewer)._heartbeat_timer is None
        await app._request_agent_transcript_close()
        app._submit_agent_selection("one")
        assert app._agent_transition_task is not None
        await app._agent_transition_task
        await pilot.pause()
        assert app.query_one(AgentTranscriptViewer)._heartbeat_timer is not None
    assert "App heartbeat drift:" not in product_log.read_text()


@pytest.mark.asyncio
async def test_selection_markers_include_generation_and_close_timing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = build_test_chartreux_app()
    with caplog.at_level(logging.DEBUG, logger="vibe"):
        async with app.run_test() as pilot:
            app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
                return_value=AgentTranscriptGetResponse(
                    state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
                )
            )
            await app._handle_turn_event(AgentsUpdate([_agent("one")]))
            await pilot.press("ctrl+shift+a", "down", "enter")
            await pilot.pause()
            await app._request_agent_transcript_close()
    messages = [record.getMessage() for record in caplog.records]
    assert any("phase=submitted generation=" in text for text in messages)
    assert any("phase=transition-start generation=" in text for text in messages)
    assert any("phase=closed generation=" in text for text in messages)


@pytest.mark.asyncio
async def test_close_main_pane_logs_teardown(
    product_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    init_file_logging(product_log)
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
        await app._request_agent_transcript_close()
    assert "Agent transcript teardown took" in product_log.read_text()


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
        composer = app._chat_input_container
        assert composer is not None and composer.input_widget is not None
        composer.input_widget.load_text("draft retained")
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert not app.query_one("#chat").display
        for chrome_id in (
            "loading-area",
            "agent-bar",
            "bottom-app-container",
            "bottom-bar",
        ):
            assert not app.query_one(f"#{chrome_id}").display
        assert viewer.region.height == app.size.height
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)
        assert app.query_one("#chat").display
        for chrome_id in ("loading-area", "bottom-app-container", "bottom-bar"):
            assert app.query_one(f"#{chrome_id}").display
        assert app.query_one("#agent-bar").display
        assert composer.input_widget.text == "draft retained"
        assert app.query_one(AgentBar).selected_agent_id == "one"
        assert app.screen.focused is app.query_one(AgentBar)


@pytest.mark.asyncio
async def test_transcript_inspection_exposes_unclipped_agent_metadata() -> None:
    app = build_test_chartreux_app()
    agent = _agent("one").model_copy(
        update={
            "initial_task_summary": "investigate a long task description",
            "current_task_summary": "finish the review",
            "effective_thinking": "high",
            "current_run_id": "run-unique-identifier",
        }
    )
    async with app.run_test(size=(40, 24)) as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([agent]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        metadata = app.query_one(".agent-transcript-metadata")
        assert "Initial task: investigate a long task description" in str(
            metadata.render()
        )
        assert "Current task: finish the review" in str(metadata.render())
        assert "Thinking: high" in str(metadata.render())
        assert "Run ID: run-unique-identifier" in str(metadata.render())
        assert metadata.region.width <= 40


@pytest.mark.asyncio
async def test_compacting_transcript_keeps_polling_across_running_transition() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        read = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        app.app_server.resources.sessions.read_agent_transcript = read
        running = _agent("one", "running")
        compacting = running.model_copy(update={"compacting": True})
        await app._handle_turn_event(AgentsUpdate([running]))
        await app._handle_turn_event(AgentsUpdate([compacting]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert viewer._live_timer is not None
        before = read.await_count
        await pilot.pause(1.2)
        assert read.await_count > before
        timer = viewer._live_timer
        await app._handle_turn_event(AgentsUpdate([running]))
        assert viewer._live_timer is timer
        await app._handle_turn_event(AgentsUpdate([compacting]))
        before = read.await_count
        await pilot.pause(1.2)
        assert read.await_count > before
        assert viewer._live_timer is timer
        await app._handle_turn_event(AgentsUpdate([running]))
        assert app.query_one(AgentTranscriptViewer) is viewer
        assert viewer._live_timer is timer


@pytest.mark.asyncio
@pytest.mark.parametrize("mouse", [False, True])
@pytest.mark.parametrize("stop", [False, True])
async def test_agent_browser_restores_non_composer_opener(
    mouse: bool, stop: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        await app._initial_history_loaded.wait()
        history = app.app_server._state.projection.state.history
        assert history is not None
        history.extend(_message(i) for i in range(45))
        await app._resume_history_from_messages()
        await app._mount_history_batch(
            history[15:25], app._messages_area, start_index=15, before=0
        )
        await pilot.pause()
        opener = Button("Conversation action")
        await app._messages_area.mount(opener)
        assert app._chat_input_container is not None
        assert app._chat_input_container.input_widget is not None
        composer = app._chat_input_container.input_widget
        composer.load_text("draft retained")
        composer.set_app_focus(False)
        opener.focus()
        await pilot.pause()
        assert app.screen.focused is opener
        unit = app._transcript.units["message-18"]
        app._chat_widget.scroll_to(
            y=app._chat_widget.scroll_y
            + unit.mounted_roots[0].region.y
            - app._chat_widget.region.y,
            animate=False,
            force=True,
            immediate=True,
        )
        await pilot.pause()
        anchor = app._transcript.capture_anchor(app._messages_area, following=False)
        assert anchor is not None and anchor.entry_id == "message-18"
        if mouse:
            await pilot.click("#agent-bar")
        else:
            await pilot.press("ctrl+shift+a")
        if stop:
            running = _agent("one", "running").model_copy(
                update={"current_run_id": "r1"}
            )
            await app._handle_turn_event(AgentsUpdate([running]))
            monkeypatch.setattr(
                app.app_server,
                "cancel_agent",
                AsyncMock(
                    return_value=AgentsCancelResponse(
                        outcome=CancelOutcome.STOP_REQUESTED, run_id="r1"
                    )
                ),
            )
            await pilot.press("down", "c", "escape")
            assert app.screen.focused is app._agent_bar
            await pilot.press("c", "left", "enter")
            assert app.screen.focused is app._agent_bar
            stopped = _agent("one").model_copy(
                update={
                    "latest_run_id": "r1",
                    "last_run_status": "cancelled",
                    "stop_reason": "user_cancelled",
                }
            )
            await app._handle_turn_event(AgentsUpdate([stopped]))
            await pilot.pause()
            assert app.screen.focused is app._agent_bar
            await pilot.press("d", "escape", "f1", "escape", "enter")
        else:
            await pilot.press("d", "escape", "f1", "escape", "down", "enter")
        await pilot.pause()
        assert app.query(AgentTranscriptViewer)
        await pilot.press("escape", "escape")
        await pilot.pause()
        assert app.screen.focused is opener
        assert composer.text == "draft retained"
        restored = app._transcript.capture_anchor(app._messages_area, following=False)
        assert restored is not None
        assert restored.entry_id == anchor.entry_id
        assert restored.row_offset == anchor.row_offset


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
async def test_same_agent_selection_keeps_existing_viewer() -> None:
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
        viewer = app.query_one(AgentTranscriptViewer)

        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one(AgentTranscriptViewer) is viewer


@pytest.mark.asyncio
async def test_agent_switch_does_not_restore_chat_between_viewers() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one"), _agent("two")]))
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        first = app.query_one(AgentTranscriptViewer)
        assert not app.query_one("#chat").display

        app.post_message(AgentBar.SelectionRequested("two"))
        await pilot.pause()

        assert app.query_one(AgentTranscriptViewer) is not first
        assert app.query_one(AgentTranscriptViewer).agent_id == "two"
        assert not app.query_one("#chat").display


@pytest.mark.asyncio
async def test_mount_failure_restores_chat_and_keeps_app_alive(
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

        async def failing_mount(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("screen is gone")

        monkeypatch.setattr(app, "mount", failing_mount)
        await pilot.press("ctrl+shift+a", "down", "enter")
        await pilot.pause()
        assert app._agent_transition_task is not None
        await app._agent_transition_task

        assert not app.query(AgentTranscriptViewer)
        assert app.query_one("#chat").display
        assert app._agent_transcript_viewer is None


@pytest.mark.asyncio
async def test_mount_failure_converges_to_newer_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one"), _agent("two")]))

        mount_entered = asyncio.Event()
        release_first_mount = asyncio.Event()
        real_mount = app.mount

        async def gated_mount(widget: Any, **kwargs: Any) -> None:
            if getattr(widget, "agent_id", None) == "one":
                mount_entered.set()
                await release_first_mount.wait()
                raise RuntimeError("screen is gone")
            await real_mount(widget, **kwargs)

        monkeypatch.setattr(app, "mount", gated_mount)
        app._submit_agent_selection("one")
        await asyncio.wait_for(mount_entered.wait(), timeout=2)

        app._submit_agent_selection("two")
        release_first_mount.set()
        task = app._agent_transition_task
        assert task is not None
        await asyncio.wait_for(task, timeout=2)
        await pilot.pause()

        assert app.query_one(AgentTranscriptViewer).agent_id == "two"
        assert not app.query_one("#chat").display


@pytest.mark.asyncio
async def test_superseded_selection_skips_mounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one"), _agent("two")]))
        teardown = app._close_agent_transcript_viewer
        teardown_count = 0

        async def counting_close(
            *, restore_focus: bool, show_chat: bool = True
        ) -> None:
            nonlocal teardown_count
            teardown_count += 1
            await teardown(restore_focus=restore_focus, show_chat=show_chat)

        monkeypatch.setattr(app, "_close_agent_transcript_viewer", counting_close)
        app._submit_agent_selection("one")
        app._submit_agent_selection("two")
        assert app._agent_transition_task is not None
        await app._agent_transition_task

        assert app.query_one(AgentTranscriptViewer).agent_id == "two"
        assert teardown_count == 1


@pytest.mark.asyncio
async def test_queued_selection_storm_converges_to_latest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one"), _agent("two")]))
        teardown = app._close_agent_transcript_viewer
        teardown_count = 0

        async def counting_close(
            *, restore_focus: bool, show_chat: bool = True
        ) -> None:
            nonlocal teardown_count
            teardown_count += 1
            await teardown(restore_focus=restore_focus, show_chat=show_chat)

        monkeypatch.setattr(app, "_close_agent_transcript_viewer", counting_close)
        app._submit_agent_selection("one")
        app._submit_agent_selection("two")
        app._submit_agent_selection(None)
        assert app._agent_transition_task is not None
        await app._agent_transition_task

        assert not app.query(AgentTranscriptViewer)
        assert app.query_one("#chat").display
        assert teardown_count <= 1


@pytest.mark.asyncio
async def test_close_request_drops_pending_selection() -> None:
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
        assert app.query_one(AgentTranscriptViewer).agent_id == "one"

        await app._request_agent_transcript_close()
        assert not app.query(AgentTranscriptViewer)
        assert app.query_one("#chat").display

        app.post_message(AgentBar.SelectionRequested("one"))
        await pilot.pause()
        assert app.query_one(AgentTranscriptViewer).agent_id == "one"


@pytest.mark.asyncio
async def test_turn_started_event_preserves_the_agent_transcript_viewer(
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
        viewer = app.query_one(AgentTranscriptViewer)
        focused = app.screen.focused
        started = asyncio.Event()

        async def events():
            yield TurnStarted(
                PublicTurn(
                    id="turn",
                    session_id=app.app_server.session_id,
                    status=PublicTurnStatus.IN_PROGRESS,
                    started_at=1,
                )
            )
            started.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(app.app_server, "events", events)
        listener = asyncio.create_task(app._listen_app_server_events())
        try:
            await started.wait()
            assert app.query_one(AgentTranscriptViewer) is viewer
            assert app.screen.focused is focused
        finally:
            listener.cancel()
            with suppress(asyncio.CancelledError):
                await listener


@pytest.mark.asyncio
async def test_rapid_selection_during_teardown_converges_after_disposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(
            AgentsUpdate([_agent("one"), _agent("two"), _agent("three")])
        )
        app._submit_agent_selection("one")
        assert app._agent_transition_task is not None
        await app._agent_transition_task
        viewer = app.query_one(AgentTranscriptViewer)
        entered = asyncio.Event()
        release = asyncio.Event()
        real_dispose = viewer.dispose

        async def gated_dispose() -> None:
            entered.set()
            await release.wait()
            await real_dispose()

        monkeypatch.setattr(viewer, "dispose", gated_dispose)
        app._submit_agent_selection("two")
        await asyncio.wait_for(entered.wait(), 2)
        app._submit_agent_selection("one")
        app._submit_agent_selection("three")
        release.set()
        assert app._agent_transition_task is not None
        await asyncio.wait_for(app._agent_transition_task, 3)
        await pilot.pause()
        assert not viewer.is_attached
        assert app.query_one(AgentTranscriptViewer).agent_id == "three"
        assert not app.query_one("#chat").display


@pytest.mark.asyncio
async def test_shutdown_during_teardown_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(AgentsUpdate([_agent("one")]))
        app._submit_agent_selection("one")
        assert app._agent_transition_task is not None
        await app._agent_transition_task
        viewer = app.query_one(AgentTranscriptViewer)
        entered = asyncio.Event()
        release = asyncio.Event()
        real_dispose = viewer.dispose

        async def gated_dispose() -> None:
            entered.set()
            await release.wait()
            await real_dispose()

        monkeypatch.setattr(viewer, "dispose", gated_dispose)
        app._submit_agent_selection(None)
        await asyncio.wait_for(entered.wait(), 2)
        app.exit()
        release.set()
        assert app._agent_transition_task is not None
        await asyncio.wait_for(app._agent_transition_task, 3)
        assert not viewer.is_attached


@pytest.mark.asyncio
@pytest.mark.parametrize("saved", [False, True])
async def test_stopped_agent_output_details_eviction_and_release(
    saved: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        read = AsyncMock(
            return_value=(
                _available(
                    _entry(
                        "partial",
                        "Retained partial output",
                        kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    )
                )
                if saved
                else AgentTranscriptGetResponse(
                    state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
                )
            )
        )
        app.app_server.resources.sessions.read_agent_transcript = read
        monkeypatch.setattr(
            app.app_server,
            "cancel_agent",
            AsyncMock(
                return_value=AgentsCancelResponse(
                    outcome=CancelOutcome.STOP_REQUESTED, run_id="r1"
                )
            ),
        )
        running = _agent("one", "running").model_copy(update={"current_run_id": "r1"})
        await app._handle_turn_event(AgentsUpdate([running]))
        await pilot.press("ctrl+shift+a", "down", "c", "left", "enter")
        stopped = _agent("one").model_copy(
            update={
                "latest_run_id": "r1",
                "last_run_status": "cancelled",
                "stop_reason": "user_cancelled",
            }
        )
        await app._handle_turn_event(AgentsUpdate([stopped]))
        bar = app._agent_bar
        assert bar is not None and not bar.stop_is_pending("one", "r1")
        await pilot.press("d")
        details = str(bar.query_one("#agent-bar-full-content").render())
        assert "user_cancelled" in details and "r1" in details
        await pilot.press("escape", "enter")
        await pilot.pause()
        viewer = app.query_one(AgentTranscriptViewer)
        assert viewer._live_timer is None
        assert "user_cancelled" in str(
            viewer.query_one(".agent-transcript-metadata").render()
        )
        if saved:
            from chartreux.cli.textual_ui.widgets.messages import AssistantMessage

            assert (
                viewer.query_one(AssistantMessage).get_content()
                == "Retained partial output"
            )
        else:
            assert viewer._status_widget is not None
            assert "No saved transcript" in str(viewer._status_widget.render())
        # Eviction leaves a browsable tombstone; result expiry is independent.
        evicted = stopped.model_copy(update={"availability": "evicted"})
        await app._handle_turn_event(AgentsUpdate([evicted]))
        await pilot.pause()
        assert app.query_one(AgentTranscriptViewer) is viewer
        assert not bar.agents[0].result_expired
        assert "Result expired: False" in str(
            viewer.query_one(".agent-transcript-metadata").render()
        )
        expired = evicted.model_copy(update={"result_expired": True})
        await app._handle_turn_event(AgentsUpdate([expired]))
        assert app.query_one(AgentTranscriptViewer) is viewer
        assert bar.agents[0].result_expired
        assert "Result expired: True" in str(
            viewer.query_one(".agent-transcript-metadata").render()
        )
        # Explicit release removes the summary and closes inspection.
        await app._handle_turn_event(AgentsUpdate([]))
        await pilot.pause()
        assert not app.query(AgentTranscriptViewer)
        assert app.query_one("#chat").display
