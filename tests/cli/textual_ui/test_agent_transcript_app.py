from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import suppress
import logging
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.events import AgentsUpdate, TurnStarted
from chartreux.app_server.models import PublicTurn, PublicTurnStatus
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.observability.logging import (
    get_effective_log_level,
    init_file_logging,
    logger,
    set_log_level,
)
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
    assert sum("Agent transcript reset remove_children:" in line for line in lines) == 1
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
