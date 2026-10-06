from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.events import HistoryEntryAdded, SessionUpdated, StatsUpdated
from chartreux.app_server.models import (
    AgentStatsSnapshot,
    PublicCheckpointEntry,
    PublicEntryGenerationStatus,
)
from chartreux.app_server.protocol import (
    StatsUpdatedParams,
    WorkspaceBranchReadResponse,
)
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine
from chartreux.core.background_jobs import BashStartArgs, BashStopArgs
from tests.cli.textual_ui.test_history_grouping import _message
from tests.conftest import build_test_agent_loop, build_test_chartreux_app


@pytest.mark.asyncio
async def test_live_background_jobs_root_children_reset_and_resize() -> None:
    root = build_test_agent_loop()
    assert root.background_jobs is not None
    child = build_test_agent_loop(
        is_subagent=True,
        inherited_workspace=root.tool_manager.workspace,
        background_jobs=root.background_jobs.borrow(),
    )
    root.config.status_line.segments = [
        "directory",
        "pid",
        "context",
        "background-jobs",
    ]
    app = build_test_chartreux_app(agent_loop=root)
    try:
        async with app.run_test(size=(100, 32)) as pilot:
            await app._session_ready.wait()
            status = app.query_one(SessionStatusLine)

            async def observe(count: int) -> None:
                async with asyncio.timeout(5):
                    while (
                        status.state.active_background_job_count != count
                        or "background-jobs" not in status.config.segments
                    ):
                        await pilot.pause(0.01)
                assert app.app_server.state.active_background_job_count == count
                assert f"Jobs {count}" in status.render().plain

            await observe(0)
            first = await root.background_jobs.start(BashStartArgs(command="sleep 60"))
            await observe(1)
            assert child.background_jobs is not None
            second = await child.background_jobs.start(
                BashStartArgs(command="sleep 60")
            )
            await child.aclose()
            await observe(2)
            await pilot.resize_terminal(20, 24)
            await pilot.pause()
            assert "Jobs" not in status.render().plain
            assert "pid" not in status.render().plain
            await pilot.resize_terminal(100, 32)
            await pilot.pause()
            await observe(2)
            await root.background_jobs.stop(BashStopArgs(job_id=first.job.job_id))
            await observe(1)
            await root.background_jobs.stop(BashStopArgs(job_id=second.job.job_id))
            await observe(0)
            await root.background_jobs.start(BashStartArgs(command="sleep 60"))
            await observe(1)
            await app.app_server.clear_history()
            await observe(0)
    finally:
        await child.aclose()
        await root.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [False, True])
async def test_quit_feedback_survives_stats_and_settings_refresh(
    timeout: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if timeout:
        monkeypatch.setattr(
            "chartreux.cli.textual_ui.quit_manager.QUIT_CONFIRM_DELAY", 0.4
        )
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        status = app.query_one(SessionStatusLine)
        app._quit_manager.request_confirmation("Ctrl+D")
        prompt = status.render().plain
        assert "again to quit" in prompt
        await app._handle_turn_event(
            StatsUpdated(
                StatsUpdatedParams(
                    event_id=1,
                    emitted_at=0,
                    session_id=app.app_server.session_id,
                    stats=AgentStatsSnapshot(context_tokens=135_000),
                    context_window=400_000,
                )
            )
        )
        app.app_server._state.stats = AgentStatsSnapshot(context_tokens=135_000)
        app.app_server._state.context_window = 400_000
        app.config.status_line.separator = "pipe"
        app._on_config_changed(app.config)
        assert status.config.separator == "pipe"
        assert status.render().plain == prompt
        if timeout:
            await pilot.pause(0.5)
        else:
            app._quit_manager.cancel_confirmation()
        assert app._quit_manager.confirm_key is None
        assert "135k/400k (34%)" in status.render().plain
        assert "again to quit" not in status.render().plain
        assert status.size.height == 1


@pytest.mark.asyncio
async def test_branch_refresh_is_nonblocking_and_discards_superseded_results() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        session = app.app_server
        status = app.query_one(SessionStatusLine)
        old_result = asyncio.Event()
        entered = asyncio.Event()

        async def delayed(*, refresh: bool = False) -> WorkspaceBranchReadResponse:
            assert refresh
            entered.set()
            await old_result.wait()
            return WorkspaceBranchReadResponse(
                session_id=session.session_id,
                cwd=session.cwd,
                branch="obsolete",
                status="branch",
            )

        session.resources.workspace.read_branch = AsyncMock(side_effect=delayed)
        app._schedule_branch_refresh()
        await entered.wait()
        # The pending resource read does not block rendering or composing.
        await pilot.press("x")
        assert app._chat_input_container is not None
        assert app._chat_input_container.input_widget is not None
        assert app._chat_input_container.input_widget.text == "x"
        session.resources.workspace.read_branch = AsyncMock(
            return_value=WorkspaceBranchReadResponse(
                session_id=session.session_id,
                cwd=session.cwd,
                branch="current",
                status="branch",
            )
        )
        app._schedule_branch_refresh()
        await pilot.pause()
        assert status.state.branch == "current"
        old_result.set()
        await pilot.pause()
        assert status.state.branch == "current"
        session.resources.workspace.read_branch = AsyncMock(return_value=None)
        app._schedule_branch_refresh()
        await pilot.pause()
        assert status.state.branch == "current"


@pytest.mark.asyncio
async def test_public_root_context_updates_main_details_and_compacting_status() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        status = app.query_one(SessionStatusLine)
        assert app._agent_bar is not None
        checkpoint = PublicCheckpointEntry(
            id="compaction",
            session_id=app.app_server.session_id,
            created_at=0,
            updated_at=0,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
            kind="compaction",
        )
        app._refresh_status_for_event(HistoryEntryAdded(checkpoint))
        await app._handle_turn_event(
            StatsUpdated(
                StatsUpdatedParams(
                    event_id=1,
                    emitted_at=0,
                    session_id=app.app_server.session_id,
                    stats=AgentStatsSnapshot(context_tokens=135_000),
                    context_window=400_000,
                )
            )
        )
        assert status.state.compacting
        assert "Compacting" in app._agent_bar.full_metadata()
        assert "135k" not in status.render().plain
        # Checkpoint completion precedes the stats notification: retain the old
        # over-threshold measurement to detect a stale-numerator flash.
        app.app_server._state.stats = AgentStatsSnapshot(context_tokens=500_000)
        app.app_server._state.context_window = 400_000
        checkpoint.generation_status = PublicEntryGenerationStatus.COMPLETED
        app._refresh_status_for_event(HistoryEntryAdded(checkpoint))
        assert not status.state.compacting
        assert status.state.context_tokens is None
        assert "500k" not in status.render().plain
        assert "500k" not in app._agent_bar.full_metadata()
        assert "—" in status.render().plain
        app._refresh_context_progress()
        assert status.state.context_tokens is None
        await app._handle_turn_event(
            StatsUpdated(
                StatsUpdatedParams(
                    event_id=2,
                    emitted_at=0,
                    session_id=app.app_server.session_id,
                    stats=AgentStatsSnapshot(context_tokens=0),
                    context_window=400_000,
                )
            )
        )
        assert "0/400k (0%)" in status.render().plain
        assert "0/400k (0%)" in app._agent_bar.full_metadata()


@pytest.mark.asyncio
async def test_workspace_change_refreshes_status_and_branch() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        session = app.app_server
        previous = session.state.session.model_copy()
        session.state.session.cwd = "/relocated/workspace"
        read = AsyncMock(
            return_value=WorkspaceBranchReadResponse(
                session_id=session.session_id,
                cwd=session.cwd,
                branch="relocated",
                status="branch",
            )
        )
        session.resources.workspace.read_branch = read
        await app._handle_turn_event(
            SessionUpdated(previous=previous, session=session.state.session, patch=[])
        )
        await pilot.pause()
        status = app.query_one(SessionStatusLine)
        assert status.state.cwd == "/relocated/workspace"
        assert status.state.branch == "relocated"
        read.assert_awaited_once_with(refresh=True)


@pytest.mark.asyncio
async def test_timestamp_preference_applies_to_history_and_live_widgets() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app.app_server.resources.config.update({"show_message_timestamps": False})
        posted_at = datetime(2025, 1, 2, 12, 34, tzinfo=UTC)
        history = app.app_server._state.projection.state.history
        assert history is not None
        history.append(
            _message(1).model_copy(update={"role": "user", "posted_at": posted_at})
        )
        await app._resume_history_from_messages()
        await pilot.pause()
        user = app.query_one(UserMessage)
        assert user.posted_at == posted_at
        assert not user.header.show_message_timestamps
        assistant = AssistantMessage("live response", posted_at=posted_at)
        await app._mount_and_scroll(assistant)
        await app.app_server.resources.config.update({"show_message_timestamps": True})
        await pilot.pause()
        assert user.header.show_message_timestamps
        assert assistant.header.show_message_timestamps
        assert user.header.posted_at == posted_at
        await app.app_server.resources.config.update({"show_message_timestamps": False})
        await pilot.pause()
        assert not user.header.show_message_timestamps
        assert not assistant.header.show_message_timestamps
        # Evicted/rematerialized history units use the latest preference too.
        roots = app._transcript.build_unit("message-1", app._history_widget_indices)
        restored = next(widget for widget in roots if isinstance(widget, UserMessage))
        assert not restored.header.show_message_timestamps
        assert restored.posted_at == posted_at
