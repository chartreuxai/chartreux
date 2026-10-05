from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import UsageTotals, UsageWindowSummaries
from chartreux.app_server.protocol import Notification, UsageUpdatedParams
from chartreux.cli.textual_ui.screens.usage import TOTAL_KEY, UsageModels, UsageScreen
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine
from tests.cli.textual_ui.test_app_server_requests import (
    _question_callback,
    _wait_until,
)
from tests.cli.textual_ui.test_message_queue_ui import _blocked_app, _wait_for_queued
from tests.cli.textual_ui.test_session_status_line import _usage_summary
from tests.cli.textual_ui.test_usage_screen import snapshot, text
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _snapshot(revision: int = 100, cost: float = 12.34) -> UsageUpdatedParams:
    summary = _usage_summary(UsageTotals(known_cost_usd=cost, has_known_cost=True))
    return UsageUpdatedParams(
        as_of=datetime(2024, 1, 1, tzinfo=UTC),
        revision=revision,
        summaries=UsageWindowSummaries(day=summary, week=summary, month=summary),
    )


@pytest.mark.asyncio
async def test_initial_usage_read_does_not_delay_mount_or_composer() -> None:
    app = build_test_chartreux_app()
    await app.prepare()
    app.config.status_line.segments = ["directory", "context", "spend-today"]
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed() -> None:
        entered.set()
        await release.wait()
        app.app_server.resources.usage._publish(_snapshot())

    app.app_server.resources.usage.read = AsyncMock(side_effect=delayed)
    async with app.run_test(size=(140, 24)) as pilot:
        await app._session_ready.wait()
        await entered.wait()
        status = app.query_one(SessionStatusLine)
        assert status.state.usage_day is None
        assert "Today —" in status.render().plain
        await pilot.press("x")
        assert app._chat_input_container is not None
        assert app._chat_input_container.input_widget is not None
        assert app._chat_input_container.input_widget.text == "x"
        release.set()
        await pilot.pause()
        assert "Today $12.34" in status.render().plain
    assert not app._usage_tasks
    assert not app.app_server.resources.usage._subscribers


@pytest.mark.asyncio
async def test_usage_notification_preserves_feedback_and_session_state(
    tmp_working_directory: Path,
) -> None:
    app = build_test_chartreux_app()
    await app.prepare()
    app.config.status_line.segments = ["directory", "context", "spend-today"]
    async with app.run_test(size=(140, 24)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        status = app.query_one(SessionStatusLine)
        await pilot.press(*"draft")
        callbacks = tuple(app._pending_callbacks)
        active_callback = app._active_callback
        queue = app.app_server.turn_queue.model_copy(deep=True)
        app._quit_manager.request_confirmation("Ctrl+D")
        feedback = status.render().plain
        snapshot = _snapshot()
        await app.app_server.resources.usage.consume_notification(
            Notification(
                method="usage/updated", params=snapshot.model_dump(mode="json")
            )
        )
        assert status.state.usage_day == snapshot.summaries.day
        assert status.render().plain == feedback
        app._quit_manager.cancel_confirmation()
        rendered = status.render().plain
        assert "Today $12.34" in rendered
        assert status.state.cwd == str(tmp_working_directory)
        assert tmp_working_directory.name in rendered
        assert "0/400k (0%)" in rendered
        assert tuple(app._pending_callbacks) == callbacks
        assert app._active_callback is active_callback
        assert app.app_server.turn_queue == queue
        assert app._chat_input_container is not None
        assert app._chat_input_container.input_widget is not None
        assert app._chat_input_container.input_widget.text == "draft"


@pytest.mark.asyncio
async def test_rebind_preserves_global_spend_and_replaces_usage_tasks() -> None:
    app = build_test_chartreux_app()
    await app.prepare()
    app.config.status_line.segments = ["directory", "context", "spend-today"]
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        resource = app.app_server.resources.usage
        resource._publish(_snapshot())
        old_poll = app._usage_poll_task
        old_callback = resource._subscribers[0]
        resource.read = AsyncMock(side_effect=RuntimeError("unavailable"))
        app.app_server.state.session.id = "replacement"
        app._refresh_context_progress()
        await pilot.pause()
        assert old_poll is not None and old_poll.cancelled()
        assert app._usage_poll_task is not old_poll
        assert len(resource._subscribers) == 1
        old_callback(_snapshot(101, 999))
        status = app.query_one(SessionStatusLine)
        assert status.state.usage_day == _snapshot().summaries.day
        app._refresh_context_progress()
        assert status.state.usage_day == _snapshot().summaries.day
    assert not resource._subscribers
    assert app._usage_poll_task is None
    assert not app._usage_tasks


@pytest.mark.asyncio
async def test_spend_config_controls_reconciliation_poll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "chartreux.cli.textual_ui.app._USAGE_POLL_INTERVAL_SECONDS", 0.02
    )
    app = build_test_chartreux_app()
    await app.prepare()
    app.config.status_line.segments = ["directory", "context"]
    read = AsyncMock(side_effect=RuntimeError("unavailable"))
    app.app_server.resources.usage.read = read
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        assert app._usage_poll_task is None
        read.assert_awaited_once_with()
        assert app.query_one(SessionStatusLine).state.usage_day is None
        for segment in ("spend-today", "spend-week", "spend-month"):
            app.config.status_line.segments = ["directory", "context", segment]
            app._on_config_changed(app.config)
            poll = app._usage_poll_task
            assert poll is not None
            before = read.await_count
            await pilot.pause(0.06)
            assert read.await_count > before
            app.config.status_line.segments = ["directory", "context"]
            app._on_config_changed(app.config)
            await pilot.pause()
            assert poll.cancelled()
            assert app._usage_poll_task is None
            stopped_count = read.await_count
            await pilot.pause(0.06)
            assert read.await_count == stopped_count
        read_provider, subscribe = app._usage_screen_providers()
        assert read_provider == app.app_server.resources.usage.read
        assert subscribe == app.app_server.resources.usage.subscribe
    assert not app._usage_tasks


@pytest.mark.asyncio
async def test_session_replacement_keeps_spend_until_new_snapshot() -> None:
    app = build_test_chartreux_app()
    replacement = build_test_chartreux_app()
    await app.prepare()
    await replacement.prepare()
    original = app.app_server
    new_session = replacement.app_server
    app.config.status_line.segments = ["directory", "context", "spend-today"]
    new_session.resources.usage.read = AsyncMock(
        side_effect=RuntimeError("unavailable")
    )
    try:
        async with app.run_test() as pilot:
            await app._session_ready.wait()
            await pilot.pause()
            original.resources.usage._publish(_snapshot())
            old_poll = app._usage_poll_task
            app._app_server = new_session
            app._refresh_context_progress()
            await pilot.pause()
            assert not original.resources.usage._subscribers
            assert old_poll is not None and old_poll.cancelled()
            status = app.query_one(SessionStatusLine)
            assert status.state.usage_day == _snapshot().summaries.day
            assert len(new_session.resources.usage._subscribers) == 1
            new_session.resources.usage._publish(_snapshot(101, 20))
            assert status.state.usage_day == _snapshot(101, 20).summaries.day
        assert not new_session.resources.usage._subscribers
        assert not app._usage_tasks
    finally:
        await original.close()
        await new_session.close()


@pytest.mark.asyncio
async def test_shutdown_cancels_pending_initial_usage_read() -> None:
    app = build_test_chartreux_app()
    await app.prepare()
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked() -> None:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    app.app_server.resources.usage.read = AsyncMock(side_effect=blocked)
    async with app.run_test():
        await app._session_ready.wait()
        await entered.wait()
    assert cancelled.is_set()
    assert not app._usage_tasks
    assert not app.app_server.resources.usage._subscribers


@pytest.mark.asyncio
async def test_usage_slash_open_preserves_idle_composer_and_focus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        resource = app.app_server.resources.usage
        resource.read = AsyncMock(return_value=snapshot())
        send = AsyncMock()
        interrupt = AsyncMock()
        monkeypatch.setattr(app, "_handle_user_message", send)
        monkeypatch.setattr(app.app_server, "interrupt", interrupt)
        composer = app.query_one(ChatInputContainer)
        composer.value = "unfinished draft"
        assert composer.input_widget is not None
        composer.input_widget.focus()
        queue = app.app_server.turn_queue.model_copy(deep=True)
        await app._dispatch_submitted_value("/usage")
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        screen = app.screen
        assert isinstance(screen, UsageScreen)
        await _wait_until(pilot, lambda: screen.snapshot is not None)
        resource.read.assert_awaited_once_with("day", None)
        assert screen.read_provider == resource.read
        assert len(resource._subscribers) == 2
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._usage_worker is None)
        assert composer.value == "unfinished draft"
        assert app.focused is composer.input_widget
        assert app.app_server.turn_queue == queue
        send.assert_not_awaited()
        interrupt.assert_not_awaited()
        assert len(resource._subscribers) == 1


@pytest.mark.asyncio
async def test_usage_open_during_turn_preserves_queue_and_approval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app, backend = _blocked_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        composer = app.query_one(ChatInputContainer)
        composer.post_message(ChatInputContainer.Submitted("start turn"))
        await _wait_until(pilot, backend.started.is_set)
        composer.post_message(ChatInputContainer.Submitted("queued prompt"))
        await _wait_for_queued(pilot, app, ["queued prompt"])
        callback = _question_callback("usage-approval")
        await app._handle_turn_event(CallbackRequested(callback))
        await _wait_until(pilot, lambda: app._active_callback is callback)
        question = app.query_one(QuestionApp)
        app.app_server.resources.usage.read = AsyncMock(return_value=snapshot())
        interrupt = AsyncMock()
        monkeypatch.setattr(app.app_server, "interrupt", interrupt)
        queue = app.app_server.turn_queue.model_copy(deep=True)
        await app._dispatch_submitted_value("/usage")
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert app._active_callback is callback
        assert app.app_server.turn_active
        assert backend.calls == 1
        assert app.app_server.turn_queue == queue
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._usage_worker is None)
        assert app.query_one(QuestionApp) is question
        assert app._active_callback is callback
        assert app.focused is not None and app.focused.screen is app.screen
        assert app.app_server.turn_active
        assert app.app_server.turn_queue == queue
        assert backend.calls == 1
        interrupt.assert_not_awaited()
        backend.release.set()


@pytest.mark.asyncio
async def test_usage_slash_open_while_streaming_does_not_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class StreamingBackend(FakeBackend):
        async def complete_streaming(self, **kwargs):
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk
                started.set()
                await release.wait()

    backend = StreamingBackend([mock_llm_chunk(content="partial response")])
    app = build_test_chartreux_app(
        agent_loop=build_test_agent_loop(backend=backend, enable_streaming=True)
    )
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        composer = app.query_one(ChatInputContainer)
        composer.post_message(ChatInputContainer.Submitted("start streaming"))
        await _wait_until(pilot, started.is_set)
        composer.post_message(ChatInputContainer.Submitted("queued prompt"))
        await _wait_for_queued(pilot, app, ["queued prompt"])
        composer.value = "draft during streaming"
        app.app_server.resources.usage.read = AsyncMock(return_value=snapshot())
        interrupt = AsyncMock()
        monkeypatch.setattr(app.app_server, "interrupt", interrupt)
        queue = app.app_server.turn_queue.model_copy(deep=True)
        composer.post_message(ChatInputContainer.Submitted("/usage"))
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert app.app_server.turn_active
        assert len(backend.requests_messages) == 1
        assert app.app_server.turn_queue == queue
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._usage_worker is None)
        assert app.app_server.turn_active
        assert app.app_server.turn_queue == queue
        assert composer.value == "draft during streaming"
        assert app.focused is composer.input_widget
        assert len(backend.requests_messages) == 1
        interrupt.assert_not_awaited()
        release.set()


@pytest.mark.asyncio
async def test_usage_duplicate_open_and_initial_read_error() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        read = AsyncMock(side_effect=RuntimeError("private failure"))
        app.app_server.resources.usage.read = read
        await app._show_usage()
        worker = app._usage_worker
        await app._show_usage()
        assert app._usage_worker is worker
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        screen = app.screen
        assert isinstance(screen, UsageScreen)
        await _wait_until(pilot, lambda: "Failed:" in text(screen, "#usage-status"))
        assert screen.snapshot is None
        assert "private failure" not in text(screen, "#usage-status")
        await app._show_usage()
        assert sum(isinstance(s, UsageScreen) for s in app.screen_stack) == 1
        read.assert_awaited_once_with("day", None)
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._usage_worker is None)
        await app._show_usage()
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        assert app.screen is not screen
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._usage_worker is None)


@pytest.mark.asyncio
async def test_usage_notification_live_updates_table_and_chrome() -> None:
    app = build_test_chartreux_app()
    await app.prepare()
    app.config.status_line.segments = ["directory", "context", "spend-today"]
    async with app.run_test(size=(140, 32)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        status = app.query_one(SessionStatusLine)
        resource = app.app_server.resources.usage
        resource.read = AsyncMock(return_value=snapshot())
        await app._show_usage()
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        screen = app.screen
        assert isinstance(screen, UsageScreen)
        await _wait_until(pilot, lambda: screen.snapshot is not None)
        assert screen.snapshot is not None
        initial = screen.snapshot.model_copy(deep=True)
        models = screen.query_one(UsageModels)
        assert models.option_count == 6
        assert "$12.34" in str(models.get_option(TOTAL_KEY).prompt)
        updated = snapshot(revision=101, count=6)
        updated.summaries.day.known_cost_usd = 99
        resource.read.return_value = updated
        notification = UsageUpdatedParams(
            as_of=updated.as_of, revision=updated.revision, summaries=updated.summaries
        )
        await resource.consume_notification(
            Notification(
                method="usage/updated", params=notification.model_dump(mode="json")
            )
        )
        await _wait_until(pilot, lambda: screen.snapshot == updated)
        assert screen.snapshot != initial
        assert models.option_count == 7
        assert "model-5" in str(models.get_option_at_index(5).prompt)
        assert "$99.00" in str(models.get_option(TOTAL_KEY).prompt)
        assert "New usage available" not in text(screen, "#usage-status")
        assert status.state.usage_day == updated.summaries.day
        assert "Today $99.00" in status.render().plain
        assert resource.read.await_count == 2
        resource.read.assert_awaited_with("day", None)
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._usage_worker is None)


@pytest.mark.asyncio
async def test_usage_close_presents_callback_received_while_open() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        app.app_server.resources.usage.read = AsyncMock(return_value=snapshot())
        await app._show_usage()
        await _wait_until(pilot, lambda: isinstance(app.screen, UsageScreen))
        callback = _question_callback("usage-incoming")
        await app._handle_turn_event(CallbackRequested(callback))
        assert app._active_callback is None
        assert tuple(app._pending_callbacks) == (callback,)
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._active_callback is callback)
        assert app.query(QuestionApp)
        assert not app._pending_callbacks
