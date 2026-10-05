from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
import signal
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from textual.widgets import Button, Static

from chartreux.cli.textual_ui.app import (
    ChartreuxApp,
    _run_app_with_cleanup,
    run_textual_ui,
)
from chartreux.cli.textual_ui.quit_manager import (
    QUIT_CONFIRM_DELAY,
    ExitConsequencesScreen,
    QuitManager,
)
from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine
from tests.conftest import build_test_chartreux_app, build_test_vibe_config
from tests.stubs.app_config import build_test_app_config


@pytest.fixture
def app() -> ChartreuxApp:
    return build_test_chartreux_app()


@pytest.fixture(autouse=True)
def app_config_view(monkeypatch: pytest.MonkeyPatch) -> None:
    config = build_test_app_config()
    monkeypatch.setattr(ChartreuxApp, "config", property(lambda _app: config))


@pytest.fixture
def qm() -> QuitManager:
    mock_app = MagicMock()
    mock_app.query_one.return_value = MagicMock(spec=SessionStatusLine)
    mock_app.set_timer.return_value = MagicMock()
    return QuitManager(mock_app)


class _SessionReadyApp:
    """Warm-path base: app_server is set so the cold-path force-quit guard is skipped."""

    @pytest.fixture(autouse=True)
    def _set_app_server(self, app: ChartreuxApp) -> None:
        app._app_server = MagicMock(turn_active=False)


class TestQuitManager:
    def test_not_confirmed_initially(self, qm: QuitManager) -> None:
        assert qm.is_confirmed("Ctrl+C") is False
        assert qm.is_confirmed("Ctrl+D") is False

    def test_confirmed_within_delay(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+C")
        assert qm.is_confirmed("Ctrl+C") is True

    def test_wrong_key_not_confirmed(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+C")
        assert qm.is_confirmed("Ctrl+D") is False

    def test_expired_not_confirmed(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+C")
        qm._confirm_time = time.monotonic() - QUIT_CONFIRM_DELAY - 0.1
        assert qm.is_confirmed("Ctrl+C") is False

    def test_request_resets_timer_on_key_switch(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+C")
        qm._confirm_time = time.monotonic() - QUIT_CONFIRM_DELAY + 0.05
        qm.request_confirmation("Ctrl+D")
        assert qm.is_confirmed("Ctrl+D") is True

    def test_confirm_key_property(self, qm: QuitManager) -> None:
        assert qm.confirm_key is None
        qm.request_confirmation("Ctrl+D")
        assert qm.confirm_key == "Ctrl+D"

    def test_request_schedules_cancel_timer(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+D")
        mock_app = qm._app
        assert isinstance(mock_app, MagicMock)
        mock_app.set_timer.assert_called_once_with(
            QUIT_CONFIRM_DELAY, qm.cancel_confirmation
        )

    def test_request_stops_previous_timer(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+C")
        first_timer = qm._confirm_timer
        assert isinstance(first_timer, MagicMock)
        qm.request_confirmation("Ctrl+D")
        first_timer.stop.assert_called_once()

    def test_cancel_confirmation_resets_state(self, qm: QuitManager) -> None:
        qm.request_confirmation("Ctrl+C")
        qm.cancel_confirmation()
        assert qm.is_confirmed("Ctrl+C") is False
        assert qm.confirm_key is None
        assert qm._confirm_timer is None
        mock_app = qm._app
        assert isinstance(mock_app, MagicMock)
        mock_app.query_one.assert_called_with(SessionStatusLine)
        mock_app.query_one.return_value.clear_feedback.assert_called_once()

    def test_cancel_confirmation_noop_when_idle(self, qm: QuitManager) -> None:
        qm.cancel_confirmation()
        assert qm.confirm_key is None


@pytest.mark.asyncio
async def test_quit_confirmation_feedback_restores_status(app: ChartreuxApp) -> None:
    async with app.run_test(size=(120, 24)) as pilot:
        status = app.query_one(SessionStatusLine)
        app._quit_manager.request_confirmation("Ctrl+C")
        await pilot.pause()
        assert "again to quit" in status.render().plain
        assert not app._quit_manager.is_confirmed("Ctrl+D")
        app._quit_manager.cancel_confirmation()
        await pilot.pause()
        assert "again to quit" not in status.render().plain
        assert app._quit_manager.confirm_key is None
        assert app._quit_manager._confirm_timer is None


class TestActionInterruptOrQuit(_SessionReadyApp):
    @pytest.mark.asyncio
    async def test_clears_input_when_has_value(self, app: ChartreuxApp) -> None:
        app._app_server = None
        async with app.run_test():
            mock_container = MagicMock()
            mock_container.value = "some text"
            with patch.object(app, "_get_chat_input", return_value=mock_container):
                app.action_interrupt_or_quit()
            assert mock_container.value == ""

    def test_skips_empty_input(self, app: ChartreuxApp) -> None:
        mock_container = MagicMock()
        mock_container.value = ""
        with (
            patch.object(app, "_get_chat_input", return_value=mock_container),
            patch.object(app, "_try_interrupt_no_job_steps", return_value=False),
            patch.object(app, "_try_interrupt_running_job", return_value=False),
            patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
        ):
            app.action_interrupt_or_quit()
        assert mock_confirm.call_args.args[0] == "Ctrl+C"
        assert "Main work:" in mock_confirm.call_args.args[1]

    @pytest.mark.asyncio
    async def test_quits_on_confirmed(self, app: ChartreuxApp) -> None:
        app._app_server = None
        async with app.run_test():
            app._quit_manager.request_confirmation("Ctrl+C")
            with (
                patch.object(app, "_get_chat_input", return_value=None),
                patch.object(app, "_force_quit") as mock_quit,
            ):
                app.action_interrupt_or_quit()
            mock_quit.assert_called_once()

    @pytest.mark.asyncio
    async def test_interrupts_before_requesting_confirmation(
        self, app: ChartreuxApp
    ) -> None:
        app._app_server = None
        async with app.run_test():
            with (
                patch.object(app, "_get_chat_input", return_value=None),
                patch.object(
                    app, "_try_interrupt_no_job_steps", return_value=True
                ) as mock_interrupt,
                patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
            ):
                app.action_interrupt_or_quit()
            mock_interrupt.assert_called_once()
            mock_confirm.assert_not_called()

    def test_requests_confirmation_when_nothing_to_interrupt(
        self, app: ChartreuxApp
    ) -> None:
        with (
            patch.object(app, "_get_chat_input", return_value=None),
            patch.object(app, "_try_interrupt_no_job_steps", return_value=False),
            patch.object(app, "_try_interrupt_running_job", return_value=False),
            patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
        ):
            app.action_interrupt_or_quit()
        assert mock_confirm.call_args.args[0] == "Ctrl+C"
        assert "Main work:" in mock_confirm.call_args.args[1]


class TestActionDeleteRightOrQuit(_SessionReadyApp):
    def test_deletes_right_when_input_has_value(self, app: ChartreuxApp) -> None:
        mock_input = MagicMock()
        mock_container = MagicMock()
        mock_container.value = "some text"
        mock_container.input_widget = mock_input
        with patch.object(app, "_get_chat_input", return_value=mock_container):
            app.action_delete_right_or_quit()
        mock_input.action_delete_right.assert_called_once()

    def test_skips_empty_input(self, app: ChartreuxApp) -> None:
        mock_container = MagicMock()
        mock_container.value = ""
        with (
            patch.object(app, "_get_chat_input", return_value=mock_container),
            patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
        ):
            app.action_delete_right_or_quit()
        assert mock_confirm.call_args.args[0] == "Ctrl+D"
        assert "pending decisions:" in mock_confirm.call_args.args[1]

    @pytest.mark.asyncio
    async def test_quits_on_confirmed(self, app: ChartreuxApp) -> None:
        app._app_server = None
        async with app.run_test():
            app._quit_manager.request_confirmation("Ctrl+D")
            with (
                patch.object(app, "_get_chat_input", return_value=None),
                patch.object(app, "_force_quit") as mock_quit,
            ):
                app.action_delete_right_or_quit()
            mock_quit.assert_called_once()

    def test_requests_confirmation_when_no_input(self, app: ChartreuxApp) -> None:
        with (
            patch.object(app, "_get_chat_input", return_value=None),
            patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
        ):
            app.action_delete_right_or_quit()
        assert mock_confirm.call_args.args[0] == "Ctrl+D"
        assert "pending decisions:" in mock_confirm.call_args.args[1]

    def test_shows_queue_warning_when_queue_non_empty(self, app: ChartreuxApp) -> None:
        with (
            patch.object(app, "_get_chat_input", return_value=None),
            patch.object(type(app._queue), "has_server_work", True),
            patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
        ):
            app.action_delete_right_or_quit()
        assert mock_confirm.call_args.args[0] == "Ctrl+D"
        assert "queued input:" in mock_confirm.call_args.args[1]

    def test_quits_immediately_when_confirmation_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = build_test_chartreux_app(
            config=build_test_vibe_config(ask_confirmation_on_exit=False)
        )
        app._app_server = MagicMock(turn_active=False)
        config = build_test_app_config().model_copy(
            update={"ask_confirmation_on_exit": False}
        )
        monkeypatch.setattr(ChartreuxApp, "config", property(lambda _app: config))
        with (
            patch.object(app, "_get_chat_input", return_value=None),
            patch.object(app, "_force_quit") as mock_quit,
            patch.object(app._quit_manager, "request_confirmation") as mock_confirm,
        ):
            app.action_delete_right_or_quit()
        mock_quit.assert_called_once()
        mock_confirm.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_cleanup_cancels_in_flight_tasks(app: ChartreuxApp) -> None:
    async def _pending() -> None:
        await asyncio.Event().wait()

    agent_task = asyncio.create_task(_pending())
    bash_task = asyncio.create_task(_pending())
    app._agent_task = agent_task
    app._bash_task = bash_task

    await asyncio.wait_for(app.shutdown_cleanup(), timeout=1.0)

    assert agent_task.cancelled()
    assert bash_task.cancelled()


@pytest.mark.asyncio
async def test_begin_shutdown_stops_the_side_channel(app: ChartreuxApp) -> None:
    await app.prepare()
    with patch.object(
        app._side_channel, "shutdown", new_callable=AsyncMock
    ) as side_channel_shutdown:
        await app._begin_shutdown()

    side_channel_shutdown.assert_awaited_once()
    await app.shutdown_cleanup()


@pytest.mark.parametrize("stage", ["bootstrap", "app"])
@pytest.mark.parametrize("outcome", ["normal", "exception", "cancelled"])
def test_run_textual_ui_closes_owned_harness_before_loop_exit(
    app: ChartreuxApp, tmp_path: Path, stage: str, outcome: str
) -> None:
    error = (
        RuntimeError("boom")
        if outcome == "exception"
        else asyncio.CancelledError()
        if outcome == "cancelled"
        else None
    )
    bootstrap = AsyncMock(
        return_value=MagicMock(), side_effect=error if stage == "bootstrap" else None
    )
    events: list[str] = []

    async def session_cleanup() -> None:
        events.append("session")

    async def close_harness() -> None:
        assert asyncio.get_running_loop().is_running()
        await asyncio.sleep(0)
        events.append("harness")

    close = AsyncMock(side_effect=close_harness)
    with (
        patch("chartreux.cli.textual_ui.app.resolve_auto_theme"),
        patch("chartreux.cli.textual_ui.app.ChartreuxApp", return_value=app),
        patch.object(
            app,
            "run_async",
            new_callable=AsyncMock,
            side_effect=error if stage == "app" else None,
        ),
        patch.object(app, "shutdown_cleanup", side_effect=session_cleanup),
    ):
        if error is None:
            run_textual_ui(bootstrap, tmp_path / "history", close_app_server=close)
        else:
            with pytest.raises(type(error)):
                run_textual_ui(bootstrap, tmp_path / "history", close_app_server=close)
    close.assert_awaited_once()
    assert events == (
        ["harness"]
        if stage == "bootstrap" and error is not None
        else ["session", "harness"]
    )


def test_run_textual_ui_closes_owned_harness_when_startup_plan_is_cancelled(
    tmp_path: Path,
) -> None:
    from chartreux.app_server.host import AppServerHost

    host = MagicMock(spec=AppServerHost)
    close = AsyncMock()
    with (
        patch("chartreux.cli.textual_ui.app.resolve_auto_theme"),
        patch(
            "chartreux.cli.textual_ui.startup.resolve_session_open_plan",
            new_callable=AsyncMock,
            return_value=None,
        ),
    ):
        assert (
            run_textual_ui(
                AsyncMock(return_value=host),
                tmp_path / "history",
                close_app_server=close,
            )
            is None
        )
    close.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_app_with_cleanup_runs_cleanup_when_run_async_raises(
    app: ChartreuxApp,
) -> None:
    with (
        patch.object(
            app, "run_async", new_callable=AsyncMock, side_effect=RuntimeError("boom")
        ),
        patch.object(app, "shutdown_cleanup", new_callable=AsyncMock) as cleanup,
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await _run_app_with_cleanup(app)

    cleanup.assert_awaited_once()


def test_force_quit_delegates_to_private(app: ChartreuxApp) -> None:
    with patch.object(app, "_force_quit") as private:
        app._force_quit()

    private.assert_called_once_with()


@pytest.mark.asyncio
async def test_run_app_with_cleanup_sigterm_triggers_force_quit(
    app: ChartreuxApp,
) -> None:
    captured: list[Callable[[], None]] = []
    loop = asyncio.get_running_loop()

    def capture_add(sig: int, callback: Callable[[], None], *args: object) -> None:
        captured.append(callback)

    async def fake_run_async() -> None:
        captured[0]()

    with (
        patch.object(loop, "add_signal_handler", side_effect=capture_add),
        patch.object(loop, "remove_signal_handler") as remove_handler,
        patch.object(app, "run_async", side_effect=fake_run_async),
        patch.object(app, "shutdown_cleanup", new_callable=AsyncMock),
        patch.object(app, "_force_quit") as force_quit,
    ):
        await _run_app_with_cleanup(app)

    force_quit.assert_called_once_with()
    remove_handler.assert_any_call(signal.SIGTERM)


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["/exit", "Ctrl+C", "Ctrl+D"])
async def test_intentional_exits_require_consequence_confirmation(
    app: ChartreuxApp, route: str
) -> None:
    app._app_server = MagicMock(turn_active=True)
    app._active_callback = MagicMock()
    with (
        patch.object(app, "_force_quit") as force,
        patch.object(app._quit_manager, "request_confirmation") as confirm,
        patch.object(app, "_get_chat_input", return_value=None),
        patch.object(app, "_try_interrupt_no_job_steps", return_value=False),
        patch.object(app, "_try_interrupt_running_job", return_value=False),
    ):
        if route == "/exit":
            await app._exit_app()
        elif route == "Ctrl+C":
            app.action_interrupt_or_quit()
        else:
            app.action_delete_right_or_quit()
    force.assert_not_called()
    assert confirm.call_args.args[0] == route
    consequences = confirm.call_args.args[1]
    assert "Main work:" in consequences
    assert "background agents:" in consequences
    assert "pending decisions:" in consequences
    assert "queued input:" in consequences


@pytest.mark.asyncio
async def test_explicit_exit_skips_idle_confirmation_with_session_attached(
    app: ChartreuxApp,
) -> None:
    app._app_server = MagicMock(turn_active=False)
    with (
        patch.object(app, "_force_quit") as force,
        patch.object(app._quit_manager, "request_confirmation") as confirm,
    ):
        await app._exit_app()

    force.assert_called_once()
    confirm.assert_not_called()


def test_disabled_confirmation_still_checks_agents_and_queued_input(
    app: ChartreuxApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_test_app_config().model_copy(
        update={"ask_confirmation_on_exit": False}
    )
    monkeypatch.setattr(ChartreuxApp, "config", property(lambda _app: config))
    app._app_server = MagicMock(turn_active=False)
    app._agent_summaries = [
        MagicMock(
            current_run_status="running", last_run_status=None, availability="running"
        )
    ]
    with (
        patch.object(app, "_get_chat_input", return_value=None),
        patch.object(type(app._queue), "has_server_work", True),
        patch.object(app._quit_manager, "request_confirmation") as confirm,
        patch.object(app, "_force_quit") as force,
    ):
        app.action_delete_right_or_quit()
    force.assert_not_called()
    assert (
        "background agents: shutdown requests their stop" in confirm.call_args.args[1]
    )
    assert "queued input: not run after exit" in confirm.call_args.args[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.parametrize("cancel", ["escape", "enter", "click"])
async def test_consequential_exit_cancel_preserves_opener_and_work(
    size: tuple[int, int], cancel: str
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=size) as pilot:
        opener_screen = app.screen
        opener = app.screen.focused
        assert opener is not None
        app._pending_turn = True
        with (
            patch.object(app, "_force_quit") as force,
            patch.object(app, "_begin_shutdown", new_callable=AsyncMock) as shutdown,
            patch.object(app, "_try_interrupt") as interrupt,
            patch.object(app._queue, "pop_last", new_callable=AsyncMock) as pop,
            patch.object(
                app._queue, "clear_server_queue", new_callable=AsyncMock
            ) as clear,
            patch.object(type(app._queue), "has_removable", True),
        ):
            await app._exit_app()
            await pilot.pause()
            dialog = app.screen
            assert isinstance(dialog, ExitConsequencesScreen)
            assert dialog.focused is dialog.query_one("#exit-cancel", Button)
            assert "Main work:" in str(
                dialog.query_one("#exit-consequences Static", Static).content
            )
            if cancel == "click":
                assert await pilot.click("#exit-cancel")
            else:
                await pilot.press(cancel)
            await pilot.pause()
            assert app.screen is opener_screen
            assert app.screen.focused is opener
            assert app._pending_turn
            force.assert_not_called()
            shutdown.assert_not_called()
            interrupt.assert_not_called()
            pop.assert_not_awaited()
            clear.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (120, 36)])
@pytest.mark.parametrize("activate", ["keyboard", "click"])
async def test_consequential_exit_requires_explicit_exit_activation(
    size: tuple[int, int], activate: str
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=size) as pilot:
        app._pending_turn = True
        with patch.object(app, "_force_quit") as force:
            await app._exit_app()
            await pilot.pause()
            if activate == "click":
                assert await pilot.click("#exit-confirm")
            else:
                await pilot.press("tab")
                assert app.screen.focused is app.screen.query_one(
                    "#exit-confirm", Button
                )
                force.assert_not_called()
                await pilot.press("enter")
            await pilot.pause()
            force.assert_called_once_with()


@pytest.mark.asyncio
async def test_quit_key_ladders_are_noops_while_exit_modal_is_up() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        app._pending_turn = True
        await app._exit_app()
        await pilot.pause()
        dialog = app.screen
        assert isinstance(dialog, ExitConsequencesScreen)
        with (
            patch.object(app, "_get_chat_input") as get_input,
            patch.object(app, "_try_interrupt_no_job_steps") as no_job,
            patch.object(app, "_try_interrupt_running_job") as job,
            patch.object(app._queue, "pop_last", new_callable=AsyncMock) as pop,
            patch.object(app, "_request_intentional_exit") as request,
            patch.object(app, "_force_quit") as force,
        ):
            await pilot.press("ctrl+c", "ctrl+d", "ctrl+c", "ctrl+d")
            # Also check the ladder entry points directly, independent of bindings.
            app.action_interrupt_or_quit()
            app.action_delete_right_or_quit()
            await pilot.pause()
            assert app.screen is dialog
            assert dialog.focused is dialog.query_one("#exit-cancel", Button)
            assert app._pending_turn
            get_input.assert_not_called()
            no_job.assert_not_called()
            job.assert_not_called()
            pop.assert_not_awaited()
            request.assert_not_called()
            force.assert_not_called()
        await pilot.press("escape")
