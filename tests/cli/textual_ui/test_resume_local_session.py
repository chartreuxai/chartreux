from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from chartreux.app_server.models import AgentStatsSnapshot, PreparedPrompt
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.app_server.session import AppServerSession
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input import ChatInputContainer
from chartreux.cli.textual_ui.widgets.context_progress import ContextProgress
from chartreux.cli.textual_ui.widgets.loading import LoadingWidget
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from chartreux.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
from tests.cli.textual_ui.test_history_grouping import _message
from tests.conftest import build_test_chartreux_app

_RESUMED_TOKENS = 50_000
_RESUMED_CONTEXT_WINDOW = 200_000


@pytest.fixture
def chartreux_app() -> ChartreuxApp:
    return build_test_chartreux_app()


def _app_with_fake_runtime(runtime: MagicMock) -> ChartreuxApp:
    app = build_test_chartreux_app()
    app_server = object.__new__(AppServerSession)
    app_server.resources = MagicMock()
    app_server.resources.runtime = runtime
    app._app_server = app_server
    return app


@pytest.mark.asyncio
async def test_finish_resume_notices_shows_notices_after_ready() -> None:
    runtime = MagicMock()
    runtime.wait_until_ready = AsyncMock()
    app = _app_with_fake_runtime(runtime)

    with (
        patch.object(app, "_show_post_init_notices_once", AsyncMock()) as notices,
        patch.object(app, "_mount_and_scroll", AsyncMock()) as mount,
        patch.object(app, "_refresh_banner", MagicMock()),
    ):
        await app._finish_resume_notices()

    runtime.wait_until_ready.assert_awaited_once()
    notices.assert_awaited_once()
    mount.assert_not_awaited()


@pytest.mark.asyncio
async def test_finish_resume_notices_defers_notices_until_ready() -> None:
    gate = asyncio.Event()

    async def _blocked() -> None:
        await gate.wait()

    runtime = MagicMock()
    runtime.wait_until_ready = AsyncMock(side_effect=_blocked)
    app = _app_with_fake_runtime(runtime)

    with (
        patch.object(app, "_show_post_init_notices_once", AsyncMock()) as notices,
        patch.object(app, "_mount_and_scroll", AsyncMock()),
        patch.object(app, "_refresh_banner", MagicMock()),
    ):
        task = asyncio.create_task(app._finish_resume_notices())
        await asyncio.sleep(0)
        notices.assert_not_awaited()
        gate.set()
        await task
        notices.assert_awaited_once()


@pytest.mark.asyncio
async def test_finish_resume_notices_surfaces_late_init_failure() -> None:
    runtime = MagicMock()
    runtime.wait_until_ready = AsyncMock(side_effect=RuntimeError("mcp boom"))
    app = _app_with_fake_runtime(runtime)

    with (
        patch.object(app, "_show_post_init_notices_once", AsyncMock()) as notices,
        patch.object(app, "_mount_and_scroll", AsyncMock()) as mount,
        patch.object(app, "_refresh_banner", MagicMock()),
    ):
        await app._finish_resume_notices()

    notices.assert_not_awaited()
    mount.assert_awaited_once()
    (message,), _ = mount.call_args
    assert isinstance(message, ErrorMessage)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", [ProtocolErrorCode.CONFLICT, ProtocolErrorCode.NOT_FOUND]
)
async def test_finish_resume_notices_is_quiet_when_superseded(
    code: ProtocolErrorCode,
) -> None:
    runtime = MagicMock()
    runtime.wait_until_ready = AsyncMock(
        side_effect=AppServerResponseError(
            ProtocolError(code=code, message="superseded by a newer resume")
        )
    )
    app = _app_with_fake_runtime(runtime)

    with (
        patch.object(app, "_show_post_init_notices_once", AsyncMock()) as notices,
        patch.object(app, "_mount_and_scroll", AsyncMock()) as mount,
        patch.object(app, "_refresh_banner", MagicMock()),
    ):
        await app._finish_resume_notices()

    notices.assert_not_awaited()
    mount.assert_not_awaited()
    assert app._post_init_notices_shown is False


@pytest.mark.asyncio
async def test_finish_resume_notices_noop_when_already_shown() -> None:
    runtime = MagicMock()
    runtime.wait_until_ready = AsyncMock()
    app = _app_with_fake_runtime(runtime)
    app._post_init_notices_shown = True

    with (
        patch.object(app, "_show_post_init_notices_once", AsyncMock()) as notices,
        patch.object(app, "_mount_and_scroll", AsyncMock()),
        patch.object(app, "_refresh_banner", MagicMock()),
    ):
        await app._finish_resume_notices()

    runtime.wait_until_ready.assert_not_awaited()
    notices.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_local_session_updates_context_progress(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        runtime = chartreux_app.app_server.resources.runtime
        assert runtime.stats.context_tokens == 0

        def _apply_resumed_stats(*args: object) -> None:
            runtime._state.stats = AgentStatsSnapshot(
                context_tokens=_RESUMED_TOKENS, session_prompt_tokens=_RESUMED_TOKENS
            )
            runtime._state.context_window = _RESUMED_CONTEXT_WINDOW

        chartreux_app.app_server.resume = AsyncMock(side_effect=_apply_resumed_stats)
        chartreux_app._resume_history_from_messages = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()

        await chartreux_app._resume_local_session("abcd1234")

        widget = chartreux_app.query_one(ContextProgress)
        assert widget.tokens.current_tokens == _RESUMED_TOKENS
        assert widget.tokens.max_tokens == _RESUMED_CONTEXT_WINDOW


@pytest.mark.asyncio
async def test_resume_local_session_updates_banner_model(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        banner = chartreux_app._banner
        assert banner is not None
        previous_model = banner.state.active_model
        resumed_model = chartreux_app.config.active_model.model_copy(
            update={
                "name": "resumed-model",
                "alias": "resumed-model",
                "display_name": "Resumed Model",
            }
        )
        resumed_config = chartreux_app.config.model_copy(
            update={"active_model": resumed_model}
        )

        def _apply_resumed_model(*args: object) -> None:
            chartreux_app.app_server._state.config = resumed_config

        chartreux_app.app_server.resume = AsyncMock(side_effect=_apply_resumed_model)
        with (
            patch.object(
                chartreux_app, "_rebuild_transcript_from_current_session", AsyncMock()
            ),
            patch.object(chartreux_app, "_mount_and_scroll", AsyncMock()),
            patch.object(chartreux_app, "_finish_resume_notices", AsyncMock()),
        ):
            await chartreux_app._resume_local_session("abcd1234")

        assert banner.state.active_model != previous_model
        assert banner.state.active_model == f"Resumed Model[{resumed_model.thinking}]"


@pytest.mark.asyncio
async def test_resume_local_session_shows_zero_when_no_llm_activity(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        chartreux_app.app_server.resume = AsyncMock()
        chartreux_app._resume_history_from_messages = AsyncMock()
        chartreux_app._mount_and_scroll = AsyncMock()

        await chartreux_app._resume_local_session("abcd1234")

        widget = chartreux_app.query_one(ContextProgress)
        assert widget.tokens.current_tokens == 0


@pytest.mark.asyncio
async def test_rebuild_discards_previous_session_admission(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        chartreux_app._transcript.admit([_message(0)], start_index=0)
        assert chartreux_app._transcript.unit_ids
        await chartreux_app._rebuild_transcript_from_current_session()
        assert not chartreux_app._transcript.unit_ids
        assert chartreux_app._active_turn_start is None


@pytest.mark.asyncio
async def test_successful_resume_replaces_stale_turn_presentation(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        await chartreux_app._ensure_loading_widget()
        stale_loading = chartreux_app._loading_widget
        assert isinstance(stale_loading, LoadingWidget)
        chartreux_app._begin_pending_turn()

        chartreux_app.app_server.resume = AsyncMock()
        clear_queue = AsyncMock()
        sync_queue = AsyncMock()
        with (
            patch.object(chartreux_app._queue, "clear_server_queue", clear_queue),
            patch.object(chartreux_app._queue, "sync_server_queue", sync_queue),
            patch.object(
                chartreux_app, "_rebuild_transcript_from_current_session", AsyncMock()
            ),
            patch.object(chartreux_app, "_mount_and_scroll", AsyncMock()),
            patch.object(chartreux_app, "_finish_resume_notices", AsyncMock()),
        ):
            await chartreux_app._resume_local_session("abcd1234")

        assert chartreux_app._loading_widget is None
        assert stale_loading.parent is None
        assert chartreux_app._pending_turn is False
        assert chartreux_app._resume_ui_ready.is_set()
        clear_queue.assert_awaited_once_with()
        sync_queue.assert_awaited_once_with(chartreux_app.app_server.turn_queue)


@pytest.mark.asyncio
async def test_failed_resume_preserves_current_turn_presentation(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test():
        await chartreux_app._ensure_loading_widget()
        loading = chartreux_app._loading_widget
        assert isinstance(loading, LoadingWidget)
        chartreux_app._begin_pending_turn()

        chartreux_app.app_server.resume = AsyncMock(side_effect=RuntimeError("boom"))
        clear_queue = AsyncMock()
        sync_queue = AsyncMock()
        with (
            patch.object(chartreux_app._queue, "clear_server_queue", clear_queue),
            patch.object(chartreux_app._queue, "sync_server_queue", sync_queue),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await chartreux_app._resume_local_session("abcd1234")

        assert chartreux_app._loading_widget is loading
        assert loading.parent is not None
        assert chartreux_app._pending_turn is True
        assert chartreux_app._resume_ui_ready.is_set()
        clear_queue.assert_not_awaited()
        sync_queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_enqueue_prompt_prepares_eagerly(chartreux_app: ChartreuxApp) -> None:
    prepared = PreparedPrompt(display_text="hello", prompt_text="hello-prepared")
    prepare = AsyncMock(return_value=prepared)
    enqueue = AsyncMock()
    with (
        patch.object(chartreux_app, "_prepare_prompt_or_abort", prepare),
        patch.object(chartreux_app._queue, "enqueue_prompt", enqueue),
    ):
        result = await chartreux_app._enqueue_prompt_with_resources("hello")

    assert result is True
    prepare.assert_awaited_once_with("hello")
    enqueue.assert_awaited_once_with(
        "hello", skill_name=None, prepared_prompt=prepared, optimistic_start=False
    )


@pytest.mark.asyncio
async def test_submit_during_fresh_bootstrap_waits_then_dispatches() -> None:
    release = asyncio.Event()
    app = build_test_chartreux_app()
    app._mount_first = True
    original_starter = app._start_app_server
    assert original_starter is not None

    async def _latched_starter() -> AppServerSession:
        await release.wait()
        return await original_starter()

    app._start_app_server = _latched_starter

    enqueue = AsyncMock()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        assert app._app_server is None
        assert not app._session_ready.is_set()

        with patch.object(app._queue, "enqueue_prompt", enqueue):
            submit = asyncio.create_task(app._handle_user_message("hello"))
            await asyncio.sleep(0.05)
            # Blocked on _session_ready inside _prepare_prompt_or_abort — no
            # RuntimeError from the unbound app_server property.
            assert not submit.done()
            assert app._loading_widget is not None
            enqueue.assert_not_awaited()

            release.set()
            await pilot.pause(0.3)
            await submit

        assert app._session_ready.is_set()
        enqueue.assert_awaited_once()


async def _assert_submit_during_resume_window_dispatches(
    app: ChartreuxApp, *, resume_session_id: str | None, continue_latest: bool
) -> None:
    async with app.run_test(size=(120, 40)):
        # Warm start pre-sets ready; clear it and re-arm the resume flags to
        # simulate the mount-first auto-resume window.
        app._session_ready.clear()
        app._resume_session_id = resume_session_id
        app._continue_latest = continue_latest

        gate = asyncio.Event()

        async def _blocked_resume(_session_id: str) -> None:
            await gate.wait()

        enqueue = AsyncMock()
        with (
            patch.object(
                app.app_server.resources.sessions,
                "resolve_continue_session",
                AsyncMock(return_value="abcd1234"),
            ),
            patch.object(
                app, "_resume_local_session", AsyncMock(side_effect=_blocked_resume)
            ),
            patch.object(app, "_process_startup_prompt_when_available", AsyncMock()),
            patch.object(app._queue, "enqueue_prompt", enqueue),
        ):
            resume_task = asyncio.create_task(app._auto_resume_on_startup())
            await asyncio.sleep(0.05)
            assert not app._session_ready.is_set()

            submit = asyncio.create_task(app._handle_user_message("go"))
            await asyncio.sleep(0.05)
            assert not submit.done()
            enqueue.assert_not_awaited()

            gate.set()
            await resume_task
            await submit

        assert app._session_ready.is_set()
        enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_submit_during_resume_window_dispatches(
    chartreux_app: ChartreuxApp,
) -> None:
    await _assert_submit_during_resume_window_dispatches(
        chartreux_app, resume_session_id="abcd1234", continue_latest=False
    )


@pytest.mark.asyncio
async def test_submit_during_continue_window_dispatches(
    chartreux_app: ChartreuxApp,
) -> None:
    await _assert_submit_during_resume_window_dispatches(
        chartreux_app, resume_session_id=None, continue_latest=True
    )


@pytest.mark.asyncio
async def test_session_ready_set_on_fresh_cold_start() -> None:
    app = build_test_chartreux_app()
    app._mount_first = True
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        assert app._app_server is not None
        assert app._session_ready.is_set()


@pytest.mark.asyncio
async def test_session_ready_set_on_warm_start(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.1)
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_auto_resume_failure_is_rendered_when_transcript_rebuild_fails(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._resume_session_id = "abcd1234"
        with (
            patch.object(
                chartreux_app,
                "_resume_local_session",
                AsyncMock(side_effect=RuntimeError("connection closed")),
            ),
            patch.object(
                chartreux_app,
                "_rebuild_transcript_from_current_session",
                AsyncMock(side_effect=RuntimeError("transcript unavailable")),
            ),
            patch.object(
                chartreux_app, "_process_startup_prompt_when_available", AsyncMock()
            ),
        ):
            await chartreux_app._auto_resume_on_startup()

        assert any(
            "Failed to resume session: connection closed" in str(error._error)
            for error in chartreux_app.query(ErrorMessage)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resume_session_id", "continue_latest", "continue_target"),
    [("abcd1234", False, "abcd1234"), (None, True, "abcd1234"), (None, True, None)],
    ids=["resume", "continue", "no-sessions-found"],
)
async def test_session_ready_set_after_auto_resume(
    chartreux_app: ChartreuxApp,
    resume_session_id: str | None,
    continue_latest: bool,
    continue_target: str | None,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._session_ready.clear()
        chartreux_app._resume_session_id = resume_session_id
        chartreux_app._continue_latest = continue_latest
        with (
            patch.object(
                chartreux_app.app_server.resources.sessions,
                "resolve_continue_session",
                AsyncMock(return_value=continue_target),
            ),
            patch.object(chartreux_app, "_resume_local_session", AsyncMock()),
            patch.object(
                chartreux_app, "_process_startup_prompt_when_available", AsyncMock()
            ),
            patch.object(
                chartreux_app,
                "_show_custom_tools_deprecation_warning_after_initial_history",
                AsyncMock(),
            ),
        ):
            await chartreux_app._auto_resume_on_startup()
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_session_ready_set_on_picker_selected_success(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._session_ready.clear()
        event = MagicMock(session_id="abcd1234")
        with (
            patch.object(chartreux_app, "_switch_to_input_app", AsyncMock()),
            patch.object(chartreux_app, "_resume_local_session", AsyncMock()),
        ):
            await chartreux_app.on_session_picker_app_session_selected(event)
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_session_ready_set_on_picker_selected_failure(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._session_ready.clear()
        event = MagicMock(session_id="abcd1234")
        with (
            patch.object(chartreux_app, "_switch_to_input_app", AsyncMock()),
            patch.object(
                chartreux_app,
                "_resume_local_session",
                AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch.object(chartreux_app, "_mount_and_scroll", AsyncMock()),
            patch.object(
                chartreux_app, "_rebuild_transcript_from_current_session", AsyncMock()
            ),
        ):
            await chartreux_app.on_session_picker_app_session_selected(event)
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_session_ready_set_on_picker_cancelled(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._session_ready.clear()
        event = MagicMock()
        with (
            patch.object(chartreux_app, "_switch_to_input_app", AsyncMock()),
            patch.object(chartreux_app, "_mount_and_scroll", AsyncMock()),
        ):
            await chartreux_app.on_session_picker_app_cancelled(event)
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_session_ready_set_on_picker_delete_last_exit(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._session_ready.clear()
        with patch.object(chartreux_app, "_switch_to_input_app", AsyncMock()):
            await chartreux_app._exit_picker_to_input()
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_picker_remains_active_when_no_sessions_are_found(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)):
        chartreux_app._session_ready.clear()
        with (
            patch.object(chartreux_app, "_switch_from_input", AsyncMock()),
            patch.object(chartreux_app, "_switch_to_input_app", AsyncMock()),
            patch.object(chartreux_app, "_mount_and_scroll", AsyncMock()),
            patch.object(
                chartreux_app.app_server.resources.sessions,
                "list",
                AsyncMock(return_value=[]),
            ),
        ):
            await chartreux_app._show_session_picker()
        assert not chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_picker_discovery_failure_escape_returns_to_ready_input(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test(size=(120, 40)) as pilot:
        chartreux_app._session_ready.clear()
        with patch.object(
            chartreux_app.app_server.resources.sessions,
            "list",
            AsyncMock(side_effect=RuntimeError("discovery failed")),
        ):
            await chartreux_app._handle_command("/resume")

        picker = chartreux_app.query_one(SessionPickerApp)
        assert picker.load_error == "discovery failed"
        assert "discovery failed" in str(
            picker.query_one(".sessionpicker-loading-status", NoMarkupStatic).content
        )
        assert not chartreux_app._session_ready.is_set()

        await pilot.press("escape")
        await pilot.pause()

        assert len(chartreux_app.query(SessionPickerApp)) == 0
        input_container = chartreux_app.query_one(ChatInputContainer)
        assert input_container.display is True
        assert input_container.disabled is False
        assert chartreux_app._session_ready.is_set()


@pytest.mark.asyncio
async def test_bootstrap_error_disables_input_and_never_deadlocks() -> None:
    app = build_test_chartreux_app()
    app._mount_first = True

    async def _failing_starter() -> AppServerSession:
        raise RuntimeError("session start failed")

    app._start_app_server = _failing_starter

    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.2)
        assert app._app_server is None
        assert app._fatal_init_error is True
        # Bootstrap error never marks ready — the input is disabled, so no
        # submit can reach the turn path to deadlock on the await.
        assert not app._session_ready.is_set()
        container = app.query_one(ChatInputContainer)
        assert container.disabled is True


@pytest.mark.asyncio
async def test_ensure_runtime_ready_blocks_until_session_ready() -> None:
    runtime = MagicMock()
    runtime.ready = True
    runtime.wait_until_ready = AsyncMock()
    app = _app_with_fake_runtime(runtime)
    app._session_ready.clear()

    task = asyncio.create_task(app._ensure_runtime_ready())
    await asyncio.sleep(0.05)
    assert not task.done()
    runtime.wait_until_ready.assert_not_awaited()

    app._mark_session_ready()
    await task
    runtime.wait_until_ready.assert_awaited_once()


@pytest.mark.asyncio
async def test_ensure_runtime_ready_instant_when_session_ready() -> None:
    runtime = MagicMock()
    runtime.ready = True
    runtime.wait_until_ready = AsyncMock()
    app = _app_with_fake_runtime(runtime)
    app._mark_session_ready()

    await asyncio.wait_for(app._ensure_runtime_ready(), timeout=1.0)
    runtime.wait_until_ready.assert_awaited_once()
