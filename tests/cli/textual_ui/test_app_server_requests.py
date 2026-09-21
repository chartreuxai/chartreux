from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    OpenCallbackState,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    PublicError,
    QuestionChoice,
    TurnErrorCode,
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
    WorkspaceTrustDetails,
)
from chartreux.app_server.protocol import WorkspaceTrustStatusResponse
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui import startup
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.context_progress import ContextProgress
from chartreux.cli.textual_ui.widgets.loading import (
    DEFAULT_LOADING_STATUS,
    LoadingWidget,
)
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    ReasoningMessage,
    SlashCommandMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.events import UserMessageEvent
from chartreux.core.llm_models import Role
from chartreux.core.session_types import ScheduledLoop
from chartreux.setup.trusted_folders.trust_folder_dialog import TrustFolderApp
from chartreux.utils import VIBE_WARNING_TAG
from chartreux.utils.retry_prompt import build_retry_prompt
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend, FakeInterruptedStreamingBackend


def _callback(detail: UserInputCallbackDetail):
    return PublicCallbackEntry(
        id="callback:callback-1",
        session_id="session-1",
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        callback_id="callback-1",
        title="Input required",
        detail=detail,
        state=OpenCallbackState(),
    )


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for UI state")
        await pilot.pause(0.01)


def _turn_error(code: TurnErrorCode) -> AppServerTurnError:
    return AppServerTurnError(PublicError(message="Network error", code=code))


async def _wait_for_retry_error(app: ChartreuxApp, pilot) -> None:
    await _wait_until(
        pilot,
        lambda: any("/retry" in str(error._error) for error in app.query(ErrorMessage)),
    )


@pytest.mark.asyncio
async def test_question_callback_opens_from_public_protocol() -> None:
    app = MagicMock()
    app._active_callback = None
    app._pending_local_question = None
    app._pending_callbacks = deque()
    app._wait_for_typing_pause = AsyncMock()
    app._switch_to_question_app = AsyncMock()
    callback = _question_callback("callback-1")

    await ChartreuxApp._show_callback(app, callback)

    assert app._active_callback is callback
    app._switch_to_question_app.assert_awaited_once_with(callback.detail.request)


@pytest.mark.asyncio
async def test_overlapping_callbacks_are_queued_until_the_active_one_resolves() -> None:
    app = MagicMock()
    first = _callback(
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="First?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        )
    )
    second = first.model_copy(
        update={"id": "callback:callback-2", "callback_id": "callback-2"}
    )
    app._active_callback = first
    app._pending_callbacks = deque()

    await ChartreuxApp._show_callback(app, second)

    assert list(app._pending_callbacks) == [second]


@pytest.mark.asyncio
async def test_callback_claim_is_atomic_during_typing_debounce() -> None:
    app = MagicMock()
    first = _callback(
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="First?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        )
    )
    second = first.model_copy(
        update={"id": "callback:callback-2", "callback_id": "callback-2"}
    )
    release = asyncio.Event()
    app._active_callback = None
    app._pending_local_question = None
    app._pending_callbacks = deque()
    app._wait_for_typing_pause = AsyncMock(side_effect=release.wait)
    app._switch_to_question_app = AsyncMock()

    first_task = asyncio.create_task(ChartreuxApp._show_callback(app, first))
    await asyncio.sleep(0)
    await ChartreuxApp._show_callback(app, second)

    assert app._active_callback is first
    assert list(app._pending_callbacks) == [second]
    release.set()
    await first_task


@pytest.mark.asyncio
async def test_callback_waits_behind_local_question() -> None:
    app = MagicMock()
    callback = _callback(
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Server question?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        )
    )
    app._active_callback = None
    app._pending_local_question = asyncio.get_running_loop().create_future()
    app._pending_callbacks = deque()

    await ChartreuxApp._show_callback(app, callback)

    assert app._active_callback is None
    assert list(app._pending_callbacks) == [callback]


@pytest.mark.asyncio
async def test_answering_active_callback_opens_the_next_queued_callback() -> None:
    app = MagicMock()
    first = _callback(
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="First?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        )
    )
    second = first.model_copy(
        update={"id": "callback:callback-2", "callback_id": "callback-2"}
    )
    app._active_callback = first
    app._pending_callbacks = deque([second])
    app.app_server.respond_to_callback = AsyncMock()
    app._show_callback = AsyncMock()

    output = UserInputCallbackOutput(
        result=UserQuestionResult(answers=[], cancelled=True)
    )
    await ChartreuxApp._respond_to_active_callback(app, output)

    app.app_server.respond_to_callback.assert_awaited_once_with(
        first.callback_id, output
    )
    app._show_callback.assert_awaited_once_with(second)


@pytest.mark.asyncio
async def test_question_answer_responds_to_public_callback() -> None:
    app = MagicMock()
    app._active_callback = _callback(
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Ship it?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        )
    )
    app._respond_to_active_callback = AsyncMock()
    answer = UserAnswer(question="Ship it?", answer="Yes")

    await ChartreuxApp.on_question_app_answered(app, QuestionApp.Answered([answer]))

    app._respond_to_active_callback.assert_awaited_once_with(
        UserInputCallbackOutput(
            result=UserQuestionResult(answers=[answer], cancelled=False)
        )
    )


def _question_callback(callback_id: str) -> PublicCallbackEntry:
    callback = _callback(
        UserInputCallbackDetail(
            request=UserQuestionRequest(
                questions=[
                    UserQuestion(
                        question="Ship it?",
                        options=[
                            QuestionChoice(label="Yes"),
                            QuestionChoice(label="No"),
                        ],
                    )
                ]
            )
        )
    )
    return callback.model_copy(
        update={"id": f"callback:{callback_id}", "callback_id": callback_id}
    )


@pytest.mark.asyncio
async def test_question_callback_replaces_loading_status_before_typing_pause() -> None:
    app = MagicMock()
    callback = _question_callback("callback-1")
    loading = LoadingWidget(status=DEFAULT_LOADING_STATUS)
    release_typing_pause = asyncio.Event()
    app._active_callback = None
    app._pending_local_question = None
    app._pending_callbacks = deque()
    app._loading_widget = loading
    app._ensure_loading_widget = AsyncMock()
    app._wait_for_typing_pause = AsyncMock(side_effect=release_typing_pause.wait)
    app._switch_to_question_app = AsyncMock()

    show = asyncio.create_task(ChartreuxApp._show_callback(app, callback))
    await asyncio.sleep(0)

    assert loading.base_status == "Input required"
    assert loading._pause_start is not None
    app._switch_to_question_app.assert_not_awaited()

    release_typing_pause.set()
    await show


@pytest.mark.asyncio
async def test_answering_final_callback_restores_loading_progress() -> None:
    app = MagicMock()
    callback = _question_callback("callback-1")
    loading = LoadingWidget(status="Running command")
    loading.begin_action_required("Input required")
    app._active_callback = callback
    app._pending_callbacks = deque()
    app._loading_widget = loading
    app.app_server.respond_to_callback = AsyncMock()
    app._switch_to_input_app = AsyncMock()
    output = UserInputCallbackOutput(
        result=UserQuestionResult(answers=[], cancelled=True)
    )

    await ChartreuxApp._respond_to_active_callback(app, output)

    assert loading.base_status == "Running command"
    assert loading._pause_start is None
    app._switch_to_input_app.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_resolving_a_callback_swaps_to_the_next_without_duplicate_mount(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(config=config)
    app = build_test_chartreux_app(agent_loop=agent_loop)
    first = _question_callback("callback-1")
    second = _question_callback("callback-2")

    async with app.run_test() as pilot:
        await _wait_until(pilot, lambda: app._app_server is not None)
        monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())

        await app._handle_turn_event(CallbackRequested(first))
        await _wait_until(pilot, lambda: app._active_callback is first)
        assert len(app.query(QuestionApp)) == 1
        assert app._loading_widget is not None
        assert app._loading_widget.base_status == "Input required"
        paused_at = app._loading_widget._pause_start
        assert paused_at is not None

        await app._handle_turn_event(CallbackRequested(second))
        await pilot.pause()
        assert list(app._pending_callbacks) == [second]
        assert len(app.query(QuestionApp)) == 1

        await app._respond_to_active_callback(
            UserInputCallbackOutput(
                result=UserQuestionResult(answers=[], cancelled=True)
            )
        )
        await _wait_until(pilot, lambda: app._active_callback is second)
        assert len(app.query(QuestionApp)) == 1
        assert app._loading_widget is not None
        assert app._loading_widget._pause_start == paused_at


@pytest.mark.asyncio
async def test_workspace_trust_round_trips_through_app_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    host = MagicMock()
    host.cwd = "/workspace"
    host.trust_status = AsyncMock(
        return_value=WorkspaceTrustStatusResponse(
            status="untrusted",
            details=WorkspaceTrustDetails(
                cwd="/workspace",
                detected_files=["AGENTS.md"],
                settings_path="/home/user/.chartreux/trusted_folders.toml",
                available_decisions=["trust_cwd", "decline"],
            ),
        )
    )
    host.decide_trust = AsyncMock(
        return_value=WorkspaceTrustStatusResponse(status="trusted")
    )
    monkeypatch.setattr(
        TrustFolderApp, "run_trust_dialog_async", AsyncMock(return_value="trust_cwd")
    )
    assert await startup._resolve_workspace_trust(host) == (True, True)

    host.trust_status.assert_awaited_once_with("/workspace")
    host.decide_trust.assert_awaited_once_with("trust_cwd", cwd="/workspace")


@pytest.mark.asyncio
async def test_escape_interrupts_unsolicited_server_turn(tmp_path: Path) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    agent_loop = build_test_agent_loop(config=config)
    started = asyncio.Event()
    interrupted = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        yield UserMessageEvent(content=msg, message_id="scheduled-user")
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            interrupted.set()

    agent_loop.act = blocking_act
    metadata = agent_loop.session_logger.session_metadata
    assert metadata is not None
    now = time.time()
    metadata.loops = [
        ScheduledLoop(
            id="scheduled-1",
            interval_seconds=30,
            prompt="scheduled prompt",
            next_fire_at=now - 1,
            created_at=now - 31,
        )
    ]
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await asyncio.wait_for(started.wait(), timeout=2)
        await _wait_until(pilot, lambda: app.app_server.turn_active)
        assert app._agent_task is None

        await pilot.press("escape")

        await asyncio.wait_for(interrupted.wait(), timeout=2)
        await _wait_until(pilot, lambda: not app.app_server.turn_active)


def test_invalid_api_key_does_not_offer_retry() -> None:
    app = MagicMock()
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()

    asyncio.run(
        ChartreuxApp._mount_turn_error(
            app, _turn_error(TurnErrorCode.INVALID_API_KEY), "Invalid API key"
        )
    )

    app.event_handler.offer_retry.assert_not_called()


def test_rate_limit_error_still_offers_retry() -> None:
    app = MagicMock()
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()

    asyncio.run(
        ChartreuxApp._mount_turn_error(
            app, _turn_error(TurnErrorCode.RATE_LIMIT), "Rate limited"
        )
    )

    app.event_handler.offer_retry.assert_called_once()


def test_invalid_api_key_clears_stale_retry_state() -> None:
    app = MagicMock()
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()

    asyncio.run(
        ChartreuxApp._mount_turn_error(
            app, _turn_error(TurnErrorCode.INVALID_API_KEY), "Invalid API key"
        )
    )

    app.event_handler.cancel_retry_presentation.assert_called_once_with()


def test_backend_error_message_hints_at_retry() -> None:
    app = MagicMock()
    app._retry_hint = ChartreuxApp._retry_hint

    message = ChartreuxApp._resolve_turn_error_message(
        app, _turn_error(TurnErrorCode.BACKEND_ERROR)
    )

    assert "/retry [additional instructions]" in message
    assert "Network error" in message


def test_internal_error_message_does_not_hint_at_retry() -> None:
    app = MagicMock()

    message = ChartreuxApp._resolve_turn_error_message(
        app, _turn_error(TurnErrorCode.INTERNAL_ERROR)
    )

    assert "/retry" not in message


@pytest.mark.asyncio
async def test_incomplete_stream_retries_and_reuses_assistant_message() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="Ran three dummy read-only", stop_reason=None)],
        [mock_llm_chunk(content=" tool calls successfully.")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        assert len(app.query(AssistantMessage)) == 1
        assert (
            app.query_one(AssistantMessage).get_content()
            == "Ran three dummy read-only tool calls successfully."
        )
        assert len(app.query(ErrorMessage)) == 0
        assert len(app.query(SlashCommandMessage)) == 0

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("")


@pytest.mark.asyncio
async def test_incomplete_stream_hides_error_while_retrying() -> None:
    gate = asyncio.Event()

    class GatedRecoveryBackend(FakeBackend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0
            self.second_started = asyncio.Event()

        async def complete_streaming(self, **kwargs):
            self.calls += 1
            if self.calls == 2:
                self.second_started.set()
                await gate.wait()
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk

    backend = GatedRecoveryBackend([
        [mock_llm_chunk(content="partial", stop_reason=None)],
        [mock_llm_chunk(content=" recovered.")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))

        # Wait until the automatic retry is in flight, then confirm no error is
        # surfaced while we are still retrying and that the loader says so.
        await _wait_until(pilot, backend.second_started.is_set)
        assert len(app.query(ErrorMessage)) == 0
        assert app._loading_widget is not None
        assert app._loading_widget._base_status == "Retrying"

        gate.set()
        await _wait_until(pilot, lambda: not app._agent_job_active())

        assert len(app.query(ErrorMessage)) == 0
        assert len(app.query(SlashCommandMessage)) == 0
        assert app.query_one(AssistantMessage).get_content() == "partial recovered."


@pytest.mark.asyncio
async def test_interrupting_auto_retry_keeps_event_listener_alive() -> None:
    retry_gate = asyncio.Event()

    class GatedRetryBackend(FakeBackend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0
            self.retry_started = asyncio.Event()

        async def complete_streaming(self, **kwargs):
            self.calls += 1
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk
                if self.calls == 2:
                    self.retry_started.set()
                    await retry_gate.wait()

    backend = GatedRetryBackend([
        [mock_llm_chunk(content="partial", stop_reason=None)],
        [mock_llm_chunk(content=" retry", stop_reason=None)],
        [mock_llm_chunk(content="after cancel")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    try:
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            chat_input = app.query_one(ChatInputContainer)
            chat_input.post_message(ChatInputContainer.Submitted("first"))
            await _wait_until(pilot, backend.retry_started.is_set)

            await pilot.press("escape")
            await _wait_until(pilot, lambda: not app._agent_job_active())

            chat_input.post_message(ChatInputContainer.Submitted("next"))
            await _wait_until(pilot, lambda: len(backend.requests_messages) == 3)
            await _wait_until(
                pilot,
                lambda: any(
                    "after cancel" in message.get_content()
                    for message in app.query(AssistantMessage)
                ),
            )
    finally:
        retry_gate.set()


@pytest.mark.asyncio
async def test_empty_incomplete_stream_retries_original_request() -> None:
    backend = FakeBackend([[], [mock_llm_chunk(content="Recovered")]])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        assert [message.get_content() for message in app.query(AssistantMessage)] == [
            "Recovered"
        ]
        assert len(app.query(ErrorMessage)) == 0

    assert all(
        message.role is not Role.assistant for message in backend.requests_messages[-1]
    )


@pytest.mark.asyncio
async def test_incomplete_stream_stops_after_two_automatic_retries() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="first", stop_reason=None)],
        [mock_llm_chunk(content=" second", stop_reason=None)],
        [mock_llm_chunk(content=" third", stop_reason=None)],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 3)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        await pilot.pause(0.1)

        assert len(backend.requests_messages) == 3
        assert len(app.query(ErrorMessage)) == 1
        assert "/retry" in str(app.query_one(ErrorMessage)._error)


@pytest.mark.asyncio
async def test_incomplete_stream_does_not_retry_ahead_of_queued_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class GatedBackend(FakeBackend):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.calls = 0
            self.started = [asyncio.Event() for _ in range(3)]
            self.released = [asyncio.Event() for _ in range(3)]
            self.release = [asyncio.Event() for _ in range(3)]

        async def complete_streaming(self, **kwargs):
            idx = self.calls
            self.calls += 1
            self.started[idx].set()
            if idx in (0, 1):
                await self.release[idx].wait()
            self.released[idx].set()
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk

    backend = GatedBackend([
        [mock_llm_chunk(content="t0 done")],
        [mock_llm_chunk(content="t1 partial", stop_reason=None)],
        [mock_llm_chunk(content="t2 done")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)
    prepare_started = asyncio.Event()
    release_prepare = asyncio.Event()
    original_prepare = app._prepare_prompt_or_abort

    async def blocked_prepare(message: str):
        if message == "t2":
            prepare_started.set()
            await release_prepare.wait()
        return await original_prepare(message)

    monkeypatch.setattr(app, "_prepare_prompt_or_abort", blocked_prepare)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)

        # t0 runs; t1 queues behind it. When t0 finishes, t1 promotes and runs.
        chat_input.post_message(ChatInputContainer.Submitted("t0"))
        await _wait_until(pilot, backend.started[0].is_set)
        chat_input.post_message(ChatInputContainer.Submitted("t1"))
        await _wait_until(
            pilot, lambda: app._queue.has_server_work and len(app._queue) == 1
        )

        backend.release[0].set()
        await _wait_until(pilot, backend.started[1].is_set)

        # t2 queues only after t1 has started, so it is a separate turn behind
        # the turn that returns an incomplete stream.
        chat_input.post_message(ChatInputContainer.Submitted("t2"))
        await _wait_until(pilot, prepare_started.is_set)
        await _wait_until(pilot, lambda: app._queue.has_server_work)
        # The submission reservation is visible before preparation produces a
        # prompt for the queue, even while the prior item is settling.
        assert app._queue._pending_enqueues == 1

        # t1 returns incomplete while t2 is still preparing: the auto-retry must
        # defer to the eventual queued prompt rather than retry ahead of it.
        backend.release[1].set()
        await backend.released[1].wait()
        assert len(backend.requests_messages) == 2

        release_prepare.set()
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 3)
        await _wait_until(pilot, backend.started[2].is_set)
        assert backend.requests_messages[2][-1].content == "t2"

        await _wait_until(
            pilot, lambda: not app._agent_job_active() and len(app._queue) == 0
        )
        await _wait_until(
            pilot,
            lambda: any(
                "t2 done" in message.get_content()
                for message in app.query(AssistantMessage)
            ),
        )
        await _wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 1)
        await _wait_until(
            pilot,
            lambda: (
                app.query_one(ContextProgress).tokens.current_tokens
                == agent_loop.stats.context_tokens
            ),
        )

        contents = " ".join(m.get_content() for m in app.query(AssistantMessage))
        assert "t2 done" in contents
        assert len(app.query(ErrorMessage)) == 1
        assert "/retry" in str(app.query_one(ErrorMessage)._error)


@pytest.mark.asyncio
async def test_retry_command_reuses_interrupted_assistant_message() -> None:
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="Ran three dummy read-only")],
        [mock_llm_chunk(content=" tool calls successfully.")],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        chat_input.post_message(
            ChatInputContainer.Submitted("/retry Keep the conclusion concise.")
        )
        await _wait_until(
            pilot,
            lambda: (
                agent_loop.messages[-1].role is Role.assistant
                and agent_loop.messages[-1].content == " tool calls successfully."
            ),
        )
        await _wait_until(
            pilot,
            lambda: (
                len(app.query(AssistantMessage)) == 1
                and app.query_one(AssistantMessage).get_content()
                == "Ran three dummy read-only tool calls successfully."
            ),
        )

        visible_turn = [
            widget
            for widget in app._messages_area.children
            if isinstance(widget, AssistantMessage | ErrorMessage | SlashCommandMessage)
        ]
        assert [type(widget) for widget in visible_turn] == [AssistantMessage]
        assert [
            message._content
            for message in app.query(UserMessage)
            if not isinstance(message, SlashCommandMessage)
        ] == ["hi"]

    assert backend.streaming_attempts == 2
    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content is not None
    assert retry_message.content.startswith(f"<{VIBE_WARNING_TAG}>")
    assert "without repeating text already produced" in retry_message.content
    assert "additional instructions from the user" in retry_message.content
    assert "Keep the conclusion concise." in retry_message.content


@pytest.mark.asyncio
async def test_retry_command_keeps_separate_assistant_after_reasoning() -> None:
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="partial")],
        [
            mock_llm_chunk(content="", reasoning_content="thinking"),
            mock_llm_chunk(content="recovered"),
        ],
    ])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        chat_input.post_message(ChatInputContainer.Submitted("/retry"))
        await _wait_until(
            pilot,
            lambda: (
                agent_loop.messages[-1].role is Role.assistant
                and agent_loop.messages[-1].content == "recovered"
            ),
        )
        await _wait_until(pilot, lambda: len(app.query(AssistantMessage)) == 2)

        assert [message.get_content() for message in app.query(AssistantMessage)] == [
            "partial",
            "recovered",
        ]
        assert len(app.query(ReasoningMessage)) == 1
        assert len(app.query(ErrorMessage)) == 0
        assert len(app.query(SlashCommandMessage)) == 0


@pytest.mark.asyncio
async def test_retry_command_keeps_diagnostics_until_retry_progress() -> None:
    backend = FakeInterruptedStreamingBackend([[mock_llm_chunk(content="partial")]])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        backend._exception_to_raise = RuntimeError("retry failed")
        chat_input.post_message(ChatInputContainer.Submitted("/retry"))
        await _wait_until(pilot, lambda: backend.streaming_attempts == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        await _wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 2)

        assert [message.get_content() for message in app.query(AssistantMessage)] == [
            "partial"
        ]
        assert len(app.query(ErrorMessage)) == 2
        assert len(app.query(SlashCommandMessage)) == 1
