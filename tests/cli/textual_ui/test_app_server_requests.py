from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from textual.widgets import Button

from chartreux.app_server.events import AgentsUpdate, CallbackRequested
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
from chartreux.app_server.protocol import (
    AgentsCancelResponse,
    AgentSummaryModel,
    AppServerResponseError,
    CancelOutcome,
    ProtocolError,
    ProtocolErrorCode,
    WorkspaceTrustStatusResponse,
)
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui import startup
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.screens.settings import SettingsScreen
from chartreux.cli.textual_ui.turn_supervisor import (
    _MAX_INCOMPLETE_STREAM_RETRIES,
    TurnSupervisor,
)
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.loading import (
    DEFAULT_LOADING_STATUS,
    LoadingWidget,
)
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    ErrorRetryRequested,
    ReasoningMessage,
    SlashCommandMessage,
    UserCommandMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from chartreux.cli.textual_ui.widgets.session_status_line import SessionStatusLine
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import RecoverySectionConfig, RootRecoveryConfig
from chartreux.core.events import UserMessageEvent
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm_models import Role
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session_types import ScheduledLoop
from chartreux.core.subagents import derive_agent_name
from chartreux.setup.trusted_folders.trust_folder_dialog import TrustFolderApp
from chartreux.utils import VIBE_WARNING_TAG
from chartreux.utils.retry_prompt import build_retry_prompt
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
    wait_until,
)
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import ControlledTransport, FakeClock
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


def _enabled_root_recovery_config() -> ChartreuxConfigSchema:
    """Explicitly-enabled ``[recovery.root]`` (round-2 amendment: never rely
    on the development default — WP6 promotes it and must not break tests).
    """
    return build_test_vibe_config().model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=True))
        }
    )


def _disabled_root_recovery_config() -> ChartreuxConfigSchema:
    """Explicitly-disabled ``[recovery.root]`` (Q, WP6 promotion ripple).

    WP6 promoted the enabled default, but these tests characterize the MANUAL
    retry surface — the error widget and the /retry affordance that only
    appear when no automatic continuation is scheduled — so the pre-promotion
    default (disabled) is pinned explicitly.
    """
    return build_test_vibe_config().model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=False))
        }
    )


def _root_recovery_clock_sleep(agent_loop: Any) -> Any:
    """R2: point the shared availability registry at a manual clock and return
    the root-recovery controller's timer sleep, which advances that clock past
    the armed deployment cooldown — the gated continuation is scheduled and
    admitted with NO real wait (fake clock + deterministic transport).
    """
    clock = FakeClock()
    agent_loop.config_orchestrator._availability_registry = AvailabilityRegistry(
        clock=clock
    )

    async def _advancing_sleep(_delay: float) -> None:
        clock.advance(40.0)

    return _advancing_sleep


def _turn_error(code: TurnErrorCode) -> AppServerTurnError:
    return AppServerTurnError(PublicError(message="Network error", code=code))


class _CallbackQuestionStandIn:
    """TurnSupervisor stand-in for the moved callback/question state (stage 10.5a).

    The moved bodies own the callback/question state under public controller
    names; this stand-in forwards them to the app's underscore attributes so
    the MagicMock setup and assertions are unchanged, and exposes the app as
    ``_app`` for the late-bound collaborator calls.
    """

    def __init__(self, app: MagicMock) -> None:
        self._app = app

    @property
    def active_callback(self) -> Any:
        return self._app._active_callback

    @active_callback.setter
    def active_callback(self, value: Any) -> None:
        self._app._active_callback = value

    @property
    def pending_callbacks(self) -> Any:
        return self._app._pending_callbacks

    @pending_callbacks.setter
    def pending_callbacks(self, value: Any) -> None:
        self._app._pending_callbacks = value

    @property
    def pending_local_question(self) -> Any:
        return self._app._pending_local_question

    @pending_local_question.setter
    def pending_local_question(self, value: Any) -> None:
        self._app._pending_local_question = value

    @property
    def callback_submitting(self) -> Any:
        return self._app._callback_submitting

    @callback_submitting.setter
    def callback_submitting(self, value: Any) -> None:
        self._app._callback_submitting = value

    @property
    def callback_delivery_uncertain(self) -> Any:
        return self._app._callback_delivery_uncertain

    @callback_delivery_uncertain.setter
    def callback_delivery_uncertain(self, value: Any) -> None:
        self._app._callback_delivery_uncertain = value


def _callback_stand_in(app: MagicMock) -> Any:
    return _CallbackQuestionStandIn(app)


async def _wait_for_retry_error(app: ChartreuxApp, pilot) -> None:
    await _wait_until(
        pilot,
        lambda: (
            any("/retry" in str(error._error) for error in app.query(ErrorMessage))
            and app.event_handler is not None
            and app.event_handler._retry_presentation is not None
        ),
    )


async def _wait_for_retry_button(pilot, error_widget: ErrorMessage) -> Button:
    """Wait for the "Retry — new turn" button to mount on the error widget.

    Textual registers a mounted widget in the DOM before its compose()
    children mount, so the button needs its own settling wait: under
    full-suite load the widget's message pump can lag the turn-state
    updates the text/job waits observe (R1).
    """
    await _wait_until(
        pilot, lambda: len(error_widget.query(".error-retry-button")) == 1
    )
    return error_widget.query_one(".error-retry-button", Button)


@pytest.mark.asyncio
async def test_question_callback_opens_from_public_protocol() -> None:
    app = MagicMock()
    app._secondary_surface_active.return_value = False
    app._active_callback = None
    app._pending_local_question = None
    app._pending_callbacks = deque()
    app._wait_for_typing_pause = AsyncMock()
    app._switch_to_question_app = AsyncMock()
    callback = _question_callback("callback-1")

    await TurnSupervisor.show_callback(_callback_stand_in(app), callback)

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

    await TurnSupervisor.show_callback(_callback_stand_in(app), second)

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
    app._secondary_surface_active.return_value = False
    app._pending_local_question = None
    app._pending_callbacks = deque()
    app._wait_for_typing_pause = AsyncMock(side_effect=release.wait)
    app._switch_to_question_app = AsyncMock()

    first_task = asyncio.create_task(
        TurnSupervisor.show_callback(_callback_stand_in(app), first)
    )
    await asyncio.sleep(0)
    await TurnSupervisor.show_callback(_callback_stand_in(app), second)

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

    await TurnSupervisor.show_callback(_callback_stand_in(app), callback)

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
    app._callback_submitting = False
    app._callback_delivery_uncertain = False
    app._pending_callbacks = deque([second])
    app.app_server.respond_to_callback = AsyncMock()
    app._show_callback = AsyncMock()

    output = UserInputCallbackOutput(
        result=UserQuestionResult(answers=[], cancelled=True)
    )
    await TurnSupervisor.respond_to_active_callback(_callback_stand_in(app), output)

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

    await TurnSupervisor.question_answered(
        _callback_stand_in(app),
        QuestionApp.Answered([answer], callback_id="callback-1"),
    )

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
    app._secondary_surface_active.return_value = False
    app._pending_local_question = None
    app._pending_callbacks = deque()
    app._loading_widget = loading
    app._ensure_loading_widget = AsyncMock()
    app._wait_for_typing_pause = AsyncMock(side_effect=release_typing_pause.wait)
    app._switch_to_question_app = AsyncMock()

    show = asyncio.create_task(
        TurnSupervisor.show_callback(_callback_stand_in(app), callback)
    )
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
    app._callback_submitting = False
    app._callback_delivery_uncertain = False
    app._pending_callbacks = deque()
    app._loading_widget = loading
    app.app_server.respond_to_callback = AsyncMock()
    app._switch_to_input_app = AsyncMock()
    output = UserInputCallbackOutput(
        result=UserQuestionResult(answers=[], cancelled=True)
    )

    await TurnSupervisor.respond_to_active_callback(_callback_stand_in(app), output)

    assert loading.base_status == "Running command"
    assert loading._pause_start is None
    app._switch_to_input_app.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_local_question_waits_for_settings_draft_to_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app._show_settings()
        await _wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        screen = app.screen
        request = _question_callback("local-settings").detail.request
        task = asyncio.create_task(app._request_local_user_input(request))
        await _wait_until(
            pilot,
            lambda: (
                bool(screen.query("#settings-pending-action"))
                and screen.query_one("#settings-pending-action").display
            ),
        )
        assert app.screen is screen
        assert not app.query(QuestionApp)
        await pilot.press("escape")
        await _wait_until(pilot, lambda: bool(app.query(QuestionApp)))
        app.query_one(QuestionApp).action_cancel()
        assert (await task).cancelled


@pytest.mark.asyncio
async def test_providers_defers_incoming_decision_preserving_draft_and_bookmark() -> (
    None
):
    from copy import deepcopy

    from textual.widgets import Input, OptionList

    from chartreux.ui.providers.workbench import ProviderWorkbenchScreen

    app = build_test_chartreux_app()
    callback = _question_callback("callback-providers")
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        assert await app._handle_command("/providers")
        await _wait_until(
            pilot, lambda: isinstance(app.screen, ProviderWorkbenchScreen)
        )
        screen = app.screen
        assert isinstance(screen, ProviderWorkbenchScreen)
        await pilot.press("enter")
        assert screen.state is not None
        screen._select_action("base")
        screen.query_one("#wb-input", Input).value = "https://pending-draft.test/v1"
        await pilot.press("enter")
        fields = screen.query_one("#wb-actions", OptionList)
        fields.highlighted = fields.get_option_index("env")
        await pilot.pause()
        frame = screen._frame()
        payload = deepcopy(screen.state.changes())
        await app._handle_turn_event(CallbackRequested(callback))
        await pilot.pause()
        assert app.screen is screen
        assert screen.state.changes() == payload
        assert screen._frame() == frame
        assert fields.has_focus
        assert screen.query_one("#wb-pending-action").display
        assert app._active_callback is None
        assert not app.query(QuestionApp)
        # Discard only after verifying the host did not disturb the draft.
        screen._confirm = "discard"
        screen._update_help()
        screen.action_confirm_yes()
        await pilot.press("escape", "escape")
        await _wait_until(pilot, lambda: app._active_callback is callback)
        assert app.query(QuestionApp)


@pytest.mark.asyncio
async def test_settings_defers_incoming_decision_without_replacing_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    callback = _question_callback("callback-settings")
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        await app._show_settings()
        await _wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        screen = app.screen
        assert isinstance(screen, SettingsScreen)
        await pilot.press("m", "o", "d", "e", "l")
        await pilot.pause()
        draft = screen._filter
        await app._handle_turn_event(CallbackRequested(callback))
        await pilot.pause()
        assert app.screen is screen
        assert screen._filter == draft
        assert screen.query_one("#settings-pending-action").display
        assert not app.query(QuestionApp)
        await pilot.press("escape", "escape")
        await _wait_until(pilot, lambda: app._active_callback is callback)
        assert app.query(QuestionApp)
        response = AsyncMock()
        monkeypatch.setattr(app.app_server, "respond_to_callback", response)
        await app._respond_to_active_callback(
            UserInputCallbackOutput(
                result=UserQuestionResult(
                    answers=[UserAnswer(question="Ship it?", answer="No")],
                    cancelled=False,
                )
            )
        )
        response.assert_awaited_once()
        assert app._active_callback is None


@pytest.mark.asyncio
async def test_transcript_defers_incoming_decision_until_close() -> None:
    from chartreux.app_server.events import AgentsUpdate
    from chartreux.app_server.protocol import (
        AgentSummaryModel,
        AgentTranscriptGetResponse,
        AgentTranscriptState,
    )

    app = build_test_chartreux_app()
    callback = _question_callback("callback-agent")
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        app.app_server.resources.sessions.read_agent_transcript = AsyncMock(
            return_value=AgentTranscriptGetResponse(
                state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
            )
        )
        await app._handle_turn_event(
            AgentsUpdate([
                AgentSummaryModel(
                    agent_id="one",
                    profile="worker",
                    availability="idle",
                    name=derive_agent_name(
                        agent_id="one",
                        profile="worker",
                        profile_display_label=None,
                        base_model=None,
                        effective_model=None,
                    ),
                )
            ])
        )
        await pilot.press("ctrl+shift+a", "down", "enter")
        await _wait_until(pilot, lambda: bool(app.query(AgentTranscriptViewer)))
        await app._handle_turn_event(CallbackRequested(callback))
        assert app.query_one("#agent-transcript-pending-action").display
        assert app._active_callback is None
        await pilot.press("escape")
        await _wait_until(pilot, lambda: app._active_callback is callback)
        assert app.query(QuestionApp)


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
    supervisor = cast(Any, SimpleNamespace(_app=app))

    asyncio.run(
        TurnSupervisor._mount_turn_error(
            supervisor, _turn_error(TurnErrorCode.INVALID_API_KEY), "Invalid API key"
        )
    )

    app.event_handler.offer_retry.assert_not_called()


def test_rate_limit_error_still_offers_retry() -> None:
    app = MagicMock()
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()
    supervisor = cast(Any, SimpleNamespace(_app=app))

    asyncio.run(
        TurnSupervisor._mount_turn_error(
            supervisor, _turn_error(TurnErrorCode.RATE_LIMIT), "Rate limited"
        )
    )

    app.event_handler.offer_retry.assert_called_once()


def test_invalid_api_key_clears_stale_retry_state() -> None:
    app = MagicMock()
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()
    supervisor = cast(Any, SimpleNamespace(_app=app))

    asyncio.run(
        TurnSupervisor._mount_turn_error(
            supervisor, _turn_error(TurnErrorCode.INVALID_API_KEY), "Invalid API key"
        )
    )

    app.event_handler.cancel_retry_presentation.assert_called_once_with()


def test_backend_error_message_hints_at_retry() -> None:
    app = MagicMock()
    app._retry_hint = TurnSupervisor._retry_hint

    message = TurnSupervisor._resolve_turn_error_message(
        app, _turn_error(TurnErrorCode.BACKEND_ERROR)
    )

    assert "/retry [additional instructions]" in message
    assert "Network error" in message


def test_internal_error_message_does_not_hint_at_retry() -> None:
    app = MagicMock()

    message = TurnSupervisor._resolve_turn_error_message(
        app, _turn_error(TurnErrorCode.INTERNAL_ERROR)
    )

    assert "/retry" not in message


@pytest.mark.asyncio
async def test_incomplete_stream_retries_and_reuses_assistant_message() -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="Ran three dummy read-only", stop_reason=None)],
        [mock_llm_chunk(content=" tool calls successfully.")],
    ])
    # R2: the continuation is gated — enabled explicitly (round-2
    # amendment) with the manual availability clock advanced past the armed
    # cooldown, so the controller's timer admits the continuation at once.
    agent_loop = build_test_agent_loop(
        config=_enabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    app._root_recovery._sleep = _root_recovery_clock_sleep(agent_loop)

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
    # R2: the continuation is gated — enabled explicitly (round-2
    # amendment) with the manual availability clock advanced past the armed
    # cooldown, so the controller's timer admits the continuation at once.
    agent_loop = build_test_agent_loop(
        config=_enabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    app._root_recovery._sleep = _root_recovery_clock_sleep(agent_loop)

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
    # R2: the continuation is gated — enabled explicitly (round-2
    # amendment) with the manual availability clock advanced past the armed
    # cooldown, so the controller's timer admits the continuation at once.
    agent_loop = build_test_agent_loop(
        config=_enabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    app._root_recovery._sleep = _root_recovery_clock_sleep(agent_loop)

    try:
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            chat_input = app.query_one(ChatInputContainer)
            chat_input.post_message(ChatInputContainer.Submitted("first"))
            await _wait_until(pilot, backend.retry_started.is_set)

            await pilot.press("escape")
            # The interrupt settle tears the act() stream down through the
            # server (a few async hops); under parallel load the gated model
            # call was observed taking 5.1s to fail — the assertion is the
            # settled state, not the latency. 8.0s stays under the 10s
            # per-test ceiling with room for the post-settle steps.
            await _wait_until(pilot, lambda: not app._agent_job_active(), timeout=8.0)

            chat_input.post_message(ChatInputContainer.Submitted("next"))
            await _wait_until(pilot, lambda: len(backend.requests_messages) == 3)
            await _wait_until(
                pilot,
                lambda: any(
                    "after cancel" in message.get_content()
                    for message in app.query(AssistantMessage)
                ),
            )

            # The interrupted retry left no error behind: the event listener
            # stayed alive and the post-cancel turn rendered cleanly.
            assert len(app.query(ErrorMessage)) == 0
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
    # R2: the continuation is gated — enabled explicitly (round-2
    # amendment) with the manual availability clock advanced past the armed
    # cooldown, so the controller's timer admits each continuation at once.
    agent_loop = build_test_agent_loop(
        config=_enabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    app._root_recovery._sleep = _root_recovery_clock_sleep(agent_loop)

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
    # R2: enabled explicitly (round-2 amendment) so this exercises the GATE's
    # queued-work barrier — the schedule is declined, never silently rearmed.
    agent_loop = build_test_agent_loop(
        config=_enabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    app._root_recovery._sleep = _root_recovery_clock_sleep(agent_loop)
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
                app.query_one(SessionStatusLine).state.context_tokens
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
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
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
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
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
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
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


@pytest.mark.asyncio
async def test_dirty_proxy_preserved_until_ordered_decisions_present() -> None:
    from chartreux.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp

    app = build_test_chartreux_app()
    first = _question_callback("proxy-first")
    second = _question_callback("proxy-second")
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app._show_proxy_setup()
        proxy = app.query_one(ProxySetupApp)
        field = next(iter(proxy.inputs.values()))
        field.value = "http://dirty.example"
        await app._handle_turn_event(CallbackRequested(first))
        await app._handle_turn_event(CallbackRequested(first))
        await app._handle_turn_event(CallbackRequested(second))
        assert app.query_one(ProxySetupApp) is proxy
        assert field.value == "http://dirty.example"
        assert app.query_one("#bottom-pending-action").display
        assert [item.callback_id for item in app._pending_callbacks] == [
            "proxy-first",
            "proxy-second",
        ]
        proxy.post_message(ProxySetupApp.ProxySetupClosed(saved=False))
        await _wait_until(pilot, lambda: app._active_callback is first)
        assert not app.query(ProxySetupApp)
        assert app.query_one(QuestionApp).args is first.detail.request
        assert [item.callback_id for item in app._pending_callbacks] == ["proxy-second"]


@pytest.mark.asyncio
async def test_proxy_apply_waits_for_one_production_write_and_blocks_cancel() -> None:
    from textual.widgets import Button

    from chartreux.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp

    app = build_test_chartreux_app()
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_update(_changes: dict[str, str | None]) -> None:
        started.set()
        await release.wait()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app._show_proxy_setup()
        proxy = app.query_one(ProxySetupApp)
        next(iter(proxy.inputs.values())).value = "http://proxy.example"
        update = AsyncMock(side_effect=slow_update)
        app.app_server.resources.config.update_proxy = update

        proxy._save_and_close()
        await asyncio.wait_for(started.wait(), timeout=2)
        assert proxy.query_one("#proxysetup-save", Button).disabled
        assert proxy.query_one("#proxysetup-cancel", Button).disabled
        proxy._save_and_close()
        proxy.action_close()
        proxy.post_message(ProxySetupApp.ProxySetupClosed(saved=False))
        await asyncio.sleep(0)
        assert app.query_one(ProxySetupApp) is proxy
        assert update.await_count == 1

        release.set()
        await _wait_until(pilot, lambda: not app.query(ProxySetupApp))
        assert update.await_count == 1
        assert not any(
            "Proxy setup cancelled" in str(message._content)
            for message in app.query(UserCommandMessage)
        )


@pytest.mark.asyncio
async def test_question_focus_hides_global_loading_shortcuts() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await app._handle_turn_event(CallbackRequested(_question_callback("hints")))
        await _wait_until(pilot, lambda: bool(app.query(QuestionApp)))
        loading = app.query_one(LoadingWidget)
        assert loading.hint_widget is not None
        assert not loading.hint_widget.display

        await app._switch_to_input_app()
        assert loading.hint_widget.display


@pytest.mark.asyncio
async def test_inspection_rejected_while_question_draft_is_visible() -> None:
    from chartreux.app_server.events import AgentsUpdate
    from chartreux.app_server.protocol import AgentSummaryModel

    app = build_test_chartreux_app()
    callback = _question_callback("inspection")
    async with app.run_test():
        await app._handle_turn_event(
            AgentsUpdate([
                AgentSummaryModel(
                    agent_id="one",
                    profile="worker",
                    availability="idle",
                    name=derive_agent_name(
                        agent_id="one",
                        profile="worker",
                        profile_display_label=None,
                        base_model=None,
                        effective_model=None,
                    ),
                )
            ])
        )
        await app._handle_turn_event(CallbackRequested(callback))
        question = app.query_one(QuestionApp)
        question.selected_option = 1
        await app.action_toggle_agent_browser()
        app._submit_agent_selection("one")
        assert app.query_one(QuestionApp) is question
        assert question.selected_option == 1
        assert not app.query(AgentTranscriptViewer)
        assert any(
            "Finish or cancel this question first" in str(item._content)
            for item in app.query(UserCommandMessage)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("uncertain", [False, True])
async def test_callback_failure_retains_draft_and_blocks_uncertain_resend(
    uncertain: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from chartreux.app_server import AppServerConnectionClosed

    app = build_test_chartreux_app()
    callback = _question_callback("failed-answer")
    entered = asyncio.Event()
    release = asyncio.Event()
    failure = (
        AppServerConnectionClosed("disconnected")
        if uncertain
        else AppServerResponseError(
            ProtocolError(code=ProtocolErrorCode.INVALID_PARAMS, message="rejected")
        )
    )

    async def respond(*_args: object) -> None:
        entered.set()
        await release.wait()
        raise failure

    async with app.run_test():
        response = AsyncMock(side_effect=respond)
        monkeypatch.setattr(app.app_server, "respond_to_callback", response)
        await app._handle_turn_event(CallbackRequested(callback))
        question = app.query_one(QuestionApp)
        question.selected_option = 1
        answer = UserInputCallbackOutput(
            result=UserQuestionResult(
                answers=[UserAnswer(question="Ship it?", answer="No")], cancelled=False
            )
        )
        submitting = asyncio.create_task(app._respond_to_active_callback(answer))
        await entered.wait()
        await app._respond_to_active_callback(answer)
        assert response.await_count == 1
        release.set()
        await submitting
        assert app._active_callback is callback
        assert app.query_one(QuestionApp) is question
        assert question.selected_option == 1
        assert question._submission_status is not None
        assert ("Warning:" if uncertain else "Failed:") in question._submission_status
        if uncertain:
            await app._respond_to_active_callback(answer)
            assert response.await_count == 1
        else:
            monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
            await app._respond_to_active_callback(answer)
            assert app._active_callback is None


def test_retry_and_empty_mcp_copy_and_availability() -> None:
    from chartreux.cli.commands import CommandContext, CommandRegistry

    registry = CommandRegistry()
    assert not registry.has_command("retry")
    registry.refresh(CommandContext(retry_available=True))
    assert registry.has_command("retry")
    assert "new turn" in registry.get_help_text()
    assert "new turn" in TurnSupervisor._retry_hint()
    assert "may repeat" in build_retry_prompt("")


@pytest.mark.asyncio
async def test_empty_mcp_and_agents_show_supported_next_action() -> None:
    from textual.widgets import Static

    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        app.app_server.resources.mcp.read = AsyncMock(
            return_value=MagicMock(sources=[], statuses={})
        )
        await app._show_mcp()
        await app._show_mcp_status()
        await app.action_toggle_agent_browser()
        messages = [str(message._content) for message in app.query(UserCommandMessage)]
        assert sum("/mcp add <url>" in text for text in messages) == 2
        # The expanded browser stays reachable on an empty inventory and shows
        # its empty-state guidance in place of a command message (design §8).
        bar = app._agent_bar
        assert bar is not None
        await pilot.pause()
        assert bar.expanded
        rendered = str(bar.query_one("#agent-bar-content", Static).render())
        assert "No background agents" in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("mouse", [False, True])
@pytest.mark.parametrize("parent_running", [False, True])
async def test_confirmed_stop_calls_client_with_pinned_run_only(
    mouse: bool, parent_running: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_loop = build_test_agent_loop()
    parent_started, parent_interrupted = asyncio.Event(), asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        yield UserMessageEvent(content=msg, message_id="parent-user")
        parent_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            parent_interrupted.set()

    agent_loop.act = blocking_act
    app = build_test_chartreux_app(agent_loop=agent_loop)
    entered, release = asyncio.Event(), asyncio.Event()

    async def cancel(agent_id: str, run_id: str) -> AgentsCancelResponse:
        entered.set()
        await release.wait()
        return AgentsCancelResponse(outcome=CancelOutcome.STOP_REQUESTED, run_id=run_id)

    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        request = AsyncMock(side_effect=cancel)
        interrupt = AsyncMock()
        monkeypatch.setattr(app.app_server, "cancel_agent", request)
        monkeypatch.setattr(app.app_server, "interrupt", interrupt)
        if parent_running:
            app.query_one(ChatInputContainer).post_message(
                ChatInputContainer.Submitted("parent work")
            )
            await asyncio.wait_for(parent_started.wait(), 2)
            await _wait_until(pilot, lambda: app.app_server.turn_active)
        target = AgentSummaryModel(
            agent_id="one",
            profile="worker",
            availability="running",
            current_run_id="r1",
            name=derive_agent_name(
                agent_id="one",
                profile="worker",
                profile_display_label=None,
                base_model=None,
                effective_model=None,
            ),
        )
        sibling = target.model_copy(update={"agent_id": "two", "current_run_id": "s1"})
        await app._handle_turn_event(AgentsUpdate([target, sibling]))
        parent_active = app.app_server.turn_active
        assert parent_active is parent_running
        await pilot.press("ctrl+shift+a", "down", "c")
        request.assert_not_awaited()
        assert not app.query(AgentTranscriptViewer)
        if mouse:
            await pilot.click("#agent-stop-submit")
        else:
            await pilot.press("left", "enter")
        await asyncio.wait_for(entered.wait(), 2)
        bar = app._agent_bar
        assert bar is not None and app.screen.focused is bar
        assert bar.stop_is_pending("one", "r1")
        assert not bar.stop_is_pending("two", "s1")
        assert bar.agents == (target, sibling)
        await pilot.press("c")
        assert bar._stop_confirmation is None
        # Authoritative replacement arrives before the old RPC response.
        replacement = target.model_copy(update={"current_run_id": "r2"})
        await app._handle_turn_event(AgentsUpdate([replacement, sibling]))
        release.set()
        await pilot.pause()
        request.assert_awaited_once_with("one", "r1")
        assert not bar.stop_is_pending("one", "r2")
        assert bar.agents == (replacement, sibling)
        assert app.app_server.turn_active is parent_active
        assert not parent_interrupted.is_set()
        interrupt.assert_not_awaited()
        assert not app.query(AgentTranscriptViewer)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "forbidden"])
async def test_stop_failure_surfaces_toast_and_restores_browser(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        request = AsyncMock(
            side_effect=RuntimeError("connection lost")
            if failure == "transport"
            else None,
            return_value=AgentsCancelResponse(
                outcome=CancelOutcome.FORBIDDEN, run_id="r1"
            ),
        )
        notify = MagicMock()
        monkeypatch.setattr(app.app_server, "cancel_agent", request)
        monkeypatch.setattr(app, "notify", notify)
        await app._handle_turn_event(
            AgentsUpdate([
                AgentSummaryModel(
                    agent_id="one",
                    profile="worker",
                    availability="running",
                    current_run_id="r1",
                    name=derive_agent_name(
                        agent_id="one",
                        profile="worker",
                        profile_display_label=None,
                        base_model=None,
                        effective_model=None,
                    ),
                )
            ])
        )
        await pilot.press("ctrl+shift+a", "down", "c", "left", "enter")
        await _wait_until(pilot, lambda: notify.called)
        request.assert_awaited_once_with("one", "r1")
        notify.assert_called_once()
        text = notify.call_args.args[0]
        assert (
            "connection lost" if failure == "transport" else "Stop forbidden"
        ) in text
        assert "one, run r1" in text and "Retained output remains inspectable" in text
        assert notify.call_args.kwargs == {"severity": "error", "markup": False}
        bar = app._agent_bar
        assert bar is not None and bar.expanded and app.screen.focused is bar
        assert not bar.stop_is_pending("one", "r1")
        assert not app.query(AgentTranscriptViewer)
        await pilot.press("c")
        assert bar._stop_confirmation == ("one", "r1")


@pytest.mark.asyncio
async def test_escape_dismisses_stop_confirmation_before_browser_and_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    interrupt, cancel = MagicMock(), AsyncMock()
    monkeypatch.setattr(app, "_try_interrupt", interrupt)
    async with app.run_test() as pilot:
        await app._session_ready.wait()
        monkeypatch.setattr(app.app_server, "cancel_agent", cancel)
        await app._handle_turn_event(
            AgentsUpdate([
                AgentSummaryModel(
                    agent_id="one",
                    profile="worker",
                    availability="running",
                    current_run_id="r1",
                    name=derive_agent_name(
                        agent_id="one",
                        profile="worker",
                        profile_display_label=None,
                        base_model=None,
                        effective_model=None,
                    ),
                )
            ])
        )
        await pilot.press("ctrl+shift+a", "down", "c", "left", "escape")
        bar = app._agent_bar
        assert bar is not None and bar.expanded
        assert bar._stop_confirmation is None and app.screen.focused is bar
        cancel.assert_not_awaited()
        interrupt.assert_not_called()
        await pilot.press("escape")
        assert not bar.expanded
        interrupt.assert_not_called()
        await pilot.press("escape")
        interrupt.assert_called_once()


def _authority_callback(request) -> PublicCallbackEntry:
    return _callback(UserInputCallbackDetail(request=request))


@pytest.mark.asyncio
async def test_shell_permissions_panel_opens_from_settings_over_app_server(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Settings → Shell Permissions inspects and revokes over the real endpoints."""
    from types import SimpleNamespace

    from textual.widgets import OptionList

    from chartreux.cli.textual_ui.widgets.shell_permissions_app import (
        ShellPermissionsApp,
    )
    from chartreux.core.tools.builtins._shell_command_policy import (
        GitRepositoryRelocation,
        resolve_repository_scope,
    )
    from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    scope = resolve_repository_scope(project, GitRepositoryRelocation())
    assert scope is not None
    loop = build_test_agent_loop(cwd=project)
    loop._dangerous_command_grants.grant_session({("git.push", scope)})
    app = build_test_chartreux_app(agent_loop=loop)
    async with app.run_test(size=(100, 32)) as pilot:
        await app._session_ready.wait()
        await app._show_settings()
        await _wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press(*"shell")
        await pilot.pause()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ShellPermissionsApp))
        panel = app.screen
        assert isinstance(panel, ShellPermissionsApp)
        await _wait_until(pilot, lambda: panel._inspection is not None)
        grants = panel.query_one("#shell-permissions-grants", OptionList)
        row_ids = [str(row.id) for row in grants.options if row.id is not None]
        assert "live:git.push" in row_ids
        project_line = str(
            panel.query_one("#shell-permissions-project", NoMarkupStatic).content
        )
        assert str(project) in project_line
        # Revoke the live grant through the real endpoint.
        await pilot.press("enter")
        await _wait_until(
            pilot,
            lambda: (
                "live session authority removed"
                in str(
                    panel.query_one(
                        "#shell-permissions-feedback", NoMarkupStatic
                    ).content
                )
            ),
        )
        feedback = str(
            panel.query_one("#shell-permissions-feedback", NoMarkupStatic).content
        )
        assert "no saved removal attempted" in feedback
        # The session store no longer carries the authority.
        assert loop._dangerous_command_grants.dangerous_command_authorities() == ()
        # The refreshed list reconciles: the live row is gone.
        await _wait_until(
            pilot,
            lambda: (
                "live:git.push"
                not in [str(row.id) for row in grants.options if row.id is not None]
            ),
        )


@pytest.mark.asyncio
async def test_shell_permissions_panel_defers_incoming_question(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A question arriving while the panel is open defers, then presents."""
    from types import SimpleNamespace

    from chartreux.cli.textual_ui.widgets.shell_permissions_app import (
        ShellPermissionsApp,
    )
    from tests.stubs.fake_config_orchestrator import FakeConfigOrchestrator

    async def preview_candidate(self: FakeConfigOrchestrator):
        return SimpleNamespace(config=self.config)

    monkeypatch.setattr(FakeConfigOrchestrator, "preview_candidate", preview_candidate)
    app = build_test_chartreux_app()
    callback = _question_callback("callback-shell-permissions")
    async with app.run_test(size=(100, 32)) as pilot:
        await app._session_ready.wait()
        await app._show_settings()
        await _wait_until(pilot, lambda: isinstance(app.screen, SettingsScreen))
        await pilot.press(*"shell")
        await pilot.pause()
        await pilot.press("enter")
        await _wait_until(pilot, lambda: isinstance(app.screen, ShellPermissionsApp))
        panel = app.screen
        await app._handle_turn_event(CallbackRequested(callback))
        await pilot.pause()
        # The panel stays open and shows the pending-action notice; no question.
        assert app.screen is panel
        assert panel.query_one("#shell-permissions-pending-action").display
        assert not app.query(QuestionApp)
        # Escape the panel, then Settings (the first Escape clears the filter,
        # the second dismisses); the deferred question presents.
        await pilot.press("escape")
        await pilot.pause()
        assert isinstance(app.screen, SettingsScreen)
        await pilot.press("escape", "escape")
        await _wait_until(pilot, lambda: app._active_callback is callback)
        assert app.query(QuestionApp)


@pytest.mark.asyncio
async def test_dangerous_command_question_presents_safely_via_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The P4 dangerous-command question renders safely through the callback."""
    from pathlib import Path
    import shlex

    from chartreux.core.agent_loop._dangerous_command_grants import (
        DANGEROUS_COMMAND_DENY_LABEL,
        DANGEROUS_COMMAND_ONCE_LABEL,
        DANGEROUS_COMMAND_PROJECT_LABEL,
        DANGEROUS_COMMAND_SESSION_LABEL,
        dangerous_command_request,
    )
    from chartreux.core.dangerous_commands import (
        DangerousCommandProposal,
        RepositoryScope,
    )
    from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

    project = Path("/home/example/projects/shop")
    command = ("git", "push", "origin", "[red]main[/red]")
    proposal = DangerousCommandProposal(
        command=command,
        project=project,
        scope=RepositoryScope(git_dir=project / ".git", work_tree=project),
        operations=("git.push",),
        consequence="Publish local commits to the remote.",
    )
    request = dangerous_command_request(proposal)
    app = build_test_chartreux_app()
    callback = _authority_callback(request)
    async with app.run_test(size=(80, 24)) as pilot:
        await app._session_ready.wait()
        await app._handle_turn_event(CallbackRequested(callback))
        await _wait_until(pilot, lambda: bool(app.query(QuestionApp)))
        question = app.query_one(QuestionApp)
        title = question.query_one(".question-title", NoMarkupStatic)
        # The literal command — including markup-looking text — is displayed.
        assert str(title.content) == request.questions[0].question
        assert shlex.join(command) in str(title.content)
        assert "[red]main[/red]" in str(title.content)
        assert "not a standing intent to publish" in str(title.content)
        labels = [
            str(widget.content) for widget in question.option_widgets if widget.display
        ]
        assert len(labels) == 4
        assert DANGEROUS_COMMAND_DENY_LABEL in labels[0]
        assert DANGEROUS_COMMAND_ONCE_LABEL in labels[1]
        assert DANGEROUS_COMMAND_SESSION_LABEL in labels[2]
        assert DANGEROUS_COMMAND_PROJECT_LABEL in labels[3]
        # Deny is default-selected; answering it denies the command.
        assert question.selected_option == 0
        response = AsyncMock()
        monkeypatch.setattr(app.app_server, "respond_to_callback", response)
        question._mount_time = 0.0
        await pilot.press("enter")
        await _wait_until(pilot, lambda: response.await_count == 1)
        assert response.await_args is not None
        output = response.await_args.args[1]
        assert output.result.cancelled is False
        assert output.result.answers[0].answer == DANGEROUS_COMMAND_DENY_LABEL
        assert output.result.answers[0].question == request.questions[0].question


def _r0_single_deployment_config() -> ChartreuxConfigSchema:
    snapshot = CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {"test-first": {"api_base": "https://first.invalid"}},
            "models": {
                "base": {"deployments": [{"provider": "test-first", "name": "first"}]}
            },
            "roles": {},
        }),
        "test-revision",
    )
    # Q (WP6 promotion ripple): both callers characterize the MANUAL retry
    # surface (the terminal exhaustion error and its affordance), so root
    # recovery is explicitly disabled — the pre-promotion default.
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": snapshot}
    ).model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=False))
        }
    )
    return config.attach_catalog_snapshot(snapshot)


def _r0_connect_fault() -> BackendError:
    """A production-faithful transport fault (D1/L): the BackendError carries
    its httpx cause, so it classifies CONNECTION and the cooldown arms.
    """
    fault = BackendError(
        provider="test-first",
        endpoint="https://first.invalid/chat",
        status=None,
        reason="network boom",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="first",
        payload_summary=PayloadSummary(
            model="first",
            message_count=1,
            approx_chars=10,
            temperature=0,
            has_tools=False,
            tool_choice=None,
        ),
    )
    fault.__cause__ = httpx.ConnectError("network boom")
    return fault


@pytest.mark.asyncio
async def test_r0_all_deployments_unavailable_is_visible_terminal_state() -> None:
    """R0 acceptance 3 (TUI): all-deployments-unavailable is a VISIBLE
    terminal state.

    Turn 1 faults on the only deployment (a retryable backend error that
    offers /retry). Turn 2 is blocked at admission and surfaces an error
    naming the model and each deployment's exclusion reason — with no hidden
    automatic continuation: exactly one transport attempt exists across both
    turns.

    R1 (plan v0.5.6-recovery) layers the manual fallback ON TOP of R0's
    characterization: the terminal exhaustion error now carries the
    keyboard-accessible "Retry — new turn" affordance. R0 pinned its pre-R1
    absence ("NO /retry affordance today — R1's design territory"); R1 makes
    the exhaustion state retryable-for-the-affordance, with honest text that
    the deployments were unavailable and a manual retry resets the model's
    cooldowns.
    """
    backend = ControlledTransport([_r0_connect_fault()])
    agent_loop = build_test_agent_loop(
        config=_r0_single_deployment_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        # Turn 1: the only deployment faults — a retryable backend error
        # whose widget offers /retry.
        await _wait_until(
            pilot,
            lambda: any(
                "/retry" in str(error._error) for error in app.query(ErrorMessage)
            ),
        )
        await _wait_until(pilot, lambda: not app._agent_job_active())

        # Turn 2: blocked at admission — the visible terminal state.
        chat_input.post_message(ChatInputContainer.Submitted("again"))
        await _wait_until(
            pilot,
            lambda: any(
                "All deployments for model 'base' are unavailable" in str(error._error)
                for error in app.query(ErrorMessage)
            ),
        )
        await _wait_until(pilot, lambda: not app._agent_job_active())

        terminal = [
            error
            for error in app.query(ErrorMessage)
            if "All deployments for model 'base' are unavailable" in str(error._error)
        ]
        assert len(terminal) == 1
        terminal_text = str(terminal[0]._error)
        assert "test-first" in terminal_text
        assert "cooldown" in terminal_text
        # R1: the terminal exhaustion error carries the retry affordance —
        # the keyboard-accessible button on the widget, the retry hint, and
        # the honest text that a manual retry resets the model's cooldowns.
        assert terminal[0]._retryable is True
        button = await _wait_for_retry_button(pilot, terminal[0])
        assert str(button.label) == "Retry — new turn"
        assert "Retry — new turn" in terminal_text
        assert "resets the model's cooldowns" in terminal_text
        assert "recent actions may repeat" in terminal_text
        assert await wait_until(pilot, lambda: app.commands.has_command("retry"))
        # Still no automatic continuation: exactly one transport attempt
        # across both turns.
        assert len(backend.requests_messages) == 1


@pytest.mark.asyncio
async def test_r0_retry_command_starts_exactly_one_new_turn() -> None:
    """R0 acceptance 4 (TUI): manual /retry is ONE injected new turn.

    One /retry submission produces exactly one additional turn: one new
    transport attempt, an injected retry-prompt message, and no new user
    message. The retry's admission also resets the cooldown the interrupted
    stream armed — the attempt exists at all; without the user-initiated
    reset the retry would be blocked at admission (F6 path 3 pins the same
    block for the automatic continuation).
    """
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="Ran three dummy read-only")],
        [mock_llm_chunk(content=" tool calls successfully.")],
    ])
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    registry = agent_loop.config_orchestrator.availability_registry
    committed = agent_loop.committed_model
    assert committed is not None

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        # The interrupted stream armed the shared deployment cooldown.
        assert (
            registry.cooldown_until(committed.base_model, committed.provider)
            is not None
        )
        # The /retry command exists only once the failed turn's retry
        # presentation is offered; the turn job can finish before the offer
        # is processed, so wait for the command gate before submitting it.
        assert await wait_until(pilot, lambda: app.commands.has_command("retry"))

        chat_input.post_message(ChatInputContainer.Submitted("/retry"))
        await _wait_until(pilot, lambda: backend.streaming_attempts == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        # Exactly ONE new turn: one additional attempt, one assistant
        # message — and no new user message (the retry turn is injected).
        assert backend.streaming_attempts == 2
        assert len(app.query(AssistantMessage)) == 1
        assert [
            message._content
            for message in app.query(UserMessage)
            if not isinstance(message, SlashCommandMessage)
        ] == ["hi"]

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("")


@pytest.mark.asyncio
async def test_r0_incomplete_stream_bound_is_per_episode() -> None:
    """R0 acceptance 5 (TUI): the incomplete-stream continuation stays
    BOUNDED — at most ``_MAX_INCOMPLETE_STREAM_RETRIES`` automatic
    continuations per user turn (each an injected new turn carrying the
    retry prompt), then a visible error. A fresh user turn starts a fresh
    bounded episode rather than extending or resetting the shared counter
    globally.
    """
    assert _MAX_INCOMPLETE_STREAM_RETRIES == 2
    backend = FakeBackend([
        [mock_llm_chunk(content="first", stop_reason=None)],
        [mock_llm_chunk(content=" second", stop_reason=None)],
        [mock_llm_chunk(content=" third", stop_reason=None)],
        [mock_llm_chunk(content="fourth", stop_reason=None)],
        [mock_llm_chunk(content=" fifth", stop_reason=None)],
        [mock_llm_chunk(content=" sixth", stop_reason=None)],
    ])
    # R2: the bound now lives in the GATE's shared episode budget — enabled
    # explicitly (round-2 amendment) with the manual availability clock
    # advanced past the armed cooldown on each timer wake.
    agent_loop = build_test_agent_loop(
        config=_enabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    app._root_recovery._sleep = _root_recovery_clock_sleep(agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        # Episode 1: the turn plus two bounded continuations, then a
        # visible error offering /retry.
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 3)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        await _wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 1)
        assert "/retry" in str(app.query_one(ErrorMessage)._error)
        # Each continuation is an injected new turn carrying the retry
        # prompt — never a user message.
        for attempt in (1, 2):
            continuation = backend.requests_messages[attempt][-1]
            assert continuation.injected is True
            assert continuation.content == build_retry_prompt("")

        # Episode 2: a fresh user turn starts a fresh bounded episode.
        chat_input.post_message(ChatInputContainer.Submitted("again"))
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 6)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        await _wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 2)

    assert len(backend.requests_messages) == 6


@pytest.mark.asyncio
async def test_r1_retry_affordance_button_starts_exactly_one_new_turn() -> None:
    """R1 (TUI): the "Retry — new turn" affordance invokes the EXISTING retry
    operation.

    One activation of the button on the terminal error widget starts exactly
    one new turn: one additional transport attempt, an injected retry-prompt
    message, and no new user message. The intentional user-requested cooldown
    reset (the only path allowed to reset it) is what admits the retry past
    the armed cooldown. The completed-actions-may-repeat warning stays on the
    error text, and the button disables itself on activation.
    """
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="Ran three dummy read-only")],
        [mock_llm_chunk(content=" tool calls successfully.")],
    ])
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    registry = agent_loop.config_orchestrator.availability_registry
    committed = agent_loop.committed_model
    assert committed is not None

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        error_widget = app.query_one(ErrorMessage)
        # The affordance is displayed with the warning preserved.
        assert error_widget._retryable is True
        assert "recent actions may repeat" in str(error_widget._error)
        button = await _wait_for_retry_button(pilot, error_widget)
        assert str(button.label) == "Retry — new turn"

        # The interrupted stream armed the shared deployment cooldown.
        assert (
            registry.cooldown_until(committed.base_model, committed.provider)
            is not None
        )

        # Activate the affordance button.
        button.press()
        await _wait_until(pilot, lambda: backend.streaming_attempts == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        # The button disabled itself on activation.
        assert button.disabled is True
        # Exactly ONE new turn: one additional attempt, one assistant
        # message — and no new user message (the retry turn is injected).
        assert backend.streaming_attempts == 2
        assert len(app.query(AssistantMessage)) == 1
        assert [
            message._content
            for message in app.query(UserMessage)
            if not isinstance(message, SlashCommandMessage)
        ] == ["hi"]
        # The manual retry retained its intentional cooldown reset: the
        # retry was admitted past the armed cooldown, and the reset cleared
        # it.
        assert registry.cooldown_until(committed.base_model, committed.provider) is None
        # The terminal error widget (affordance included) was resolved away
        # once the retry turn produced output.
        assert len(app.query(ErrorMessage)) == 0

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("")


@pytest.mark.asyncio
async def test_r1_retry_affordance_keyboard_shortcut_starts_one_turn() -> None:
    """R1 (TUI): the affordance is keyboard-accessible — Ctrl+R invokes the
    EXISTING retry operation, gated exactly like /retry.

    The chat input's focus policy reclaims focus from transcript widgets, so
    the displayed button is pointer-activated and Ctrl+R is the keyboard
    path; the error text names both. With no retry offered the shortcut is a
    no-op; with the affordance offered it starts exactly one new turn with
    the intentional user-requested cooldown reset.
    """
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="partial")],
        [mock_llm_chunk(content="recovered")],
    ])
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    registry = agent_loop.config_orchestrator.availability_registry

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        # Gate: with no retry offered, Ctrl+R starts nothing.
        await pilot.press("ctrl+r")
        await pilot.pause(0.2)
        assert backend.streaming_attempts == 0
        assert not app.commands.has_command("retry")

        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        committed = agent_loop.committed_model
        assert committed is not None
        assert (
            registry.cooldown_until(committed.base_model, committed.provider)
            is not None
        )
        error_widget = app.query_one(ErrorMessage)
        # The error text names the keyboard path and the button.
        assert "Ctrl+R" in str(error_widget._error)
        assert "Retry — new turn" in str(error_widget._error)
        assert await wait_until(pilot, lambda: app.commands.has_command("retry"))

        # Keyboard activation starts exactly one new turn.
        await pilot.press("ctrl+r")
        await _wait_until(pilot, lambda: backend.streaming_attempts == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        assert backend.streaming_attempts == 2
        assert registry.cooldown_until(committed.base_model, committed.provider) is None

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("")


@pytest.mark.asyncio
async def test_r1_retry_affordance_double_activation_starts_one_turn() -> None:
    """R1 (TUI): double activation starts AT MOST one turn.

    Two rapid activations of the affordance — the widget-level guard (the
    button disables itself and posts once) and the operation-level guard (the
    retry operation's turn-ownership claim refuses a second turn) both apply.
    """
    backend = FakeInterruptedStreamingBackend([
        [mock_llm_chunk(content="partial")],
        [mock_llm_chunk(content="recovered")],
    ])
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        error_widget = app.query_one(ErrorMessage)
        button = await _wait_for_retry_button(pilot, error_widget)

        # Two rapid activations: a real button press plus a directly posted
        # affordance message (exercising the operation-level guard).
        button.press()
        error_widget.post_message(ErrorRetryRequested())
        await _wait_until(pilot, lambda: backend.streaming_attempts == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())
        await pilot.pause(0.2)

        # At most one new turn started.
        assert backend.streaming_attempts == 2
        assert button.disabled is True

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("")


@pytest.mark.asyncio
async def test_r1_exhaustion_affordance_retries_with_cooldown_reset() -> None:
    """R1 (TUI): the affordance's retryable set INCLUDES
    exhaustion-after-cooldown — the terminal all-deployments-unavailable
    error offers "Retry — new turn", and activating it starts exactly one
    new turn.

    The retry is a genuine user action, so the EXISTING retry operation's
    intentional cooldown reset (the only path allowed to reset it) admits the
    retry past the armed cooldown: without the reset the turn would be
    blocked at admission (zero additional transport attempts). The error text
    stays honest that the deployments were unavailable and that a manual
    retry resets the model's cooldowns.
    """
    backend = ControlledTransport([
        _r0_connect_fault(),
        [mock_llm_chunk(content="recovered")],
    ])
    agent_loop = build_test_agent_loop(
        config=_r0_single_deployment_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    registry = agent_loop.config_orchestrator.availability_registry

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        # Turn 1: the only deployment faults — a retryable backend error.
        await _wait_until(
            pilot,
            lambda: any(
                "/retry" in str(error._error) for error in app.query(ErrorMessage)
            ),
        )
        await _wait_until(pilot, lambda: not app._agent_job_active())
        committed = agent_loop.committed_model
        assert committed is not None
        # The fault armed the shared deployment cooldown.
        assert (
            registry.cooldown_until(committed.base_model, committed.provider)
            is not None
        )

        # Turn 2: blocked at admission — the terminal exhaustion error,
        # which (R1) carries the affordance with honest text.
        chat_input.post_message(ChatInputContainer.Submitted("again"))
        await _wait_until(
            pilot,
            lambda: any(
                "All deployments for model 'base' are unavailable" in str(error._error)
                for error in app.query(ErrorMessage)
            ),
        )
        await _wait_until(pilot, lambda: not app._agent_job_active())
        assert len(backend.requests_messages) == 1

        terminal = [
            error
            for error in app.query(ErrorMessage)
            if "All deployments for model 'base' are unavailable" in str(error._error)
        ]
        assert len(terminal) == 1
        terminal_text = str(terminal[0]._error)
        assert "test-first" in terminal_text
        assert "cooldown" in terminal_text
        assert "Retry — new turn" in terminal_text
        assert "resets the model's cooldowns" in terminal_text
        assert "recent actions may repeat" in terminal_text
        assert terminal[0]._retryable is True
        await _wait_for_retry_button(pilot, terminal[0])
        assert await wait_until(pilot, lambda: app.commands.has_command("retry"))

        # Activate the affordance (keyboard path): the user-initiated retry
        # resets the cooldown and is admitted — exactly one new turn.
        await pilot.press("ctrl+r")
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        assert len(backend.requests_messages) == 2
        assert registry.cooldown_until(committed.base_model, committed.provider) is None
        assert app.query_one(AssistantMessage).get_content() == "recovered"

    retry_message = backend.requests_messages[-1][-1]
    assert retry_message.injected is True
    assert retry_message.content == build_retry_prompt("")


@pytest.mark.asyncio
async def test_r1_no_retry_action_with_unresolved_runtime_approval() -> None:
    """R1 (TUI): no retry action while an unresolved runtime approval is
    present — activating the affordance under an open callback starts no
    turn.
    """
    backend = FakeInterruptedStreamingBackend([[mock_llm_chunk(content="partial")]])
    agent_loop = build_test_agent_loop(
        config=_disabled_root_recovery_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))
        await _wait_for_retry_error(app, pilot)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        error_widget = app.query_one(ErrorMessage)
        button = await _wait_for_retry_button(pilot, error_widget)

        # An unresolved runtime approval arrives while the affordance is shown.
        await app._handle_turn_event(CallbackRequested(_question_callback("cb-1")))
        await _wait_until(pilot, lambda: app._active_callback is not None)

        button.press()
        await pilot.pause(0.2)

        # No retry turn started: still exactly one transport attempt, and
        # the approval is still unresolved.
        assert backend.streaming_attempts == 1
        assert app._active_callback is not None


@pytest.mark.asyncio
async def test_r1_disconnect_offers_inspection_guidance_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R1 (TUI): a transport disconnect offers inspection/reconnect guidance,
    not automatic continuation — the error widget carries the guidance text,
    no retry affordance, and no new turn starts.

    The disconnect is injected at the client-driven turn path (the same path
    the retry operation drives): ``app_server.act``'s event stream raises
    ``AppServerConnectionClosed`` on first read, so the turn fails with a
    generic (non-typed) exception.
    """
    from chartreux.app_server import AppServerConnectionClosed

    app = build_test_chartreux_app()
    act_calls = 0

    def _disconnecting_act(*_args: object, **_kwargs: object):
        nonlocal act_calls
        act_calls += 1

        async def _stream():
            for _item in ():
                yield _item
            raise AppServerConnectionClosed("App server connection closed")

        return _stream()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        monkeypatch.setattr(app.app_server, "act", _disconnecting_act)
        await pilot.pause(0.1)
        turn = asyncio.create_task(app._handle_turn("hi"))
        await _wait_until(pilot, lambda: len(app.query(ErrorMessage)) == 1)
        await turn
        await _wait_until(pilot, lambda: not app._agent_job_active())

        error_widget = app.query_one(ErrorMessage)
        text = str(error_widget._error)
        assert "App server connection closed" in text
        assert "Inspect the session and server state" in text
        assert "reconnects automatically" in text
        assert "Nothing is continued automatically" in text
        # No retry affordance, no retry hint, no automatic continuation.
        assert error_widget._retryable is False
        assert len(error_widget.query(".error-retry-button")) == 0
        assert "/retry" not in text
        assert "Retry — new turn" not in text
        assert "Ctrl+R" not in text
        assert app.event_handler is not None
        assert app.event_handler._retry_presentation is None
        assert not app.commands.has_command("retry")
        await pilot.pause(0.2)
        assert act_calls == 1
