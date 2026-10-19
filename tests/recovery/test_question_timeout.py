"""Unattended ask_user_question expiry — end-to-end behavior (plan U/OD1, stage 3).

The authoritative timer lives beside the app-server callback record, so these
tests drive real app-server sessions over an in-process transport with a real
agent loop and a scripted fake backend. All time-based behavior uses a manual
monotonic clock and a gated wakeup patched into the session's TurnController
(the production ``_clock``/``_wakeup`` seams): no test sleeps through a real
deadline.

Coverage maps to the plan's U acceptance matrix: (1) disabled config (set
EXPLICITLY per the round-2 amendment), an absent unattended-default marker,
and mixed/protected batches never expire; (2) eligible silence produces
exactly one unanswered result and one deferred record; (3)
answer/expiry/keep-waiting/cancellation races have exactly one winner; (4) no
late answer changes a result or answers a newer callback; (5) root-access
approvals stay unanswered indefinitely with no side effect (the
dangerous-command half lives in
tests/app_server/test_dangerous_command_acceptance.py); (6) server-side expiry
works with a detached client and a new client sees terminal history, not a
ghost callback; (7) a late answer gets a definite expired response and may
continue as a new user message; (8) compaction carries the deferred record
across the structural boundary; (9) timeout causes no tool auto-retry or
repeated asking. The [user_questions] config validation and the admission
matrix are pinned at unit level; the TUI half of (7)/(10) lives in
tests/cli/textual_ui/test_app_server_requests.py.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
import pytest

from chartreux.app_server.client import AppServerClient
from chartreux.app_server.events import CallbackRequested, HistoryEntryUpdated
from chartreux.app_server.models import (
    AnsweredCallbackState,
    CancelledCallbackState,
    CompletedEffectState,
    ExpiredCallbackState,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicEffectEntry,
    PublicMessageEntry,
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    CallbackResultError,
    ClientCapabilities,
    ClientInfo,
    ProtocolErrorCode,
)
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop._root_grants import root_grant_request
from chartreux.core.config.models import UserQuestionsConfig
from chartreux.core.deferred_questions import DeferredQuestionStore
from chartreux.core.events import UserInputOrigin, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.tools.builtins.ask_user_question import AskUserQuestionArgs
from chartreux.questions import (
    QuestionChoice,
    QuestionKind,
    UnansweredQuestions,
    UnattendedDefaultAction,
)
from tests.app_server._session_runtime_support import _consume, _next_event
from tests.app_server._subagent_test_support import _GatedWakeup, _ManualClock
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import build_test_app_server, legacy_backend
from tests.stubs.fake_backend import FakeBackend

if TYPE_CHECKING:
    from chartreux.app_server.server import AppServer
    from chartreux.core.agent_loop import AgentLoop
    from chartreux.core.config import ChartreuxConfigSchema

_DEFAULT_TIMEOUT = 60.0
_MAX_TIMEOUT = 120.0
_QUESTION_TEXT = "Ship it?"
_DEFAULT_DESCRIPTION = "Defer the release until the user answers."


def _config_with_user_questions(
    *,
    enabled: bool,
    default_timeout: float = _DEFAULT_TIMEOUT,
    max_timeout: float = _MAX_TIMEOUT,
) -> ChartreuxConfigSchema:
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    return config.model_copy(
        update={
            "user_questions": UserQuestionsConfig(
                unattended_timeout_enabled=enabled,
                default_timeout_seconds=default_timeout,
                max_timeout_seconds=max_timeout,
            )
        }
    )


def _enabled_config() -> ChartreuxConfigSchema:
    """[user_questions] enabled with a 60s default / 120s ceiling."""
    return _config_with_user_questions(enabled=True)


def _explicitly_disabled_config() -> ChartreuxConfigSchema:
    """Expiry disabled EXPLICITLY.

    The round-2 amendment: disabled-state tests never rely on the development
    default (WP6 promotes it per OD4b and must not break these tests).
    """
    return _config_with_user_questions(enabled=False)


def _defaultable_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": _QUESTION_TEXT,
        "options": [{"label": "Yes"}, {"label": "No"}],
        "kind": "preference",
        "unattended_default": {
            "action": "defer_affected_work",
            "description": _DEFAULT_DESCRIPTION,
        },
    }
    payload.update(overrides)
    return payload


def _requires_user_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "question": "Authorize this?",
        "options": [{"label": "Yes"}, {"label": "No"}],
    }
    payload.update(overrides)
    return payload


def _question_tool_call(
    questions: list[dict[str, Any]],
    *,
    call_id: str = "question-1",
    timeout: float | None = None,
) -> ToolCall:
    arguments: dict[str, Any] = {"questions": questions}
    if timeout is not None:
        arguments["timeout_seconds"] = timeout
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="ask_user_question", arguments=json.dumps(arguments)
        ),
    )


def _read_tool_call(target: Path, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="read_file", arguments=json.dumps({"file_path": str(target)})
        ),
    )


def _request_event(
    questions: list[dict[str, Any]],
    *,
    origin: UserInputOrigin = UserInputOrigin.MODEL_QUESTION,
    request_id: str = "req-1",
    timeout: float | None = None,
) -> UserInputRequestEvent:
    arguments: dict[str, Any] = {"questions": questions}
    if timeout is not None:
        arguments["timeout_seconds"] = timeout
    return UserInputRequestEvent(
        request_id=request_id,
        args=AskUserQuestionArgs.model_validate(arguments),
        tool_call_id="tool-1",
        origin=origin,
    )


def _raw_request_event(
    args: Any,
    *,
    origin: UserInputOrigin = UserInputOrigin.MODEL_QUESTION,
    request_id: str = "req-1",
) -> UserInputRequestEvent:
    return UserInputRequestEvent(
        request_id=request_id, args=args, tool_call_id="tool-1", origin=origin
    )


def _answer_output(text: str = "Yes") -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question=_QUESTION_TEXT, answer=text)]
        )
    )


def _cancelled_output() -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(answers=[], cancelled=True)
    )


async def _open_session(agent_loop: AgentLoop) -> tuple[AppServer, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)
    session = await AppServerSession.start(
        AppServerClient(client_transport, run_peer=server.serve),
        client_info=ClientInfo(name="question-timeout-test", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
    )
    return server, session


async def _open_reconnectable_session(
    agent_loop: AgentLoop, *, reconnect_gate: asyncio.Event
) -> tuple[AppServer, AppServerSession]:
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(agent_loop, server_transport)

    def make_client(
        next_client_transport, next_server_transport, *, wait_for_gate: bool
    ) -> AppServerClient:
        async def serve_connection() -> None:
            if wait_for_gate:
                await reconnect_gate.wait()
            await server.serve_connection(
                next_server_transport, close_on_disconnect=False
            )

        return AppServerClient(next_client_transport, run_peer=serve_connection)

    session = await AppServerSession.start(
        make_client(client_transport, server_transport, wait_for_gate=False),
        client_info=ClientInfo(name="question-timeout-reconnect", version="0"),
        capabilities=ClientCapabilities(callback_kinds=["user_input"]),
        client_factory=lambda: make_client(
            *memory_transport_pair(), wait_for_gate=True
        ),
    )
    return server, session


@dataclass
class _SessionContext:
    loop: AgentLoop
    session: AppServerSession
    clock: _ManualClock
    wakeup: _GatedWakeup


def _patch_timer(server: AppServer, clock: _ManualClock, wakeup: _GatedWakeup) -> None:
    """Patch the TurnController's clock/wakeup seams (the plan U test seam)."""
    turns = legacy_backend(server).session.turns
    turns._clock = clock
    turns._wakeup = wakeup


def _release_gates(wakeup: _GatedWakeup) -> None:
    for gate in wakeup.gates:
        gate.set()


async def _wait_until(predicate: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for the server-side state")
        await asyncio.sleep(0.01)


@asynccontextmanager
async def _patched_session(config, backend, **loop_kwargs):
    """A real session whose TurnController runs on a manual clock."""
    loop = build_test_agent_loop(config=config, backend=backend, **loop_kwargs)
    server, session = await _open_session(loop)
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    _patch_timer(server, clock, wakeup)
    try:
        yield _SessionContext(loop=loop, session=session, clock=clock, wakeup=wakeup)
    finally:
        await session.close()
        await loop.aclose()


@asynccontextmanager
async def _admission_turns(config, **loop_kwargs):
    """A live TurnController for the admission matrix (no turn is driven)."""
    loop = build_test_agent_loop(config=config, **loop_kwargs)
    server, session = await _open_session(loop)
    try:
        yield loop, legacy_backend(server).session.turns
    finally:
        await session.close()
        await loop.aclose()


def _callback_entry_for(history: list[Any], callback_id: str) -> PublicCallbackEntry:
    matches = [
        entry
        for entry in history
        if isinstance(entry, PublicCallbackEntry) and entry.callback_id == callback_id
    ]
    assert len(matches) == 1
    return matches[0]


def _callback_entry(history: list[Any]) -> PublicCallbackEntry:
    matches = [entry for entry in history if isinstance(entry, PublicCallbackEntry)]
    assert len(matches) == 1
    return matches[0]


def _effect_entries(history: list[Any]) -> list[PublicEffectEntry]:
    return [entry for entry in history if isinstance(entry, PublicEffectEntry)]


def _unanswered_from_effect(effect: PublicEffectEntry) -> UnansweredQuestions:
    assert isinstance(effect.state, CompletedEffectState)
    output = effect.state.output
    assert isinstance(output, dict)
    return UnansweredQuestions.model_validate(output)


def _completed_effect_output(history: list[Any]) -> Any:
    """The single completed effect's output payload (narrows the state union)."""
    effects = _effect_entries(history)
    assert len(effects) == 1
    assert isinstance(effects[0].state, CompletedEffectState)
    return effects[0].state.output


class TestUserQuestionsConfig:
    """[user_questions] strict validation (plan U)."""

    def test_defaults_enable_expiry_since_the_v056_promotion(self) -> None:
        # WP6 default promotion (OD4b): the shipped default is enabled.
        config = UserQuestionsConfig()
        assert config.unattended_timeout_enabled is True
        assert config.default_timeout_seconds == 300.0
        assert config.max_timeout_seconds == 3600.0
        # The schema default is enabled too.
        assert build_test_vibe_config().user_questions.unattended_timeout_enabled is (
            True
        )

    def test_schema_section_is_typed_and_validated(self) -> None:
        config = build_test_vibe_config(
            user_questions={
                "unattended_timeout_enabled": True,
                "default_timeout_seconds": 60.0,
                "max_timeout_seconds": 120.0,
            }
        )
        assert config.user_questions.unattended_timeout_enabled is True
        assert config.user_questions.default_timeout_seconds == 60.0
        assert config.user_questions.max_timeout_seconds == 120.0
        with pytest.raises(ValidationError):
            build_test_vibe_config(user_questions={"default_timeout_seconds": 1.0})

    def test_rejects_boolean_as_number(self) -> None:
        with pytest.raises(ValidationError):
            UserQuestionsConfig(default_timeout_seconds=True)
        with pytest.raises(ValidationError):
            UserQuestionsConfig(max_timeout_seconds=False)
        with pytest.raises(ValidationError):
            UserQuestionsConfig.model_validate({"unattended_timeout_enabled": 1})

    def test_rejects_non_finite_durations(self) -> None:
        with pytest.raises(ValidationError):
            UserQuestionsConfig(default_timeout_seconds=float("nan"))
        with pytest.raises(ValidationError):
            UserQuestionsConfig(max_timeout_seconds=float("inf"))

    def test_rejects_zero_and_negative(self) -> None:
        with pytest.raises(ValidationError):
            UserQuestionsConfig(default_timeout_seconds=0.0)
        with pytest.raises(ValidationError):
            UserQuestionsConfig(max_timeout_seconds=-1.0)

    def test_rejects_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            UserQuestionsConfig(default_timeout_seconds=59.0)
        with pytest.raises(ValidationError):
            UserQuestionsConfig(max_timeout_seconds=3601.0)

    def test_rejects_default_above_max(self) -> None:
        with pytest.raises(ValidationError):
            UserQuestionsConfig(default_timeout_seconds=120.0, max_timeout_seconds=60.0)

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            UserQuestionsConfig.model_validate({"unknown_field": 1})


class TestExpiryAdmission:
    """The admission matrix: every gate is provenance- or policy-based."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_model_question_with_declared_defaults_is_admitted(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            expiry = turns._unattended_expiry(_request_event([_defaultable_payload()]))
            assert expiry is not None
            assert expiry.timeout_seconds == _DEFAULT_TIMEOUT
            assert len(expiry.questions) == 1
            echoed = expiry.questions[0]
            assert echoed.question_id == "req-1:0"
            assert echoed.question == _QUESTION_TEXT
            assert echoed.kind is QuestionKind.PREFERENCE
            assert (
                echoed.unattended_default.action
                is UnattendedDefaultAction.DEFER_AFFECTED_WORK
            )
            assert echoed.unattended_default.description == _DEFAULT_DESCRIPTION

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_runtime_approval_origin_is_permanently_ineligible(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            # An explicit runtime approval is permanently ineligible...
            assert (
                turns._unattended_expiry(
                    _request_event(
                        [_defaultable_payload()],
                        origin=UserInputOrigin.RUNTIME_APPROVAL,
                    )
                )
                is None
            )
            # ...including the real root-grant approval request shape.
            assert (
                turns._unattended_expiry(
                    _raw_request_event(
                        root_grant_request(Path("/tmp/some-root")),
                        origin=UserInputOrigin.RUNTIME_APPROVAL,
                    )
                )
                is None
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_default_origin_fails_closed(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            event = UserInputRequestEvent(
                request_id="req-1",
                args=AskUserQuestionArgs.model_validate({
                    "questions": [_defaultable_payload()]
                }),
                tool_call_id="tool-1",
            )
            assert event.origin is UserInputOrigin.RUNTIME_APPROVAL
            assert turns._unattended_expiry(event) is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_explicitly_disabled_config_never_admits(self) -> None:
        async with _admission_turns(
            _explicitly_disabled_config(), user_input_capability=True
        ) as (_loop, turns):
            assert (
                turns._unattended_expiry(_request_event([_defaultable_payload()]))
                is None
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_missing_default_never_admits(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            payload = _defaultable_payload()
            del payload["unattended_default"]
            assert turns._unattended_expiry(_request_event([payload])) is None

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_requires_user_blocks_the_whole_batch(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            # A requires_user question never expires, even with a default.
            protected = _requires_user_payload(
                kind="requires_user",
                unattended_default={
                    "action": "defer_affected_work",
                    "description": "deferred",
                },
            )
            assert turns._unattended_expiry(_request_event([protected])) is None
            # A mixed batch (one defaultable, one protected) never expires.
            assert (
                turns._unattended_expiry(
                    _request_event([_defaultable_payload(), _requires_user_payload()])
                )
                is None
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_subagent_loop_never_admits(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            loop,
            turns,
        ):
            # The gate reads the loop's subagent flag; a runtime child loop is
            # built with it set, and its questions never expire.
            loop._is_subagent = True
            assert (
                turns._unattended_expiry(_request_event([_defaultable_payload()]))
                is None
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_loop_without_user_input_capability_never_admits(self) -> None:
        async with _admission_turns(_enabled_config()) as (_loop, turns):
            # user_input_capability defaults to False: the config alone is
            # never enough to arm an expiry.
            assert (
                turns._unattended_expiry(_request_event([_defaultable_payload()]))
                is None
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_non_question_request_never_admits(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            assert (
                turns._unattended_expiry(_raw_request_event(QuestionChoice(label="x")))
                is None
            )

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_timeout_override_within_range_is_honored(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            expiry = turns._unattended_expiry(
                _request_event([_defaultable_payload()], timeout=90.0)
            )
            assert expiry is not None
            assert expiry.timeout_seconds == 90.0
            # The configured bounds are inclusive.
            at_max = turns._unattended_expiry(
                _request_event([_defaultable_payload()], timeout=_MAX_TIMEOUT)
            )
            assert at_max is not None
            assert at_max.timeout_seconds == _MAX_TIMEOUT
            at_min = turns._unattended_expiry(
                _request_event([_defaultable_payload()], timeout=60.0)
            )
            assert at_min is not None
            assert at_min.timeout_seconds == 60.0

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_timeout_override_outside_range_is_rejected_not_clamped(self) -> None:
        async with _admission_turns(_enabled_config(), user_input_capability=True) as (
            _loop,
            turns,
        ):
            with pytest.raises(ValueError, match="rejected, not clamped"):
                turns._unattended_expiry(
                    _request_event([_defaultable_payload()], timeout=30.0)
                )
            with pytest.raises(ValueError, match="rejected, not clamped"):
                turns._unattended_expiry(
                    _request_event([_defaultable_payload()], timeout=_MAX_TIMEOUT + 1.0)
                )


class TestEligibleSilenceExpires:
    """Acceptance (2): eligible silence produces exactly one unanswered
    result and exactly one deferred record; expiry never approves anything.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_silence_expires_into_one_unanswered_result_and_one_record(
        self,
    ) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            events: list[Any] = []
            callback: PublicCallbackEntry | None = None
            async for event in ctx.session.act("ask me"):
                events.append(event)
                if callback is None and isinstance(event, CallbackRequested):
                    callback = event.callback
                    # The public expiry snapshot rides the callback detail.
                    expiry = callback.detail.unattended_expiry
                    assert expiry is not None
                    assert expiry.timeout_seconds == _DEFAULT_TIMEOUT
                    assert [q.question_id for q in expiry.questions] == [
                        f"{callback.callback_id}:0"
                    ]
                    # Wait until the timer is armed and parked on the gate.
                    await _wait_until(lambda: bool(ctx.wakeup.calls))
                    assert ctx.wakeup.calls == [_DEFAULT_TIMEOUT]
                    # Silence: advance past the deadline and release the timer.
                    ctx.clock.advance(_DEFAULT_TIMEOUT + 1.0)
                    _release_gates(ctx.wakeup)
            assert callback is not None
            # Exactly one unanswered result: the tool effect output IS the
            # server-owned disposition, with no fabricated answer.
            effects = _effect_entries(ctx.session.history)
            assert len(effects) == 1
            unanswered = _unanswered_from_effect(effects[0])
            assert unanswered.disposition == "unanswered"
            assert unanswered.reason == "unanswered_timeout"
            assert unanswered.answers == []
            assert unanswered.cancelled is False
            assert unanswered.timeout_seconds == _DEFAULT_TIMEOUT
            assert unanswered.elapsed_seconds == _DEFAULT_TIMEOUT + 1.0
            assert unanswered.questions[0].question_id == f"{callback.callback_id}:0"
            assert unanswered.continuation_guidance
            # The closure is published: terminal expired history, no live
            # callback, and the client saw the terminal update.
            entry = _callback_entry(ctx.session.history)
            assert isinstance(entry.state, ExpiredCallbackState)
            assert entry.state.reason == "unanswered_timeout"
            assert ctx.session.state.active_callbacks == []
            assert any(
                isinstance(event, HistoryEntryUpdated)
                and isinstance(event.entry, PublicCallbackEntry)
                and event.entry.callback_id == callback.callback_id
                for event in events
            )
            # Exactly one deferred record, keyed by session and request.
            records = ctx.loop.deferred_questions.records()
            assert len(records) == 1
            record = records[0]
            assert record.session_id == ctx.loop.session_id
            assert record.request_id == callback.callback_id
            assert record.question_ids == [f"{callback.callback_id}:0"]
            assert record.timeout_seconds == _DEFAULT_TIMEOUT
            assert record.elapsed_seconds == _DEFAULT_TIMEOUT + 1.0
            assert record.reason == "unanswered_timeout"
            assert _DEFAULT_DESCRIPTION in record.affected_work
            # The turn completed normally after the deferral.
            assert any(
                isinstance(entry, PublicMessageEntry)
                and entry.role == "assistant"
                and "Done" in entry.text
                for entry in ctx.session.history
            )


class TestIneligibleQuestionsNeverExpire:
    """Acceptance (1): disabled config (explicit), absent marker, and mixed
    protected batches never expire.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_explicitly_disabled_config_never_expires(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _explicitly_disabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            # No unattended-expiry policy is attached to the callback...
            assert callback.detail.unattended_expiry is None
            # ...and no timer is armed at all.
            await asyncio.sleep(0.05)
            assert ctx.wakeup.calls == []
            # Silence far past any deadline: the question stays open.
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(
                _callback_entry(ctx.session.history).state, OpenCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []
            # The user can still answer normally.
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "answered"
            assert result.answers[0].answer == "Yes"
            assert isinstance(
                _callback_entry(ctx.session.history).state, AnsweredCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_absent_unattended_default_never_expires(self) -> None:
        # A defaultable kind WITHOUT the unattended-default marker: the
        # per-question opt-in is absent, so the question never expires.
        payload = _defaultable_payload()
        del payload["unattended_default"]
        backend = FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_question_tool_call([payload])])],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            assert callback.detail.unattended_expiry is None
            await asyncio.sleep(0.05)
            assert ctx.wakeup.calls == []
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert ctx.loop.deferred_questions.records() == []
            # Cancelling still works and produces the cancelled disposition.
            await ctx.session.respond_to_callback(
                callback.callback_id, _cancelled_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "cancelled"
            # An answered-with-cancellation result lands as an answered
            # callback carrying the cancelled disposition (CancelledCallbackState
            # is reserved for callbacks left unresolved at turn finalize).
            entry = _callback_entry(ctx.session.history)
            assert isinstance(entry.state, AnsweredCallbackState)
            assert isinstance(entry.state.output, UserInputCallbackOutput)
            assert entry.state.output.result.disposition == "cancelled"
            assert ctx.loop.deferred_questions.records() == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_mixed_protected_batch_never_expires(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _question_tool_call([
                            _defaultable_payload(),
                            _requires_user_payload(),
                        ])
                    ],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            assert callback.detail.unattended_expiry is None
            await asyncio.sleep(0.05)
            assert ctx.wakeup.calls == []
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert ctx.loop.deferred_questions.records() == []
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "answered"
            assert ctx.loop.deferred_questions.records() == []


class TestArbitrationRaces:
    """Acceptance (3): answer/expiry/keep-waiting/cancellation races have
    exactly one winner; the loser leaves no trace.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_answer_before_the_deadline_wins(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            # Let the timer arm and park on the gate before answering.
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            assert ctx.wakeup.calls == [_DEFAULT_TIMEOUT]
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            assert isinstance(
                _callback_entry(ctx.session.history).state, AnsweredCallbackState
            )
            result = UserQuestionResult.model_validate(
                _completed_effect_output(ctx.session.history)
            )
            assert result.disposition == "answered"
            # The cancelled timer never fires, even far past the deadline.
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert not any(
                isinstance(entry.state, ExpiredCallbackState)
                for entry in ctx.session.history
                if isinstance(entry, PublicCallbackEntry)
            )
            assert ctx.loop.deferred_questions.records() == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_keep_waiting_lifts_the_deadline(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            # Keep waiting: the deadline is lifted (OPEN_WITHOUT_DEADLINE).
            assert (
                await ctx.session.keep_waiting_callback(callback.callback_id) == "kept"
            )
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            # The question stays open with no deadline; nothing expired.
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(
                _callback_entry(ctx.session.history).state, OpenCallbackState
            )
            assert ctx.loop.deferred_questions.records() == []
            # The user can still answer afterwards.
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            assert isinstance(
                _callback_entry(ctx.session.history).state, AnsweredCallbackState
            )
            # A terminal callback reports already_terminal.
            assert (
                await ctx.session.keep_waiting_callback(callback.callback_id)
                == "already_terminal"
            )
            assert ctx.loop.deferred_questions.records() == []

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_keep_waiting_on_question_without_deadline(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_requires_user_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            assert callback.detail.unattended_expiry is None
            assert (
                await ctx.session.keep_waiting_callback(callback.callback_id)
                == "no_deadline"
            )
            await ctx.session.respond_to_callback(
                callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(stream), timeout=10)

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_cancellation_before_the_deadline_wins(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            await ctx.session.reject_callback(
                callback.callback_id, CallbackResultError(message="user cancelled")
            )
            # The rejection fails the turn; the stream surfaces the error.
            with pytest.raises(AppServerTurnError, match="user cancelled"):
                await asyncio.wait_for(_consume(stream), timeout=10)
            # Exactly one terminal disposition: cancelled, never expired.
            assert isinstance(
                _callback_entry(ctx.session.history).state, CancelledCallbackState
            )
            ctx.clock.advance(10_000.0)
            _release_gates(ctx.wakeup)
            await asyncio.sleep(0.05)
            assert not any(
                isinstance(entry.state, ExpiredCallbackState)
                for entry in ctx.session.history
                if isinstance(entry, PublicCallbackEntry)
            )
            assert ctx.loop.deferred_questions.records() == []


class TestLateAnswers:
    """Acceptance (4)/(7): a late answer gets a definite expired response,
    changes nothing, never answers a newer callback, and the user can still
    follow up as a new message.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_late_answer_is_rejected_and_changes_nothing(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
            [mock_llm_chunk(content="acknowledged")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            ctx.clock.advance(_DEFAULT_TIMEOUT + 1.0)
            _release_gates(ctx.wakeup)
            await asyncio.wait_for(_consume(stream), timeout=10)
            # A late submission receives a definite expired response.
            with pytest.raises(AppServerResponseError) as exc_info:
                await ctx.session.respond_to_callback(
                    callback.callback_id, _answer_output()
                )
            assert exc_info.value.error.code is ProtocolErrorCode.CALLBACK_CLOSED
            # Nothing changed: the result, the history, and the record stand.
            unanswered = _unanswered_from_effect(
                _effect_entries(ctx.session.history)[0]
            )
            assert unanswered.disposition == "unanswered"
            assert unanswered.answers == []
            assert isinstance(
                _callback_entry(ctx.session.history).state, ExpiredCallbackState
            )
            assert len(ctx.loop.deferred_questions.records()) == 1
            # The late answer may continue as a NEW user message through the
            # normal queue — never retroactive, never granting the old ask.
            await asyncio.wait_for(
                _consume(ctx.session.act("The answer is Yes")), timeout=10
            )
            assert any(
                isinstance(entry, PublicMessageEntry)
                and entry.role == "user"
                and "The answer is Yes" in entry.text
                for entry in ctx.session.history
            )
            assert len(ctx.loop.deferred_questions.records()) == 1
            assert len(backend.requests_messages) == 3

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_late_answer_never_answers_a_newer_callback(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _question_tool_call(
                            [_defaultable_payload()], call_id="question-2"
                        )
                    ],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            first_stream = ctx.session.act("first ask")
            first_callback = (
                await _next_event(first_stream, CallbackRequested)
            ).callback
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            ctx.clock.advance(_DEFAULT_TIMEOUT + 1.0)
            _release_gates(ctx.wakeup)
            await asyncio.wait_for(_consume(first_stream), timeout=10)
            assert isinstance(
                _callback_entry_for(
                    ctx.session.history, first_callback.callback_id
                ).state,
                ExpiredCallbackState,
            )
            # A newer question opens in a second turn.
            second_stream = ctx.session.act("second ask")
            second_callback = (
                await _next_event(second_stream, CallbackRequested)
            ).callback
            assert second_callback.callback_id != first_callback.callback_id
            # The late answer for the expired callback is rejected...
            with pytest.raises(AppServerResponseError) as exc_info:
                await ctx.session.respond_to_callback(
                    first_callback.callback_id, _answer_output()
                )
            assert exc_info.value.error.code is ProtocolErrorCode.CALLBACK_CLOSED
            # ...and does NOT answer the newer callback.
            assert [c.callback_id for c in ctx.session.state.active_callbacks] == [
                second_callback.callback_id
            ]
            assert isinstance(
                _callback_entry_for(
                    ctx.session.history, second_callback.callback_id
                ).state,
                OpenCallbackState,
            )
            # Answering the newer callback works normally.
            await ctx.session.respond_to_callback(
                second_callback.callback_id, _answer_output()
            )
            await asyncio.wait_for(_consume(second_stream), timeout=10)
            assert isinstance(
                _callback_entry_for(
                    ctx.session.history, second_callback.callback_id
                ).state,
                AnsweredCallbackState,
            )
            # Exactly one deferred record: only the first question expired.
            records = ctx.loop.deferred_questions.records()
            assert len(records) == 1
            assert records[0].request_id == first_callback.callback_id


class TestDetachedClientExpiry:
    """Acceptance (6): server-side expiry works with a detached client; a new
    client sees terminal history, not a ghost callback.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_detached_client_expires_and_reconnect_sees_terminal_history(
        self,
    ) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        loop = build_test_agent_loop(
            config=_enabled_config(),
            backend=backend,
            enable_streaming=True,
            user_input_capability=True,
        )
        reconnect_gate = asyncio.Event()
        server, session = await _open_reconnectable_session(
            loop, reconnect_gate=reconnect_gate
        )
        clock = _ManualClock()
        wakeup = _GatedWakeup()
        _patch_timer(server, clock, wakeup)
        try:
            stream = session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            await _wait_until(lambda: bool(wakeup.calls))
            # Detach: delivery becomes a silent no-op, but the server-side
            # timer is authoritative beside the callback record.
            client = session._connection.current
            assert client is not None
            await client.close()
            clock.advance(_DEFAULT_TIMEOUT + 1.0)
            _release_gates(wakeup)
            await _wait_until(lambda: len(loop.deferred_questions.records()) == 1)
            await _wait_until(lambda: len(backend.requests_messages) == 2)
            # Reconnect: the new client sees terminal history, not a ghost.
            reconnect_gate.set()
            rest = await asyncio.wait_for(_consume(stream), timeout=10)
            assert not any(
                isinstance(event, CallbackRequested)
                and event.callback.callback_id == callback.callback_id
                for event in rest
            )
            assert session.state.active_callbacks == []
            entry = _callback_entry(session.history)
            assert isinstance(entry.state, ExpiredCallbackState)
            assert entry.state.reason == "unanswered_timeout"
            unanswered = _unanswered_from_effect(_effect_entries(session.history)[0])
            assert unanswered.disposition == "unanswered"
            assert unanswered.timeout_seconds == _DEFAULT_TIMEOUT
            assert len(loop.deferred_questions.records()) == 1
        finally:
            reconnect_gate.set()
            await session.close()
            await loop.aclose()


class TestNoRetryOrReask:
    """Acceptance (9): timeout causes no tool auto-retry and no repeated
    asking.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_expiry_does_not_retry_or_reask(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            # This question is eligible: the expiry snapshot rides the detail.
            assert callback.detail.unattended_expiry is not None
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            ctx.clock.advance(_DEFAULT_TIMEOUT + 1.0)
            _release_gates(ctx.wakeup)
            await asyncio.wait_for(_consume(stream), timeout=10)
            # Exactly two model calls: the question turn and the closing
            # content — no auto-retry, no repeated asking.
            assert len(backend.requests_messages) == 2
            # The tool was called exactly once and produced exactly one result.
            tool_messages = [
                message for message in ctx.loop.messages if message.role == Role.tool
            ]
            assert len(tool_messages) == 1
            assert "unanswered_timeout" in (tool_messages[0].content or "")
            assert len(_effect_entries(ctx.session.history)) == 1
            assert len(ctx.loop.deferred_questions.records()) == 1


class TestCompactionCarriesDeferredRecords:
    """Acceptance (8): compaction/reload preserves deferred-report references."""

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_compaction_boundary_carries_and_replay_restores(self) -> None:
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_question_tool_call([_defaultable_payload()])],
                )
            ],
            [mock_llm_chunk(content="Done")],
            [mock_llm_chunk(content="<summary>summary of the conversation</summary>")],
        ])
        async with _patched_session(
            _enabled_config(),
            backend,
            enable_streaming=True,
            user_input_capability=True,
        ) as ctx:
            stream = ctx.session.act("ask me")
            callback = (await _next_event(stream, CallbackRequested)).callback
            await _wait_until(lambda: bool(ctx.wakeup.calls))
            ctx.clock.advance(_DEFAULT_TIMEOUT + 1.0)
            _release_gates(ctx.wakeup)
            await asyncio.wait_for(_consume(stream), timeout=10)
            record = ctx.loop.deferred_questions.records()[0]
            # Compaction carries the deferred record across the boundary.
            summary = await ctx.loop.compact()
            assert summary
            boundaries = [
                message
                for message in ctx.loop.messages
                if message.context_boundary == "compaction"
            ]
            assert len(boundaries) == 1
            tool_result = boundaries[0].tool_result
            assert tool_result is not None
            output = tool_result.output
            assert "deferred_questions" in output
            carried = output["deferred_questions"]
            assert isinstance(carried, list)
            assert len(carried) == 1
            first = carried[0]
            assert isinstance(first, dict)
            assert first["request_id"] == record.request_id
            assert first["session_id"] == record.session_id
            # A reload replays the envelope into a fresh store: the
            # deferred-report reference survives the structural boundary.
            replayed = DeferredQuestionStore.replay(list(ctx.loop.messages))
            assert [r.request_id for r in replayed.records()] == [record.request_id]
            assert replayed.records()[0].affected_work == record.affected_work
            assert replayed.records()[0].request_id == callback.callback_id


class TestRuntimeApprovalsNeverExpire:
    """Acceptance (5), root-access half: a root-grant approval shares the
    callback plumbing but is permanently ineligible for expiry — it stays
    unanswered indefinitely and grants nothing. The dangerous-command half
    lives in tests/app_server/test_dangerous_command_acceptance.py.
    """

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_root_access_approval_never_expires_and_grants_nothing(
        self, tmp_path: Path
    ) -> None:
        project = tmp_path / "project"
        project.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        target = outside / "note.txt"
        target.write_text("granted content")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    content="", tool_calls=[_read_tool_call(target, "read-1")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        config = _enabled_config().model_copy(update={"enabled_tools": ["read_file"]})
        loop = build_test_agent_loop(
            config=config, backend=backend, cwd=project, user_input_capability=True
        )
        server, session = await _open_session(loop)
        clock = _ManualClock()
        wakeup = _GatedWakeup()
        _patch_timer(server, clock, wakeup)
        try:
            stream = session.act("read the note")
            callback = (await _next_event(stream, CallbackRequested)).callback
            detail = callback.detail
            assert isinstance(detail, UserInputCallbackDetail)
            # The runtime approval is permanently ineligible: no expiry policy.
            assert detail.unattended_expiry is None
            question = detail.request.questions[0]
            assert question.header == "Grant access"
            assert str(outside.resolve()) in question.question
            # No timer is armed even though the config is enabled.
            await asyncio.sleep(0.05)
            assert wakeup.calls == []
            # Silence far past any deadline: still open, nothing granted.
            clock.advance(10_000.0)
            _release_gates(wakeup)
            await asyncio.sleep(0.05)
            assert [c.callback_id for c in session.state.active_callbacks] == [
                callback.callback_id
            ]
            assert isinstance(_callback_entry(session.history).state, OpenCallbackState)
            assert loop._session_root_grants.roots == set()
            assert not loop.tool_manager.workspace.allows(target)
            assert loop.deferred_questions.records() == []
            # Deny: the refusal is remembered; still nothing was granted.
            await session.respond_to_callback(
                callback.callback_id,
                UserInputCallbackOutput(
                    result=UserQuestionResult(
                        answers=[UserAnswer(question="grant", answer="Deny")]
                    )
                ),
            )
            await asyncio.wait_for(_consume(stream), timeout=10)
            assert loop._session_root_grants.roots == set()
            assert loop._session_root_grants.denied_roots == {outside.resolve()}
            assert loop.deferred_questions.records() == []
        finally:
            await session.close()
            await loop.aclose()
