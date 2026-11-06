"""Q2c: the TUI dismisses dead and closed callback widgets (incident replay).

The overnight incident: a peer-seat child's misfired ``ask_user_question``
rendered a blocking widget in the parent TUI. Cancel teardown closed the child,
but nothing unmounted the widget, and Enter/Esc both routed to the dead
callback id, whose rejection branch returned with a draft-and-retry hint. The
widget stayed undismissible for ~9 hours.

These tests replay that chain through the callback seam (Q1's denial removes
the natural trigger) and pin the dismissal boundary: dead codes dismiss, a
protocol rejection (CONFLICT/INVALID_PARAMS) still keeps the draft, and a late
terminal event never touches a newer callback.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.events import (
    CallbackRequested,
    ChildCallbackClosed,
    HistoryEntryUpdated,
)
from chartreux.app_server.models import (
    AnsweredCallbackState,
    CancelledCallbackState,
    OpenCallbackState,
    PublicCallbackEntry,
    PublicEntryGenerationStatus,
    QuestionChoice,
    UserAnswer,
    UserInputCallbackDetail,
    UserInputCallbackOutput,
    UserQuestion,
    UserQuestionRequest,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ChildCallbackClosedParams,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.turn_supervisor import (
    _ANSWERED_ELSEWHERE_MESSAGE,
    _CLOSED_CALLBACK_MESSAGE,
    _DEAD_CALLBACK_MESSAGE,
)
from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
from tests.conftest import build_test_chartreux_app


def _question_callback(
    callback_id: str, *, session_id: str = "session-1"
) -> PublicCallbackEntry:
    return PublicCallbackEntry(
        id=f"callback:{callback_id}",
        session_id=session_id,
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        callback_id=callback_id,
        title="Input required",
        detail=UserInputCallbackDetail(
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
        ),
        state=OpenCallbackState(),
    )


def _child_closure(callback_id: str, *, session_id: str = "child-session-1"):
    return ChildCallbackClosed(
        ChildCallbackClosedParams(
            session_id=session_id, callback_id=callback_id, reason="child cancelled"
        )
    )


def _answer(callback_id: str) -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer="No")], cancelled=False
        )
    )


def _type_draft(question: QuestionApp, text: str) -> None:
    other_input = question.other_input
    assert other_input is not None
    other_input.value = text


def _draft_input_value(question: QuestionApp) -> str:
    other_input = question.other_input
    assert other_input is not None
    return other_input.value


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for UI state")
        await pilot.pause(0.01)


async def _show_question(app: ChartreuxApp, pilot, callback: PublicCallbackEntry):
    await app._handle_turn_event(CallbackRequested(callback))
    await _wait_until(pilot, lambda: bool(app.query(QuestionApp)))
    question = app.query_one(QuestionApp)
    # Past the mount grace period, so Enter/Esc act immediately.
    question._mount_time = 0.0
    return question


async def _show_newer_behind_active(
    app: ChartreuxApp,
    pilot,
    monkeypatch: pytest.MonkeyPatch,
    older: PublicCallbackEntry,
    newer: PublicCallbackEntry,
) -> QuestionApp:
    """Answer ``older`` successfully so ``newer`` becomes the active widget."""
    await _show_question(app, pilot, older)
    await app._handle_turn_event(CallbackRequested(newer))
    assert [pending.callback_id for pending in app._pending_callbacks] == [
        newer.callback_id
    ]
    monkeypatch.setattr(app.app_server, "respond_to_callback", AsyncMock())
    await pilot.press("enter")
    await _wait_until(pilot, lambda: app._active_callback is newer)
    await _wait_until(
        pilot,
        lambda: (
            bool(app.query(QuestionApp))
            and app.query_one(QuestionApp)._callback_id == newer.callback_id
        ),
    )
    return app.query_one(QuestionApp)


@pytest.mark.asyncio
async def test_child_question_closure_dismisses_the_widget_without_user_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The incident replay: widget renders, child closes, widget goes away."""
    app = build_test_chartreux_app()
    child_question = _question_callback("child-question", session_id="child-session-1")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_question(app, pilot, child_question)
        assert app._active_callback is child_question
        assert question._callback_id == "child-question"
        monkeypatch.setattr(app, "notify", notify)

        # The child was cancelled: the server reports the closure (Q2b). No
        # key press, no user action — the widget must dismiss on its own.
        await app._handle_turn_event(_child_closure("child-question"))

        await _wait_until(pilot, lambda: not app.query(QuestionApp))
        assert app._active_callback is None
        assert not app._pending_callbacks
        assert _CLOSED_CALLBACK_MESSAGE in [
            call.args[0] for call in notify.call_args_list
        ]

        # A duplicate closure is idempotent: nothing left to dismiss.
        await app._handle_turn_event(_child_closure("child-question"))
        assert not app.query(QuestionApp)
        assert _CLOSED_CALLBACK_MESSAGE in [
            call.args[0] for call in notify.call_args_list
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_state", "expected_message"),
    [
        (CancelledCallbackState(reason="turn ended"), _CLOSED_CALLBACK_MESSAGE),
        (AnsweredCallbackState(output=_answer("cb-1")), _ANSWERED_ELSEWHERE_MESSAGE),
    ],
    ids=["cancelled", "answered"],
)
async def test_projected_terminal_state_dismisses_the_matching_widget(
    terminal_state, expected_message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any non-open projection state unmounts the live widget (not just expired).

    The copy is outcome-specific (plan U): a cancellation reads as closed and an
    answer landed elsewhere reads as answered elsewhere — in both cases quietly,
    as information rather than a warning.
    """
    app = build_test_chartreux_app()
    callback = _question_callback("projected")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        await _show_question(app, pilot, callback)
        monkeypatch.setattr(app, "notify", notify)

        closed = callback.model_copy(update={"state": terminal_state})
        await app._handle_turn_event(
            HistoryEntryUpdated(previous=callback, entry=closed, patch=[])
        )

        await _wait_until(pilot, lambda: not app.query(QuestionApp))
        assert app._active_callback is None
        matched = [
            call for call in notify.call_args_list if call.args[0] == expected_message
        ]
        assert matched, notify.call_args_list
        assert matched[0].kwargs["severity"] == "information"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", [ProtocolErrorCode.NOT_FOUND, ProtocolErrorCode.CALLBACK_CLOSED]
)
async def test_enter_on_a_dead_callback_dismisses_without_a_resend_hint(
    code: ProtocolErrorCode, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    callback = _question_callback(f"dead-{code.value}")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_question(app, pilot, callback)
        monkeypatch.setattr(app, "notify", notify)
        response = AsyncMock(
            side_effect=AppServerResponseError(
                ProtocolError(code=code, message="callback is gone")
            )
        )
        monkeypatch.setattr(app.app_server, "respond_to_callback", response)

        await pilot.press("enter")

        await _wait_until(pilot, lambda: not app.query(QuestionApp))
        assert response.await_count == 1
        assert app._active_callback is None
        # A typed dead code is a definite outcome, never delivery-uncertain.
        assert not app._callback_delivery_uncertain
        messages = [call.args[0] for call in notify.call_args_list]
        assert _DEAD_CALLBACK_MESSAGE in messages
        assert not any("submit again" in message for message in messages)
        assert not any("Draft kept" in message for message in messages)
        assert "submit again" not in (question._submission_status or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code", [ProtocolErrorCode.NOT_FOUND, ProtocolErrorCode.CALLBACK_CLOSED]
)
async def test_escape_on_a_dead_callback_dismisses_without_a_resend_hint(
    code: ProtocolErrorCode, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = build_test_chartreux_app()
    callback = _question_callback(f"dead-{code.value}")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_question(app, pilot, callback)
        monkeypatch.setattr(app, "notify", notify)
        response = AsyncMock(
            side_effect=AppServerResponseError(
                ProtocolError(code=code, message="callback is gone")
            )
        )
        monkeypatch.setattr(app.app_server, "respond_to_callback", response)

        await pilot.press("escape")

        await _wait_until(pilot, lambda: not app.query(QuestionApp))
        assert response.await_count == 1
        assert app._active_callback is None
        assert not app._callback_delivery_uncertain
        messages = [call.args[0] for call in notify.call_args_list]
        assert _DEAD_CALLBACK_MESSAGE in messages
        assert not any("submit again" in message for message in messages)
        assert "submit again" not in (question._submission_status or "")


@pytest.mark.asyncio
async def test_conflict_rejection_keeps_the_draft_and_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CONFLICT is a definite rejection, not a dead callback: retry stays open."""
    app = build_test_chartreux_app()
    callback = _question_callback("conflict")

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_question(app, pilot, callback)
        question.selected_option = 1
        reject = AsyncMock(
            side_effect=AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.CONFLICT,
                    message="question is not open for answers",
                )
            )
        )
        monkeypatch.setattr(app.app_server, "respond_to_callback", reject)

        await pilot.press("enter")

        await _wait_until(pilot, lambda: reject.await_count == 1)
        await _wait_until(pilot, lambda: question._submission_status is not None)
        assert app._active_callback is callback
        assert app.query_one(QuestionApp) is question
        assert question.selected_option == 1
        status = question._submission_status
        assert status is not None
        assert status.startswith("Failed:")
        assert "Draft kept" in status
        assert "edit and submit again" in status

        # The hint is honest: a corrected resubmission really does go out.
        retry = AsyncMock()
        monkeypatch.setattr(app.app_server, "respond_to_callback", retry)
        await pilot.press("enter")
        await _wait_until(pilot, lambda: retry.await_count == 1)
        retry_call = retry.await_args
        assert retry_call is not None
        assert retry_call.args[0] == callback.callback_id
        assert retry_call.args[1].result.answers[0].answer == "No"


@pytest.mark.asyncio
async def test_late_child_closure_leaves_the_newer_callback_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale closure never dismisses the newer widget or clears its draft."""
    app = build_test_chartreux_app()
    older = _question_callback("older", session_id="child-session-1")
    newer = _question_callback("newer")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_newer_behind_active(
            app, pilot, monkeypatch, older, newer
        )
        _type_draft(question, "keep this draft")
        await pilot.pause()
        assert question.other_texts[0] == "keep this draft"
        monkeypatch.setattr(app, "notify", notify)

        # The closure for the OLD callback arrives after the newer one is live.
        await app._handle_turn_event(_child_closure("older"))
        await pilot.pause()

        assert app._active_callback is newer
        assert app.query_one(QuestionApp) is question
        assert question.other_texts[0] == "keep this draft"
        assert _draft_input_value(question) == "keep this draft"
        assert not notify.called


@pytest.mark.asyncio
async def test_late_terminal_patch_leaves_the_newer_callback_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    older = _question_callback("older")
    newer = _question_callback("newer")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_newer_behind_active(
            app, pilot, monkeypatch, older, newer
        )
        _type_draft(question, "keep this draft")
        await pilot.pause()
        monkeypatch.setattr(app, "notify", notify)

        closed = older.model_copy(
            update={"state": CancelledCallbackState(reason="turn ended")}
        )
        await app._handle_turn_event(
            HistoryEntryUpdated(previous=older, entry=closed, patch=[])
        )
        await pilot.pause()

        assert app._active_callback is newer
        assert app.query_one(QuestionApp) is question
        assert question.other_texts[0] == "keep this draft"
        assert not notify.called


@pytest.mark.asyncio
async def test_closure_for_a_queued_callback_prunes_it_and_keeps_the_active_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    active = _question_callback("active")
    queued = _question_callback("queued", session_id="child-session-1")
    notify = MagicMock()

    async with app.run_test() as pilot:
        await app._session_ready.wait()
        question = await _show_question(app, pilot, active)
        _type_draft(question, "keep this draft")
        await pilot.pause()
        await app._handle_turn_event(CallbackRequested(queued))
        assert [pending.callback_id for pending in app._pending_callbacks] == ["queued"]
        monkeypatch.setattr(app, "notify", notify)

        await app._handle_turn_event(_child_closure("queued"))
        await pilot.pause()

        assert not app._pending_callbacks
        assert app._active_callback is active
        assert app.query_one(QuestionApp) is question
        assert question.other_texts[0] == "keep this draft"
        assert not notify.called
