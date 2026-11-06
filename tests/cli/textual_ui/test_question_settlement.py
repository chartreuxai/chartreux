"""UQ: terminal-event/RPC reconciliation and quiet outcome presentation (plan U).

The server publishes the terminal callback patch from inside the resolution
lock, before it answers the settlement RPC. The client therefore normally sees
a non-open ``HistoryEntryUpdated`` while its own answer is still in flight. The
old presentation treated every such event as an out-of-band closure, showed the
user "This question was closed; it can no longer be answered." as a warning,
and bypassed the answer RPC's own return path — the reported error-on-answer
bug.

These nodes pin the redesigned contract:

* an echo matching this client's in-flight settlement is reconciled by callback
  identity AND exact output, and never reported as a closure;
* a closure the user did not cause is presented quietly, with outcome-specific
  copy (answered elsewhere vs cancelled);
* expiry keeps its unchanged "Unanswered — deferred" warning.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.models import (
    AnsweredCallbackState,
    CallbackState,
    CancelledCallbackState,
    ExpiredCallbackState,
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
from chartreux.cli.textual_ui.turn_supervisor import (
    _ANSWERED_ELSEWHERE_MESSAGE,
    _CLOSED_CALLBACK_MESSAGE,
    TurnSupervisor,
)


def _question_request() -> UserQuestionRequest:
    return UserQuestionRequest(
        questions=[
            UserQuestion(
                question="Ship it?",
                options=[QuestionChoice(label="Yes"), QuestionChoice(label="No")],
            )
        ]
    )


def _callback_entry(
    callback_id: str = "callback-1", state: CallbackState | None = None
) -> PublicCallbackEntry:
    return PublicCallbackEntry(
        id=f"callback:{callback_id}",
        session_id="session-1",
        turn_id="turn-1",
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        callback_id=callback_id,
        title="Input required",
        detail=UserInputCallbackDetail(request=_question_request()),
        state=OpenCallbackState() if state is None else state,
    )


def _answer(text: str = "Yes") -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(
            answers=[UserAnswer(question="Ship it?", answer=text)], cancelled=False
        )
    )


def _cancelled() -> UserInputCallbackOutput:
    return UserInputCallbackOutput(
        result=UserQuestionResult(answers=[], cancelled=True)
    )


def _app() -> MagicMock:
    app = MagicMock()
    app._show_callback = AsyncMock()
    app._switch_to_input_app = AsyncMock()
    app.query_one.return_value = MagicMock(_callback_id="callback-1")
    return app


def _supervisor(app: MagicMock) -> TurnSupervisor:
    supervisor = TurnSupervisor(
        interrupt_wait_timeout=lambda: 2.0, interrupt_settle_timeout=lambda: 30.0
    )
    supervisor._bind_app(app)
    return supervisor


def _answered(
    callback_id: str = "callback-1", output: UserInputCallbackOutput | None = None
) -> PublicCallbackEntry:
    return _callback_entry(
        callback_id,
        AnsweredCallbackState(output=output if output is not None else _answer()),
    )


@pytest.mark.asyncio
async def test_answering_a_question_shows_no_closure_warning() -> None:
    """The reported bug: answering must not surface a closed-question warning.

    The fake server delivers the terminal patch from inside the settlement RPC,
    exactly as the app server does (patch before response).
    """
    app = _app()
    supervisor = _supervisor(app)
    supervisor.active_callback = _callback_entry()
    output = _answer()

    async def respond(callback_id: str, submitted: UserInputCallbackOutput) -> None:
        await supervisor._question_closed(_answered(output=submitted))

    app.app_server.respond_to_callback = AsyncMock(side_effect=respond)

    await supervisor.respond_to_active_callback(output)

    app.notify.assert_not_called()
    assert supervisor.active_callback is None
    assert supervisor.settling_callback is None
    app._switch_to_input_app.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_own_settlement_echo_is_not_a_closure_while_the_rpc_is_in_flight() -> (
    None
):
    app = _app()
    supervisor = _supervisor(app)
    callback = _callback_entry()
    output = _answer()
    supervisor.active_callback = callback
    supervisor.callback_submitting = True
    supervisor.settling_callback = (callback.callback_id, output)

    await supervisor._question_closed(_answered(output=output))

    assert supervisor.active_callback is callback
    app.notify.assert_not_called()
    app.query_one.return_value.close_for_expiry.assert_not_called()
    app._switch_to_input_app.assert_not_awaited()


@pytest.mark.asyncio
async def test_own_output_is_reconciled_by_identity_and_exact_output() -> None:
    """A matching callback id with a different output is a foreign settlement."""
    app = _app()
    supervisor = _supervisor(app)
    callback = _callback_entry()
    supervisor.active_callback = callback
    supervisor.callback_submitting = True
    supervisor.settling_callback = (callback.callback_id, _answer("No"))

    await supervisor._question_closed(_answered(output=_answer("Yes")))

    assert supervisor.active_callback is None
    app.notify.assert_called_once_with(
        _ANSWERED_ELSEWHERE_MESSAGE, severity="information"
    )


@pytest.mark.asyncio
async def test_own_echo_after_an_uncertain_delivery_closes_silently() -> None:
    """The echo proves the uncertain delivery landed: close, but stay quiet."""
    app = _app()
    supervisor = _supervisor(app)
    callback = _callback_entry()
    output = _answer()
    supervisor.active_callback = callback
    supervisor.callback_submitting = False
    supervisor.callback_delivery_uncertain = True
    supervisor.settling_callback = (callback.callback_id, output)

    await supervisor._question_closed(_answered(output=output))

    assert supervisor.active_callback is None
    assert supervisor.callback_delivery_uncertain is False
    assert supervisor.settling_callback is None
    app.notify.assert_not_called()
    app._switch_to_input_app.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_answer_from_elsewhere_is_informational() -> None:
    app = _app()
    supervisor = _supervisor(app)
    supervisor.active_callback = _callback_entry()

    await supervisor._question_closed(_answered())

    assert supervisor.active_callback is None
    app.notify.assert_called_once_with(
        _ANSWERED_ELSEWHERE_MESSAGE, severity="information"
    )
    app._switch_to_input_app.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_out_of_band_cancellation_is_informational() -> None:
    app = _app()
    supervisor = _supervisor(app)
    supervisor.active_callback = _callback_entry()
    cancelled = _callback_entry(state=CancelledCallbackState(reason="Turn ended"))

    await supervisor._question_closed(cancelled)

    assert supervisor.active_callback is None
    app.notify.assert_called_once_with(_CLOSED_CALLBACK_MESSAGE, severity="information")


@pytest.mark.asyncio
async def test_expiry_keeps_its_deferred_warning() -> None:
    """Expiry/consent semantics unchanged (plan U/OD1)."""
    app = _app()
    supervisor = _supervisor(app)
    supervisor.active_callback = _callback_entry()
    expired = _callback_entry(state=ExpiredCallbackState(reason="unanswered_timeout"))

    await supervisor._question_closed(expired)

    assert supervisor.active_callback is None
    app.notify.assert_called_once_with("Unanswered — deferred", severity="warning")


@pytest.mark.asyncio
async def test_terminal_event_for_a_foreign_callback_dismisses_nothing() -> None:
    app = _app()
    supervisor = _supervisor(app)
    active = _callback_entry("callback-1")
    supervisor.active_callback = active

    await supervisor._question_closed(_answered("callback-2"))

    assert supervisor.active_callback is active
    app.notify.assert_not_called()
    app._switch_to_input_app.assert_not_awaited()


@pytest.mark.asyncio
async def test_submitted_cancel_is_reconciled_as_own_settlement() -> None:
    """Esc records answered(cancelled=True): the echo is this client's own."""
    app = _app()
    supervisor = _supervisor(app)
    callback = _callback_entry()
    output = _cancelled()
    supervisor.active_callback = callback
    supervisor.callback_submitting = True
    supervisor.settling_callback = (callback.callback_id, output)

    await supervisor._question_closed(_answered(output=output))

    assert supervisor.active_callback is callback
    app.notify.assert_not_called()
