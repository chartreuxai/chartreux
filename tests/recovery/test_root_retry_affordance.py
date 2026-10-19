"""R1 — terminal root-failure retry affordance (plan v0.5.6-recovery, package R1).

R1 (the manual fallback that must exist before R2's automation): on an
authoritative TERMINAL RETRYABLE root failure, the TUI displays a
keyboard-accessible "Retry — new turn" affordance that invokes the EXISTING
retry operation (one injected new turn, ``user_initiated_retry=True``,
``build_retry_prompt("")``) — never ``resume_agent``, never automatic.

Retryable-set decision (pinned here): the affordance's retryable set is the
existing ``_RETRYABLE_TURN_ERROR_CODES`` PLUS exhaustion-after-cooldown
(``ALL_DEPLOYMENTS_UNAVAILABLE``). The user pressing Retry is a genuine user
action, and the existing manual retry operation's intentional cooldown reset
(the only path allowed to reset it) is exactly what makes a retry meaningful
once every deployment was excluded. The error text stays honest that the
deployments were unavailable and that a manual retry resets the model's
cooldowns.

Coverage map (R1 behavior bullet -> tests):

- Affordance display conditions -> ``test_affordance_offered_for_each_retryable_code``,
  ``test_no_affordance_for_non_retryable_codes``,
  ``test_affordance_retryable_set_is_pinned``.
- Keyboard access + existing-operation invocation + warning preservation +
  cooldown-reset retention + double activation -> the TUI-level pins in
  ``tests/cli/textual_ui/test_app_server_requests.py`` (the R1 block), plus
  ``test_retry_invokes_existing_operation``,
  ``test_double_activation_starts_at_most_one_turn``,
  ``test_begin_retry_refuses_second_activation``,
  ``test_retry_hint_preserves_completed_actions_warning`` here.
- Suppression conditions (stopping / active execution / unresolved runtime
  approval) -> ``test_retry_suppressed_while_stopping``,
  ``test_retry_suppressed_while_turn_active``,
  ``test_retry_suppressed_with_unresolved_runtime_approval``, plus the
  TUI-level unresolved-approval pin in ``test_app_server_requests.py``.
- Transport disconnect / suspected hang offers inspection/reconnect guidance,
  not automatic continuation -> ``test_disconnect_guidance_offers_inspection_not_retry``,
  ``test_generic_exception_gets_no_disconnect_guidance``, plus the TUI-level
  pin in ``test_app_server_requests.py``.
- Honest exhaustion text -> ``test_exhaustion_error_text_is_honest``.

No live provider calls: everything here is unit-level against the real
``TurnSupervisor``/``EventHandler``/``ErrorMessage`` with mock app delegates.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server import AppServerConnectionClosed
from chartreux.app_server.models import PublicError, TurnErrorCode
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.turn_supervisor import (
    _RETRY_AFFORDANCE_TURN_ERROR_CODES,
    _RETRYABLE_TURN_ERROR_CODES,
    TurnSupervisor,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, ErrorRetryRequested
from chartreux.utils.retry_prompt import build_retry_prompt

_AFFORDANCE_CODES = [
    TurnErrorCode.BACKEND_ERROR,
    TurnErrorCode.RATE_LIMIT,
    TurnErrorCode.RESPONSE_TOO_LONG,
    TurnErrorCode.INCOMPLETE_STREAM,
    TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE,
]

_NON_RETRYABLE_CODES = [
    TurnErrorCode.CONTEXT_TOO_LONG,
    TurnErrorCode.REFUSAL,
    TurnErrorCode.INVALID_API_KEY,
    TurnErrorCode.INTERNAL_ERROR,
    TurnErrorCode.EMPTY_LLM_RESPONSE,
    TurnErrorCode.INVALID_MODEL,
]


def _turn_error(
    code: TurnErrorCode, message: str = "Network error"
) -> AppServerTurnError:
    return AppServerTurnError(PublicError(message=message, code=code))


def _mock_app() -> MagicMock:
    app = MagicMock()
    app._tools_collapsed = False
    app._mount_and_scroll = AsyncMock()
    app.event_handler = MagicMock()
    return app


def test_affordance_retryable_set_is_pinned() -> None:
    """R1 decision: the affordance's retryable set is the existing retryable
    set PLUS exhaustion-after-cooldown (ALL_DEPLOYMENTS_UNAVAILABLE). The
    legacy retryable set itself is unchanged — the affordance extends it.
    """
    assert _RETRYABLE_TURN_ERROR_CODES == {
        TurnErrorCode.BACKEND_ERROR,
        TurnErrorCode.RATE_LIMIT,
        TurnErrorCode.RESPONSE_TOO_LONG,
        TurnErrorCode.INCOMPLETE_STREAM,
    }
    assert _RETRY_AFFORDANCE_TURN_ERROR_CODES == _RETRYABLE_TURN_ERROR_CODES | {
        TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE
    }
    assert set(_AFFORDANCE_CODES) == _RETRY_AFFORDANCE_TURN_ERROR_CODES


@pytest.mark.parametrize("code", _AFFORDANCE_CODES)
def test_affordance_offered_for_each_retryable_code(code: TurnErrorCode) -> None:
    """An authoritative terminal retryable root failure displays the affordance:
    the error widget carries the retry button and the retry presentation is
    offered (so the affordance invokes the EXISTING retry operation).
    """
    app = _mock_app()
    supervisor = cast(Any, SimpleNamespace(_app=app))

    asyncio.run(TurnSupervisor._mount_turn_error(supervisor, _turn_error(code), "boom"))

    app.event_handler.offer_retry.assert_called_once()
    widget = app.event_handler.offer_retry.call_args.args[0]
    assert isinstance(widget, ErrorMessage)
    assert widget._retryable is True
    app._refresh_command_registry.assert_called_once()


@pytest.mark.parametrize("code", _NON_RETRYABLE_CODES)
def test_no_affordance_for_non_retryable_codes(code: TurnErrorCode) -> None:
    """Non-retryable terminal failures get no retry affordance and no retry
    presentation.
    """
    app = _mock_app()
    supervisor = cast(Any, SimpleNamespace(_app=app))

    asyncio.run(TurnSupervisor._mount_turn_error(supervisor, _turn_error(code), "boom"))

    app.event_handler.offer_retry.assert_not_called()
    widget = app._mount_and_scroll.call_args.args[0]
    assert isinstance(widget, ErrorMessage)
    assert widget._retryable is False
    if code is TurnErrorCode.INVALID_API_KEY:
        # The pre-existing stale-presentation cancel is preserved.
        app.event_handler.cancel_retry_presentation.assert_called_once_with()


def test_exhaustion_error_text_is_honest() -> None:
    """The exhaustion (all-deployments-unavailable) error text is honest: the
    deployments really were unavailable, a manual retry resets the model's
    cooldowns, and the completed-actions-may-repeat warning is preserved.
    """
    app = MagicMock()
    app._retry_hint = TurnSupervisor._retry_hint
    app._exhaustion_retry_hint = TurnSupervisor._exhaustion_retry_hint

    message = TurnSupervisor._resolve_turn_error_message(
        app,
        _turn_error(
            TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE,
            "All deployments for model 'base' are unavailable (test-first: cooldown)",
        ),
    )

    assert "All deployments for model 'base' are unavailable" in message
    assert "test-first" in message
    assert "Retry — new turn" in message
    assert "/retry [additional instructions]" in message
    assert "resets the model's cooldowns" in message
    assert "Completed tool results are kept; recent actions may repeat." in message


def test_retry_hint_preserves_completed_actions_warning() -> None:
    """The retry hint names the affordance (button + Ctrl+R + /retry) and
    keeps the completed-actions warning verbatim; the injected retry turn
    carries the same warning (the affordance must not weaken it).
    """
    hint = TurnSupervisor._retry_hint()
    assert "Retry — new turn" in hint
    assert "Ctrl+R" in hint
    assert "/retry [additional instructions]" in hint
    assert "Completed tool results are kept; recent actions may repeat." in hint
    assert "recent actions may repeat" in build_retry_prompt("")


class _RetryAffordanceApp(App[None]):
    """Minimal host proving the affordance widget itself is keyboard-accessible."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list[ErrorRetryRequested] = []

    def compose(self) -> ComposeResult:
        yield ErrorMessage("boom", retryable=True)

    def on_error_retry_requested(self, message: ErrorRetryRequested) -> None:
        self.received.append(message)


@pytest.mark.asyncio
async def test_affordance_widget_is_keyboard_accessible() -> None:
    """The "Retry — new turn" button is a focusable, keyboard-activatable
    affordance: focusing it and pressing Enter posts ErrorRetryRequested
    exactly once (double activation is refused at the widget level too).

    In the full app the chat input's focus policy reclaims focus from
    transcript widgets, so the app-level Ctrl+R binding is the keyboard path
    there (pinned in tests/cli/textual_ui/test_app_server_requests.py); this
    test pins the widget's own keyboard accessibility.
    """
    from textual.widgets import Button

    app = _RetryAffordanceApp()
    async with app.run_test() as pilot:
        widget = app.query_one(ErrorMessage)
        assert widget._retryable is True
        button = widget.query_one(".error-retry-button", Button)
        assert str(button.label) == "Retry — new turn"
        assert button.can_focus is True

        button.focus()
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert len(app.received) == 1
        assert button.disabled is True

        # Double activation at the widget level: refused.
        button.press()
        await pilot.pause(0.1)
        assert len(app.received) == 1


@pytest.mark.parametrize(
    "error",
    [
        AppServerConnectionClosed("App server connection closed"),
        ConnectionError("connection reset"),
        TimeoutError("read timed out"),
    ],
    ids=["app-server-closed", "connection-error", "timeout"],
)
def test_disconnect_guidance_offers_inspection_not_retry(error: Exception) -> None:
    """A transport disconnect / suspected hang offers inspection/reconnect
    guidance — never a retry affordance, never an automatic continuation.
    """
    app = _mock_app()
    supervisor = cast(Any, SimpleNamespace(_app=app))

    message = TurnSupervisor._resolve_turn_error_message(app, error)

    assert "Inspect the session and server state" in message
    assert "reconnects automatically" in message
    assert "Nothing is continued automatically" in message
    assert "/retry" not in message
    assert "Retry — new turn" not in message

    asyncio.run(TurnSupervisor._mount_turn_error(supervisor, error, message))

    app.event_handler.offer_retry.assert_not_called()
    widget = app._mount_and_scroll.call_args.args[0]
    assert isinstance(widget, ErrorMessage)
    assert widget._retryable is False


def test_generic_exception_gets_no_disconnect_guidance() -> None:
    """A generic (non-transport) exception keeps its plain message — no
    guidance, no affordance.
    """
    app = MagicMock()

    message = TurnSupervisor._resolve_turn_error_message(app, RuntimeError("boom"))

    assert message == "boom"


def _supervisor_with_mock_app() -> tuple[TurnSupervisor, MagicMock, EventHandler]:
    """A real TurnSupervisor + real EventHandler over a mock app, mirroring
    the production wiring closely enough to exercise the retry admission.
    """
    handler = EventHandler(
        mount_callback=AsyncMock(), get_tools_collapsed=lambda: False
    )
    app = MagicMock()
    app.event_handler = handler
    app._handle_turn = AsyncMock()
    app._refresh_command_registry = MagicMock()
    app._queue.notify_busy_changed = MagicMock()
    supervisor = TurnSupervisor(
        interrupt_wait_timeout=lambda: 2.0, interrupt_settle_timeout=lambda: 30.0
    )
    supervisor._bind_app(app)
    app._agent_job_active = lambda: (
        supervisor.agent_task is not None and not supervisor.agent_task.done()
    )
    return supervisor, app, handler


@pytest.mark.asyncio
async def test_retry_invokes_existing_operation() -> None:
    """The affordance path IS the existing retry operation: exactly ONE
    injected new turn with user_initiated_retry=True and build_retry_prompt("")
    — never resume_agent, never automatic.
    """
    supervisor, app, handler = _supervisor_with_mock_app()
    handler.offer_retry()

    await supervisor.retry()

    app._handle_turn.assert_called_once()
    args, kwargs = app._handle_turn.call_args
    assert args[0] == build_retry_prompt("")
    assert kwargs["injected"] is True
    assert kwargs["user_initiated_retry"] is True
    task = supervisor.agent_task
    assert task is not None
    await task
    app._handle_turn.assert_awaited_once()


@pytest.mark.asyncio
async def test_retry_suppressed_while_stopping() -> None:
    """No retry action while stopping (an interrupt is still settling)."""
    supervisor, app, handler = _supervisor_with_mock_app()
    handler.offer_retry()
    supervisor.interrupt_requested = True

    await supervisor.retry()

    app._handle_turn.assert_not_called()
    assert supervisor.agent_task is None


@pytest.mark.asyncio
async def test_retry_suppressed_while_turn_active() -> None:
    """No retry action while active execution is present."""
    supervisor, app, handler = _supervisor_with_mock_app()
    handler.offer_retry()
    app._agent_job_active = lambda: True

    await supervisor.retry()

    app._handle_turn.assert_not_called()
    assert supervisor.agent_task is None


@pytest.mark.asyncio
async def test_retry_suppressed_with_unresolved_runtime_approval() -> None:
    """No retry action while an unresolved runtime approval is present."""
    supervisor, app, handler = _supervisor_with_mock_app()
    handler.offer_retry()
    supervisor.active_callback = MagicMock()

    await supervisor.retry()

    app._handle_turn.assert_not_called()
    assert supervisor.agent_task is None


@pytest.mark.asyncio
async def test_double_activation_starts_at_most_one_turn() -> None:
    """Double activation starts AT MOST one turn: the first activation claims
    turn ownership (the created agent task), so the second is refused.
    """
    supervisor, app, handler = _supervisor_with_mock_app()
    handler.offer_retry()

    await supervisor.retry()
    await supervisor.retry()

    app._handle_turn.assert_called_once()
    task = supervisor.agent_task
    assert task is not None
    await task
    app._handle_turn.assert_awaited_once()


def test_begin_retry_refuses_second_activation() -> None:
    """Presentation-level double-activation guard: a second begin_retry is
    refused while a retry is already admitted; a fresh offer re-arms it.
    """
    handler = EventHandler(
        mount_callback=AsyncMock(), get_tools_collapsed=lambda: False
    )

    handler.offer_retry()
    assert handler.begin_retry()
    assert not handler.begin_retry()
    # A fresh offer (the previous retry concluded with another error) re-arms.
    handler.offer_retry()
    assert handler.begin_retry()


def test_error_retry_requested_routes_to_existing_operation() -> None:
    """The affordance message is wired to the app handler that invokes the
    EXISTING retry operation (``app._retry`` -> ``turn_supervisor.retry``).
    """
    assert ErrorRetryRequested.handler_name == "on_error_retry_requested"
    from chartreux.cli.textual_ui.app import ChartreuxApp

    assert hasattr(ChartreuxApp, "on_error_retry_requested")
