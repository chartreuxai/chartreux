"""R2 — root automatic-recovery races and the end-to-end continuation path.

The race matrix of the bounded automatic root-turn recovery pair: the TUI
controller's dedup, single-flight, cancellation, and meta-recovery behavior
(items 2, 3, 11, 15, 16, 17), the end-to-end continuation the harness
schedules itself after a terminal root failure with NO user action (items 1
and 7 — including the proof that an injected continuation NEVER resets
availability cooldowns), user input before admission cancelling the pending
continuation while processing normally (item 11), and the milestone notice
riding the normal history with a deterministic id (item 18).

Harness: controller-level tests drive ``RootRecoveryController`` over a fake
app with mocked wire methods (deterministic, no sleeps); the end-to-end tests
run a real TUI app over the in-memory transport against a scripted
controlled transport (``tests/recovery/conftest.py``); the shared
availability registry runs on the opt-in ``fake_clock`` and the controller's
timer sleep is patched to advance it, so every cooldown window is exact. All
transports are controlled fakes — an unmocked provider call cannot occur
(fail-closed egress).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from chartreux.app_server.events import TurnCompleted
from chartreux.app_server.models import (
    PublicError,
    PublicNoticeEntry,
    PublicTurn,
    PublicTurnStatus,
    RootRecoveryNoticeDetail,
)
from chartreux.app_server.protocol import (
    RootRecoveryAdmitResponse,
    RootRecoveryScheduleResponse,
)
from chartreux.app_server.session import AppServerTurnError
from chartreux.cli.textual_ui.root_recovery import RootRecoveryController
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ErrorMessage,
    UserCommandMessage,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import RecoverySectionConfig, RootRecoveryConfig
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.utils.retry_prompt import build_retry_prompt
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import ControlledTransport, FakeClock
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


async def _wait_until(pilot: Any, predicate: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for UI state")
        await pilot.pause(0.01)


async def _consume(events: Any) -> list[Any]:
    return [event async for event in events]


async def _drain() -> None:
    """Let in-memory mock chains and armed timer tasks run to a fixed point."""
    for _ in range(25):
        await asyncio.sleep(0)


async def _no_sleep(_delay: float) -> None:
    return None


async def _blocked_sleep(_delay: float) -> None:
    # Never returns on its own: the armed timer parks until the test drives
    # the wake (or cancels the timer).
    await asyncio.Event().wait()


def _enabled_config() -> ChartreuxConfigSchema:
    """Explicitly-enabled ``[recovery.root]`` (round-2 amendment: never rely
    on the development default — WP6 promotes it and must not break tests).
    """
    config = build_test_vibe_config()
    return config.model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=True))
        }
    )


# ---------------------------------------------------------------------------
# The controller over a fake app (the wire surface mocked)
# ---------------------------------------------------------------------------


def _turn_error(turn_id: str = "turn-1") -> AppServerTurnError:
    return AppServerTurnError(
        PublicError(message="Network error", code="backend_error"), turn_id=turn_id
    )


def _turn_completed(
    turn_id: str = "turn-1",
    *,
    status: PublicTurnStatus = PublicTurnStatus.FAILED,
    error: PublicError | None = None,
    next_turn_id: str | None = None,
) -> TurnCompleted:
    return TurnCompleted(
        turn=PublicTurn(
            id=turn_id,
            session_id="session-1",
            status=status,
            started_at=1,
            error=error,
            next_turn_id=next_turn_id,
        )
    )


def _scheduled_response(**overrides: Any) -> RootRecoveryScheduleResponse:
    base: dict[str, Any] = {
        "decision": "scheduled",
        "episode_id": "root-auto-turn-1",
        "idempotency_key": "root-auto-recovery:root-auto-turn-1:1",
        "attempt": 1,
        "limit": 2,
        "delay_seconds": 30.0,
        "eligible_class": "timeout",
        "affected_deployment": "test-first/first",
    }
    base.update(overrides)
    return RootRecoveryScheduleResponse(**base)


def _admitted_response(**overrides: Any) -> RootRecoveryAdmitResponse:
    base: dict[str, Any] = {
        "decision": "admitted",
        "attempt": 1,
        "limit": 2,
        "predecessor_turn_id": "turn-1",
    }
    base.update(overrides)
    return RootRecoveryAdmitResponse(**base)


def _make_app(
    *,
    schedule_response: Any = None,
    admit_response: Any = None,
    session_id: str = "session-1",
) -> Any:
    app = MagicMock()
    app.app_server.session_id = session_id
    app.app_server.schedule_root_recovery = AsyncMock(
        return_value=schedule_response or _scheduled_response()
    )
    app.app_server.admit_root_recovery = AsyncMock(
        return_value=admit_response or _admitted_response()
    )
    app.app_server.cancel_root_recovery = AsyncMock(return_value=True)
    app._input_generation = 0
    # The local queued-work barrier's input: clean by default (the barrier
    # tests flip it explicitly).
    app._queue.has_server_work = False
    app._auto_retry_incomplete_stream = AsyncMock()
    supervisor = MagicMock()
    supervisor._resolve_turn_error_message = MagicMock(return_value="Network error")
    supervisor._mount_turn_error = AsyncMock()
    app._turn_supervisor = supervisor
    return app


def _make_controller(app: Any) -> RootRecoveryController:
    controller = RootRecoveryController()
    controller._bind_app(app)
    return controller


@pytest.mark.asyncio
async def test_controller_deduplicates_solicited_and_unsolicited_observations() -> None:
    """Item 2 (controller half): the solicited ``AppServerTurnError`` and the
    unsolicited ``TurnCompleted`` for the same failed turn produce ONE
    schedule — the (session, terminal turn) dedup key.
    """
    app = _make_app()
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    solicited = await controller.observe_terminal(_turn_error(), solicited=True)
    assert solicited is True
    assert controller.pending is not None

    unsolicited = await controller.observe_terminal(
        _turn_completed(error=PublicError(message="Network error")), solicited=False
    )
    # The duplicate observation neither re-schedules nor drops the armed
    # continuation.
    assert unsolicited is True
    assert app.app_server.schedule_root_recovery.await_count == 1

    await _drain()
    # The armed timer drove exactly one admission and one continuation.
    app.app_server.admit_root_recovery.assert_awaited_once()
    app._auto_retry_incomplete_stream.assert_awaited_once_with(
        1, idempotency_key="root-auto-recovery:root-auto-turn-1:1"
    )
    assert controller.pending is None


@pytest.mark.asyncio
async def test_controller_never_schedules_without_a_typed_terminal_error() -> None:
    """Items 5/6 (controller half): never schedule from generic TUI
    exceptions, disconnects, missing spinners, or non-terminal turns.
    """
    app = _make_app()
    controller = _make_controller(app)

    # A generic TUI exception is not a typed terminal turn error.
    assert (
        await controller.observe_terminal(RuntimeError("boom"), solicited=True) is False
    )
    # A typed error without the terminal turn identity never schedules.
    assert (
        await controller.observe_terminal(
            AppServerTurnError(PublicError(message="x")), solicited=True
        )
        is False
    )
    # A COMPLETED unsolicited turn is not a failure.
    assert (
        await controller.observe_terminal(
            _turn_completed(status=PublicTurnStatus.COMPLETED), solicited=False
        )
        is False
    )
    # A FAILED turn with a queued successor is not terminal.
    assert (
        await controller.observe_terminal(
            _turn_completed(error=PublicError(message="x"), next_turn_id="turn-2"),
            solicited=False,
        )
        is False
    )
    # A FAILED turn without an error payload carries no typed evidence.
    assert (
        await controller.observe_terminal(
            _turn_completed(turn_id="turn-4"), solicited=False
        )
        is False
    )

    app.app_server.schedule_root_recovery.assert_not_awaited()
    assert controller.pending is None
    assert controller._timer is None


@pytest.mark.asyncio
async def test_controller_user_input_before_admission_cancels() -> None:
    """Item 11 (controller half): user input after the schedule cancels the
    pending continuation — the generation token's defense-in-depth half.
    """
    app = _make_app()
    controller = _make_controller(app)
    controller._sleep = _blocked_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    assert controller.pending is not None

    # The user typed / submitted / retried: the generation moved.
    app._input_generation += 1
    await controller._on_wake()

    assert controller.pending is None
    app.app_server.admit_root_recovery.assert_not_awaited()
    app.app_server.cancel_root_recovery.assert_awaited_once_with(
        "root-auto-turn-1", reason="user_input"
    )
    app._auto_retry_incomplete_stream.assert_not_awaited()
    controller.shutdown()


@pytest.mark.asyncio
async def test_controller_session_change_at_wake_drops_the_schedule() -> None:
    """Item 15 (controller half): a session change invalidates the schedule
    at the wake (the server-side gate is invalidated by the reset too).
    """
    app = _make_app()
    controller = _make_controller(app)
    controller._sleep = _blocked_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    app.app_server.session_id = "session-2"
    await controller._on_wake()

    assert controller.pending is None
    app.app_server.admit_root_recovery.assert_not_awaited()
    controller.shutdown()


@pytest.mark.asyncio
async def test_controller_shutdown_cancels_the_timer_and_drops_references() -> None:
    """Item 15 (controller half): shutdown cancels the local timer and drops
    its references (an admitted successor in flight is never cancelled).
    """
    app = _make_app()
    controller = _make_controller(app)
    controller._sleep = _blocked_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    timer = controller._timer
    assert timer is not None

    controller.shutdown()
    assert controller.pending is None
    assert controller._timer is None
    # shutdown() cancels the timer task; the cancellation settles on the next
    # loop iterations (the blocked sleep is cancelled, not awaited).
    await _drain()
    assert timer.done()

    # A wake after shutdown is a no-op.
    await controller._on_wake()
    app.app_server.admit_root_recovery.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_already_admitted_reconciles_without_a_second_successor() -> (
    None
):
    """Item 16 (controller half): an ambiguous admission redelivery
    reconciles to the original successor — never a second one, never a loop.
    """
    app = _make_app(
        admit_response=_admitted_response(decision="already_admitted", turn_id="turn-2")
    )
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    await _drain()

    app._auto_retry_incomplete_stream.assert_not_awaited()
    app._turn_supervisor._mount_turn_error.assert_not_awaited()
    assert controller.pending is None
    assert controller._timer is None


@pytest.mark.asyncio
async def test_controller_consumed_admission_without_a_successor_parks_once() -> None:
    """Items 16/17 (controller half): an admission consumed with no observed
    successor start parks to the manual fallback — one state, never a loop.
    """
    app = _make_app(
        admit_response=_admitted_response(decision="already_admitted", turn_id=None)
    )
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    await _drain()

    app._auto_retry_incomplete_stream.assert_not_awaited()
    # Exactly one manual-fallback mount; the timer is never re-armed.
    app._turn_supervisor._mount_turn_error.assert_awaited_once()
    assert controller.pending is None
    assert controller._timer is None


@pytest.mark.asyncio
async def test_controller_declined_admission_parks_to_the_manual_fallback_once() -> (
    None
):
    """Item 17 (controller half): a declined admission surfaces the stashed
    terminal failure with the manual R1 fallback — once, never a loop.
    """
    app = _make_app(
        admit_response=RootRecoveryAdmitResponse(
            decision="declined", reason="deadline_exceeded"
        )
    )
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    await _drain()

    app._auto_retry_incomplete_stream.assert_not_awaited()
    app._turn_supervisor._mount_turn_error.assert_awaited_once()
    assert controller.pending is None
    assert controller._timer is None


@pytest.mark.asyncio
async def test_controller_deferred_admission_re_arms_within_the_episode() -> None:
    """A deferred admission (a read-only availability wait) re-arms the
    timer within the episode — it never consumes a retry and never starts a
    successor.
    """
    app = _make_app(
        admit_response=RootRecoveryAdmitResponse(
            decision="deferred", attempt=1, limit=2, delay_seconds=1.0
        )
    )
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    await _drain()

    app._auto_retry_incomplete_stream.assert_not_awaited()
    assert app.app_server.admit_root_recovery.await_count >= 1
    assert controller.pending is not None
    controller.shutdown()


@pytest.mark.asyncio
async def test_controller_wire_failure_parks_without_a_self_retry_loop() -> None:
    """Item 17 (controller half): a controller/wire failure is not another
    provider failure — the manual fallback is the one state, never a loop.
    """
    app = _make_app()
    app.app_server.schedule_root_recovery = AsyncMock(
        side_effect=RuntimeError("wire down")
    )
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is False
    assert controller.pending is None
    assert controller._timer is None

    # A duplicate observation of the same failed turn does not retry the wire.
    assert await controller.observe_terminal(_turn_error(), solicited=True) is False
    assert app.app_server.schedule_root_recovery.await_count == 1


@pytest.mark.asyncio
async def test_controller_declined_schedule_never_arms_a_timer() -> None:
    """A gate decline (disabled / not eligible / barrier) never arms a timer
    and never mounts a duplicate error — the caller mounts the R1 fallback.
    """
    app = _make_app(
        schedule_response=RootRecoveryScheduleResponse(
            decision="declined", reason="disabled"
        )
    )
    controller = _make_controller(app)
    controller._sleep = _no_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is False
    await _drain()

    assert controller.pending is None
    assert controller._timer is None
    app.app_server.admit_root_recovery.assert_not_awaited()
    app._turn_supervisor._mount_turn_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_controller_queued_user_work_declines_the_schedule() -> None:
    """Item 12 (controller half): queued user work visible only locally — a
    submission still preparing its prompt holds an enqueue reservation the
    server-side gate cannot see — declines the schedule at observation time.
    The caller mounts the terminal error with the manual R1 fallback, exactly
    like the legacy incomplete-stream continuation (never retry ahead of
    queued prompts).
    """
    app = _make_app()
    app._queue.has_server_work = True
    controller = _make_controller(app)

    assert await controller.observe_terminal(_turn_error(), solicited=True) is False
    assert (
        await controller.observe_terminal(
            _turn_completed(error=PublicError(message="Network error")), solicited=False
        )
        is False
    )
    app.app_server.schedule_root_recovery.assert_not_awaited()
    assert controller.pending is None
    assert controller._timer is None


@pytest.mark.asyncio
async def test_controller_queued_work_at_wake_cancels_and_offers_the_fallback() -> None:
    """Item 12 (controller half): queued user work at the timer wake cancels
    the pending schedule — cancel remotely and show why, offer the manual
    fallback, never silently rearm and never drive ahead of the queued work.
    """
    app = _make_app()
    controller = _make_controller(app)
    controller._sleep = _blocked_sleep

    assert await controller.observe_terminal(_turn_error(), solicited=True) is True
    assert controller.pending is not None

    # A submission is mid-preparation at the wake: it holds an enqueue
    # reservation but no server turn, so the gate's barrier cannot see it.
    app._queue.has_server_work = True
    await controller._on_wake()

    assert controller.pending is None
    assert controller._timer is None
    app.app_server.admit_root_recovery.assert_not_awaited()
    app.app_server.cancel_root_recovery.assert_awaited_once_with(
        "root-auto-turn-1", reason="queued_work"
    )
    app._auto_retry_incomplete_stream.assert_not_awaited()
    await _drain()
    app._turn_supervisor._mount_turn_error.assert_awaited_once()
    controller.shutdown()


# ---------------------------------------------------------------------------
# End-to-end over the real TUI app (in-memory transport, controlled backend)
# ---------------------------------------------------------------------------


def _timeout_fault() -> BackendError:
    """A streaming timeout fault with its httpx cause (classifies TIMEOUT)."""
    fault = BackendError(
        provider="test-first",
        endpoint="https://first.invalid/chat",
        status=None,
        reason="read timed out",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="first",
        payload_summary=PayloadSummary(
            model="first",
            message_count=1,
            approx_chars=2,
            temperature=0.0,
            has_tools=False,
            tool_choice=None,
        ),
    )
    fault.__cause__ = httpx.ReadTimeout("read timed out")
    return fault


def _single_deployment_config() -> ChartreuxConfigSchema:
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
    config = ChartreuxConfigSchema.model_validate(
        {"active_model": "base"}, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    return config.model_copy(
        update={
            "recovery": RecoverySectionConfig(root=RootRecoveryConfig(enabled=True))
        }
    )


@pytest.mark.asyncio
async def test_terminal_timeout_schedules_and_starts_a_continuation_with_no_user_action() -> (
    None
):
    """Item 1 (TUI) + item 7 (TUI): a terminal root TIMEOUT schedules and
    starts the continuation turn with NO user action, and the injected
    continuation NEVER resets the shared availability cooldown (WP0 F6).
    """
    backend = ControlledTransport([
        _timeout_fault(),
        [mock_llm_chunk(content="Recovered")],
    ])
    started, release = backend.gate_attempt(1)
    agent_loop = build_test_agent_loop(
        config=_single_deployment_config(), backend=backend, enable_streaming=True
    )
    # The shared registry runs on a manual clock; the controller's timer
    # sleep advances it past the armed cooldown before the admission.
    clock = FakeClock()
    agent_loop.config_orchestrator._availability_registry = AvailabilityRegistry(
        clock=clock
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)

    async def _advancing_sleep(_delay: float) -> None:
        clock.advance(40.0)

    app._root_recovery._sleep = _advancing_sleep

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))

        # The continuation is in flight (gated): the first failure armed the
        # deployment cooldown, and the injected turn did NOT reset it.
        await _wait_until(pilot, started.is_set)
        registry = agent_loop.config_orchestrator.availability_registry
        assert registry.cooldown_until("base", "test-first") == 30.0

        release.set()
        await _wait_until(pilot, lambda: not app._agent_job_active())

        # One continuation: an injected new turn carrying the retry prompt.
        assert len(backend.requests_messages) == 2
        continuation = backend.requests_messages[1][-1]
        assert continuation.injected is True
        assert continuation.content == build_retry_prompt("")
        assert app.query_one(AssistantMessage).get_content() == "Recovered"
        assert len(app.query(ErrorMessage)) == 0

        # The milestone notices mounted as history lines (never fresh
        # instructions that would trigger another turn).
        notices = " ".join(
            str(message._content) for message in app.query(UserCommandMessage)
        )
        assert "Automatic retry 1 of 2 scheduled" in notices
        assert "started as turn" in notices
        assert "succeeded" in notices


@pytest.mark.asyncio
async def test_user_input_before_admission_cancels_and_processes_normally() -> None:
    """Item 11 (TUI): user input before admission cancels the pending
    continuation (the input-generation token's synchronous half) and the
    input processes normally.
    """
    backend = FakeBackend([
        [mock_llm_chunk(content="partial", stop_reason=None)],
        [mock_llm_chunk(content="Answer")],
    ])
    agent_loop = build_test_agent_loop(
        config=_enabled_config(), backend=backend, enable_streaming=True
    )
    app = build_test_chartreux_app(agent_loop=agent_loop)
    armed = asyncio.Event()
    release = asyncio.Event()

    async def _gated_sleep(_delay: float) -> None:
        armed.set()
        await release.wait()

    app._root_recovery._sleep = _gated_sleep

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("hi"))

        # The failed turn's continuation is scheduled and armed.
        await _wait_until(pilot, lambda: app._root_recovery.pending is not None)

        # User input arrives before the timer fires: the pending continuation
        # is cancelled synchronously and the input processes normally.
        chat_input.post_message(ChatInputContainer.Submitted("there"))
        release.set()
        await _wait_until(pilot, lambda: len(backend.requests_messages) == 2)
        await _wait_until(pilot, lambda: not app._agent_job_active())

        assert app._root_recovery.pending is None
        # The second request is the user's new turn — not a continuation.
        second = backend.requests_messages[1][-1]
        assert second.content == "there"
        assert second.injected is False
        # The failed turn's partial output stays mounted (failed turns are
        # immutable); the user's new turn answers in its own message.
        assistant_contents = [
            message.get_content() for message in app.query(AssistantMessage)
        ]
        assert assistant_contents == ["partial", "Answer"]
        assert len(app.query(ErrorMessage)) == 0


# ---------------------------------------------------------------------------
# The app-server surface: the solicited error's identity + the notice history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_root_recovery_notice_rides_the_normal_history() -> None:
    """Item 18 (app-server half): the solicited ``AppServerTurnError``
    carries the terminal turn identity, and the milestone notice is a durable
    history entry with a deterministic id — readable via the normal history
    (across a reconnect), never injected as a fresh instruction.
    """
    backend = FakeBackend([[mock_llm_chunk(content="partial", stop_reason=None)]])
    agent_loop = build_test_agent_loop(
        config=_enabled_config(), backend=backend, enable_streaming=True
    )
    session = await create_test_app_server_session(agent_loop)
    try:
        with pytest.raises(AppServerTurnError) as raised:
            await _consume(session.act("hi", client_message_id="u1"))
        turn_id = raised.value.turn_id
        assert turn_id is not None

        scheduled = await session.schedule_root_recovery(
            turn_id, solicited=True, input_generation=0
        )
        assert scheduled.decision == "scheduled"
        assert scheduled.episode_id == f"root-auto-{turn_id}"

        history = await session.resources.sessions.get_session_history(
            session.session_id
        )
        notices = [
            entry
            for entry in history
            if isinstance(entry, PublicNoticeEntry)
            and isinstance(entry.detail, RootRecoveryNoticeDetail)
        ]
        assert len(notices) == 1
        entry = notices[0]
        assert isinstance(entry.detail, RootRecoveryNoticeDetail)
        # The deterministic notice id: exactly-once at the projection layer.
        assert entry.id == f"root-recovery:scheduled:root-auto-{turn_id}:1"
        assert entry.level == "info"
        assert entry.detail.milestone == "scheduled"
        assert entry.detail.episode_id == f"root-auto-{turn_id}"
        assert entry.detail.attempt == 1
        assert entry.detail.limit == 2
        assert "Automatic retry 1 of 2 scheduled" in entry.message
        assert "No user action is required" in entry.message
    finally:
        await session.close()
        await agent_loop.aclose()


@pytest.mark.asyncio
async def test_act_consumer_close_interrupts_the_uncompleted_server_turn() -> None:
    """The consumer-close race: ``aclosing`` closes the act() stream with
    GeneratorExit at the yield — which a CancelledError-only arm bypasses —
    so the server turn would keep running unobserved (the TUI's agent-job
    state keys on ``turn_active`` and would wedge busy forever, e.g. an Escape
    landing while the driving task processes an event instead of waiting on
    the stream). The stream close must interrupt the uncompleted turn.
    """
    gate = asyncio.Event()

    class GatedBackend(FakeBackend):
        async def complete_streaming(self, **kwargs):
            async for chunk in super().complete_streaming(**kwargs):
                yield chunk
            # Keep the stream open past its chunks: the turn stays active.
            await gate.wait()

    backend = GatedBackend([[mock_llm_chunk(content="partial", stop_reason=None)]])
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)
    try:
        stream = session.act("hi", client_message_id="u1")
        await stream.__anext__()
        assert session.turn_active

        # Consumer close with the turn uncompleted: GeneratorExit at the
        # yield, NOT a cancellation thrown into the stream.
        await stream.aclose()

        deadline = time.monotonic() + 2.0
        while session.turn_active and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert not session.turn_active
    finally:
        gate.set()
        await session.close()
        await agent_loop.aclose()
