"""Turn/interrupt/retry supervisor controller for the TUI app (W4b C1, stage 10)."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing, suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from chartreux.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
    TurnCompleted,
    TurnStarted,
)
from chartreux.app_server.models import (
    PublicCallbackEntry,
    PublicMessageEntry,
    PublicTurnStatus,
    TurnErrorCode,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import AppServerResponseError
from chartreux.cli.textual_ui.notifications import NotificationContext
from chartreux.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from chartreux.cli.textual_ui.widgets.loading import (
    DEFAULT_LOADING_STATUS,
    INTERRUPTING_LOADING_STATUS,
    paused_timer,
)
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, InterruptMessage
from chartreux.observability.logging import logger
from chartreux.utils.retry_prompt import build_retry_prompt

if TYPE_CHECKING:
    from chartreux.app_server.models import (
        PreparedPrompt,
        PublicError,
        PublicHistoryEntry,
        UserQuestionRequest,
    )
    from chartreux.cli.textual_ui.app import ChartreuxApp
    from chartreux.cli.textual_ui.widgets.messages import SlashCommandMessage
    from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

# Expected turn outcomes with bespoke user messages; not worth reporting to Sentry.
_BENIGN_TURN_ERROR_CODES = {
    TurnErrorCode.RATE_LIMIT,
    TurnErrorCode.CONTEXT_TOO_LONG,
    TurnErrorCode.RESPONSE_TOO_LONG,
    TurnErrorCode.REFUSAL,
    TurnErrorCode.INCOMPLETE_STREAM,
    TurnErrorCode.INVALID_MODEL,
    TurnErrorCode.INVALID_API_KEY,
}

_RETRYABLE_TURN_ERROR_CODES = {
    TurnErrorCode.BACKEND_ERROR,
    TurnErrorCode.RATE_LIMIT,
    TurnErrorCode.RESPONSE_TOO_LONG,
    TurnErrorCode.INCOMPLETE_STREAM,
}

# R1 (plan v0.5.6-recovery): the terminal root-failure codes that display the
# keyboard-accessible "Retry — new turn" affordance. The affordance's
# retryable set is the existing retryable set PLUS exhaustion-after-cooldown
# (ALL_DEPLOYMENTS_UNAVAILABLE): the user pressing Retry is a genuine user
# action, and the existing manual retry operation's intentional cooldown reset
# (the only path allowed to reset it, ``_loop.py``) is exactly what makes a
# retry meaningful once every deployment was excluded. The error text stays
# honest that deployments were unavailable and that a manual retry resets the
# model's cooldowns (``_exhaustion_retry_hint``).
_RETRY_AFFORDANCE_TURN_ERROR_CODES = _RETRYABLE_TURN_ERROR_CODES | {
    TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE
}

# R1: transport-disconnect / suspected-hang signals. A lost app-server
# connection or a stalled transport surfaces as a generic exception, not a
# typed turn error. These get inspection/reconnect GUIDANCE — never a retry
# affordance and never an automatic continuation.
_DISCONNECT_ERROR_TYPES: tuple[type[Exception], ...] = (ConnectionError, TimeoutError)

# R2/OD3: the mirrored default of the gate-controlled episode budget
# ([recovery.root] auto_retries_max=2) — the legacy incomplete-stream
# continuation is MERGED into the gate, so this constant documents the shared
# limit rather than deciding it (kept for the app.py re-export and the R0
# per-episode bound test).
_MAX_INCOMPLETE_STREAM_RETRIES = 2

# Interrupt wait/settle timeouts. app.py re-exports these names, so the
# module-global seam is preserved: tests patch
# ``chartreux.cli.textual_ui.app._INTERRUPT_WAIT_TIMEOUT`` /
# ``_INTERRUPT_SETTLE_TIMEOUT``, and the controller reads the values through
# app-side wrappers that resolve the app module global at call time.
_INTERRUPT_WAIT_TIMEOUT = 2.0
_INTERRUPT_SETTLE_TIMEOUT = 30.0
_INTERRUPT_STILL_STOPPING_WARNING = (
    "The turn is still stopping. New turns will wait until it has fully stopped."
)

# A provider error string can embed a multi-kilobyte (or megabyte) single-line
# excerpt. The first line (provider, model, error head) stays visible and the
# ErrorMessage disclosure shows the retained excerpt; anything beyond this
# bound is dropped here so a megabyte blob is never carried into the TUI.
_MAX_TURN_ERROR_CHARS = 4000


def _truncate_error_payload(message: str) -> str:
    if len(message) <= _MAX_TURN_ERROR_CHARS:
        return message
    omitted = len(message) - _MAX_TURN_ERROR_CHARS
    return (
        f"{message[:_MAX_TURN_ERROR_CHARS]}\n…[truncated; {omitted} characters omitted]"
    )


def _public_entry(event: AppServerEvent) -> PublicHistoryEntry | None:
    match event:
        case HistoryEntryAdded(entry=entry) | HistoryEntryUpdated(entry=entry):
            return entry
        case _:
            return None


def _transport_disconnect_guidance(error: Exception) -> str:
    """R1: inspection/reconnect guidance for a transport disconnect or a
    suspected hang — never a retry affordance, never an automatic
    continuation.

    A lost app-server connection surfaces as a generic exception (not a typed
    ``AppServerTurnError``), so it never enters the retry affordance set. This
    is not a watchdog: a stalled turn with no terminal failure produces no
    error widget at all.
    """
    from chartreux.app_server import AppServerConnectionClosed

    if not isinstance(error, (AppServerConnectionClosed, *_DISCONNECT_ERROR_TYPES)):
        return ""
    return (
        "\n\nThe connection to the app server was lost or the turn stalled. "
        "Inspect the session and server state; the client reconnects "
        "automatically, and a new turn can be started once the connection is "
        "back. Nothing is continued automatically."
    )


def _message_matches_active_callback(
    active_callback: PublicCallbackEntry | None,
    message: QuestionApp.Answered | QuestionApp.Cancelled,
) -> bool:
    """Correlate QuestionApp messages with callback identity (plan U).

    A server-originated question posts its callback id; a local question
    posts None. A stale message (an expired or replaced question) never
    answers the active callback and never resolves a local question.
    """
    if message.callback_id is None:
        # Local question: only valid while no server callback is active.
        return active_callback is None
    return (
        active_callback is not None
        and active_callback.callback_id == message.callback_id
    )


class TurnSupervisor:
    """Owns the turn/interrupt/retry state and flow for one app.

    Plain object (no mixins): the app binds itself late via ``_bind_app``
    (never constructor-captured), and collaborators are invoked as
    ``self._app._method(...)`` at call time so instance patches keep
    intercepting. The app keeps forwarding properties for every moved state
    field, so direct-attribute consumers (tests, other controllers) see no
    change.

    The interrupt triplet ``interrupt_turn`` / ``finish_interrupt`` /
    ``finalize_turn_ui`` lives in this one module by invariant: there is no
    await between cancelling the owned agent task and settling turn
    ownership, so a late interrupt cannot cancel a replacement turn.

    Turn driving (``handle_turn``), the unsolicited-turn lifecycle, turn-error
    presentation, and outcome tracking also live here: the controller owns
    ``_agent_task`` so a retry claims turn ownership immediately after
    admission, and the interrupt supervisor can observe idle ownership to
    settle the interrupt gate.
    """

    # Late-bound host reference, assigned via _bind_app after construction.
    _app: ChartreuxApp

    def __init__(
        self,
        *,
        interrupt_wait_timeout: Callable[[], float],
        interrupt_settle_timeout: Callable[[], float],
    ) -> None:
        # Late-bound reads of the re-exported app module-global seams: tests
        # patch chartreux.cli.textual_ui.app._INTERRUPT_WAIT_TIMEOUT /
        # _INTERRUPT_SETTLE_TIMEOUT, and the app-side wrappers resolve the
        # module global at call time so those patches stay effective.
        self._interrupt_wait_timeout = interrupt_wait_timeout
        self._interrupt_settle_timeout = interrupt_settle_timeout
        self.interrupt_requested = False
        # A locally submitted turn is "in flight" from the moment the user hits
        # Enter until the server confirms it started (TurnStarted). The unified
        # backend runs turns server-side (enqueue -> promote -> TurnStarted), so
        # unlike the legacy local `agent_task` there is a gap where nothing looks
        # active. This flag bridges that gap so the spinner, busy-routing, and
        # interruptibility all derive from one "turn in flight" notion; the event
        # lets an interrupt fired during the gap wait for the turn to surface.
        self.pending_turn = False
        self.turn_started_event = asyncio.Event()
        # An interrupt settles asynchronously: the server pauses the queue and
        # completes the turn out-of-band, so a submit fired right after Escape
        # would otherwise race that propagation and read stale queue state.
        self.interrupt_pending = False
        self.interrupt_settled = asyncio.Event()
        self.interrupt_settled.set()
        # Owns a timed-out cancellation/interrupt until it actually completes.
        # Keeping this separate from turn activity closes the interval where the
        # server reports idle before the interrupt request itself has returned.
        self.interrupt_operation: asyncio.Task[None] | None = None
        # Deferred submits (those that waited out an interrupt) run in workers;
        # this lock keeps them ordered so a fast double-Enter stays FIFO, and the
        # counter routes later submits through the same lock until they drain.
        self.deferred_submit_lock = asyncio.Lock()
        self.deferred_submits = 0
        self.agent_task: asyncio.Task | None = None
        self.turn_ui_generation = 0
        self.turn_ui_mutex: asyncio.Lock | None = None
        self.turn_has_final_prose = False
        self.active_callback: PublicCallbackEntry | None = None
        self.pending_callbacks: deque[PublicCallbackEntry] = deque()
        self.callback_submitting = False
        self.callback_delivery_uncertain = False
        self.pending_local_question: asyncio.Future[UserQuestionResult] | None = None
        self.active_turn_start: int | None = None
        # Unsent drafts preserved locally when a question expires unanswered
        # (plan U/OD1); keyed by callback id, never auto-delivered.
        self.expired_question_drafts: dict[str, dict[int, str]] = {}

    def _bind_app(self, app: ChartreuxApp) -> None:
        """Bind the host app after construction."""
        self._app = app

    async def finalize_turn_ui(
        self, *, notify_complete: bool = True, turn_ui_generation: int | None = None
    ) -> None:
        async with self._app._turn_ui_lock():
            if (
                turn_ui_generation is not None
                and turn_ui_generation != self.turn_ui_generation
            ):
                self.interrupt_requested = False
                self.agent_task = None
                self._clear_pending_turn()
                self._app._queue.notify_busy_changed()
                self.maybe_settle_interrupt()
                return
            self.interrupt_requested = False
            self.agent_task = None
            self._clear_pending_turn()
            self.maybe_settle_interrupt()
            if self._app._loading_widget:
                await self._app._loading_widget.remove()
            self._app._loading_widget = None
            if self._app.event_handler:
                await self._app.event_handler.finalize_streaming()
                self._app.event_handler.escalate_unresolved_errors()
            self._app._queue.notify_busy_changed()
            if not notify_complete:
                self._app._sync_retry_command_availability()
                return
            if self._app.event_handler:
                self._app.event_handler.clear_tool_call_anchors()
            try:
                self._app._admit_live_history()
            except Exception:
                logger.exception("Failed to admit live history while finalizing turn")
            if self._app.event_handler:
                self._app.event_handler.settle_turn()
            self._app._sync_retry_command_availability()
            await self._app._refresh_windowing_from_history()
            self._app._clear_active_turn_pins()
            self._app._request_transcript_reconcile()
            self._app._terminal_notifier.notify(NotificationContext.COMPLETE)
            self._app.call_after_refresh(self._app._maybe_show_graduation)

    async def interrupt_turn(self) -> None:
        if self.interrupt_operation is not None and not self.interrupt_operation.done():
            return
        if self.interrupt_requested:
            return
        if not self._app._agent_job_active():
            # The turn finished before this worker ran, so there is nothing to
            # interrupt. The Escape handler already opened the settle window via
            # _begin_interrupt_settle; close it here (no turn/finalize event is
            # coming to do it) so a racing submit resumes on current state
            # instead of waiting out the full _await_interrupt_settled timeout.
            # Settle unconditionally rather than via _maybe_settle_interrupt:
            # this path never issues interrupt(), so no queue pause is coming,
            # and _maybe_settle_interrupt would keep the window open while
            # accepted-but-unpaused items exist -- waiting on a pause that a
            # real interrupt would emit but this no-op never will.
            self._app._clear_active_turn_pins()
            self.settle_interrupt()
            return

        self.interrupt_requested = True

        # Acknowledge the keypress immediately: the cancel below blocks until the
        # in-flight turn unwinds (LLM stream, tool call, server ack), and only
        # then does the spinner tear down. Flipping the status now, on the message
        # pump, gives instant feedback instead of a spinner that keeps running.
        if self._app._loading_widget is not None:
            self._app._loading_widget.set_status(INTERRUPTING_LOADING_STATUS)

        self.active_callback = None
        self.callback_delivery_uncertain = False
        self.pending_callbacks.clear()
        self._app._indicate_pending_action()
        self._cancel_pending_question(UserQuestionResult(answers=[], cancelled=True))

        # A just-submitted idle turn may still be promoting (enqueued, no
        # TurnStarted yet). Wait briefly for it to surface so we interrupt the
        # real turn instead of no-opping; _clear_pending_turn on TurnStarted or
        # finalize wakes this. Bounded so a turn that never starts can't hang.
        if (
            self.pending_turn
            and not self._app.app_server.turn_active
            and (self.agent_task is None or self.agent_task.done())
        ):
            with suppress(TimeoutError):
                await asyncio.wait_for(self.turn_started_event.wait(), timeout=30.0)
            # The turn surfacing remounts the loading widget as "Generating";
            # re-apply the interrupting label to the fresh one.
            if self._app._loading_widget is not None:
                self._app._loading_widget.set_status(INTERRUPTING_LOADING_STATUS)
        self._clear_pending_turn()

        # Invariant: a turn is driven EITHER by a client task (``_agent_task``,
        # used for retry/auto-retry) OR by the server promoting an enqueued turn
        # (``turn_active`` with no client task, the normal path for both the
        # legacy and unified backends) -- never both at once. The two branches
        # are therefore mutually exclusive by design. If that ever breaks (a turn
        # live locally AND server-side), the server turn would not be interrupted
        # here and the settle would wait out its full timeout; keep this an elif
        # rather than two ifs so a late interrupt() cannot cancel a replacement
        # turn a deferred submit promoted during the cancel await.
        interrupt: Awaitable[None] | None = None
        if self.agent_task and not self.agent_task.done():
            # Cancelling the client task tears down its act() stream, which
            # interrupts the server turn -- no explicit interrupt() needed.
            self.agent_task.cancel()
            interrupt = self.agent_task
        elif self._app.app_server.turn_active:
            interrupt = self._app.app_server.interrupt()

        if interrupt is None:
            self._app._clear_active_turn_pins()
            self.settle_interrupt()
            return

        self.interrupt_operation = asyncio.create_task(self.finish_interrupt(interrupt))
        try:
            await asyncio.wait_for(
                asyncio.shield(self.interrupt_operation),
                timeout=self._interrupt_wait_timeout(),
            )
        except TimeoutError:
            self._app.notify(
                _INTERRUPT_STILL_STOPPING_WARNING,
                severity="warning",
                markup=False,
                timeout=10,
            )

    async def finish_interrupt(self, interrupt: Awaitable[None]) -> None:
        """Consume an interrupt operation and finalize only after it really settles."""
        succeeded = False
        try:
            try:
                await interrupt
                succeeded = True
            except asyncio.CancelledError:
                # Cancellation is the expected completion for a client-owned turn.
                # A cancellation of this supervisor itself is shutdown, however,
                # and must propagate after the ownership state is settled below.
                if (task := asyncio.current_task()) is not None and task.cancelling():
                    raise
                succeeded = True
            except Exception as exc:
                logger.warning("Interrupt failed", exc_info=exc)
                self._app.notify(
                    f"Failed to stop the turn: {exc}",
                    severity="warning",
                    markup=False,
                    timeout=10,
                )

            if succeeded:
                if self._app.event_handler:
                    self._app.event_handler.stop_current_tool_call(cancelled=True)
                    self._app.event_handler.stop_current_compact()
                    await self._app.event_handler.finalize_streaming()

                await self._app._loading_area.remove_children()
                self._app._loading_widget = None
                await self._app._mount_and_scroll(InterruptMessage())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # The supervisor owns UI finalization as well as the transport
            # operation. Consume late DOM errors so timed-out supervisors cannot
            # become unobserved task failures.
            logger.warning("Interrupt finalization failed", exc_info=exc)
        finally:
            self.interrupt_requested = False
            self.interrupt_operation = None
            self._app._queue.notify_busy_changed()
            self.maybe_settle_interrupt()

    def handle_agent_running_escape(self) -> None:
        self.begin_interrupt_settle()
        self._app.run_worker(self.interrupt_turn(), exclusive=False)

    def begin_interrupt_settle(self) -> None:
        """Open the interrupt window so a racing submit waits for fresh state.

        A submit that lands before the server pauses the queue and completes the
        turn would branch on stale ``paused``/``turn_active`` values, losing the
        prompt or flashing it as queued. ``_maybe_settle_interrupt`` closes the
        window once the queue reflects the interrupt.
        """
        if self.interrupt_pending:
            return
        self.interrupt_pending = True
        self.interrupt_settled.clear()

    def settle_interrupt(self) -> None:
        """Close the interrupt window, waking any submit deferred behind it."""
        self.interrupt_pending = False
        self.interrupt_settled.set()

    def maybe_settle_interrupt(self) -> None:
        if not self.interrupt_pending:
            return
        if self._app._agent_job_active():
            return
        # Accepted server items that are not yet paused mean the interrupt's
        # queue transition is still in flight: wait so a submit resumes cleanly
        # instead of enqueuing into a queue that is about to pause. An empty
        # queue (or one already paused) emits nothing more, so this settles.
        # Reading the live queue rather than a snapshot avoids stalling when the
        # server never pauses (e.g. in-flight enqueues or a turn that raced to a
        # natural completion): those leave no accepted items to wait on.
        if self._app._queue.has_removable and not self._app._queue.paused:
            return
        self.settle_interrupt()

    async def await_interrupt_settled(self) -> None:
        if not self.interrupt_pending:
            return
        try:
            await asyncio.wait_for(
                self.interrupt_settled.wait(), timeout=self._interrupt_settle_timeout()
            )
        except TimeoutError:
            # The settle window is only a stale-state guard when no interrupt
            # request remains owned. A timed-out request must keep submissions
            # deferred until its supervisor consumes the eventual outcome.
            operation = self.interrupt_operation
            if operation is not None and not operation.done():
                await self.interrupt_settled.wait()
            else:
                self.settle_interrupt()

    def agent_job_active(self) -> bool:
        if self.pending_turn:
            return True
        if self.interrupt_operation is not None and not self.interrupt_operation.done():
            return True
        if self._app._app_server is not None and self._app._app_server.turn_active:
            return True
        return self.agent_task is not None and not self.agent_task.done()

    def begin_pending_turn(self) -> None:
        """Enter the 'turn in flight' state for a just-submitted idle prompt."""
        self.pending_turn = True
        self.turn_has_final_prose = False
        if hasattr(self._app, "_turn_outcome_notice"):
            self._app._turn_outcome_notice.hide()
        if self.active_turn_start is None:
            self.active_turn_start = self._app._transcript.admitted_end_index
        self.turn_started_event.clear()

    def _clear_pending_turn(self) -> None:
        """Leave the in-flight state and wake anything waiting for turn start."""
        self.pending_turn = False
        self.turn_started_event.set()

    async def _handle_turn_error(self, *, cancelled: bool = False) -> None:
        if self._app._loading_widget and self._app._loading_widget.parent:
            await self._app._loading_widget.remove()
        if self._app.event_handler:
            self._app.event_handler.stop_current_tool_call(
                success=False, cancelled=cancelled
            )

    async def ensure_runtime_ready(self) -> None:
        # Must precede any app_server access — the session may still be unbound
        # when a turn is dispatched (only _handle_turn reaches here).
        await self._app._session_ready.wait()
        show_init_spinner = not self._app.app_server.resources.runtime.ready
        if show_init_spinner:
            await self._app._ensure_loading_widget("Initializing", show_hint=False)
        await self._app.app_server.resources.runtime.wait_until_ready()
        if show_init_spinner:
            await self._app._remove_loading_widget()
            self._app._refresh_banner()

    async def _handle_turn_events(
        self, events: AsyncGenerator[AppServerEvent, None]
    ) -> None:
        async for event in events:
            await self._app._handle_turn_event(event)

    def show_no_output_outcome(self) -> None:
        # agent_is_active is read through the app wrapper so tests patching
        # chartreux.cli.textual_ui.app.agent_is_active keep intercepting.
        agents_running = any(
            self._app._agent_is_active(agent) for agent in self._app._agent_summaries
        )
        self._app._turn_outcome_notice.show(
            "Info: Main turn completed without a final response; "
            + (
                "background agents are still running."
                if agents_running
                else "ready for another prompt."
            ),
            timeout=None,
        )

    def _track_turn_outcome(self, event: AppServerEvent) -> None:
        if isinstance(event, TurnStarted):
            self.turn_has_final_prose = False
            self._app._turn_outcome_notice.hide()
        elif isinstance(event, (HistoryEntryAdded, HistoryEntryUpdated)):
            entry = _public_entry(event)
            if isinstance(entry, PublicMessageEntry) and entry.role == "assistant":
                self.turn_has_final_prose |= bool(entry.text.strip())
        elif isinstance(event, TurnCompleted):
            if (
                event.turn.next_turn_id is None
                and event.turn.status is PublicTurnStatus.COMPLETED
                and not self.turn_has_final_prose
            ):
                self.show_no_output_outcome()

    async def _begin_unsolicited_turn(self) -> None:
        async with self._app._turn_ui_lock():
            self.turn_ui_generation += 1
            if self._app.event_handler:
                await self._app.event_handler.finalize_streaming()
                self._app.event_handler.escalate_unresolved_errors()
            await self._app._remove_loading_widget()
            # If Escape landed during the enqueue -> promote -> TurnStarted gap,
            # an interrupt is already in flight; mount the fresh spinner as
            # "Interrupting" so surfacing the turn does not clobber that feedback
            # back to "Generating" (the remount races _interrupt_turn's own label).
            await self._app._ensure_loading_widget(
                INTERRUPTING_LOADING_STATUS
                if self.interrupt_requested
                else DEFAULT_LOADING_STATUS
            )
            self._app._queue.notify_busy_changed()

    async def _complete_unsolicited_turn(self, event: TurnCompleted) -> None:
        from chartreux.app_server.session import AppServerTurnError

        if event.turn.next_turn_id is not None:
            return
        recovery_pending = False
        if event.turn.status is PublicTurnStatus.FAILED:
            error = AppServerTurnError(event.turn.error, turn_id=event.turn.id)
            await self._handle_turn_error()
            # R2/OD3: the one terminal observation path — the app-server
            # gate decides eligibility/budget/barriers; the TUI controller
            # owns the local timer. The legacy incomplete-stream immediate
            # continuation is MERGED into that controller (OD4a: when
            # [recovery.root] is disabled no automatic continuation runs).
            recovery_pending = await self._app._root_recovery.observe_terminal(
                event, solicited=False
            )
            if recovery_pending:
                if self._app.event_handler is not None:
                    self._app.event_handler.offer_retry()
            else:
                message = self._resolve_turn_error_message(error)
                await self._mount_turn_error(error, message)
        elif event.turn.status is PublicTurnStatus.INTERRUPTED:
            await self._handle_turn_error(cancelled=True)
        await self.finalize_turn_ui(notify_complete=not recovery_pending)

    async def handle_turn(
        self,
        prompt: str,
        *,
        prepared_prompt: PreparedPrompt | None = None,
        client_message_id: str | None = None,
        injected: bool = False,
        user_initiated_retry: bool = False,
        incomplete_stream_retries: int = 0,
        idempotency_key: str | None = None,
    ) -> None:
        turn_ui_generation = self.turn_ui_generation
        if self.active_turn_start is None:
            self.active_turn_start = self._app._transcript.admitted_end_index
        await self._app._remove_loading_widget()
        recovery_pending = False

        try:
            await self.ensure_runtime_ready()
            await self._app._ensure_loading_widget(
                "Retrying" if incomplete_stream_retries else DEFAULT_LOADING_STATUS
            )
            if injected:
                prompt_text = prompt
                auto_title = None
                images = None
                mentions = None
            else:
                prepared = prepared_prompt or await self._app._prepare_prompt_or_abort(
                    prompt
                )
                if prepared is None:
                    return
                prompt_text = prepared.prompt_text
                auto_title = prepared.auto_title
                images = prepared.images or None
                mentions = prepared.mentions
            message_id = None if injected else client_message_id or str(uuid4())
            async with aclosing(
                self._app.app_server.act(
                    prompt_text,
                    client_message_id=message_id,
                    auto_title=auto_title,
                    images=images,
                    mention_stats=mentions,
                    injected=injected,
                    user_initiated_retry=user_initiated_retry,
                    idempotency_key=idempotency_key,
                )
            ) as events:
                await self._handle_turn_events(events)
        except asyncio.CancelledError:
            await self._handle_turn_error(cancelled=True)
            raise
        except Exception as e:
            await self._handle_turn_error()

            # _watch_init_completion already rendered the fatal startup error
            # and told the user to exit -- don't duplicate the message.
            if self._app._fatal_init_error:
                return

            # R2/OD3: the one terminal observation path — the app-server
            # gate decides eligibility/budget/barriers; the TUI controller
            # owns the local timer. The legacy incomplete-stream immediate
            # continuation is MERGED into that controller (OD4a: when
            # [recovery.root] is disabled no automatic continuation runs at
            # all — the manual R1 retry affordance is the fallback).
            recovery_pending = await self._app._root_recovery.observe_terminal(
                e, solicited=True
            )
            if recovery_pending:
                # Auto-retry silently: don't surface the error while we retry.
                # Arm the retry presentation (without an error widget) so the
                # partial assistant message is reused when the retry streams in.
                if self._app.event_handler is not None:
                    self._app.event_handler.offer_retry()
            else:
                logger.exception("App server turn failed")

                message = self._resolve_turn_error_message(e)

                await self._mount_turn_error(e, message)
        finally:
            await self.finalize_turn_ui(
                notify_complete=not recovery_pending,
                turn_ui_generation=turn_ui_generation,
            )

    async def _mount_turn_error(self, error: Exception, message: str) -> None:
        from chartreux.app_server.session import AppServerTurnError

        event_handler = self._app.event_handler
        if (
            event_handler is not None
            and isinstance(error, AppServerTurnError)
            and error.error.code == TurnErrorCode.INVALID_API_KEY
        ):
            event_handler.cancel_retry_presentation()
        # R1: an authoritative terminal retryable root failure displays the
        # keyboard-accessible "Retry — new turn" affordance on the error
        # widget and arms the retry presentation, so the affordance invokes
        # the EXISTING retry operation (never resume_agent, never automatic).
        offer_retry = (
            event_handler is not None
            and isinstance(error, AppServerTurnError)
            and error.error.code in _RETRY_AFFORDANCE_TURN_ERROR_CODES
        )
        widget = ErrorMessage(
            message, collapsed=self._app._tools_collapsed, retryable=offer_retry
        )
        await self._app._mount_and_scroll(widget)
        if offer_retry and event_handler is not None:
            event_handler.offer_retry(widget)
            self._app._refresh_command_registry()

    def _resolve_turn_error_message(self, e: Exception) -> str:
        from chartreux.app_server.session import AppServerTurnError

        if not isinstance(e, AppServerTurnError):
            # R1: a transport disconnect / suspected hang gets
            # inspection/reconnect guidance — never a retry affordance.
            return (
                f"{_truncate_error_payload(str(e))}{_transport_disconnect_guidance(e)}"
            )
        code = e.error.code
        match code:
            case TurnErrorCode.RATE_LIMIT:
                base = self._rate_limit_message()
            case TurnErrorCode.CONTEXT_TOO_LONG:
                return self._context_too_long_message()
            case TurnErrorCode.REFUSAL:
                return self._refusal_message(e.error)
            case _:
                base = _truncate_error_payload(str(e))
        if code in _RETRY_AFFORDANCE_TURN_ERROR_CODES:
            # PublicError.code arrives as a plain str (pydantic enum
            # coercion), so compare by value, never by identity.
            if code == TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE:
                return f"{base}{self._exhaustion_retry_hint()}"
            return f"{base}{self._retry_hint()}"
        return base

    @staticmethod
    def _retry_hint() -> str:
        return (
            "\n\nRetry — new turn: use the button below, press Ctrl+R, or run "
            "/retry [additional instructions]. Completed tool results are "
            "kept; recent actions may repeat."
        )

    @staticmethod
    def _exhaustion_retry_hint() -> str:
        # R1: honest exhaustion text — the deployments really were
        # unavailable, and only a manual user retry resets the model's
        # cooldowns (the sole path allowed to do so).
        return (
            "\n\nAll deployments for this model were unavailable. Retry — new "
            "turn: use the button below, press Ctrl+R, or run /retry "
            "[additional instructions]; a manual retry is your request and "
            "resets the model's cooldowns. Completed tool results are kept; "
            "recent actions may repeat."
        )

    def _rate_limit_message(self) -> str:
        return "Rate limits exceeded. Please wait a moment before trying again."

    def _context_too_long_message(self) -> str:
        return (
            "The conversation context exceeds the model's maximum limit. "
            "The last messages and output of agent actions went above the allowed size.\n\n"
            "To recover:\n"
            "1. Use /rewind to undo recent messages and tool outputs\n"
            "2. Then use /compact to summarize the remaining conversation\n\n"
            "This will free up context space so you can continue working."
        )

    def _refusal_message(self, error: PublicError) -> str:
        details = error.details if isinstance(error.details, dict) else {}
        category = details.get("category")
        explanation = details.get("explanation")
        lead = "The model declined to respond and stopped early (refusal)."
        if isinstance(category, str):
            lead += f"\nCategory: {category}."
        detail = (
            explanation
            if isinstance(explanation, str)
            else (
                "This can happen with certain prompts or content. "
                "Try rephrasing your request or starting a new conversation."
            )
        )
        return f"{lead}\n\n{detail}"

    async def auto_retry_incomplete_stream(
        self, incomplete_stream_retries: int, *, idempotency_key: str | None = None
    ) -> None:
        # R2/OD3: the continuation DRIVER for the bounded automatic
        # root-turn recovery (the legacy incomplete-stream continuation is
        # MERGED into the gate-controlled episode). ``idempotency_key`` is
        # the gate's single-flight admission key; the continuation is an
        # injected, NON-user-initiated new turn, so it never resets
        # availability cooldowns (WP0 F6).
        # Automatic retries are new turns too. Keep them behind the same
        # interrupt admission gate as submitted prompts so late interrupt
        # finalization cannot mutate the replacement turn's presentation.
        await self._app._await_interrupt_settled()
        if self._app.event_handler is None or not self._app.event_handler.begin_retry():
            return
        self.agent_task = asyncio.current_task()
        self._app._queue.notify_busy_changed()
        await self._app._handle_turn(
            build_retry_prompt(""),
            injected=True,
            incomplete_stream_retries=incomplete_stream_retries,
            idempotency_key=idempotency_key,
        )

    async def retry(
        self,
        cmd_args: str = "",
        command_message: SlashCommandMessage | None = None,
        incomplete_stream_retries: int = 0,
        **_kwargs: Any,
    ) -> None:
        # _agent_job_active and _handle_turn are read through the app
        # delegates so instance patches on them keep intercepting.
        if self._app._agent_job_active():
            return
        # R1: no retry action while stopping (an interrupt is still settling)
        # or while a runtime approval / user question is unresolved.
        if self.interrupt_requested:
            return
        if self.active_callback is not None:
            return
        if self._app.event_handler is None or not self._app.event_handler.begin_retry(
            command_message
        ):
            return
        self._app._refresh_command_registry()
        self.agent_task = asyncio.create_task(
            self._app._handle_turn(
                build_retry_prompt(cmd_args),
                injected=True,
                user_initiated_retry=True,
                incomplete_stream_retries=incomplete_stream_retries,
            )
        )
        self._app._queue.notify_busy_changed()

    async def question_answered(self, message: QuestionApp.Answered) -> None:
        if not _message_matches_active_callback(self.active_callback, message):
            return
        result = UserQuestionResult(answers=message.answers, cancelled=False)
        if self.active_callback is not None:
            await self._app._respond_to_active_callback(
                UserInputCallbackOutput(result=result)
            )
            return
        self._cancel_pending_question(result)

    async def question_cancelled(self, message: QuestionApp.Cancelled) -> None:
        if not _message_matches_active_callback(self.active_callback, message):
            return
        if self.active_callback is not None and (
            self.callback_submitting or self.callback_delivery_uncertain
        ):
            return
        result = UserQuestionResult(answers=[], cancelled=True)
        if self.active_callback is not None:
            await self._app._respond_to_active_callback(
                UserInputCallbackOutput(result=result)
            )
            return
        self._cancel_pending_question(result)

    async def question_keep_waiting(self, message: QuestionApp.KeepWaiting) -> None:
        """Lift the unattended-expiry deadline for the active callback (plan U)."""
        if message.callback_id is None:
            return
        active = self.active_callback
        if active is None or active.callback_id != message.callback_id:
            return
        try:
            await self._app.app_server.keep_waiting_callback(active.callback_id)
        except Exception as exc:
            # The server timer still fires on failure; the question may
            # expire even though the user asked to keep waiting.
            logger.warning("Keep-waiting request failed", exc_info=exc)
            self._app.notify(
                "Keep-waiting request failed; the question may still expire.",
                severity="warning",
            )

    async def question_expired(self, entry: PublicCallbackEntry) -> None:
        """Close the UI for a callback the server expired unanswered (plan U).

        The live input is closed, the unsent draft is preserved locally (never
        auto-delivered), and the next pending callback (if any) is shown.
        """
        callback_id = entry.callback_id
        self._prune_expired_pending()
        active = self.active_callback
        if active is not None and active.callback_id == callback_id:
            from chartreux.cli.textual_ui.app import _get_question_app_class

            try:
                question = self._app.query_one(_get_question_app_class())
            except Exception:
                question = None
            if question is not None and question._callback_id == callback_id:
                # Preserve the unsent draft locally; never auto-deliver it.
                self.expired_question_drafts[callback_id] = dict(question.other_texts)
                question.close_for_expiry()
            self.active_callback = None
            self._app.notify("Unanswered — deferred", severity="warning")
            if self.pending_callbacks:
                await self._app._show_callback(self.pending_callbacks.popleft())
                return
            if self._app._loading_widget is not None:
                self._app._loading_widget.end_action_required()
            await self._app._switch_to_input_app()
            return
        # A queued callback expired before it was ever displayed.
        self.pending_callbacks = deque(
            pending
            for pending in self.pending_callbacks
            if pending.callback_id != callback_id
        )

    def _prune_expired_pending(self) -> None:
        """Drop queued callbacks the server already closed (plan U/OD1).

        A queued callback can expire before it is ever displayed. Only prune
        callbacks the client projection has actually seen reach a non-open
        state: a callback the projection has not seen yet (for example one
        delivered before the projection caught up) must still present.
        """
        closed_ids = {
            entry.callback_id
            for entry in (self._app.app_server.state.history or [])
            if isinstance(entry, PublicCallbackEntry) and entry.state.status != "open"
        }
        if not closed_ids:
            return
        self.pending_callbacks = deque(
            pending
            for pending in self.pending_callbacks
            if pending.callback_id not in closed_ids
        )

    def _cancel_pending_question(self, result: UserQuestionResult) -> None:
        """Resolve a pending local question with ``result`` if one is open."""
        if self.pending_local_question and not self.pending_local_question.done():
            self.pending_local_question.set_result(result)

    async def request_local_user_input(
        self, request: UserQuestionRequest
    ) -> UserQuestionResult:
        if self.active_callback is not None:
            raise RuntimeError("Cannot open local input while a callback is active")
        self.pending_local_question = asyncio.get_running_loop().create_future()
        opened = False
        try:
            await self._app._wait_for_typing_pause()
            if self._app._secondary_surface_active():
                self._app._secondary_returned.clear()
                self._app._indicate_pending_action()
                await self._app._secondary_returned.wait()
            self._app._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            with paused_timer(self._app._loading_widget):
                await self._app._switch_to_question_app(request)
                opened = True
                return await self.pending_local_question
        finally:
            self.pending_local_question = None
            if opened:
                if self.pending_callbacks and self.active_callback is None:
                    await self._app._show_callback(self.pending_callbacks.popleft())
                else:
                    await self._app._switch_to_input_app()

    async def present_pending_callback(self) -> None:
        self._prune_expired_pending()
        if self._app._secondary_surface_active():
            self._app._indicate_pending_action()
            return
        self._app._secondary_returned.set()
        if (
            self.pending_callbacks
            and self.active_callback is None
            and self.pending_local_question is None
        ):
            await self._app._show_callback(self.pending_callbacks.popleft())
        self._app._indicate_pending_action()

    async def show_callback(self, callback: PublicCallbackEntry) -> None:
        if (
            self.active_callback is not None
            and self.active_callback.callback_id == callback.callback_id
        ) or any(
            pending.callback_id == callback.callback_id
            for pending in self.pending_callbacks
        ):
            return
        if self._app._secondary_surface_active() or (
            self.active_callback is not None or self.pending_local_question is not None
        ):
            self.pending_callbacks.append(callback)
            self._app._indicate_pending_action()
            return
        self.active_callback = callback
        try:
            loading = self._app._loading_widget
            if loading is None or loading.parent is None:
                await self._app._ensure_loading_widget()
                loading = self._app._loading_widget
            if loading is not None:
                loading.begin_action_required(callback.title)
            await self._app._wait_for_typing_pause()
            if self._app._secondary_surface_active():
                self.active_callback = None
                self.pending_callbacks.appendleft(callback)
                self._app._indicate_pending_action()
                return
            self._app._terminal_notifier.notify(NotificationContext.ACTION_REQUIRED)
            await self._app._switch_to_question_app(callback.detail.request)
            # Attach the callback identity and the armed unattended-expiry
            # policy so the widget can correlate messages and count down
            # honestly (plan U/OD1).
            from chartreux.cli.textual_ui.app import _get_question_app_class

            self._app.query_one(_get_question_app_class()).attach_callback(
                callback.callback_id, callback.detail.unattended_expiry
            )
        except BaseException:
            if self.active_callback is callback:
                self.active_callback = None
            raise

    async def respond_to_active_callback(self, output: UserInputCallbackOutput) -> None:
        from chartreux.app_server import AppServerConnectionClosed
        from chartreux.cli.textual_ui.app import _get_question_app_class

        callback = self.active_callback
        if (
            callback is None
            or self.callback_submitting
            or self.callback_delivery_uncertain
        ):
            return
        self.callback_submitting = True
        question = self._app.query_one(_get_question_app_class())
        question.set_submission_status("Running: Sending answer")
        try:
            await self._app.app_server.respond_to_callback(callback.callback_id, output)
        except AppServerResponseError as exc:
            # A protocol rejection is definite: the same draft may be corrected
            # and submitted again without guessing whether the server accepted it.
            question.set_submission_status(
                f"Failed: Answer rejected ({exc}). Draft kept; edit and submit again, or Esc to cancel."
            )
            return
        except (AppServerConnectionClosed, OSError, TimeoutError) as exc:
            self.callback_delivery_uncertain = True
            question.set_submission_status(
                "Warning: Answer delivery is uncertain. Draft kept; do not resend. "
                "Ctrl+D exits; inspect the session before deciding what to do next. "
                f"({exc})"
            )
            return
        except Exception as exc:
            self.callback_delivery_uncertain = True
            logger.warning("Callback delivery outcome unknown", exc_info=exc)
            question.set_submission_status(
                "Warning: Answer delivery is uncertain. Draft kept; do not resend. "
                "Ctrl+D exits; inspect the session before deciding what to do next."
            )
            return
        finally:
            self.callback_submitting = False
        if self.active_callback is callback:
            question.disabled = True
            self.active_callback = None
            self.callback_delivery_uncertain = False
            if self.pending_callbacks:
                await self._app._show_callback(self.pending_callbacks.popleft())
                return
            if self._app._loading_widget is not None:
                self._app._loading_widget.end_action_required()
            await self._app._switch_to_input_app()

    async def wait_for_typing_pause(self) -> None:
        # The debounce resolver stays an app module global (test seam); the
        # call-time import reads it — and the patched _DEFAULT_TYPING_DEBOUNCE_MS
        # module global it reads — at call time.
        from chartreux.cli.textual_ui.app import _resolve_typing_debounce_s

        try:
            text_area = self._app.query_one(ChatTextArea)
        except Exception:
            return

        debounce_s = _resolve_typing_debounce_s()
        if text_area.time_since_last_keystroke() >= debounce_s:
            return

        if self._app._loading_widget:
            self._app._loading_widget.show_debounce_hint()

        try:
            while True:
                elapsed = text_area.time_since_last_keystroke()
                if elapsed >= debounce_s:
                    return
                await asyncio.sleep(debounce_s - elapsed)
        finally:
            if self._app._loading_widget:
                self._app._loading_widget.hide_debounce_hint()

    async def submit_or_defer(self, raw_value: str) -> None:
        # A submit fired right after Escape must see the settled post-interrupt
        # queue state (paused/idle), not the transient state mid-interrupt, so
        # it resumes or starts a turn instead of being dropped or trapped. The
        # settle path completes turn/queue UI on the message pump, so a pending
        # interrupt is drained in a worker to keep this handler from blocking it.
        # While any deferred submit is still draining, later submits take the
        # same ordered path so a fast double-Enter cannot reorder ahead of it.
        if self.interrupt_pending or self.deferred_submits:
            self.deferred_submits += 1
            self._app.run_worker(self._deferred_dispatch(raw_value), exclusive=False)
            return
        await self._app._dispatch_submitted_value(raw_value)

    async def _deferred_dispatch(self, raw_value: str) -> None:
        try:
            async with self.deferred_submit_lock:
                await self.await_interrupt_settled()
                await self._app._dispatch_submitted_value(raw_value)
        finally:
            self.deferred_submits -= 1
