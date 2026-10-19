"""R2 — the TUI root automatic-recovery controller (plan state.md R2, OD3).

The TUI half of the bounded automatic root-turn recovery pair. The app-server
gate (``chartreux/app_server/_root_recovery.py``) owns the authoritative
decisions (eligibility, episode accounting, single-flight admission); THIS
controller owns the LOCAL half: an in-process asyncio timer that schedules
the continuation turn itself after an authoritative terminal root failure, and
local cancellation via an input-generation token invalidated synchronously on
typing/submission/question actions/manual retry/interrupt/session change. The
controller also enforces the LOCAL queued-work barrier: the gate's server-side
``queued_work`` cannot see client-side work (a submission still preparing its
prompt holds a reservation without a server turn), so the controller checks
``app._queue.has_server_work`` — the same local truth the legacy incomplete-
stream continuation checked — at observation time and again at the timer wake.

A healthy process/event loop is the precondition — a process crash stays on
the existing manual fresh-process resume (documented honestly in the plan).

One terminal observation path feeds BOTH the solicited ``AppServerTurnError``
(carrying the terminal turn identity) and the unsolicited ``TurnCompleted``,
deduplicated by (session identity, terminal turn id) — never scheduled from
generic TUI exceptions, disconnects, or missing spinners. On admission the
continuation is driven through the EXISTING incomplete-stream continuation
driver (``turn_supervisor.auto_retry_incomplete_stream`` with the gate's
idempotency key) as an injected, non-user-initiated turn — so it NEVER resets
availability cooldowns (WP0 F6). Meta-recovery: a controller/notice/admission
failure is not another provider failure — ambiguous starts reconcile via the
idempotency identity, and anything unreconcilable parks to the manual R1
fallback, never a self-retry loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from chartreux.app_server.events import TurnCompleted
from chartreux.app_server.models import PublicTurnStatus
from chartreux.app_server.session import AppServerTurnError
from chartreux.observability.logging import logger

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.app import ChartreuxApp
    from chartreux.cli.textual_ui.turn_supervisor import TurnSupervisor


@dataclass(slots=True)
class _PendingSchedule:
    """One armed automatic-continuation schedule (the gate owns the truth;
    this is the local timer's handle on it).
    """

    episode_id: str
    idempotency_key: str
    terminal_turn_id: str
    attempt: int
    limit: int
    session_id: str
    input_generation: int
    # The stashed original terminal failure — mounted as the manual fallback
    # if the gate declines at admission time (after the pending window).
    error: AppServerTurnError


class RootRecoveryController:
    """Owns the local timer and local cancellation for root turn recovery."""

    # Late-bound host reference, assigned via _bind_app after construction.
    _app: ChartreuxApp

    def __init__(self) -> None:
        self._timer: asyncio.Task[None] | None = None
        self._pending: _PendingSchedule | None = None
        # (session identity, terminal turn id) — one observation per failed
        # turn, across BOTH the solicited and unsolicited paths.
        self._observed: set[tuple[str, str]] = set()
        self._shutdown = False
        # Test seam: the timer sleep (production: asyncio.sleep).
        self._sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def _bind_app(self, app: ChartreuxApp) -> None:
        """Bind the host app after construction."""
        self._app = app

    @property
    def pending(self) -> _PendingSchedule | None:
        """The armed schedule, if any (tests and diagnostics)."""
        return self._pending

    # ------------------------------------------------------------------
    # The one terminal observation path
    # ------------------------------------------------------------------

    async def observe_terminal(  # noqa: PLR0911 - one early exit per guard
        self, source: object, *, solicited: bool
    ) -> bool:
        """Observe one terminal root-turn failure; arm the continuation.

        Returns True when an automatic continuation is pending (the caller
        suppresses the error presentation and keeps the retry presentation
        armed), False when the gate declines (the caller mounts the error
        with the manual R1 fallback). Only a typed terminal turn error
        carrying the terminal turn identity ever schedules — generic
        exceptions, disconnects, and missing spinners never do.
        """
        error = self._terminal_error(source, solicited=solicited)
        if error is None or not error.turn_id:
            return False
        app = self._app
        try:
            session_id = app.app_server.session_id
        except Exception:
            return False
        key = (session_id, error.turn_id)
        if key in self._observed:
            # Duplicate observation (solicited + unsolicited, or a replay):
            # one schedule per failed turn.
            return self._pending is not None
        self._observed.add(key)
        if app._queue.has_server_work:
            # LOCAL queued-work barrier (plan R2 interaction gates): the
            # gate's server-side ``queued_work`` cannot see client-side
            # work — a submission still preparing its prompt holds an
            # enqueue reservation without a server turn yet. Never
            # schedule ahead of queued user work: decline here and the
            # caller mounts the terminal error with the manual R1
            # fallback (the legacy incomplete-stream behavior).
            return False
        try:
            response = await app.app_server.schedule_root_recovery(
                error.turn_id,
                solicited=solicited,
                input_generation=app._input_generation,
            )
        except Exception:
            # Meta-recovery: a controller/wire failure is not a provider
            # failure — park to the manual fallback, never a self-retry loop.
            logger.warning("root recovery schedule request failed", exc_info=True)
            return False
        if response.decision not in {"scheduled", "already_scheduled"}:
            return False
        if response.episode_id is None or response.idempotency_key is None:
            return False
        # A new schedule replaces any stale pending one (its episode was
        # cancelled server-side by the intervening turn start).
        self._pending = _PendingSchedule(
            episode_id=response.episode_id,
            idempotency_key=response.idempotency_key,
            terminal_turn_id=error.turn_id,
            attempt=response.attempt,
            limit=response.limit,
            session_id=session_id,
            input_generation=app._input_generation,
            error=error,
        )
        self._arm_timer(response.delay_seconds)
        return True

    @staticmethod
    def _terminal_error(
        source: object, *, solicited: bool
    ) -> AppServerTurnError | None:
        """The typed terminal turn error carrying the terminal turn identity,
        or None (never schedule without it).
        """
        if solicited:
            if isinstance(source, AppServerTurnError):
                return source
            return None
        if isinstance(source, TurnCompleted):
            event = source
            if (
                event.turn.status is PublicTurnStatus.FAILED
                and event.turn.error is not None
                and event.turn.next_turn_id is None
            ):
                return AppServerTurnError(event.turn.error, turn_id=event.turn.id)
        return None

    # ------------------------------------------------------------------
    # The local timer
    # ------------------------------------------------------------------

    def _arm_timer(self, delay: float) -> None:
        self._cancel_timer()
        self._timer = asyncio.create_task(self._run_timer(delay))

    def _cancel_timer(self) -> None:
        timer = self._timer
        self._timer = None
        if (
            timer is not None
            and not timer.done()
            and timer is not asyncio.current_task()
        ):
            # A timer whose wake is running THIS task has already fired:
            # cancelling ourselves would kill the continuation _on_wake is
            # driving, so only the reference is dropped.
            timer.cancel()

    def _drop_pending(self) -> None:
        self._pending = None
        self._cancel_timer()

    async def _run_timer(self, delay: float) -> None:
        try:
            await self._sleep(delay)
        except asyncio.CancelledError:
            return
        await self._on_wake()

    async def _on_wake(self) -> None:
        """Timer wake: recheck locally, then ask the gate for admission."""
        pending = self._pending
        if pending is None or self._shutdown:
            return
        app = self._app
        # Reconcile the session identity (a session change invalidates the
        # schedule; the server-side gate was invalidated by the reset).
        try:
            session_id = app.app_server.session_id
        except Exception:
            self._drop_pending()
            return
        if session_id != pending.session_id:
            self._drop_pending()
            return
        # Defense in depth behind the synchronous token invalidation: user
        # input after the schedule cancels the pending continuation.
        if app._input_generation != pending.input_generation:
            await self._cancel_remote(pending, reason="user_input")
            self._drop_pending()
            return
        # Defense in depth behind the observation-time barrier: queued user
        # work visible only locally (a submission still preparing its prompt
        # holds an enqueue reservation the server-side gate cannot see)
        # cancels the schedule — cancel and show why, never silently rearm.
        if app._queue.has_server_work:
            await self._cancel_remote(pending, reason="queued_work")
            self._drop_pending()
            self._offer_manual_fallback(pending.error)
            return
        try:
            response = await app.app_server.admit_root_recovery(
                pending.episode_id,
                pending.idempotency_key,
                pending.terminal_turn_id,
                input_generation=app._input_generation,
            )
        except Exception:
            # Meta-recovery: park to the manual fallback — one state, no loop.
            logger.warning("root recovery admission request failed", exc_info=True)
            self._drop_pending()
            self._offer_manual_fallback(pending.error)
            return
        match response.decision:
            case "admitted":
                self._pending = None
                self._cancel_timer()
                # Drive the continuation through the EXISTING driver: an
                # injected, non-user-initiated new turn (never a cooldown
                # reset — WP0 F6). The drive runs in THIS timer task; the
                # reference was dropped above, so local cancellation no
                # longer touches it (an admitted successor settles via
                # normal interruption rules, never as rollback).
                try:
                    await app._auto_retry_incomplete_stream(
                        pending.attempt, idempotency_key=pending.idempotency_key
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Meta-recovery: a drive failure is not a provider
                    # failure — handle_turn has already presented it; never
                    # a self-retry loop.
                    logger.exception("root recovery continuation drive failed")
            case "already_admitted":
                # Ambiguous delivery reconciled via the idempotency
                # identity: never start a second successor.
                self._pending = None
                self._cancel_timer()
                if response.turn_id is None:
                    # The admission was consumed but no successor start was
                    # observed: park to the manual fallback, never loop.
                    self._offer_manual_fallback(pending.error)
            case "deferred":
                # A read-only availability wait: re-arm within the episode
                # (never consumes a retry).
                self._arm_timer(response.delay_seconds)
            case _:
                # Declined at admission (deadline/barrier/budget): surface
                # the stashed failure with the manual R1 fallback.
                self._pending = None
                self._cancel_timer()
                self._offer_manual_fallback(pending.error)

    # ------------------------------------------------------------------
    # Local cancellation (the input-generation token's other half)
    # ------------------------------------------------------------------

    def cancel_local(self, *, reason: str = "user_input") -> None:
        """Cancel the local timer and pending schedule synchronously.

        Called on typing/submission/question actions/manual retry/interrupt
        (via ``app._invalidate_user_input``) and on session change/shutdown.
        The wire cancel is fire-and-forget: the gate also rechecks the input
        generation at admission, and refuses to cancel an admitted successor
        in flight (it settles via normal interruption rules).
        """
        pending = self._pending
        self._pending = None
        self._cancel_timer()
        if pending is None:
            return
        try:
            asyncio.create_task(
                self._app.app_server.cancel_root_recovery(
                    pending.episode_id, reason=reason
                )
            )
        except Exception:
            logger.warning("root recovery cancel request failed", exc_info=True)

    def shutdown(self) -> None:
        """App shutdown: cancel the local timer and drop references. The
        server-side gate is invalidated by the session close; an admitted
        successor in flight settles via normal interruption rules (never
        cancelled as rollback).
        """
        self._shutdown = True
        self._pending = None
        self._cancel_timer()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _cancel_remote(self, pending: _PendingSchedule, *, reason: str) -> None:
        try:
            await self._app.app_server.cancel_root_recovery(
                pending.episode_id, reason=reason
            )
        except Exception:
            logger.warning("root recovery cancel request failed", exc_info=True)

    def _offer_manual_fallback(self, error: AppServerTurnError) -> None:
        """Surface the stashed terminal failure with the manual R1 fallback
        (the error widget's retry affordance where the code offers it).
        """
        supervisor: TurnSupervisor = self._app._turn_supervisor
        asyncio.create_task(self._mount_fallback(error, supervisor))

    async def _mount_fallback(
        self, error: AppServerTurnError, supervisor: TurnSupervisor
    ) -> None:
        try:
            message = supervisor._resolve_turn_error_message(error)
            await supervisor._mount_turn_error(error, message)
        except Exception:
            logger.exception("root recovery manual fallback mount failed")
