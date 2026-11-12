"""Agent-side turn control: steering admission and injection.

The coordinator owns the turn queue and decides when a turn starts or is
promoted. What has no loop-level equivalent is the arbitration that decides
whether ONE steering delivery may commit — idempotent replay by key, the
waiting-only precondition, and the commit evidence a settlement reports — so it
lives here, inside the agent process, next to the loop that commits it.

This is the port of the in-process controller's steering machinery
(:mod:`chartreux.app_server._turns`: its receipt pruning, commit evidence,
waiting-only validation, and the injection that drives the commit). The
differences are the ones the process boundary forces: there is no completed-turn
history here (so receipt eviction that loses evidence reports "ambiguous", never
"not committed"), and the coordinator's own admission state — queue ownership,
handoff, closing, and the session-execution registry — stays with the
coordinator, which never sends a delivery it has already refused.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from chartreux.app_server._utils import decode_content_blocks
from chartreux.app_server.agent_wire import SteerSettlement
from chartreux.app_server.protocol import ContextInjectParams, TurnSteerParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.events import BaseEvent, UserMessageEvent
from chartreux.core.llm_models import ManualShellContext

STEER_RECEIPT_CAPACITY = 128
"""The receipts one agent keeps for replay before refusing a new keyed delivery."""

_RETIRED_RECEIPT_RETENTION = 64
"""Settled receipts retained beyond the cap so a late replay still settles."""

type EmitEvent = Callable[[str, BaseEvent], Awaitable[None]]


class SteeringConflictError(RuntimeError):
    """One steering or injection delivery this agent refuses, typed."""


@dataclass(slots=True)
class _SteerReceipt:
    """One keyed steering delivery: its params, task, and commit evidence."""

    params: TurnSteerParams
    task: asyncio.Task[SteerSettlement] | None = None
    event: UserMessageEvent | None = None
    events: list[BaseEvent] = field(default_factory=list)


class AgentSteeringArbiter:
    """The steering admission and commit evidence of one agent loop."""

    def __init__(
        self,
        loop: AgentLoop,
        *,
        active_turn_id: Callable[[], str | None],
        emit_event: EmitEvent,
    ) -> None:
        self._loop = loop
        self._active_turn_id = active_turn_id
        self._emit_event = emit_event
        self._receipts: dict[str, _SteerReceipt] = {}
        self._evidence_lost = False
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[SteerSettlement]] = set()

    def known(self, params: TurnSteerParams) -> bool:
        """Whether this delivery's key is the one an earlier delivery used."""
        self._prune()
        receipt = self._receipts.get(params.idempotency_key or "")
        return receipt is not None and receipt.params == params

    def committed(self, params: TurnSteerParams) -> bool | None:
        """Whether one delivery committed: ``True``, ``False``, or ambiguous.

        A changed session identity is compaction, not a rejection, and an
        evicted receipt whose projections may already have reached a client is
        not evidence either: both answer ``None`` so no caller enqueues what the
        agent may already have seen.
        """
        receipt = self._receipts.get(params.idempotency_key or "")
        if receipt is not None and receipt.params.session_id != params.session_id:
            return None
        if receipt is None and self._evidence_lost:
            return None
        return (
            receipt is not None
            and receipt.params == params
            and receipt.event is not None
        )

    async def steer(self, params: TurnSteerParams) -> SteerSettlement:
        """Arbitrate and commit one steering delivery, idempotently by key."""
        async with self._lock:
            self._prune()
            receipt = self._receipts.get(params.idempotency_key or "")
            if receipt is not None:
                if receipt.params != params:
                    raise SteeringConflictError(
                        "Steering idempotency key has a different payload"
                    )
                if (
                    receipt.task is not None
                    and receipt.task.done()
                    and receipt.event is not None
                ):
                    return SteerSettlement(committed=True, events=len(receipt.events))
            else:
                if params.require_waiting_only and not params.idempotency_key:
                    raise SteeringConflictError(
                        "Conditional steering requires an idempotency key"
                    )
                self._validate(params)
                if (
                    params.idempotency_key
                    and len(self._receipts) >= STEER_RECEIPT_CAPACITY
                ):
                    raise SteeringConflictError("Steering receipt capacity reached")
                receipt = _SteerReceipt(params=params.model_copy(deep=True))
                receipt.task = asyncio.create_task(self._run(receipt))
                self._tasks.add(receipt.task)
                receipt.task.add_done_callback(self._tasks.discard)
                receipt.task.add_done_callback(lambda _: self._prune())
                # Observe failures even if the coordinator stopped waiting.
                receipt.task.add_done_callback(_observe)
                if params.idempotency_key:
                    self._receipts[params.idempotency_key] = receipt
            assert receipt.task is not None
        return await asyncio.shield(receipt.task)

    async def inject(
        self, params: ContextInjectParams, manual_shell: ManualShellContext | None
    ) -> list[BaseEvent]:
        """Inject one delivery between turns, carrying its manual-shell record.

        The manual ``!`` command's record text is ordinary user input; the
        context beside it is message metadata the timeline reads, and it rides
        the committed message the loop appends.
        """
        if self._active_turn_id() is not None:
            raise SteeringConflictError("Use turn/steer while a turn is active")
        decoded = decode_content_blocks(params.input, session_dir=self._session_dir())
        return await self._loop.inject_user_context(
            decoded.prompt,
            as_message=params.as_message,
            inject_implicit=params.inject_invoked_skill,
            images=decoded.images or None,
            input_text=decoded.input_text if decoded.resources else None,
            resources=decoded.resources or None,
            client_message_id=params.client_user_message_id,
            manual_shell=manual_shell,
        )

    def _validate(self, params: TurnSteerParams) -> None:
        """Refuse a delivery the active turn cannot accept right now."""
        turn_id = self._active_turn_id()
        if turn_id is None or turn_id != params.expected_turn_id:
            raise SteeringConflictError(
                f"steering names turn {params.expected_turn_id!r}, which this "
                "agent is not running"
            )
        if not params.require_waiting_only:
            return
        if params.session_id != self._loop.session_id:
            raise SteeringConflictError(
                "Steering session does not match the active turn"
            )
        if not self._loop.is_waiting_only(turn_id):
            raise SteeringConflictError("Active turn is not waiting-only")

    async def _run(self, receipt: _SteerReceipt) -> SteerSettlement:
        params = receipt.params

        def on_injected_event(event: BaseEvent) -> None:
            # Retain the expansion work before persistence can fail.
            receipt.events.append(event)

        def on_commit(event: UserMessageEvent) -> None:
            receipt.event = event
            on_injected_event(event)

        decoded = decode_content_blocks(params.message, session_dir=self._session_dir())
        try:
            await self._loop.inject_user_context(
                decoded.prompt,
                as_message=True,
                inject_implicit=params.inject_invoked_skill,
                images=decoded.images or None,
                input_text=decoded.input_text if decoded.resources else None,
                resources=decoded.resources or None,
                client_message_id=params.client_user_message_id,
                before_commit=lambda: self._validate(params),
                on_commit=on_commit,
                on_injected_event=on_injected_event,
                on_context_ready=lambda: (
                    self._loop.cancel_outstanding_waits_for_steering(
                        params.expected_turn_id
                    )
                ),
            )
        except BaseException as exc:
            if receipt.event is not None:
                raise SteeringConflictError(
                    "Steering committed; accepted-operation recovery failed"
                ) from exc
            raise
        # The committed events ride the turn's own event stream, in the order
        # the loop produced them, before the settlement receipt answers.
        for event in receipt.events:
            await self._emit_event(params.expected_turn_id, event)
        return SteerSettlement(committed=True, events=len(receipt.events))

    def _prune(self) -> None:
        """Keep settled receipts for replay without growing without bound."""
        active_id = self._active_turn_id()
        retired = [
            key
            for key, receipt in self._receipts.items()
            if receipt.params.expected_turn_id != active_id
            and receipt.task is not None
            and receipt.task.done()
        ]
        for key in retired[:-_RETIRED_RECEIPT_RETENTION]:
            del self._receipts[key]
        for key, receipt in list(self._receipts.items()):
            if len(self._receipts) < STEER_RECEIPT_CAPACITY:
                break
            if (
                receipt.params.expected_turn_id != active_id
                and receipt.task is not None
                and receipt.task.done()
            ):
                del self._receipts[key]
                # A settled delivery's projections may already have reached a
                # client. Losing its receipt must not become evidence that it
                # never committed.
                self._evidence_lost = True

    def _session_dir(self) -> Path | None:
        return self._loop.session_logger.session_dir


def _observe(task: asyncio.Task[SteerSettlement]) -> None:
    if not task.cancelled():
        task.exception()
