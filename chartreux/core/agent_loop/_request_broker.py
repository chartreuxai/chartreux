"""Agent-side user-input request broker and callback arbitration.

The broker is the executing agent's own decision state for user input (design
§Permission and approval split): every open request is a :class:`CallbackRecord`
here — the waiter future, the resolution lock, the terminal disposition, and the
unattended-expiry timer — so a host driving the agent through the execution port
only relays requests and answers. Settlement is arbitrated under the record's
resolution lock (answer/expiry/keep-waiting/cancellation have exactly one
winner), and the terminal disposition is published through the bound
:class:`SettlementPublisher` BEFORE the waiter unblocks, preserving
publication-before-unblock ordering across the host/agent boundary.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import time
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel

from chartreux.core.events import BaseEvent, UserInputOrigin, UserInputRequestEvent
from chartreux.core.tools.base import InteractionRequestPort, ToolError
from chartreux.questions import (
    UnansweredQuestion,
    UnansweredQuestions,
    UnattendedExpiryPolicy,
)

# The unattended-expiry floor mirrors the [user_questions] config validation
# (both configured timeout bounds are >= 60s); an override outside the
# configured range is rejected, never clamped (plan U/OD1).
MIN_UNATTENDED_TIMEOUT_SECONDS = 60.0

# The terminal history reason for an expired-unanswered callback (plan U/OD1).
EXPIRED_CALLBACK_REASON = "unanswered_timeout"

# Model-facing denial for a child-origin model question refused at admission
# (plan D2). Identical to the ask tool's own child refusal so the model reads
# one contract whichever layer stops the question.
CHILD_QUESTION_REFUSAL = (
    "ask_user_question is not available to subagents; return a blocker naming "
    "the question instead of asking the user"
)

# Model-facing continuation guidance carried by the server-owned unanswered
# disposition (plan U/OD1).
UNANSWERED_CONTINUATION_GUIDANCE = (
    "The user did not answer in time, so no choice was made and no permission "
    "was granted. The declared unattended default describes how to proceed "
    "without the answer: defer the affected work or keep the existing "
    "choice. Do not re-ask the same question in a loop; continue with the "
    "deferred outcome, or ask a new, different question if one is still "
    "needed. If the user later provides the answer explicitly, treat it as "
    "a new user message."
)


class ChildQuestionRefusedError(RuntimeError):
    """A child (subagent) loop's model question, refused at admission.

    Raised into the broker waiter so the asking tool call resolves with a
    typed denial instead of opening a callback or hanging (plan D2).
    """


type UserInputSettleOutcome = Literal[
    "accepted", "duplicate", "not_found", "expired", "closed", "conflict"
]
type KeepWaitingOutcome = Literal[
    "kept", "already_terminal", "no_deadline", "not_found"
]


@dataclass(frozen=True, slots=True)
class UserInputSettlement:
    """The terminal disposition of one user-input request.

    Published through the settlement publisher before the waiter unblocks so a
    client or a reconnect sees the terminal state, never a ghost open callback.
    """

    kind: Literal["answered", "rejected", "expired", "cancelled"]
    result: BaseModel | None = None
    message: str | None = None
    was_open: bool = True


class SettlementPublisher(Protocol):
    """Host hook the broker awaits around settlement (publication-before-unblock)."""

    async def publish_user_input_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        """Publish the terminal state BEFORE the waiter is settled."""
        ...

    async def after_user_input_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        """Run after the waiter settled (e.g. the resumed status emission)."""
        ...


class UserInputAdmission(Protocol):
    """Agent-side admission policy for user-input requests (loop-injected)."""

    def child_refusal(self, event: UserInputRequestEvent) -> str | None:
        """The child-question refusal message, or None when the request may open."""
        ...

    def expiry_policy(
        self, event: UserInputRequestEvent
    ) -> UnattendedExpiryPolicy | None:
        """The unattended-expiry admission snapshot, or None when ineligible.

        Raises ValueError when a model-supplied timeout override does not fit
        the configured range (rejected, never clamped).
        """
        ...

    def session_id(self) -> str: ...

    def session_generation(self) -> int: ...

    def active_turn_id(self) -> str | None: ...

    def record_unanswered(
        self, request_id: str, unanswered: UnansweredQuestions
    ) -> None:
        """Record exactly one deferred-question record for an expiry."""
        ...


@dataclass(slots=True)
class CallbackExpiry:
    """Authoritative unattended-expiry timer state (plan U/OD1).

    Owned by the callback record — not the tool and not the host. The policy
    is snapshotted at request admission; the monotonic deadline arbitrates
    against answer/keep-waiting/cancel/reconnect under the record's
    resolution lock, so exactly one terminal disposition wins. A stale timer
    (session or turn generation changed) never fires.
    """

    deadline: float
    opened_at: float
    timeout_seconds: float
    questions: list[UnansweredQuestion]
    session_id: str
    session_generation: int
    turn_id: str
    timer: asyncio.Task[None] | None = None
    kept_waiting: bool = False


@dataclass(slots=True)
class CallbackRecord:
    """One open user-input request: the agent-side arbitration state."""

    event: UserInputRequestEvent
    future: asyncio.Future[BaseModel]
    resolution: BaseModel | BaseException | None = None
    core_resolved: bool = False
    resolution_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    expiry: CallbackExpiry | None = None
    expired: bool = False
    refusal: str | None = None
    admission_error: BaseException | None = None


def _same_resolution(left: object, right: object) -> bool:
    if isinstance(left, BaseModel) and isinstance(right, BaseModel):
        return left.model_dump(mode="json") == right.model_dump(mode="json")
    return str(left) == str(right)


class InteractionRequestBroker(InteractionRequestPort):
    """Own the open user-input requests and arbitrate their settlement."""

    def __init__(self) -> None:
        self._events: asyncio.Queue[BaseEvent | None] | None = None
        self._records: dict[str, CallbackRecord] = {}
        self._admission: UserInputAdmission | None = None
        self._settlement_publisher: SettlementPublisher | None = None
        # Monotonic clock / wakeup seams for the unattended-expiry timer
        # (plan U/OD1): the host binds its own seams per controller.
        self._clock_getter: Callable[[], float] = time.monotonic
        self._wakeup_getter: Callable[[], Callable[[float], Awaitable[None]]] = lambda: (
            asyncio.sleep
        )

    def bind(self, events: asyncio.Queue[BaseEvent | None]) -> None:
        if self._events is not None:
            raise RuntimeError("An interaction request stream is already active")
        self._events = events

    def unbind(self, events: asyncio.Queue[BaseEvent | None]) -> None:
        if self._events is not events:
            raise RuntimeError("Cannot close an inactive interaction request stream")
        self._events = None

    def bind_admission(self, admission: UserInputAdmission) -> None:
        self._admission = admission

    def bind_settlement_publisher(self, publisher: SettlementPublisher) -> None:
        self._settlement_publisher = publisher

    def bind_timer_seams(
        self,
        *,
        clock: Callable[[], float],
        wakeup: Callable[[], Callable[[float], Awaitable[None]]],
    ) -> None:
        self._clock_getter = clock
        self._wakeup_getter = wakeup

    def records(self) -> dict[str, CallbackRecord]:
        """Live view of the open request records (the host relays through this)."""
        return self._records

    def record(self, request_id: str) -> CallbackRecord | None:
        return self._records.get(request_id)

    def expiry_policy(
        self, event: UserInputRequestEvent
    ) -> UnattendedExpiryPolicy | None:
        """Relay the admission computation (raises on a rejected override)."""
        admission = self._admission
        return admission.expiry_policy(event) if admission is not None else None

    async def request_user_input(
        self,
        args: BaseModel,
        tool_call_id: str,
        *,
        origin: UserInputOrigin = UserInputOrigin.RUNTIME_APPROVAL,
    ) -> BaseModel:
        events = self._events
        if events is None:
            raise ToolError("User input is not available outside an active turn")

        request_id = str(uuid4())
        record = self._open_record(
            UserInputRequestEvent(
                request_id=request_id,
                args=args,
                tool_call_id=tool_call_id,
                origin=origin,
            ),
            request_id=request_id,
        )
        await events.put(record.event)
        try:
            return await record.future
        finally:
            # A refused or admission-failed request never became a callback,
            # so its record is dropped with the waiter. A request that was
            # relayed as a callback keeps its record after settling (and a
            # cancelled waiter keeps it for the host's terminalization): a
            # late answer must see the terminal disposition, not a vanished
            # id, until the host's reset drops it.
            if record.refusal is not None or record.admission_error is not None:
                self._records.pop(request_id, None)

    def relay_user_input_request(self, event: UserInputRequestEvent) -> None:
        """Adopt a host-relayed request: its record and timer live agent-side."""
        if event.request_id in self._records:
            raise RuntimeError(f"Duplicate user-input request: {event.request_id}")
        self._open_record(event, request_id=event.request_id)

    def _open_record(
        self, event: UserInputRequestEvent, *, request_id: str
    ) -> CallbackRecord:
        admission = self._admission
        refusal = admission.child_refusal(event) if admission is not None else None
        policy: UnattendedExpiryPolicy | None = None
        admission_error: BaseException | None = None
        if refusal is None and admission is not None:
            try:
                policy = admission.expiry_policy(event)
            except Exception as exc:
                admission_error = exc
        if policy is not None:
            event = event.model_copy(update={"unattended_expiry": policy})
        record = CallbackRecord(
            event=event,
            future=asyncio.get_running_loop().create_future(),
            refusal=refusal,
            admission_error=admission_error,
        )
        self._records[request_id] = record
        if policy is not None:
            turn_id = admission.active_turn_id() if admission is not None else None
            record.expiry = self._arm_expiry(request_id, policy, turn_id)
        return record

    def resolve_user_input(self, request_id: str, result: BaseModel) -> None:
        """Raw settle: resolve the open request's waiter with *result*."""
        record = self._records.get(request_id)
        if record is not None and not record.future.done():
            record.future.set_result(result)

    def reject(self, request_id: str, error: BaseException) -> None:
        """Raw settle: fail the open request's waiter with *error*."""
        record = self._records.get(request_id)
        if record is None:
            return
        record.core_resolved = True
        self._cancel_expiry_timer(record)
        if not record.future.done():
            record.future.set_exception(error)

    async def settle_answer(
        self, request_id: str, result: BaseModel
    ) -> UserInputSettleOutcome:
        """Arbitrated answer: settle the open request with *result* exactly once."""
        record = self._records.get(request_id)
        if record is None:
            return "not_found"
        async with record.resolution_lock:
            if record.expired:
                # A definite expired response: the unattended timer already
                # resolved this request (plan U/OD1).
                return "expired"
            if record.resolution is None and (
                record.core_resolved or record.future.done()
            ):
                # The waiter is gone (its turn was cancelled) or the request
                # was already closed without a resolution: no answer can land,
                # so this is closed rather than a settlement that would strand
                # the turn's state.
                return "closed"
            if record.resolution is not None:
                if _same_resolution(record.resolution, result):
                    return "duplicate"
                return "conflict"
            # Settlement is authoritative under the resolution lock: the
            # terminal disposition is published BEFORE the waiter resolves
            # (plan U). Presentation failure never loses an accepted
            # settlement — the answer is settled even when the terminal
            # patch cannot be delivered.
            record.resolution = result
            record.core_resolved = True
            self._cancel_expiry_timer(record)
            try:
                await self._publish_settlement(
                    request_id, UserInputSettlement(kind="answered", result=result)
                )
            finally:
                if not record.future.done():
                    record.future.set_result(result)
            return "accepted"

    async def settle_rejection(
        self, request_id: str, error: BaseException
    ) -> UserInputSettleOutcome:
        """Arbitrated rejection: fail the open request with *error* exactly once."""
        record = self._records.get(request_id)
        if record is None:
            return "not_found"
        async with record.resolution_lock:
            if record.expired:
                return "expired"
            if record.core_resolved and record.resolution is None:
                return "closed"
            if record.resolution is not None:
                if _same_resolution(record.resolution, error):
                    return "duplicate"
                return "conflict"
            # The rejection is authoritative; the terminal cancelled state is
            # published BEFORE the waiter settles so a client or a reconnect
            # sees a closed entry, never a ghost open callback.
            record.resolution = error
            try:
                await self._publish_settlement(
                    request_id,
                    # A rejection never reports a child callback closed;
                    # only a cancellation of an open request does.
                    UserInputSettlement(
                        kind="rejected", message=str(error), was_open=False
                    ),
                )
            finally:
                record.core_resolved = True
                self._cancel_expiry_timer(record)
                if not record.future.done():
                    record.future.set_exception(error)
            return "accepted"

    async def keep_waiting(self, request_id: str) -> KeepWaitingOutcome:
        """Lift the unattended-expiry deadline for a request (plan U/OD1).

        The question stays open without a deadline (OPEN_WITHOUT_DEADLINE);
        the expiry timer is cancelled under the record's resolution lock so
        the answer/expiry/keep-waiting race still has exactly one winner.
        """
        record = self._records.get(request_id)
        if record is None:
            return "not_found"
        async with record.resolution_lock:
            expiry = record.expiry
            if expiry is None:
                return "no_deadline"
            if (
                record.resolution is not None
                or record.core_resolved
                or record.future.done()
            ):
                return "already_terminal"
            expiry.kept_waiting = True
            if expiry.timer is not None:
                expiry.timer.cancel()
                expiry.timer = None
            return "kept"

    async def cancel(self, request_id: str, error: BaseException) -> None:
        """Cancel an open request (turn end / host close), fail-closed."""
        record = self._records.get(request_id)
        if record is None:
            return
        async with record.resolution_lock:
            # Settle under the record's resolution lock (plan U): an answer,
            # expiry, or rejection that won the race owns the request, so a
            # cancellation can neither publish a second terminal state over it
            # nor race the entry-state check that guards the patch. Terminal
            # projection BEFORE the waiter settles: a reconnect, or a client
            # answering a dead request, must see a closed entry, never a ghost.
            was_open = not record.core_resolved
            await self._publish_settlement(
                request_id,
                UserInputSettlement(
                    kind="cancelled", message=str(error), was_open=was_open
                ),
            )
            if record.core_resolved:
                return
            record.core_resolved = True
            self._cancel_expiry_timer(record)
            if not record.future.done():
                record.future.set_exception(error)
        # The record stays: a late answer to a cancelled request must see the
        # closed disposition (CALLBACK_CLOSED), not a vanished id, until the
        # host's reset drops it.

    def _arm_expiry(
        self, request_id: str, policy: UnattendedExpiryPolicy, turn_id: str | None
    ) -> CallbackExpiry:
        """Arm the authoritative unattended-expiry timer (plan U/OD1)."""
        admission = self._admission
        opened_at = self._clock_getter()
        expiry = CallbackExpiry(
            deadline=opened_at + policy.timeout_seconds,
            opened_at=opened_at,
            timeout_seconds=policy.timeout_seconds,
            questions=policy.questions,
            session_id=admission.session_id() if admission is not None else "",
            session_generation=(
                admission.session_generation() if admission is not None else 0
            ),
            turn_id=turn_id or "",
        )
        expiry.timer = asyncio.create_task(
            self._expire(request_id, expiry),
            name=f"agent-user-input-expiry:{request_id}",
        )
        return expiry

    async def _expire(self, request_id: str, expiry: CallbackExpiry) -> None:
        """Fire the unattended-expiry timer (plan U/OD1).

        Waits out the monotonic deadline, then arbitrates under the record's
        resolution lock: answer, expiry, keep-waiting, cancellation, and
        reconnect races have exactly one winner. A stale timer (session or
        turn generation changed) never fires, and there is no timer
        restoration after a restart.
        """
        try:
            await self._wakeup_getter()(
                max(0.0, expiry.deadline - self._clock_getter())
            )
        except asyncio.CancelledError:
            # Keep-waiting, cancellation, or host close won the race.
            return
        record = self._records.get(request_id)
        if record is None:
            return
        async with record.resolution_lock:
            if (
                record.resolution is not None
                or record.core_resolved
                or record.future.done()
                or record.expiry is not expiry
                or expiry.kept_waiting
            ):
                return
            admission = self._admission
            if admission is not None and (
                admission.session_id() != expiry.session_id
                or admission.session_generation() != expiry.session_generation
            ):
                return
            if admission is not None and admission.active_turn_id() != expiry.turn_id:
                return
            record.expired = True
            record.core_resolved = True
            unanswered = UnansweredQuestions(
                questions=expiry.questions,
                elapsed_seconds=self._clock_getter() - expiry.opened_at,
                timeout_seconds=expiry.timeout_seconds,
                continuation_guidance=UNANSWERED_CONTINUATION_GUIDANCE,
            )
            # Expiry never approves anything: the tool result IS the
            # unanswered disposition. Closure is published BEFORE the waiter
            # resolves, so the effect-unblock patch lands while the related
            # effect is still running.
            await self._publish_settlement(
                request_id, UserInputSettlement(kind="expired")
            )
            record.future.set_result(unanswered)
            if admission is not None:
                admission.record_unanswered(request_id, unanswered)
        await self._after_settlement(request_id, UserInputSettlement(kind="expired"))

    def _cancel_expiry_timer(self, record: CallbackRecord) -> None:
        expiry = record.expiry
        if expiry is not None and expiry.timer is not None:
            expiry.timer.cancel()
            expiry.timer = None

    async def _publish_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        publisher = self._settlement_publisher
        if publisher is not None:
            await publisher.publish_user_input_settlement(request_id, settlement)

    async def _after_settlement(
        self, request_id: str, settlement: UserInputSettlement
    ) -> None:
        publisher = self._settlement_publisher
        if publisher is not None:
            await publisher.after_user_input_settlement(request_id, settlement)
