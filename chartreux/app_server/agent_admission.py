"""Agent-side long-operation admission and the incarnation-scoped marker table.

A long operation (an open that resumes a stored session here; compaction and
the identity operations that follow) crosses the process boundary as one
request, but its deadline is two-sided: the coordinator mints an ABSOLUTE
``CLOCK_MONOTONIC`` deadline at send, and this end admits the operation only
over a live fence, records the coordinator's request id in the marker table,
arms the agent-side timer, and announces the admission before any body work
starts.

The marker record carries the coordinator's request id (S9-C1), so correlation
is direct: a coordinator that stopped waiting re-reads a snapshot and finds the
record for its own request id, or proves never-admitted from its absence in a
payload constructed after its deadline. Records are never pruned within an
incarnation, so absence inside one incarnation can only mean never admitted.

At deadline expiry the timer cancels whatever the operation is awaiting -- a
body step or the admission announcement -- and this end settles the record
TERMINAL, releases the fence, and answers the request id typed (S9-C2), so a
coordinator always has a terminal answer to correlate and the fence can never
stay wedged.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
import itertools
import time
from typing import Any

OP_STATE_ACTIVE = "active"
"""Admission: the operation was recorded, but nothing is published by it yet."""

OP_STATE_TERMINAL = "terminal"
"""The operation settled agent-side; ``outcome`` names the terminal class."""

OP_OUTCOME_COMPLETED = "completed"
OP_OUTCOME_FAILED = "failed"
OP_OUTCOME_ABORTED = "aborted"
"""The deadline abort: the operation was cancelled and never claimed a result."""


type MonotonicClock = Callable[[], float]
type DeadlineSleep = Callable[[float], Awaitable[None]]
type AdmissionAnnouncement = Callable[["OperationMarker"], Awaitable[None]]
type AdmissionObserver = Callable[["OperationMarker"], None]
type OperationBody = Callable[[], Awaitable[dict[str, Any]]]


async def _default_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


class OperationConflictError(RuntimeError):
    """A second long operation this agent refuses while one holds the fence."""


class OperationExpiredError(RuntimeError):
    """An admission over an already-spent absolute deadline; no body starts."""


class OperationAbortedError(RuntimeError):
    """The operation was aborted at its deadline; its TERMINAL record settles it."""


@dataclass(frozen=True, slots=True)
class OperationMarker:
    """One long operation's admission record, keyed by the coordinator request id.

    ``state`` is ``active`` from admission onward or ``terminal`` once the
    operation settled; ``outcome`` names the terminal class and stays ``None``
    while the record is active. ``deadline`` is the coordinator's absolute
    monotonic deadline, kept so a re-read payload can be judged against it.
    """

    request_id: str
    op_id: str
    method: str
    state: str
    deadline: float
    outcome: str | None = None

    @property
    def incarnation_id(self) -> str:
        """The incarnation whose counter minted this op id."""
        return self.op_id.partition(":")[0]

    def settled(self, outcome: str) -> OperationMarker:
        """The record this one becomes once its operation settles."""
        return replace(self, state=OP_STATE_TERMINAL, outcome=outcome)


class OperationMarkerTable:
    """The incarnation-scoped in-memory marker table: records are never pruned.

    The table is the only marker store (no durable file carries it): its only
    consumers are the live coordinator reading the live agent through the
    snapshot payload, and agent-process death is terminal for the session, so
    there is no post-mortem consumer and no location to publish. It dies with
    the process, and it is cleared at an identity rebind except the rebind
    operation's own TERMINAL record, which seeds the new incarnation so the
    rebind's one-shot re-check can always find its own evidence.
    """

    def __init__(self, incarnation_id: str) -> None:
        if not incarnation_id:
            raise ValueError("An incarnation id is required")
        self._incarnation_id = incarnation_id
        self._records: dict[str, OperationMarker] = {}
        self._counter = itertools.count(1)

    @property
    def incarnation_id(self) -> str:
        """The incarnation whose counter mints this table's op ids."""
        return self._incarnation_id

    def admit(
        self, *, request_id: str, method: str, deadline: float
    ) -> OperationMarker:
        """Record one operation as admitted, before any body step runs."""
        if not request_id:
            raise ValueError("A marker record requires the coordinator's request id")
        if request_id in self._records:
            raise OperationConflictError(
                f"request {request_id!r} is already recorded in this incarnation"
            )
        marker = OperationMarker(
            request_id=request_id,
            op_id=f"{self._incarnation_id}:{next(self._counter)}",
            method=method,
            state=OP_STATE_ACTIVE,
            deadline=deadline,
        )
        self._records[request_id] = marker
        return marker

    def settle(self, op_id: str, outcome: str) -> OperationMarker | None:
        """Settle one operation TERMINAL; settling a settled record is a no-op."""
        for request_id, marker in self._records.items():
            if marker.op_id != op_id:
                continue
            if marker.state == OP_STATE_TERMINAL:
                return marker
            settled = marker.settled(outcome)
            self._records[request_id] = settled
            return settled
        return None

    def records(self) -> tuple[OperationMarker, ...]:
        """Every record this incarnation admitted, in admission order."""
        return tuple(self._records.values())

    def for_request(self, request_id: str) -> OperationMarker | None:
        """The record this incarnation admitted for one request id, if any."""
        return self._records.get(request_id)

    def for_op(self, op_id: str) -> OperationMarker | None:
        """The record one op id names, if this incarnation minted it."""
        return next(
            (marker for marker in self._records.values() if marker.op_id == op_id), None
        )

    def rebind(self, *, incarnation_id: str, carry_op_id: str | None = None) -> None:
        """Clear the table for a new incarnation, retaining one TERMINAL seed.

        The carried record keeps its minted op id: the old incarnation's UUID
        namespace is never continued by the new incarnation's counter, so the
        seed cannot collide with a record the new incarnation mints.
        """
        if not incarnation_id:
            raise ValueError("An incarnation id is required")
        seed = self.for_op(carry_op_id) if carry_op_id is not None else None
        if carry_op_id is not None and (
            seed is None or seed.state != OP_STATE_TERMINAL
        ):
            raise OperationConflictError(
                f"cannot carry {carry_op_id!r} across the rebind: it is not a "
                "settled record of this incarnation"
            )
        self._incarnation_id = incarnation_id
        self._counter = itertools.count(1)
        self._records = {} if seed is None else {seed.request_id: seed}


class LongOperationFence:
    """The one-long-op admission fence with the two-sided absolute deadline.

    At most one long operation is in flight per root. A second admission is
    refused typed before anything runs, and an admission over an already-spent
    deadline is refused typed before any body starts, so a late body is
    excluded by construction rather than racing the coordinator's timeout.
    """

    def __init__(
        self,
        *,
        table: OperationMarkerTable,
        announce: AdmissionAnnouncement,
        on_admitted: AdmissionObserver | None = None,
        clock: MonotonicClock = time.monotonic,
        sleep: DeadlineSleep = _default_sleep,
    ) -> None:
        self._table = table
        self._announce = announce
        self._on_admitted = on_admitted
        self._clock = clock
        self._sleep = sleep
        self._active_op_id: str | None = None
        self._op_task: asyncio.Task[dict[str, Any]] | None = None
        self._aborted_op_id: str | None = None
        self._cancel_requested_op_id: str | None = None
        self._timers: set[asyncio.Task[None]] = set()

    @property
    def table(self) -> OperationMarkerTable:
        """The marker table this fence admits into."""
        return self._table

    @property
    def active_op_id(self) -> str | None:
        """The op id currently holding the fence, or ``None`` when it is free."""
        return self._active_op_id

    def records(self) -> tuple[OperationMarker, ...]:
        """Every marker record this incarnation admitted, in admission order."""
        return self._table.records()

    async def admit(
        self, *, request_id: str, method: str, deadline: float, body: OperationBody
    ) -> dict[str, Any]:
        """Admit one long operation and run it under its armed deadline.

        The caller answers ``request_id`` with the returned reply, or with the
        typed failure this raises; either way the fence is free and the record
        is TERMINAL by the time this returns.
        """
        if self._active_op_id is not None:
            raise OperationConflictError(
                f"{method} is refused while operation {self._active_op_id} is in flight"
            )
        if deadline <= self._clock():
            raise OperationExpiredError(
                f"{method} arrived after its absolute deadline had passed; "
                "no operation body started"
            )
        marker = self._table.admit(
            request_id=request_id, method=method, deadline=deadline
        )
        self._active_op_id = marker.op_id
        if self._on_admitted is not None:
            # Synchronous, before any task exists: a caller that needs the
            # minted op id (to settle a body's own publication class, or to
            # recognise its own cancel) holds it before the body can run.
            self._on_admitted(marker)
        # Arm the deadline before any awaited emission: the timer and the body
        # are both created without an intervening await, so the deadline is
        # live from the moment the fence is held.
        self._op_task = asyncio.create_task(self._run(marker, body))
        timer = asyncio.create_task(self._expire(marker))
        self._timers.add(timer)
        timer.add_done_callback(self._timers.discard)
        try:
            return await self._op_task
        except asyncio.CancelledError:
            if self._aborted_op_id == marker.op_id:
                # The timer cancelled the operation at its deadline. Settlement
                # is synchronous from here, so the TERMINAL record and the
                # released fence precede the reply this raises into.
                self._settle_abort(marker)
                raise OperationAbortedError(
                    f"{method} was aborted at its absolute deadline; its "
                    "TERMINAL marker records the aborted outcome"
                ) from None
            raise
        finally:
            self._aborted_op_id = None
            self._cancel_requested_op_id = None
            self._op_task = None
            self._release(marker.op_id)
            if not timer.done():
                timer.cancel()
            with suppress(BaseException):
                await timer

    def cancel(self, op_id: str) -> bool:
        """Cancel the operation this id names, if it is the one holding the fence.

        The frame is idempotent: a cancel that names a settled, unknown or
        absent operation is a no-op, because the operation's own settlement is
        the only assertable outcome and it has already happened. A live
        operation is cancelled at its next await point, exactly as the deadline
        timer would, and the body itself decides what that means for its
        publication class.
        """
        task = self._op_task
        if self._active_op_id != op_id or task is None or task.done():
            return False
        self._cancel_requested_op_id = op_id
        task.cancel()
        return True

    def cancel_requested(self, op_id: str) -> bool:
        """Whether a coordinator cancel -- rather than the deadline -- cancelled this."""
        return self._cancel_requested_op_id == op_id

    async def _run(
        self, marker: OperationMarker, body: OperationBody
    ) -> dict[str, Any]:
        """Announce the admission, then run the body, under the armed timer."""
        try:
            await self._announce(marker)
            reply = await body()
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._table.settle(marker.op_id, OP_OUTCOME_FAILED)
            raise
        self._table.settle(marker.op_id, OP_OUTCOME_COMPLETED)
        return reply

    async def _expire(self, marker: OperationMarker) -> None:
        """Cancel the operation's pending await once its absolute deadline lands."""
        remaining = marker.deadline - self._clock()
        if remaining > 0:
            await self._sleep(remaining)
        task = self._op_task
        if task is None or task.done():
            return
        if self._aborted_op_id is not None:
            return
        self._aborted_op_id = marker.op_id
        task.cancel()
        # Join the cancellation so the body's own cleanup runs before the
        # admission handler settles the record and answers the request id.
        with suppress(BaseException):
            await task

    def _settle_abort(self, marker: OperationMarker) -> None:
        """Settle the aborted operation TERMINAL and release its fence."""
        self._table.settle(marker.op_id, OP_OUTCOME_ABORTED)
        self._release(marker.op_id)

    def _release(self, op_id: str) -> None:
        if self._active_op_id == op_id:
            self._active_op_id = None
