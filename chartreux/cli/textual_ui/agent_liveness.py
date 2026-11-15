"""Reliable background-agent liveness observation (v0.7.9 point 5, C7b).

Design of record: tasks/v0.7.9/session-status-design-final.md §3.2. The
``AgentsUpdate`` presentation event is a low-latency hint, never proof that
the inventory is current: a coalesced publication can still be shed downstream.
This module owns the CONTROL-CLASS observation the composition trusts — the
one-second poller over the C6b ``agents/inventory/read`` carrier.

Discipline (design §3.2): at most one request is in flight with a coalesced
pending refresh; the poll runs once a second even when the last snapshot says
nothing is active; a failed or timed-out read is UNKNOWN, never evidence of no
agents (the last accepted snapshot survives it and the poller retries while
attached); a confirmed root death stops the poller; an explicit resume or a
replacement generation re-arms it with a fresh floor.

The accepted-revision floor is PER-INCARNATION: a revision compares only
within one identical ``(session_id, generation)`` pair; the first snapshot
accepted for a new incarnation establishes the fresh floor, and an old
incarnation's floor can never reject a new incarnation's snapshot. Numeric
generations of the same session are ordered, so a snapshot of an older
generation never resurrects over a newer one.

The COMPOSED EMISSION RULE legs live here as queries, applied by the settle
and snapshot-completion sites: LEG 1 — FENCE MATCH / ACCEPTANCE RECENCY (the
fence counter increments on every ``AgentsUpdate`` arrival and every
pending-refresh mark; a terminal emission needs a response whose fence equals
the current value, and a settlement input needs no fence increment since its
ACCEPTANCE with the acceptance within one poll interval of the settlement);
LEG 2 — ISSUANCE AGE (the full issuance -> acceptance -> emission chain must
sit within one poll interval end to end). A response failing either leg alone
is admitted as a retain-input only.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
import time

from chartreux.app_server.protocol import (
    AgentsInventoryReadResponse,
    AgentSummaryModel,
    AppServerResponseError,
)
from chartreux.observability.logging import logger

DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_READ_TIMEOUT = 30.0

# The typed confirmed-root-death receipt marker (RootFailureSignature
# DEATH_RECEIPT, root_agent_process.py): a dead generation answers every
# carrier request with it. Matched as text so this module needs no
# root-process import; the receipt contract is the message, not the enum.
_DEATH_RECEIPT_MARKER = "death-receipt"

_terminal_states = frozenset({"done", "idle"})


def is_terminal_state(state: str) -> bool:
    """Whether a composed state is the terminal class the emission rule gates."""
    return state in _terminal_states


@dataclass(frozen=True)
class LivenessSnapshot:
    """One accepted inventory plus the metadata the emission legs read."""

    session_id: str | None
    generation: int | None
    revision: int
    agents: tuple[AgentSummaryModel, ...]
    issued_at: float
    accepted_at: float
    fence_at_issuance: int
    fence_at_acceptance: int

    @property
    def incarnation(self) -> tuple[str | None, int | None]:
        """The root identity the revision floor compares within."""
        return (self.session_id, self.generation)


@dataclass(frozen=True)
class SettlementInput:
    """The composed emission rule's verdict for one settlement offer.

    ``qualified`` names whether the accepted snapshot satisfies BOTH legs as
    a settlement input. ``child_active`` is the snapshot-carried activity an
    unqualified offer may not drive a terminal from (the caller retains the
    current non-terminal status instead).
    """

    qualified: bool
    child_active: bool


def is_death_receipt(exc: BaseException) -> bool:
    """Whether an error carries the typed confirmed-root-death receipt."""
    return isinstance(exc, AppServerResponseError) and _DEATH_RECEIPT_MARKER in str(exc)


class AgentLivenessPoller:
    """The TUI's one-second liveness poller over the inventory carrier.

    Constructed beside the composed-ownership record; attached where the
    prompt returns. The read callable, the current-session filter, the
    active predicate and the accept callback are late-bound so the poller
    follows the app's own attachment lifecycle without owning any of it.
    """

    def __init__(
        self,
        *,
        read: Callable[[], Awaitable[AgentsInventoryReadResponse]],
        current_session_id: Callable[[], str | None],
        is_active: Callable[[AgentSummaryModel], bool],
        on_accept: Callable[[LivenessSnapshot], None] | None = None,
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        clock: Callable[[], float] = time.monotonic,
        diagnose: Callable[[str], None] | None = None,
    ) -> None:
        self._read = read
        self._current_session_id = current_session_id
        self._is_active = is_active
        self._on_accept = on_accept
        self.poll_interval = poll_interval
        self.read_timeout = read_timeout
        self._clock = clock
        self._diagnose = diagnose or (lambda message: logger.debug(message))
        self._task: asyncio.Task[None] | None = None
        self._wake = asyncio.Event()
        # The fence increments on every AgentsUpdate arrival and every
        # pending-refresh mark: a read carries its issue-time fence, and a
        # terminal emission may ride only a response whose fence equals the
        # current value (design §3.2 leg 1).
        self._fence = 0
        # The invalidation epoch: session-scope resets, rotation and a rearm
        # discard every read still in flight across them.
        self._epoch = 0
        self._accepted: LivenessSnapshot | None = None
        self._dead = False
        self._last_diagnostic = 0.0

    # -- lifecycle --------------------------------------------------------

    @property
    def attached(self) -> bool:
        """Whether the polling loop is running against a live session."""
        return self._task is not None and not self._task.done()

    @property
    def accepted(self) -> LivenessSnapshot | None:
        """The last accepted snapshot; retained across unknown failures."""
        return self._accepted

    @property
    def confirmed_dead(self) -> bool:
        """Whether the carrier answered the typed confirmed-death receipt."""
        return self._dead

    @property
    def fence(self) -> int:
        """The current fence value (test seam and diagnostics)."""
        return self._fence

    def attach(self) -> None:
        """Start (or keep) polling while the session stays attached."""
        if self.attached:
            return
        self._wake.set()
        self._task = asyncio.get_running_loop().create_task(self._poll())

    def detach(self) -> None:
        """Stop polling (app shutdown): the task is cancelled and dropped."""
        task = self._task
        self._task = None
        if task is not None and not task.done():
            task.cancel()

    def invalidate(self) -> None:
        """A session-scope reset invalidates every old read and old snapshot.

        The /clear, plan-execution and /compact resets carry the record to a
        new conversation scope: the accepted inventory of the old scope must
        never feed the new one's composition. The per-incarnation floor means
        the next snapshot of whatever incarnation answers establishes its own
        fresh floor — no numeric revision ever crosses the reset.
        """
        self._epoch += 1
        self._accepted = None
        self.request_refresh()

    def rearm(self) -> None:
        """Rearm after an explicit resume or a replacement generation.

        The incumbent loop is gone when death stopped it, so the rearm
        restarts polling: fresh floor, fresh snapshot, live task.
        """
        self._dead = False
        self._epoch += 1
        self._accepted = None
        self.request_refresh()
        self.attach()

    # -- event and refresh marks -----------------------------------------

    def note_event(self) -> None:
        """An AgentsUpdate arrived: bump the fence and want a fresh read.

        The event remains a refresh hint for the browser store; for the
        composition it is only a fence increment plus a pending mark, exactly
        the two things leg 1 keys on.
        """
        self.request_refresh()

    def request_refresh(self) -> None:
        """Mark a coalesced pending refresh (design §3.2's second fence input)."""
        self._fence += 1
        self._wake.set()

    # -- emission rule queries -------------------------------------------

    def settlement_input(self) -> SettlementInput:
        """Evaluate the composed emission rule for a settlement offer.

        NOT ATTACHED: no backend means no liveness claim is involved —
        children cannot be active through a server that is not there, so the
        offer stands with no active child. CONFIRMED DEATH: the record
        emits best-effort per the receipt-scoped clause (design §3.2) —
        there is no backend left to interrogate, and no fence leg can ever
        qualify again. Otherwise the accepted snapshot must satisfy both
        named legs; an unqualified offer is a retain-input only.
        """
        if not self.attached:
            return SettlementInput(qualified=True, child_active=False)
        if self._dead:
            return SettlementInput(qualified=True, child_active=False)
        snapshot = self._accepted
        if snapshot is None:
            return SettlementInput(qualified=False, child_active=False)
        now = self._clock()
        interval = self.poll_interval
        fence_settled = (
            snapshot.fence_at_issuance == self._fence
            and snapshot.fence_at_acceptance == self._fence
        )
        issued_to_accepted_ok = snapshot.accepted_at - snapshot.issued_at <= interval
        accepted_recent = now - snapshot.accepted_at <= interval
        issued_recent = now - snapshot.issued_at <= interval
        return SettlementInput(
            qualified=(
                fence_settled
                and issued_to_accepted_ok
                and accepted_recent
                and issued_recent
            ),
            child_active=self._child_active(snapshot),
        )

    def completion_qualifies(self) -> bool:
        """Whether the accepted snapshot may drive the emission the
        snapshot-completion site computes — the in-flight-response legs: no
        fence increment since issuance, and the full
        issuance -> acceptance -> emission chain inside one poll interval.
        """
        snapshot = self._accepted
        if not self.attached or self._dead or snapshot is None:
            return True
        now = self._clock()
        interval = self.poll_interval
        return (
            snapshot.fence_at_issuance == self._fence
            and snapshot.fence_at_acceptance == self._fence
            and snapshot.accepted_at - snapshot.issued_at <= interval
            and now - snapshot.issued_at <= interval
        )

    def _child_active(self, snapshot: LivenessSnapshot) -> bool:
        return any(self._is_active(agent) for agent in snapshot.agents)

    # -- the poll loop ---------------------------------------------------

    async def _poll(self) -> None:
        while self._task is not None and not self._dead:
            if not self._wake.is_set():
                try:
                    await asyncio.wait_for(self._wake.wait(), self.poll_interval)
                except TimeoutError:
                    pass
            self._wake.clear()
            if self._task is None or self._dead:
                return
            epoch = self._epoch
            fence = self._fence
            issued_at = self._clock()
            try:
                response = await asyncio.wait_for(self._read(), self.read_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if is_death_receipt(exc):
                    # Confirmed root death: the root process and backend are
                    # dead. Stop interrogating; the settle paths emit the
                    # saved deferred record receipt-scoped and best-effort,
                    # and an explicit resume re-arms (design §3.2).
                    self._dead = True
                    return
                # Unknown, never evidence of no agents: retain the accepted
                # snapshot, surface one bounded diagnostic, keep retrying.
                self._bounded_diagnostic(exc)
                continue
            accepted_at = self._clock()
            if epoch != self._epoch:
                # A reset, rotation or rearm retired this read mid-flight.
                continue
            snapshot = self._accept(response, fence, issued_at, accepted_at)
            if snapshot is not None and self._on_accept is not None:
                try:
                    self._on_accept(snapshot)
                except Exception:
                    logger.exception("Liveness snapshot reconciliation failed")

    def _bounded_diagnostic(self, exc: BaseException) -> None:
        now = self._clock()
        if now - self._last_diagnostic < self.poll_interval:
            return
        self._last_diagnostic = now
        self._diagnose(f"Agent liveness read failed (retained last snapshot): {exc}")

    def _accept(
        self,
        response: AgentsInventoryReadResponse,
        fence_at_issuance: int,
        issued_at: float,
        accepted_at: float,
    ) -> LivenessSnapshot | None:
        """Accept one response against the current session and floor.

        A response of another session than the app's current one is never
        accepted (no old session's inventory can survive a rotation). Within
        the accepted incarnation the revision may not regress; a new
        incarnation of the same session — a higher generation — establishes a
        fresh floor and is always accepted (an old incarnation's floor can
        never reject it, and an older generation never resurrects over a
        newer one).

        A response carrying NO session id is the registry's pre-identity
        projection — the in-process read before the first root generation
        and in the post-release gap, where the registry clears its identity
        while the app side may still know its session — so the session
        filter has nothing to compare and it is accepted as a bounded
        deviation: the floor below still polices its own no-identity
        incarnation (a regressing revision is rejected like any
        same-incarnation regression), a session-carrying successor is the
        usual fresh-incarnation reset, and the composed emission rule's
        legs (design §3.2) demote it to a retain-input within one poll
        interval, so it cannot persist a false idle or done.
        """
        current = self._current_session_id()
        if (
            current is not None
            and response.session_id is not None
            and response.session_id != current
        ):
            return None
        previous = self._accepted
        if previous is not None and response.session_id == previous.session_id:
            new_generation = response.generation
            previous_generation = previous.generation
            if (
                new_generation is not None
                and previous_generation is not None
                and new_generation < previous_generation
            ):
                return None
            if (
                new_generation == previous_generation
                and response.revision < previous.revision
            ):
                return None
        snapshot = LivenessSnapshot(
            session_id=response.session_id,
            generation=response.generation,
            revision=response.revision,
            agents=tuple(response.agents),
            issued_at=issued_at,
            accepted_at=accepted_at,
            fence_at_issuance=fence_at_issuance,
            fence_at_acceptance=self._fence,
        )
        self._accepted = snapshot
        return snapshot
