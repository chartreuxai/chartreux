"""Fence, floor, emission-leg, coalescing and death tests for the liveness
poller (v0.7.9 point 5, C7b; design §3.2 and §5 acceptance 4).

The poller is self-contained: these drive it directly with a scripted read
callable and a controllable clock (the recovery-entry unit tests for the
fence/floor logic, then the loop disciplines — one in-flight with coalesced
pending, timeout as unknown-not-idle, confirmed root death, rearm).
"""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

import pytest

from chartreux.app_server.protocol import (
    AgentsInventoryReadResponse,
    AgentSummaryModel,
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
)
from chartreux.cli.textual_ui.agent_liveness import (
    AgentLivenessPoller,
    LivenessSnapshot,
    is_death_receipt,
    is_terminal_state,
)
from chartreux.cli.textual_ui.widgets.agent_bar import agent_is_active
from chartreux.core.subagents import derive_agent_name

_INTERVAL = 0.2


def _agent(agent_id: str = "agent-1", **kwargs: Any) -> AgentSummaryModel:
    values: dict[str, Any] = {"identity": "coder", "availability": "idle"}
    values.update(kwargs)
    values.setdefault(
        "name",
        derive_agent_name(
            agent_id=agent_id,
            identity=values["identity"],
            identity_display_label=None,
            base_model=values.get("base_model"),
            effective_model=values.get("effective_model"),
        ),
    )
    return AgentSummaryModel(agent_id=agent_id, **values)


def _response(
    revision: int = 0,
    *,
    session_id: str | None = "session-1",
    generation: int | None = 1,
    availability: str = "idle",
) -> AgentsInventoryReadResponse:
    return AgentsInventoryReadResponse(
        session_id=session_id,
        generation=generation,
        revision=revision,
        agents=[_agent(availability=availability)],
    )


def _death_receipt() -> AppServerResponseError:
    return AppServerResponseError(
        ProtocolError(
            code=ProtocolErrorCode.INTERNAL_ERROR,
            message="death-receipt: the root agent process exited — restart",
        )
    )


class _Clock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _snapshot(
    *,
    revision: int = 1,
    session_id: str | None = "session-1",
    generation: int | None = 1,
    availability: str = "idle",
    fence: int = 0,
    issued_at: float = 1000.0,
    accepted_at: float = 1000.1,
) -> LivenessSnapshot:
    agents = (_agent(availability=availability),)
    return LivenessSnapshot(
        session_id=session_id,
        generation=generation,
        revision=revision,
        agents=agents,
        issued_at=issued_at,
        accepted_at=accepted_at,
        fence_at_issuance=fence,
        fence_at_acceptance=fence,
    )


def _poller(
    *,
    read: Any = None,
    current_session: str | None = "session-1",
    clock: Any = None,
    poll_interval: float = _INTERVAL,
    diagnose: Any = None,
) -> AgentLivenessPoller:
    async def _hanging_read() -> AgentsInventoryReadResponse:
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    return AgentLivenessPoller(
        read=read if read is not None else _hanging_read,
        current_session_id=lambda: current_session,
        is_active=agent_is_active,
        poll_interval=poll_interval,
        clock=clock if clock is not None else _Clock(),
        diagnose=diagnose or (lambda message: None),
    )


class TestReceiptsAndStateClasses:
    def test_a_death_receipt_is_recognized_by_its_marker(self) -> None:
        assert is_death_receipt(_death_receipt())
        assert not is_death_receipt(ValueError("death-receipt-like? no"))
        assert not is_death_receipt(
            AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INTERNAL_ERROR,
                    message="an ordinary internal failure",
                )
            )
        )

    def test_only_done_and_idle_are_terminal_states(self) -> None:
        assert is_terminal_state("done")
        assert is_terminal_state("idle")
        assert not is_terminal_state("working")
        assert not is_terminal_state("blocked")
        assert not is_terminal_state("error")


@pytest.mark.asyncio
class TestPollerNotAttachedOrDead:
    async def test_not_attached_settlements_make_no_liveness_claim(self) -> None:
        poller = _poller()
        view = poller.settlement_input()
        assert view.qualified is True
        assert view.child_active is False
        assert poller.completion_qualifies() is True

    async def test_confirmed_death_settles_receipt_scoped(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            poller._dead = True
            poller._accepted = _snapshot(availability="running")
            view = poller.settlement_input()
            assert view.qualified is True
            assert view.child_active is False
        finally:
            poller.detach()


@pytest.mark.asyncio
class TestRevisionFloor:
    """The per-incarnation floor: a revision compares only within one
    identical (session, generation); an old incarnation's floor never
    rejects a new incarnation; numeric generations order resurrections.
    """

    async def test_same_incarnation_revision_regression_is_rejected(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            assert poller._accept(_response(revision=3), 0, 1000.0, 1000.1) is not None
            assert poller._accept(_response(revision=2), 0, 1000.2, 1000.3) is None
        finally:
            poller.detach()

    async def test_same_incarnation_higher_revision_is_accepted(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            poller._accept(_response(revision=3), 0, 1000.0, 1000.1)
            snapshot = poller._accept(_response(revision=4), 0, 1000.2, 1000.3)
            assert snapshot is not None and snapshot.revision == 4
        finally:
            poller.detach()

    async def test_a_new_generation_establishes_a_fresh_floor(self) -> None:
        # The new incarnation's revision 0 is accepted over the old
        # incarnation's revision 9: never a cross-incarnation compare.
        poller = _poller()
        poller.attach()
        try:
            poller._accept(_response(revision=9, generation=1), 0, 1000.0, 1000.1)
            snapshot = poller._accept(
                _response(revision=0, generation=2), 0, 1000.2, 1000.3
            )
            assert snapshot is not None and snapshot.generation == 2
        finally:
            poller.detach()

    async def test_an_older_generation_of_the_same_session_is_rejected(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            poller._accept(_response(revision=9, generation=2), 0, 1000.0, 1000.1)
            assert (
                poller._accept(_response(revision=99, generation=1), 0, 1000.2, 1000.3)
                is None
            )
        finally:
            poller.detach()

    async def test_a_foreign_session_response_is_rejected(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            poller._accept(
                _response(revision=1, session_id="session-1"), 0, 1000.0, 1000.1
            )
            assert (
                poller._accept(
                    _response(revision=2, session_id="old-session"), 0, 1000.2, 1000.3
                )
                is None
            )
            assert poller.accepted is not None
            assert poller.accepted.session_id == "session-1"
        finally:
            poller.detach()

    async def test_a_none_session_revision_regression_is_rejected(self) -> None:
        # The registry's pre-identity projection (``_sessions.py`` clears its
        # generation identity while the app side may still know its session)
        # carries no session id, so the session filter has nothing to
        # compare and it is accepted as a bounded deviation — but its own
        # no-identity incarnation still owns a floor: a regressing revision
        # is rejected like any same-incarnation regression.
        poller = _poller()
        poller.attach()
        try:
            assert (
                poller._accept(
                    _response(revision=3, session_id=None, generation=None),
                    0,
                    1000.0,
                    1000.1,
                )
                is not None
            )
            assert (
                poller._accept(
                    _response(revision=2, session_id=None, generation=None),
                    0,
                    1000.2,
                    1000.3,
                )
                is None
            )
            assert poller.accepted is not None
            assert poller.accepted.revision == 3
        finally:
            poller.detach()

    async def test_a_none_session_higher_revision_is_accepted(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            poller._accept(
                _response(revision=3, session_id=None, generation=None),
                0,
                1000.0,
                1000.1,
            )
            snapshot = poller._accept(
                _response(revision=4, session_id=None, generation=None),
                0,
                1000.2,
                1000.3,
            )
            assert snapshot is not None and snapshot.revision == 4
        finally:
            poller.detach()

    async def test_identity_arriving_is_the_usual_fresh_incarnation_reset(self) -> None:
        # Leaving the pre-identity window is the fresh-incarnation reset: a
        # session-carrying successor is accepted regardless of the
        # no-identity floor (the bounded deviation never blocks the resume
        # path), never compared to it.
        poller = _poller()
        poller.attach()
        try:
            poller._accept(
                _response(revision=9, session_id=None, generation=None),
                0,
                1000.0,
                1000.1,
            )
            snapshot = poller._accept(_response(revision=0), 0, 1000.2, 1000.3)
            assert snapshot is not None and snapshot.session_id == "session-1"
        finally:
            poller.detach()

    async def test_invalidate_discards_the_accepted_snapshot(self) -> None:
        poller = _poller()
        poller.attach()
        try:
            poller._accept(_response(revision=1), poller._fence, 1000.0, 1000.1)
            assert poller.accepted is not None
            poller.invalidate()
            assert poller.accepted is None
            assert poller._fence >= 1
        finally:
            poller.detach()


@pytest.mark.asyncio
class TestEmissionLegs:
    def _setup(self) -> tuple[AgentLivenessPoller, _Clock]:
        clock = _Clock()
        poller = _poller(clock=clock)
        # The legs belong to an attached poller; the hanging read parks the
        # loop so the manual snapshot below is the one the queries read.
        poller.attach()
        return poller, clock

    def _teardown(self, poller: AgentLivenessPoller) -> None:
        poller.detach()

    async def test_a_fresh_unfenced_snapshot_qualifies_both_legs(self) -> None:
        poller, clock = self._setup()
        try:
            poller._accepted = _snapshot(fence=poller._fence)
            assert poller.settlement_input().qualified is True
            assert poller.completion_qualifies() is True
            assert poller.settlement_input().child_active is False
        finally:
            self._teardown(poller)

    async def test_an_active_child_snapshot_qualifies_with_activity(self) -> None:
        poller, _clock = self._setup()
        try:
            poller._accepted = _snapshot(availability="running", fence=poller._fence)
            assert poller.settlement_input().qualified is True
            assert poller.settlement_input().child_active is True
        finally:
            self._teardown(poller)

    async def test_leg_one_a_fence_increment_since_issuance_disqualifies(self) -> None:
        poller, _clock = self._setup()
        try:
            poller._accepted = _snapshot(fence=poller._fence)
            poller.note_event()
            assert poller.settlement_input().qualified is False
            assert poller.completion_qualifies() is False
        finally:
            self._teardown(poller)

    async def test_leg_two_an_over_age_issuance_is_a_retain_input_only(self) -> None:
        poller, clock = self._setup()
        try:
            poller._accepted = _snapshot(fence=poller._fence, accepted_at=1000.5)
            clock.advance(2 * _INTERVAL)
            assert poller.settlement_input().qualified is False
            assert poller.completion_qualifies() is False
        finally:
            self._teardown(poller)

    async def test_leg_two_over_age_acceptance_window_disqualifies_a_settlement(
        self,
    ) -> None:
        poller, clock = self._setup()
        try:
            # Issuance-to-acceptance within the bound, but the acceptance is
            # now older than one poll interval: the chain no longer sits
            # inside it.
            poller._accepted = _snapshot(
                fence=poller._fence,
                issued_at=1000.0,
                accepted_at=1000.0 + _INTERVAL / 2,
            )
            clock.advance(_INTERVAL + 0.01)
            assert poller.settlement_input().qualified is False
        finally:
            self._teardown(poller)

    async def test_a_slow_read_fails_the_issuance_to_acceptance_bound(self) -> None:
        poller, _clock = self._setup()
        try:
            poller._accepted = _snapshot(
                fence=poller._fence,
                issued_at=1000.0,
                accepted_at=1000.0 + 2 * _INTERVAL,
            )
            assert poller.settlement_input().qualified is False
        finally:
            self._teardown(poller)


class _ScriptedReads:
    """A read callable with a hand-pumpable, parkable response queue."""

    def __init__(self) -> None:
        self.queue: deque[AgentsInventoryReadResponse | Exception] = deque()
        self.issued = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self._arrived = asyncio.Event()

    def append(self, item: AgentsInventoryReadResponse | Exception) -> None:
        self.queue.append(item)
        self._arrived.set()

    async def __call__(self) -> AgentsInventoryReadResponse:
        self.issued += 1
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            while not self.queue:
                # No script left: park — the read stays the one in flight
                # until a test appends a response (the coalescing seam).
                await self._arrived.wait()
            self._arrived.clear()
            item = self.queue.popleft()
            if isinstance(item, Exception):
                raise item
            return item
        finally:
            self.in_flight -= 1


@pytest.mark.asyncio
class TestPollLoopDisciplines:
    async def test_a_accepted_read_drives_the_completion_callback(self) -> None:
        reads = _ScriptedReads()
        reads.queue.append(_response(revision=1))
        accepted: list[LivenessSnapshot] = []
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            on_accept=accepted.append,
            poll_interval=_INTERVAL,
        )
        poller.attach()
        try:

            async def _wait() -> None:
                while not accepted:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_wait(), timeout=2.0)
            assert accepted[0].revision == 1
        finally:
            poller.detach()

    async def test_one_in_flight_with_coalesced_pending(self) -> None:
        reads = _ScriptedReads()
        reads.queue.append(_response(revision=1))
        reads.queue.append(_response(revision=2))
        accepted: list[LivenessSnapshot] = []
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            on_accept=accepted.append,
            poll_interval=_INTERVAL,
            read_timeout=5.0,
        )
        poller.attach()
        try:
            # The first read completes; two events coalesce behind it.
            async def _first() -> None:
                while not accepted:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_first(), timeout=2.0)
            poller.note_event()
            poller.note_event()

            async def _second() -> None:
                while len(accepted) < 2:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_second(), timeout=2.0)
            assert [snapshot.revision for snapshot in accepted] == [1, 2]
            # The two event marks coalesced behind one outstanding read at a
            # time: the poller never had two requests in flight.
            assert reads.max_in_flight == 1
        finally:
            poller.detach()

    async def test_failure_is_unknown_keeps_the_last_snapshot_and_retries(self) -> None:
        reads = _ScriptedReads()
        reads.queue.append(_response(revision=1))
        reads.queue.append(ValueError("transport hiccup"))
        reads.queue.append(_response(revision=2))
        diagnostics: list[str] = []
        accepted: list[LivenessSnapshot] = []
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            on_accept=accepted.append,
            poll_interval=_INTERVAL,
            diagnose=diagnostics.append,
        )
        poller.attach()
        try:

            async def _second() -> None:
                while len(accepted) < 2:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_second(), timeout=2.0)
            # The failed read never cleared the accepted inventory.
            assert [snapshot.revision for snapshot in accepted] == [1, 2]
            assert diagnostics, "a bounded diagnostic was expected"
        finally:
            poller.detach()

    async def test_a_timeout_is_unknown_keeps_the_snapshot_and_retries(self) -> None:
        # The bounded read timeout never clears the accepted inventory: the
        # hung read is unknown, the last safe snapshot survives, and the
        # poller keeps issuing (retrying continues while attached).
        first = _response(revision=1)
        diagnostics: list[str] = []
        accepted: list[LivenessSnapshot] = []

        class _OneThenHang:
            def __init__(self) -> None:
                self.issued = 0

            async def __call__(self) -> AgentsInventoryReadResponse:
                self.issued += 1
                if self.issued == 1:
                    return first
                await asyncio.sleep(30.0)
                raise AssertionError("unreachable")

        reads = _OneThenHang()
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            on_accept=accepted.append,
            poll_interval=_INTERVAL,
            read_timeout=0.01,
            diagnose=diagnostics.append,
        )
        poller.attach()
        try:

            async def _first() -> None:
                while not accepted:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_first(), timeout=2.0)

            async def _diagnosed() -> None:
                while not diagnostics:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_diagnosed(), timeout=2.0)
            assert poller.accepted is not None
            assert poller.accepted.revision == 1
            assert diagnostics
        finally:
            poller.detach()

    async def test_death_receipt_stops_the_poller(self) -> None:
        reads = _ScriptedReads()
        reads.queue.append(_death_receipt())
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            poll_interval=_INTERVAL,
        )
        poller.attach()
        task = poller._task
        try:
            await asyncio.wait_for(_finished(task), timeout=2.0)
            assert poller.confirmed_dead is True
            assert poller.attached is False
            # Receipt-scoped: a settlement emits from the record best-effort.
            view = poller.settlement_input()
            assert view.qualified is True
            assert view.child_active is False
        finally:
            poller.detach()

    async def test_rearm_resumes_polling_after_death(self) -> None:
        reads = _ScriptedReads()
        reads.queue.append(_death_receipt())
        accepted: list[LivenessSnapshot] = []
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            on_accept=accepted.append,
            poll_interval=_INTERVAL,
        )
        poller.attach()
        try:
            await asyncio.wait_for(_finished(poller._task), timeout=2.0)
            assert poller.confirmed_dead is True
            poller.rearm()
            reads.append(_response(revision=1, generation=2))

            async def _rearmed() -> None:
                while not accepted:
                    await asyncio.sleep(0)

            await asyncio.wait_for(_rearmed(), timeout=2.0)
            assert poller.confirmed_dead is False
            assert accepted[0].generation == 2
        finally:
            poller.detach()

    async def test_invalidate_rejects_a_read_that_was_in_flight(self) -> None:
        reads = _ScriptedReads()
        accepted: list[LivenessSnapshot] = []
        poller = AgentLivenessPoller(
            read=reads,
            current_session_id=lambda: "session-1",
            is_active=agent_is_active,
            on_accept=accepted.append,
            poll_interval=_INTERVAL,
        )
        poller.attach()
        try:
            # The loop issues its first read and parks on the empty script.
            await asyncio.sleep(0.05)
            poller.invalidate()
            reads.append(_response(revision=1))

            async def _after() -> None:
                for _ in range(50):
                    await asyncio.sleep(0.02)
                    if poller._fence >= 2:
                        return

            await asyncio.wait_for(_after(), timeout=2.0)
            # The parked read answers only after the invalidation retired it:
            # the epoch check discards it, so nothing from the old scope is
            # accepted (the poller re-reads on its next tick).
            assert poller.accepted is None
        finally:
            poller.detach()


async def _finished(task: asyncio.Task | None) -> None:
    if task is None:
        return
    while not task.done():
        await asyncio.sleep(0)
