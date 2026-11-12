from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from chartreux.app_server.agent_admission import (
    OP_OUTCOME_ABORTED,
    OP_OUTCOME_COMPLETED,
    OP_STATE_ACTIVE,
    OP_STATE_TERMINAL,
    LongOperationFence,
    OperationAbortedError,
    OperationConflictError,
    OperationExpiredError,
    OperationMarkerTable,
)
from chartreux.app_server.agent_snapshot import (
    MARKER_STALE,
    MARKER_TERMINAL,
    build_session_snapshot,
    marker_verdict,
)
from chartreux.app_server.relay import (
    ControlCapacityExhausted,
    FrameClass,
    FrameSchemaError,
    IncarnationState,
    RelayHop,
)
from tests.conftest import build_test_agent_loop


def test_retired_frames_do_not_refresh_replacement_liveness() -> None:
    now = [0.0]
    state = IncarnationState("first", clock=lambda: now[0])
    assert state.retire("first")
    assert state.deadline is None
    assert state.replace("second", "first")
    now[0] = 9.0
    assert not state.receive("first")
    now[0] = 10.0
    assert state.expired()
    assert state.receive("second")
    assert not state.expired()


def test_replacement_and_unannounced_frames() -> None:
    state = IncarnationState("first")
    assert state.replace("second", "first")
    assert not state.retire("first")
    assert not state.replace("second", "first")
    with pytest.raises(FrameSchemaError):
        state.receive("unknown")


class Sink:
    async def send(self, message: dict[str, Any]) -> None:
        pass

    def abort(self, reason: str) -> int:
        return 0


def test_retirement_invalidates_queue_and_precedes_new_frames() -> None:
    hop = RelayHop(lambda _: Sink())
    stream = hop.open_stream("s", incarnation=True)
    frame = {"jsonrpc": "2.0", "method": "session/updated", "params": {}}
    stream.enqueue(frame)
    assert stream.retire_incarnation() == 1
    assert stream.invalidated_frames == 1
    assert hop.presentation_bytes == 0
    assert hop.control_bytes > 0
    stream.enqueue(frame)
    first = hop.dequeue()
    second = hop.dequeue()
    assert first is not None and second is not None
    assert first[1].frame_class is FrameClass.CONTROL
    assert first[1].message["method"] == "relay/retire"
    assert first[1].message["relay_incarnation"] == 0
    assert second[1].message["relay_incarnation"] == 1
    assert hop.dequeue() is None


@pytest.mark.parametrize("reverse", [False, True])
def test_receive_drops_stale_in_both_directions(reverse: bool) -> None:
    now = [0.0]
    left = RelayHop(lambda _: Sink(), clock=lambda: now[0])
    right = RelayHop(lambda _: Sink(), clock=lambda: now[0])
    a = left.open_stream("s", incarnation=True)
    b = right.open_stream("s", incarnation=True)
    sender, receiver = (b, a) if reverse else (a, b)
    sender_hop = right if reverse else left
    frame = {"jsonrpc": "2.0", "method": "session/updated", "params": {}}
    sender.enqueue(frame)
    stale = sender_hop.dequeue()
    assert stale is not None
    sender.retire_incarnation()
    notice = sender_hop.dequeue()
    assert notice is not None
    assert receiver.receive(notice[1].message) is None
    now[0] = 9.0
    assert receiver.receive(stale[1].message) is None
    assert sender.receive(stale[1].message) is None
    assert receiver.invalidated_frames == 1
    assert sender.invalidated_frames == 1
    now[0] = 10.0
    assert receiver.idle_expired()
    sender.enqueue(frame)
    current = sender_hop.dequeue()
    assert current is not None
    assert receiver.receive(current[1].message) == frame
    assert not receiver.idle_expired()


def test_retirement_capacity_failure_is_atomic() -> None:
    hop = RelayHop(lambda _: Sink(), control_capacity=1)
    stream = hop.open_stream("s", incarnation=True)
    stream.enqueue({"jsonrpc": "2.0", "method": "session/updated", "params": {}})
    with pytest.raises(ControlCapacityExhausted):
        stream.retire_incarnation()
    assert stream.generation == 0
    assert stream.queued_frames == 1
    assert stream.invalidated_frames == 0


# ---------------------------------------------------------------------------
# The long-operation marker table: one incarnation, never pruned
# ---------------------------------------------------------------------------
#
# A marker record carries the coordinator's request id (S9-C1), so correlation
# and absence are direct. Records are never pruned inside an incarnation, which
# is exactly what lets a post-deadline snapshot prove never-admitted from an
# absence; the table is cleared at an identity rebind except the rebind
# operation's own TERMINAL record, which seeds the new incarnation.


def test_marker_records_are_never_pruned_within_an_incarnation() -> None:
    table = OperationMarkerTable("inc-1")

    first = table.admit(request_id="coordinator:1", method="agent/open", deadline=100.0)
    table.settle(first.op_id, OP_OUTCOME_COMPLETED)
    second = table.admit(
        request_id="coordinator:2", method="agent/open", deadline=101.0
    )

    assert [record.request_id for record in table.records()] == [
        "coordinator:1",
        "coordinator:2",
    ]
    assert table.for_request("coordinator:1") == first.settled(OP_OUTCOME_COMPLETED)
    assert table.for_op(second.op_id) == second
    assert second.op_id == f"{first.incarnation_id}:2"
    assert first.op_id != second.op_id


def test_a_rebind_clears_the_table_but_seeds_the_rebind_record() -> None:
    table = OperationMarkerTable("inc-1")
    retired = table.admit(
        request_id="coordinator:1", method="agent/open", deadline=100.0
    )
    table.settle(retired.op_id, OP_OUTCOME_COMPLETED)
    rebind = table.admit(
        request_id="coordinator:2", method="agent/session/replace", deadline=101.0
    )
    table.settle(rebind.op_id, OP_OUTCOME_COMPLETED)

    table.rebind(incarnation_id="inc-2", carry_op_id=rebind.op_id)

    # The seed keeps the op id its old incarnation minted; the cleared table
    # proves nothing about the retired incarnation's requests.
    assert table.records() == (rebind.settled(OP_OUTCOME_COMPLETED),)
    assert table.for_request("coordinator:1") is None
    fresh = table.admit(request_id="coordinator:3", method="agent/open", deadline=102.0)
    assert fresh.incarnation_id == "inc-2"
    assert fresh.op_id == "inc-2:1"
    assert fresh.op_id != rebind.op_id


def test_a_rebind_refuses_to_carry_an_unsettled_record() -> None:
    table = OperationMarkerTable("inc-1")
    active = table.admit(
        request_id="coordinator:1", method="agent/open", deadline=100.0
    )

    with pytest.raises(OperationConflictError):
        table.rebind(incarnation_id="inc-2", carry_op_id=active.op_id)
    with pytest.raises(OperationConflictError):
        table.admit(request_id="coordinator:1", method="agent/open", deadline=101.0)

    assert active.state == OP_STATE_ACTIVE


def test_a_carried_rebind_seed_reads_terminal_across_the_incarnation() -> None:
    """The replacement's re-check reads its own seed out of the NEW table.

    A rebind clears the marker table except the rebind operation's own TERMINAL
    record, which keeps the retired incarnation's op id. Presence therefore
    outranks the incarnation check: the coordinator still expects the old
    incarnation, and the carried seed is exactly the evidence it needs.
    """
    table = OperationMarkerTable("inc-1")
    rebind = table.admit(
        request_id="coordinator:9", method="agent/session/replace", deadline=10.0
    )
    table.settle(rebind.op_id, OP_OUTCOME_COMPLETED)
    table.rebind(incarnation_id="inc-2", carry_op_id=rebind.op_id)
    loop = build_test_agent_loop()

    replacement = build_session_snapshot(
        loop, markers=table.records(), incarnation_id="inc-2", clock=lambda: 11.0
    )
    verdict = marker_verdict(
        replacement, request_id="coordinator:9", deadline=10.5, incarnation_id="inc-1"
    )

    assert verdict.state == MARKER_TERMINAL
    assert verdict.outcome == OP_OUTCOME_COMPLETED
    assert verdict.op_id == "inc-1:1"

    # Absence from the replacement's table proves nothing about the retired
    # incarnation, so the same read falls back to stale instead of absent.
    cleared = build_session_snapshot(
        loop, markers=(), incarnation_id="inc-2", clock=lambda: 11.0
    )
    absent = marker_verdict(
        cleared, request_id="coordinator:1", deadline=10.5, incarnation_id="inc-1"
    )
    assert absent.state == MARKER_STALE
    assert absent.never_admitted is False


@pytest.mark.asyncio
async def test_a_second_long_operation_is_refused_while_the_fence_is_held() -> None:
    table = OperationMarkerTable("inc-1")
    release = asyncio.Event()

    async def announce(marker: Any) -> None:
        return None

    async def blocked_body() -> dict[str, Any]:
        await release.wait()
        return {"session_id": "held"}

    fence = LongOperationFence(table=table, announce=announce)
    held = asyncio.create_task(
        fence.admit(
            request_id="coordinator:1",
            method="agent/open",
            deadline=time.monotonic() + 30.0,
            body=blocked_body,
        )
    )
    while fence.active_op_id is None:
        await asyncio.sleep(0)

    with pytest.raises(OperationConflictError):
        await fence.admit(
            request_id="coordinator:2",
            method="agent/open",
            deadline=time.monotonic() + 30.0,
            body=blocked_body,
        )

    release.set()
    assert await held == {"session_id": "held"}
    assert fence.active_op_id is None
    settled = table.for_request("coordinator:1")
    assert settled is not None
    assert settled.state == OP_STATE_TERMINAL
    assert settled.outcome == OP_OUTCOME_COMPLETED


@pytest.mark.asyncio
async def test_an_admission_past_its_absolute_deadline_never_starts_the_body() -> None:
    table = OperationMarkerTable("inc-1")
    started: list[str] = []

    async def announce(marker: Any) -> None:
        return None

    async def body() -> dict[str, Any]:
        started.append("ran")
        return {}

    fence = LongOperationFence(table=table, announce=announce)

    with pytest.raises(OperationExpiredError):
        await fence.admit(
            request_id="coordinator:1",
            method="agent/open",
            deadline=time.monotonic() - 0.1,
            body=body,
        )

    assert started == []
    assert table.records() == ()
    assert fence.active_op_id is None


@pytest.mark.asyncio
async def test_the_deadline_abort_cancels_a_blocked_announcement() -> None:
    """S9-C2: the timer cancels what the operation AWAITS, then settles it.

    A blocked admission announcement is exactly the case the coordinator's
    notification would otherwise be lost behind: the timer cancels the
    emission itself, writes the TERMINAL record, releases the fence, and the
    request id gets its typed answer.
    """
    table = OperationMarkerTable("inc-1")
    announced = asyncio.Event()
    release = asyncio.Event()
    body_ran: list[str] = []

    async def blocked_announce(marker: Any) -> None:
        announced.set()
        await release.wait()

    async def body() -> dict[str, Any]:
        body_ran.append("ran")
        return {}

    fence = LongOperationFence(table=table, announce=blocked_announce)
    admitted = asyncio.create_task(
        fence.admit(
            request_id="coordinator:1",
            method="agent/open",
            deadline=time.monotonic() + 0.1,
            body=body,
        )
    )
    await asyncio.wait_for(announced.wait(), 5)

    with pytest.raises(OperationAbortedError):
        await asyncio.wait_for(admitted, 10)

    record = table.for_request("coordinator:1")
    assert record is not None
    assert record.state == OP_STATE_TERMINAL
    assert record.outcome == OP_OUTCOME_ABORTED
    assert body_ran == []
    assert fence.active_op_id is None


@pytest.mark.asyncio
async def test_the_deadline_abort_cancels_a_body_at_its_await_point() -> None:
    table = OperationMarkerTable("inc-1")
    entered = asyncio.Event()
    never = asyncio.Event()

    async def announce(marker: Any) -> None:
        return None

    async def body() -> dict[str, Any]:
        entered.set()
        await never.wait()
        return {}

    fence = LongOperationFence(table=table, announce=announce)
    admitted = asyncio.create_task(
        fence.admit(
            request_id="coordinator:1",
            method="agent/open",
            deadline=time.monotonic() + 0.1,
            body=body,
        )
    )
    await asyncio.wait_for(entered.wait(), 5)

    with pytest.raises(OperationAbortedError):
        await asyncio.wait_for(admitted, 10)

    record = table.for_request("coordinator:1")
    assert record is not None and record.outcome == OP_OUTCOME_ABORTED
    assert fence.active_op_id is None

    # The fence is released for real: a live admission over it succeeds.
    later = await fence.admit(
        request_id="coordinator:2",
        method="agent/open",
        deadline=time.monotonic() + 30.0,
        body=_completed_body,
    )
    assert later == {"session_id": "second"}
    settled = table.for_request("coordinator:2")
    assert settled is not None and settled.outcome == OP_OUTCOME_COMPLETED


async def _completed_body() -> dict[str, Any]:
    return {"session_id": "second"}
