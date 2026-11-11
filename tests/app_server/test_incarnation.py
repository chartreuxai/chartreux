from __future__ import annotations

from typing import Any

import pytest

from chartreux.app_server.relay import (
    ControlCapacityExhausted,
    FrameClass,
    FrameSchemaError,
    IncarnationState,
    RelayHop,
)


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
