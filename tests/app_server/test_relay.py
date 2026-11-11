"""Relay foundation: closed frame schema, byte accounting, fair scheduling."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from typing import Any, TypedDict, Unpack

import pytest

from chartreux.app_server.demux import StreamAbortedError, StreamDemultiplexer
from chartreux.app_server.protocol import server_notification_registry
from chartreux.app_server.relay import (
    CONTROL_BYTES_PER_HOP,
    PRESENTATION_BYTES_PER_HOP,
    PRESENTATION_BYTES_PER_STREAM,
    PRESENTATION_DELTA_METHODS,
    ByteBudget,
    ControlCapacityExhausted,
    FrameClass,
    FrameDisposition,
    FrameSchemaError,
    QueuedFrame,
    RelayClosed,
    RelayConnectionFailed,
    RelayHop,
    RelayStreamAborted,
    classify_frame,
    encoded_frame_size,
    frame_classification,
)

_PRESENTATION = "session/updated"
_CONTROL = "session/snapshot"


def _frame(method: str, tag: str = "tag", *, pad: int = 1) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": {"tag": tag, "pad": "x" * pad},
    }


def _size(method: str, *, pad: int = 1) -> int:
    return encoded_frame_size(_frame(method, pad=pad))


class _RecordingSink:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.aborted: list[str] = []
        self.started = asyncio.Event()
        self.gate: asyncio.Event | None = None
        self.failure: BaseException | None = None

    async def send(self, message: dict[str, Any]) -> None:
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.failure is not None:
            raise self.failure
        self.messages.append(message)

    def abort(self, reason: str) -> int:
        self.aborted.append(reason)
        return 0


class _SinkTable:
    def __init__(self) -> None:
        self.sinks: dict[str, _RecordingSink] = {}

    def __call__(self, stream_id: str) -> _RecordingSink:
        return self.sinks.setdefault(stream_id, _RecordingSink())


class _HopLimits(TypedDict, total=False):
    presentation_per_stream: int
    presentation_per_hop: int
    control_capacity: int


def _hop(**limits: Unpack[_HopLimits]) -> RelayHop:
    return RelayHop(_SinkTable(), **limits)


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not met before the deadline")
        await asyncio.sleep(0.001)


def _message(method: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": {"pad": "x"}}


def _tags(sink: _RecordingSink, prefix: str) -> list[str]:
    return [
        message["params"]["tag"]
        for message in sink.messages
        if message["params"]["tag"].startswith(prefix)
    ]


def test_frame_schema_is_closed_over_the_declared_notifications() -> None:
    registry = server_notification_registry()
    schema = frame_classification()

    assert schema is frame_classification()
    assert set(schema) == set(registry)
    assert PRESENTATION_DELTA_METHODS <= set(schema)
    for method in schema:
        expected = (
            FrameClass.PRESENTATION
            if method in PRESENTATION_DELTA_METHODS
            else FrameClass.CONTROL
        )
        assert schema[method] is expected


@pytest.mark.parametrize(
    "method",
    ["session/snapshot", "session/compacted", "turn/completed", "callback/childClosed"],
)
def test_terminal_state_and_snapshot_pages_are_control_class(method: str) -> None:
    assert classify_frame(_message(method)) is FrameClass.CONTROL


@pytest.mark.parametrize("method", sorted(PRESENTATION_DELTA_METHODS))
def test_declared_deltas_are_droppable(method: str) -> None:
    assert classify_frame(_message(method)) is FrameClass.PRESENTATION


def test_requests_and_responses_are_control_class() -> None:
    request = {"jsonrpc": "2.0", "id": 7, "method": "turn/start", "params": {}}
    success = {"jsonrpc": "2.0", "id": 7, "result": {}}
    failure = {"jsonrpc": "2.0", "id": 7, "error": {"code": "internal_error"}}

    assert classify_frame(request) is FrameClass.CONTROL
    assert classify_frame(success) is FrameClass.CONTROL
    assert classify_frame(failure) is FrameClass.CONTROL


@pytest.mark.parametrize(
    "frame",
    [
        {"jsonrpc": "2.0", "method": "relay/unknown", "params": {}},
        {"jsonrpc": "2.0"},
        {"jsonrpc": "2.0", "id": 3},
        {"jsonrpc": "1.0", "method": _PRESENTATION, "params": {}},
        {"method": _PRESENTATION, "params": {}},
        "not-an-object",
    ],
)
def test_frames_outside_the_closed_schema_fail_typed(frame: object) -> None:
    with pytest.raises(FrameSchemaError):
        classify_frame(frame)  # type: ignore[arg-type]


def test_undeclared_notification_is_rejected_by_the_hop() -> None:
    hop = _hop()
    stream = hop.open_stream("agent-a")

    with pytest.raises(FrameSchemaError, match="undeclared notification method"):
        stream.enqueue({"jsonrpc": "2.0", "method": "relay/unknown", "params": {}})
    assert stream.queued_frames == 0


def test_encoded_frame_size_is_the_compact_wire_size() -> None:
    frame = _frame(_PRESENTATION, "a-1", pad=3)

    assert encoded_frame_size(frame) == len(
        json.dumps(frame, separators=(",", ":")).encode()
    )
    with pytest.raises(FrameSchemaError, match="JSON-encodable"):
        encoded_frame_size({"jsonrpc": "2.0", "params": {"bad": {1, 2}}})


def test_module_bounds_are_the_plan_table_values() -> None:
    hop = _hop()

    assert PRESENTATION_BYTES_PER_STREAM == 4 * 1024 * 1024
    assert PRESENTATION_BYTES_PER_HOP == 16 * 1024 * 1024
    assert CONTROL_BYTES_PER_HOP == 1024 * 1024
    assert hop.presentation_capacity == PRESENTATION_BYTES_PER_HOP
    assert hop.control_capacity == CONTROL_BYTES_PER_HOP


def test_default_per_stream_bound_sheds_one_oversized_delta() -> None:
    hop = _hop()
    stream = hop.open_stream("agent-a")

    outcome = stream.enqueue(_frame(_PRESENTATION, pad=PRESENTATION_BYTES_PER_STREAM))

    assert outcome.disposition is FrameDisposition.SHED
    assert stream.queued_frames == 0
    assert hop.presentation_bytes == 0


def test_per_stream_budget_sheds_presentation_but_admits_control() -> None:
    size = _size(_PRESENTATION)
    control_size = _size(_CONTROL)
    hop = _hop(
        presentation_per_stream=size * 2,
        presentation_per_hop=size * 100,
        control_capacity=control_size * 2,
    )
    stream = hop.open_stream("agent-a")

    first = stream.enqueue(_frame(_PRESENTATION, "a-1"))
    second = stream.enqueue(_frame(_PRESENTATION, "a-2"))
    third = stream.enqueue(_frame(_PRESENTATION, "a-3"))

    assert first.disposition is FrameDisposition.QUEUED
    assert second.disposition is FrameDisposition.QUEUED
    assert third.frame_class is FrameClass.PRESENTATION
    assert third.disposition is FrameDisposition.SHED
    assert third.shed_reason == "stream presentation budget exhausted"
    assert stream.shed_frames == 1
    assert hop.shed_frames == 1
    assert hop.presentation_bytes == size * 2

    control = stream.enqueue(_frame(_CONTROL, "a-c"))
    assert control.disposition is FrameDisposition.QUEUED
    assert hop.presentation_bytes == size * 2
    assert hop.control_bytes == control_size

    hop.dequeue()
    assert hop.presentation_bytes == size
    assert stream.enqueue(_frame(_PRESENTATION, "a-4")).disposition is (
        FrameDisposition.QUEUED
    )


def test_hop_aggregate_budget_sheds_across_streams() -> None:
    size = _size(_PRESENTATION)
    hop = _hop(presentation_per_stream=size * 10, presentation_per_hop=size * 2)
    first = hop.open_stream("agent-a")
    second = hop.open_stream("agent-b")

    assert first.enqueue(_frame(_PRESENTATION, "a-1")).disposition is (
        FrameDisposition.QUEUED
    )
    assert first.enqueue(_frame(_PRESENTATION, "a-2")).disposition is (
        FrameDisposition.QUEUED
    )
    shed = second.enqueue(_frame(_PRESENTATION, "b-1"))

    assert shed.disposition is FrameDisposition.SHED
    assert shed.shed_reason == "hop presentation budget exhausted"
    assert second.queued_frames == 0
    assert first.queued_frames == 2
    assert hop.presentation_bytes == size * 2


def test_reserved_control_capacity_is_independent_of_presentation() -> None:
    size = _size(_PRESENTATION)
    control_size = _size(_CONTROL)
    hop = _hop(
        presentation_per_stream=size,
        presentation_per_hop=size,
        control_capacity=control_size * 2,
    )
    stream = hop.open_stream("agent-a")

    assert stream.enqueue(_frame(_CONTROL, "c-1")).disposition is (
        FrameDisposition.QUEUED
    )
    assert stream.enqueue(_frame(_CONTROL, "c-2")).disposition is (
        FrameDisposition.QUEUED
    )
    with pytest.raises(ControlCapacityExhausted, match="reserved control capacity"):
        stream.enqueue(_frame(_CONTROL, "c-3"))
    assert hop.control_bytes == control_size * 2
    assert hop.shed_frames == 0

    assert stream.enqueue(_frame(_PRESENTATION, "a-1")).disposition is (
        FrameDisposition.QUEUED
    )
    assert hop.presentation_bytes == size
    assert stream.enqueue(_frame(_PRESENTATION, "a-2")).disposition is (
        FrameDisposition.SHED
    )


def test_streams_are_fifo_and_scheduled_fairly() -> None:
    hop = _hop()
    first = hop.open_stream("agent-a")
    second = hop.open_stream("agent-b")
    for tag in ("a-1", "a-2", "a-3", "a-4"):
        first.enqueue(_frame(_PRESENTATION, tag))
    second.enqueue(_frame(_PRESENTATION, "b-1"))

    order = []
    while (item := hop.dequeue()) is not None:
        stream, queued = item
        order.append((stream.stream_id, queued.message["params"]["tag"]))

    assert order == [
        ("agent-a", "a-1"),
        ("agent-b", "b-1"),
        ("agent-a", "a-2"),
        ("agent-a", "a-3"),
        ("agent-a", "a-4"),
    ]
    assert hop.presentation_bytes == 0


@pytest.mark.asyncio
async def test_scheduler_writes_each_stream_in_order() -> None:
    table = _SinkTable()
    hop = RelayHop(table)
    first = hop.open_stream("agent-a")
    second = hop.open_stream("agent-b")
    hop.start()
    try:
        first.enqueue(_frame(_PRESENTATION, "a-1"))
        first.enqueue(_frame(_PRESENTATION, "a-2"))
        second.enqueue(_frame(_CONTROL, "b-1"))
        await _wait_until(
            lambda: (
                len(table.sinks["agent-a"].messages)
                + len(table.sinks["agent-b"].messages)
                == 3
            )
        )
    finally:
        await hop.close()

    tags = [message["params"]["tag"] for message in table.sinks["agent-a"].messages]
    assert tags == ["a-1", "a-2"]
    assert [m["params"]["tag"] for m in table.sinks["agent-b"].messages] == ["b-1"]
    assert hop.presentation_bytes == 0
    assert hop.control_bytes == 0


@pytest.mark.asyncio
async def test_close_drains_frames_already_queued() -> None:
    table = _SinkTable()
    sink = table("agent-a")
    gate = asyncio.Event()
    sink.gate = gate
    hop = RelayHop(table)
    stream = hop.open_stream("agent-a")
    hop.start()
    stream.enqueue(_frame(_PRESENTATION, "a-1"))
    stream.enqueue(_frame(_PRESENTATION, "a-2"))
    await asyncio.wait_for(sink.started.wait(), timeout=2)

    closing = asyncio.create_task(hop.close())
    await asyncio.sleep(0.01)
    assert not closing.done()

    gate.set()
    await closing

    assert [m["params"]["tag"] for m in sink.messages] == ["a-1", "a-2"]
    assert hop.closed is True


@pytest.mark.asyncio
async def test_abortive_close_discards_every_queue() -> None:
    table = _SinkTable()
    sink = table("agent-a")
    gate = asyncio.Event()
    sink.gate = gate
    hop = RelayHop(table)
    stream = hop.open_stream("agent-a")
    hop.start()
    stream.enqueue(_frame(_PRESENTATION, "a-1"))
    stream.enqueue(_frame(_PRESENTATION, "a-2"))
    stream.enqueue(_frame(_PRESENTATION, "a-3"))
    await asyncio.wait_for(sink.started.wait(), timeout=2)

    await hop.close(drain=False)
    gate.set()
    await asyncio.sleep(0.01)

    assert sink.messages == []
    assert stream.queued_frames == 0
    assert hop.presentation_bytes == 0
    assert isinstance(stream.terminal_error, RelayStreamAborted)
    with pytest.raises(RelayStreamAborted):
        stream.enqueue(_frame(_PRESENTATION, "a-4"))


@pytest.mark.asyncio
async def test_isolated_stream_discard_leaves_other_streams_working() -> None:
    sink = _RecordingSink()
    demux = StreamDemultiplexer(sink)
    hop = RelayHop.over_demux(demux)
    aborted = hop.open_stream("agent-a")
    survivor = hop.open_stream("agent-b")
    hop.start()
    try:
        aborted.enqueue(_frame(_PRESENTATION, "a-1"))
        survivor.enqueue(_frame(_PRESENTATION, "b-1"))
        await _wait_until(lambda: len(sink.messages) == 2)

        aborted.discard("heartbeat enqueue failed")

        assert isinstance(aborted.terminal_error, RelayStreamAborted)
        assert aborted.terminal_error.reason == "heartbeat enqueue failed"
        demux_stream = demux.stream("agent-a")
        assert demux_stream is not None
        assert isinstance(demux_stream.terminal_error, StreamAbortedError)
        survivor.enqueue(_frame(_PRESENTATION, "b-2"))
        await _wait_until(lambda: _tags(sink, "b-") == ["b-1", "b-2"])
        assert [m["params"]["tag"] for m in sink.messages].count("a-1") == 1
        assert survivor.terminal_error is None
        assert demux.connection_error is None
        assert demux.closed is False
        with pytest.raises(RelayStreamAborted):
            aborted.enqueue(_frame(_PRESENTATION, "a-2"))
    finally:
        await hop.close()
        await demux.close()


@pytest.mark.asyncio
async def test_demux_abort_reaches_only_its_own_relay_stream() -> None:
    sink = _RecordingSink()
    demux = StreamDemultiplexer(sink)
    hop = RelayHop.over_demux(demux)
    aborted = hop.open_stream("agent-a")
    survivor = hop.open_stream("agent-b")
    hop.start()
    try:
        demux_stream = demux.stream("agent-a")
        assert demux_stream is not None
        demux_stream.abort("heartbeat enqueue failed")
        aborted.enqueue(_frame(_PRESENTATION, "a-1"))
        survivor.enqueue(_frame(_PRESENTATION, "b-1"))
        await _wait_until(lambda: aborted.terminal_error is not None)

        assert isinstance(aborted.terminal_error, RelayStreamAborted)
        assert aborted.terminal_error.reason == "heartbeat enqueue failed"
        survivor.enqueue(_frame(_PRESENTATION, "b-2"))
        await _wait_until(
            lambda: [m["params"]["tag"] for m in sink.messages] == ["b-1", "b-2"]
        )
        assert survivor.terminal_error is None
        assert hop.failure is None
    finally:
        await hop.close()
        await demux.close()


@pytest.mark.asyncio
async def test_shared_connection_failure_terminates_every_stream() -> None:
    sink = _RecordingSink()
    sink.failure = OSError("shared pipe closed")
    demux = StreamDemultiplexer(sink)
    hop = RelayHop.over_demux(demux)
    first = hop.open_stream("agent-a")
    second = hop.open_stream("agent-b")
    hop.start()
    try:
        first.enqueue(_frame(_PRESENTATION, "a-1"))
        second.enqueue(_frame(_PRESENTATION, "b-1"))
        await _wait_until(lambda: demux.connection_error is not None)

        first.enqueue(_frame(_PRESENTATION, "a-2"))
        await _wait_until(lambda: hop.failure is not None)

        assert isinstance(hop.failure, RelayConnectionFailed)
        assert "shared pipe closed" in str(hop.failure)
        assert first.terminal_error is hop.failure
        assert second.terminal_error is hop.failure
        assert hop.closed is True
        assert hop.dequeue() is None
        assert hop.presentation_bytes == 0
        with pytest.raises(RelayConnectionFailed):
            first.enqueue(_frame(_PRESENTATION, "a-3"))
        with pytest.raises(RelayConnectionFailed):
            hop.open_stream("agent-c")
    finally:
        await hop.close()
        await demux.close()


def test_hop_validates_its_inputs_and_stream_lifecycle() -> None:
    with pytest.raises(ValueError, match="presentation_per_stream must be positive"):
        _hop(presentation_per_stream=0)
    with pytest.raises(ValueError, match="byte budget must be positive"):
        _hop(control_capacity=0)

    hop = _hop()
    with pytest.raises(ValueError, match="stream_id must be a non-empty string"):
        hop.open_stream("")
    hop.open_stream("agent-a")
    with pytest.raises(ValueError, match="already open"):
        hop.open_stream("agent-a")
    with pytest.raises(ValueError, match="is not open"):
        hop.enqueue("agent-b", _frame(_PRESENTATION))
    assert hop.stream("agent-b") is None
    assert [stream.stream_id for stream in hop.streams] == ["agent-a"]


@pytest.mark.asyncio
async def test_close_without_a_writer_discards_queued_frames() -> None:
    hop = _hop()
    stream = hop.open_stream("agent-a")
    stream.enqueue(_frame(_PRESENTATION, "a-1"))

    await hop.close()

    assert hop.closed is True
    assert stream.queued_frames == 0
    assert stream.terminal_error is not None
    with pytest.raises(RelayClosed):
        hop.open_stream("agent-b")


def test_byte_budget_accounts_and_validates() -> None:
    budget = ByteBudget(10)

    assert budget.available == 10
    assert budget.try_reserve(4) is True
    assert budget.try_reserve(7) is False
    assert budget.reserved == 4
    budget.release(4)
    assert budget.reserved == 0
    with pytest.raises(ValueError, match="negative size"):
        budget.try_reserve(-1)
    with pytest.raises(ValueError, match="not reserved"):
        budget.release(1)


def test_dequeue_returns_queued_frames_with_their_class() -> None:
    hop = _hop()
    stream = hop.open_stream("agent-a")
    stream.enqueue(_frame(_CONTROL, "c-1"))

    item = hop.dequeue()

    assert item is not None
    dequeued_stream, queued = item
    assert dequeued_stream is stream
    assert isinstance(queued, QueuedFrame)
    assert queued.frame_class is FrameClass.CONTROL
    assert queued.message["params"]["tag"] == "c-1"
    assert hop.dequeue() is None
