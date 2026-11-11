"""Symmetric terminal propagation and the connection-fallback blast radius."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import os
from typing import Any

import pytest

from chartreux.app_server.demux import StreamDemultiplexer
from chartreux.app_server.relay import (
    CONTROL_PROPAGATION_TIMEOUT,
    TERMINAL_METHOD,
    ControlCapacityExhausted,
    FrameClass,
    FrameSchemaError,
    RelayConnectionFailed,
    RelayHop,
    RelayStreamAborted,
    TerminalDisposition,
    classify_frame,
    terminal_disposition,
)
from chartreux.app_server.transport import StdioJsonRpcTransport


def _frame(method: str, tag: str = "tag") -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": method, "params": {"tag": tag, "pad": "x"}}


def _notice(disposition: TerminalDisposition, reason: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": TERMINAL_METHOD,
        "params": {"disposition": disposition.value, "reason": reason},
    }


class _RecordingSink:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
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

    async def send_confirmed(self, message: dict[str, Any]) -> None:
        await self.send(message)

    def abort(self, reason: str) -> int:
        return 0


class _SinkTable:
    def __init__(self) -> None:
        self.sinks: dict[str, _RecordingSink] = {}

    def __call__(self, stream_id: str) -> _RecordingSink:
        return self.sinks.setdefault(stream_id, _RecordingSink())


class _PeerSink:
    """Hand every written frame to the same stream on the peer hop."""

    def __init__(self, peer: RelayHop, stream_id: str) -> None:
        self._peer = peer
        self._stream_id = stream_id
        self.delivered: list[dict[str, Any]] = []
        self.accepted: list[dict[str, Any] | None] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.delivered.append(message)
        stream = self._peer.stream(self._stream_id)
        assert stream is not None
        self.accepted.append(stream.accept_inbound(message))

    async def send_confirmed(self, message: dict[str, Any]) -> None:
        await self.send(message)

    def abort(self, reason: str) -> int:
        return 0


class _Link:
    """Two hops wired back to back, stream for stream."""

    def __init__(self) -> None:
        self.left_sinks: dict[str, _PeerSink] = {}
        self.right_sinks: dict[str, _PeerSink] = {}
        self.left = RelayHop(self._left_factory)
        self.right = RelayHop(self._right_factory)

    def _left_factory(self, stream_id: str) -> _PeerSink:
        sink = _PeerSink(self.right, stream_id)
        self.left_sinks[stream_id] = sink
        return sink

    def _right_factory(self, stream_id: str) -> _PeerSink:
        sink = _PeerSink(self.left, stream_id)
        self.right_sinks[stream_id] = sink
        return sink

    async def close(self) -> None:
        await self.left.close()
        await self.right.close()


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not met before the deadline")
        await asyncio.sleep(0.001)


def test_terminal_notices_are_control_class() -> None:
    notice = _notice(TerminalDisposition.STREAM_ABORTED, "idle expired")

    assert classify_frame(notice) is FrameClass.CONTROL
    assert terminal_disposition(notice) is TerminalDisposition.STREAM_ABORTED
    assert terminal_disposition(_frame("session/updated")) is None


@pytest.mark.parametrize(
    "notice",
    [
        {"jsonrpc": "2.0", "method": TERMINAL_METHOD},
        {"jsonrpc": "2.0", "method": TERMINAL_METHOD, "params": {"reason": "why"}},
        {
            "jsonrpc": "2.0",
            "method": TERMINAL_METHOD,
            "params": {"disposition": "stream_aborted"},
        },
        {
            "jsonrpc": "2.0",
            "method": TERMINAL_METHOD,
            "params": {"disposition": "relay_stream_aborted", "reason": "why"},
        },
    ],
)
def test_malformed_terminal_notices_fail_typed(notice: dict[str, Any]) -> None:
    with pytest.raises(FrameSchemaError):
        terminal_disposition(notice)


def test_control_propagation_bound_is_a_validated_hop_parameter() -> None:
    now = [100.0]
    hop = RelayHop(lambda _: _RecordingSink(), clock=lambda: now[0])
    stream = hop.open_stream("agent-a")

    propagation = stream.terminate("idle expired")

    assert CONTROL_PROPAGATION_TIMEOUT == 2.0
    assert hop.control_propagation_timeout == CONTROL_PROPAGATION_TIMEOUT
    assert propagation.deadline == 100.0 + CONTROL_PROPAGATION_TIMEOUT
    assert propagation.stream_id == "agent-a"
    now[0] = 101.5
    assert propagation.remaining(now[0]) == 0.5
    assert not propagation.expired()
    now[0] = 102.0
    assert propagation.expired()
    for invalid in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="control_propagation_timeout"):
            RelayHop(lambda _: _RecordingSink(), control_propagation_timeout=invalid)


def test_terminal_notice_capacity_failure_leaves_the_stream_live() -> None:
    hop = RelayHop(lambda _: _RecordingSink(), control_capacity=1)
    stream = hop.open_stream("agent-a")

    with pytest.raises(ControlCapacityExhausted):
        stream.terminate("heartbeat enqueue failed")

    assert stream.terminal_error is None
    assert stream.queued_frames == 0
    assert hop.control_bytes == 0


@pytest.mark.asyncio
async def test_terminal_propagation_is_confirmed_only_by_the_completed_write() -> None:
    table = _SinkTable()
    hop = RelayHop(table)
    stream = hop.open_stream("agent-a")
    gate = asyncio.Event()
    table("agent-a").gate = gate
    hop.start()
    try:
        propagation = stream.terminate("heartbeat enqueue failed")

        # Admission into the stream's FIFO is not delivery.
        assert stream.queued_frames == 1
        assert propagation.confirmed is False
        await asyncio.wait_for(table("agent-a").started.wait(), timeout=2)
        assert propagation.confirmed is False

        gate.set()
        assert await asyncio.wait_for(hop.settle_terminal(propagation), timeout=2)
    finally:
        gate.set()
        await hop.close()

    assert propagation.confirmed is True
    assert [message["method"] for message in table("agent-a").messages] == [
        TERMINAL_METHOD
    ]
    assert hop.failure is None


@pytest.mark.asyncio
async def test_stream_terminal_reaches_the_peer_with_the_same_disposition() -> None:
    link = _Link()
    left = link.left.open_stream("agent-a")
    right = link.right.open_stream("agent-a")
    link.left.start()
    link.right.start()
    try:
        propagation = left.terminate("heartbeat enqueue failed")
        confirmed = await asyncio.wait_for(
            link.left.settle_terminal(propagation), timeout=2
        )
    finally:
        await link.close()

    assert confirmed is True
    assert propagation.confirmed is True
    assert isinstance(left.terminal_error, RelayStreamAborted)
    assert left.terminal_error.reason == "heartbeat enqueue failed"
    assert isinstance(right.terminal_error, RelayStreamAborted)
    assert right.terminal_error.reason == "heartbeat enqueue failed"
    assert link.left.failure is None
    assert link.right.failure is None
    assert len(link.left_sinks["agent-a"].delivered) == 1
    assert link.left_sinks["agent-a"].accepted == [None]


@pytest.mark.asyncio
async def test_terminal_propagation_leaves_the_peer_other_streams_working() -> None:
    link = _Link()
    left_terminal = link.left.open_stream("agent-a")
    left_survivor = link.left.open_stream("agent-b")
    right_terminal = link.right.open_stream("agent-a")
    right_survivor = link.right.open_stream("agent-b")
    link.left.start()
    link.right.start()
    try:
        propagation = left_terminal.terminate("heartbeat enqueue failed")
        assert await asyncio.wait_for(link.left.settle_terminal(propagation), timeout=2)
        survivor_frame = _frame("session/updated", "b-1")
        left_survivor.enqueue(survivor_frame)
        await _wait_until(lambda: len(link.left_sinks["agent-b"].accepted) == 1)
        assert right_survivor.terminal_error is None
    finally:
        await link.close()

    assert isinstance(right_terminal.terminal_error, RelayStreamAborted)
    assert link.left_sinks["agent-b"].accepted == [survivor_frame]
    assert link.right.failure is None
    assert link.left.failure is None
    with pytest.raises(RelayStreamAborted):
        left_terminal.enqueue(_frame("session/updated", "a-2"))
    assert link.left_sinks["agent-a"].delivered == [
        _notice(TerminalDisposition.STREAM_ABORTED, "heartbeat enqueue failed")
    ]


@pytest.mark.asyncio
async def test_unconfirmed_terminal_propagation_fails_every_stream() -> None:
    now = [0.0]
    table = _SinkTable()
    aborts: list[str] = []

    async def abort(reason: str) -> None:
        aborts.append(reason)

    hop = RelayHop(table, clock=lambda: now[0], connection_abort=abort)
    first = hop.open_stream("agent-a")
    second = hop.open_stream("agent-b")
    gate = asyncio.Event()
    table("agent-a").gate = gate
    hop.start()
    try:
        first.enqueue(_frame("session/updated", "a-1"))
        await asyncio.wait_for(table("agent-a").started.wait(), timeout=2)
        propagation = first.terminate("heartbeat enqueue failed")

        now[0] = CONTROL_PROPAGATION_TIMEOUT
        confirmed = await asyncio.wait_for(hop.settle_terminal(propagation), timeout=2)
    finally:
        gate.set()
        await hop.close(drain=False)

    assert confirmed is False
    assert propagation.confirmed is False
    assert propagation.settled is True
    failure = hop.failure
    assert isinstance(failure, RelayConnectionFailed)
    assert propagation.error is failure
    assert first.terminal_error is failure
    assert second.terminal_error is failure
    assert len(aborts) == 1
    assert "was not confirmed within" in aborts[0]
    assert "'agent-a'" in aborts[0]
    assert hop.closed is True


@pytest.mark.asyncio
async def test_shared_connection_failure_is_not_a_stream_abort() -> None:
    table = _SinkTable()
    hop = RelayHop(table)
    first = hop.open_stream("agent-a")
    second = hop.open_stream("agent-b")
    table("agent-a").failure = OSError("shared pipe closed")
    hop.start()
    try:
        first.enqueue(_frame("session/updated", "a-1"))
        second.enqueue(_frame("session/updated", "b-1"))
        await _wait_until(lambda: hop.failure is not None)
    finally:
        await hop.close()

    failure = hop.failure
    assert isinstance(failure, RelayConnectionFailed)
    assert "shared pipe closed" in str(failure)
    assert first.terminal_error is failure
    assert second.terminal_error is failure
    assert not isinstance(first.terminal_error, RelayStreamAborted)
    assert not isinstance(second.terminal_error, RelayStreamAborted)


@pytest.mark.asyncio
async def test_unconfirmed_propagation_closes_the_connection_so_the_peer_sees_eof() -> (
    None
):
    read_fd, write_fd = os.pipe()
    peer_reader = os.fdopen(read_fd, "rb")
    writer = os.fdopen(write_fd, "wb")
    transport = StdioJsonRpcTransport(
        os.fdopen(os.open(os.devnull, os.O_RDONLY), "rb"), writer
    )
    now = [0.0]
    hop = RelayHop.over_transport(transport, clock=lambda: now[0])
    stream = hop.open_stream("agent-a")
    try:
        propagation = stream.terminate("heartbeat enqueue failed")
        now[0] = CONTROL_PROPAGATION_TIMEOUT
        confirmed = await asyncio.wait_for(hop.settle_terminal(propagation), timeout=2)
        eof = await asyncio.wait_for(asyncio.to_thread(peer_reader.read, 1), timeout=2)
    finally:
        await hop.close(drain=False)
        await transport.abortive_close()
        peer_reader.close()
        writer.close()

    assert confirmed is False
    assert eof == b""
    assert isinstance(hop.failure, RelayConnectionFailed)


@pytest.mark.asyncio
async def test_demux_receipt_waits_for_underlying_write_and_automatic_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Transport:
        def __init__(self) -> None:
            self.receipt: asyncio.Future[None] = (
                asyncio.get_running_loop().create_future()
            )
            self.admitted = asyncio.Event()
            self.aborted = asyncio.Event()

        async def send(self, message: dict[str, Any]) -> None:
            pass

        async def send_confirmed(self, message: dict[str, Any]) -> asyncio.Future[None]:
            self.admitted.set()
            return self.receipt

        async def abortive_close(self) -> None:
            self.aborted.set()
            self.receipt.cancel()

    now = [asyncio.get_running_loop().time()]
    monkeypatch.setattr(asyncio.get_running_loop(), "time", lambda: now[0])
    transport = Transport()
    demux = StreamDemultiplexer(transport)
    hop = RelayHop.over_demux(demux, clock=lambda: now[0])
    first = hop.open_stream("first")
    second = hop.open_stream("second")
    hop.start()
    propagation = first.terminate("expired")
    await transport.admitted.wait()
    assert not propagation.confirmed
    now[0] += CONTROL_PROPAGATION_TIMEOUT
    await transport.aborted.wait()
    assert isinstance(first.terminal_error, RelayConnectionFailed)
    assert second.terminal_error is first.terminal_error
    assert not propagation.confirmed
    await hop.close(drain=False)
    await demux.close()


@pytest.mark.asyncio
async def test_scheduled_timeout_surfaces_missing_abortive_close() -> None:
    class Sink:
        async def send(self, message: dict[str, Any]) -> None:
            pass

        async def close(self) -> None:
            pass

        async def messages(self):
            if False:
                yield {}

    now = [0.0]
    hop = RelayHop.over_transport(Sink(), clock=lambda: now[0])
    first = hop.open_stream("first")
    second = hop.open_stream("second")
    propagation = first.terminate("expired")
    now[0] = CONTROL_PROPAGATION_TIMEOUT
    try:
        with pytest.raises(RelayConnectionFailed, match="abortive close"):
            await hop.settle_terminal(propagation)
        assert isinstance(hop.failure, RelayConnectionFailed)
        assert first.terminal_error is hop.failure
        assert second.terminal_error is hop.failure
        assert propagation.error is hop.failure
    finally:
        await hop.close(drain=False)


@pytest.mark.asyncio
async def test_abort_hook_rejects_transport_without_abortive_close() -> None:
    from chartreux.app_server.relay import transport_abort

    class Sink:
        async def close(self) -> None:
            raise AssertionError("draining close must not be used")

    with pytest.raises(RelayConnectionFailed, match="abortive close"):
        await transport_abort(Sink())("deadline")


def test_terminal_notice_for_a_closed_stream_is_stale_not_a_failure() -> None:
    hop = RelayHop(lambda _: _RecordingSink())
    stream = hop.open_stream("agent-a")
    stream.discard("heartbeat enqueue failed")
    disposition = stream.terminal_error
    assert isinstance(disposition, RelayStreamAborted)

    assert (
        stream.accept_inbound(_notice(TerminalDisposition.STREAM_ABORTED, "later"))
        is None
    )

    assert stream.terminal_error is disposition
    assert hop.failure is None


def test_teardown_raced_by_retirement_is_rejected_as_stale() -> None:
    hop = RelayHop(lambda _: _RecordingSink())
    stream = hop.open_stream("agent-a", incarnation=True)
    retired = _notice(TerminalDisposition.STREAM_ABORTED, "first incarnation")
    retired["relay_incarnation"] = 0
    stream.retire_incarnation()

    assert stream.accept_inbound(retired) is None
    assert stream.terminal_error is None
    assert stream.invalidated_frames == 1

    current = _notice(TerminalDisposition.STREAM_ABORTED, "second incarnation")
    current["relay_incarnation"] = 1
    assert stream.accept_inbound(current) is None

    assert isinstance(stream.terminal_error, RelayStreamAborted)
    assert stream.terminal_error.reason == "second incarnation"
    assert hop.failure is None
