from __future__ import annotations

import asyncio
from collections.abc import Callable
from io import BytesIO
import json
from typing import Any

import pytest

from chartreux.app_server.demux import (
    DemuxClosedError,
    DemuxConnectionError,
    StreamAbortedError,
    StreamDemultiplexer,
)
from chartreux.app_server.transport import StdioJsonRpcTransport


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


async def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition was not met before the deadline")
        await asyncio.sleep(0.001)


@pytest.mark.asyncio
async def test_abort_discards_queued_frames_and_preserves_other_streams() -> None:
    sink = _RecordingSink()
    demux = StreamDemultiplexer(sink)
    aborted = demux.open_stream("agent-a")
    survivor = demux.open_stream("agent-b")

    aborted.send({"stream": "agent-a", "id": 1})
    aborted.send({"stream": "agent-a", "id": 2})
    survivor.send({"stream": "agent-b", "id": 1})
    discarded = aborted.abort("heartbeat enqueue failed")
    survivor.send({"stream": "agent-b", "id": 2})

    await _wait_until(lambda: len(sink.messages) == 2)

    assert discarded == 2
    assert aborted.discarded_frames == 2
    assert isinstance(aborted.terminal_error, StreamAbortedError)
    assert aborted.terminal_error.reason == "heartbeat enqueue failed"
    assert survivor.terminal_error is None
    assert demux.closed is False
    assert sink.messages == [
        {"stream": "agent-b", "id": 1},
        {"stream": "agent-b", "id": 2},
    ]
    with pytest.raises(StreamAbortedError, match="heartbeat enqueue failed"):
        aborted.send({"stream": "agent-a", "id": 3})


@pytest.mark.asyncio
async def test_abort_lets_the_in_flight_frame_finish_but_drops_queued_frames() -> None:
    gate = asyncio.Event()
    sink = _RecordingSink()
    sink.gate = gate
    demux = StreamDemultiplexer(sink)
    stream = demux.open_stream("agent-a")

    stream.send({"id": 1})
    await asyncio.wait_for(sink.started.wait(), timeout=2)
    stream.send({"id": 2})
    stream.send({"id": 3})
    discarded = stream.abort("relay enqueue failed")

    gate.set()
    await _wait_until(lambda: sink.messages == [{"id": 1}])
    await asyncio.sleep(0.01)

    assert discarded == 2
    assert sink.messages == [{"id": 1}]


@pytest.mark.asyncio
async def test_shared_connection_failure_terminates_every_stream() -> None:
    sink = _RecordingSink()
    sink.failure = OSError("shared pipe closed")
    demux = StreamDemultiplexer(sink)
    first = demux.open_stream("agent-a")
    second = demux.open_stream("agent-b")

    first.send({"id": 1})
    second.send({"id": 2})
    first.send({"id": 3})

    await asyncio.wait_for(
        asyncio.gather(first.wait_closed(), second.wait_closed()), timeout=2
    )

    failure = demux.connection_error
    assert isinstance(failure, DemuxConnectionError)
    assert "shared pipe closed" in str(failure)
    assert first.terminal_error is failure
    assert second.terminal_error is failure
    assert sink.messages == []
    with pytest.raises(DemuxConnectionError, match="shared pipe closed"):
        first.send({"id": 4})
    with pytest.raises(DemuxConnectionError, match="shared pipe closed"):
        demux.open_stream("agent-c")


@pytest.mark.asyncio
async def test_graceful_demux_close_drains_frames_instead_of_discarding() -> None:
    sink = _RecordingSink()
    demux = StreamDemultiplexer(sink)
    stream = demux.open_stream("agent-a")

    stream.send({"id": 1})
    stream.send({"id": 2})
    await demux.close()
    await demux.close()

    assert sink.messages == [{"id": 1}, {"id": 2}]
    assert stream.discarded_frames == 0
    assert isinstance(stream.terminal_error, DemuxClosedError)
    assert demux.closed is True
    with pytest.raises(DemuxClosedError):
        stream.send({"id": 3})
    with pytest.raises(DemuxClosedError):
        demux.open_stream("agent-b")


@pytest.mark.asyncio
async def test_demux_abort_over_stdio_transport_leaves_the_connection_usable() -> None:
    output = BytesIO()
    transport = StdioJsonRpcTransport(BytesIO(), output)
    demux = StreamDemultiplexer(transport)
    aborted = demux.open_stream("agent-a")
    survivor = demux.open_stream("agent-b")

    aborted.send({"jsonrpc": "2.0", "method": "a/one"})
    survivor.send({"jsonrpc": "2.0", "method": "b/one"})
    aborted.abort("heartbeat enqueue failed")
    survivor.send({"jsonrpc": "2.0", "method": "b/two"})

    await _wait_until(lambda: len(output.getvalue().splitlines()) == 2)
    await transport.close()

    lines = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [line["method"] for line in lines] == ["b/one", "b/two"]


@pytest.mark.asyncio
async def test_stream_outbox_exhaustion_closes_the_stream_abortively() -> None:
    gate = asyncio.Event()
    sink = _RecordingSink()
    sink.gate = gate
    demux = StreamDemultiplexer(sink, stream_queue_size=1)
    stream = demux.open_stream("agent-a")

    stream.send({"id": 1})
    await asyncio.wait_for(sink.started.wait(), timeout=2)
    stream.send({"id": 2})

    with pytest.raises(StreamAbortedError, match="outbox exhausted"):
        stream.send({"id": 3})

    assert stream.discarded_frames == 1
    assert isinstance(stream.terminal_error, StreamAbortedError)
    gate.set()
    await _wait_until(lambda: sink.messages == [{"id": 1}])


@pytest.mark.asyncio
async def test_aborted_stream_can_be_replaced_by_a_fresh_registration() -> None:
    sink = _RecordingSink()
    demux = StreamDemultiplexer(sink)
    first = demux.open_stream("agent-a")
    first.abort("relay enqueue failed")

    replacement = demux.open_stream("agent-a")
    replacement.send({"id": 1})

    await _wait_until(lambda: sink.messages == [{"id": 1}])
    assert replacement is not first
    assert replacement.terminal_error is None


def test_demux_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="stream_queue_size must be positive"):
        StreamDemultiplexer(_RecordingSink(), stream_queue_size=0)

    demux = StreamDemultiplexer(_RecordingSink())
    with pytest.raises(ValueError, match="stream_id must be a non-empty string"):
        demux.open_stream("")

    demux.open_stream("agent-a")
    with pytest.raises(ValueError, match="already open"):
        demux.open_stream("agent-a")
