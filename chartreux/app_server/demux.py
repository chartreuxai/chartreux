"""Per-stream demultiplexing with abortive, stream-scoped discard.

The shared transport's ``close()`` is a graceful drain: frames already queued
are written and only then does the connection close. A logical stream failing
on a multiplexed connection must not be handled that way, because the
connection also carries unrelated streams. Aborting a stream here discards the
frames still queued for it, stops it from emitting anything further, and leaves
the shared connection and every other stream working.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol


class FrameSink(Protocol):
    """The shared connection writer; ``StdioJsonRpcTransport`` satisfies it."""

    async def send(self, message: dict[str, Any]) -> None: ...


class StreamTerminatedError(RuntimeError):
    """Base class for a terminal logical-stream disposition."""

    def __init__(self, stream_id: str, reason: str) -> None:
        super().__init__(f"stream {stream_id!r} is terminally closed: {reason}")
        self.stream_id = stream_id
        self.reason = reason


class StreamAbortedError(StreamTerminatedError):
    """The stream was closed abortively and its queued frames were discarded."""


class DemuxClosedError(StreamTerminatedError):
    """The demultiplexer was closed after draining the stream's frames."""


class DemuxConnectionError(RuntimeError):
    """The shared connection failed, which is terminal for every stream."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"shared demultiplexer connection failed: {reason}")
        self.reason = reason


_STOP = object()


class DemuxStream:
    """One logical stream multiplexed over a shared connection."""

    def __init__(
        self, demultiplexer: StreamDemultiplexer, stream_id: str, *, queue_size: int
    ) -> None:
        self._demultiplexer = demultiplexer
        self._stream_id = stream_id
        self._queue_size = queue_size
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=queue_size)
        self._terminal: RuntimeError | None = None
        self._accepting = True
        self._discarded = 0
        self._writer_task: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def terminal_error(self) -> RuntimeError | None:
        """The disposition that ended this stream, or ``None`` while it lives."""
        return self._terminal

    @property
    def discarded_frames(self) -> int:
        """Frames dropped by abortive close; frames already sent are not counted."""
        return self._discarded

    def send(self, message: dict[str, Any]) -> None:
        """Queue one frame for the shared writer without blocking on it."""
        if not self._accepting:
            raise self._reject_error()
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull:
            self.abort(f"stream outbox exhausted at {self._queue_size} frames")
            raise self._reject_error() from None
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(self._write_queued())

    def send_confirmed(self, message: dict[str, Any]) -> asyncio.Future[None]:
        if not self._accepting:
            raise self._reject_error()
        receipt: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait((message, receipt))
        except asyncio.QueueFull:
            self.abort(f"stream outbox exhausted at {self._queue_size} frames")
            raise self._reject_error() from None
        if self._writer_task is None:
            self._writer_task = asyncio.create_task(self._write_queued())
        return receipt

    def abort(self, reason: str) -> int:
        """Abortively close the stream and return the discarded frame count.

        A frame already handed to the shared writer finishes writing; every
        frame still queued for this stream is dropped and nothing further is
        written on it. The shared connection is left open for other streams.
        """
        if self._terminal is not None:
            return 0
        return self._terminate(StreamAbortedError(self._stream_id, reason))

    async def wait_closed(self) -> None:
        await self._closed.wait()

    async def _write_queued(self) -> None:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            receipt: asyncio.Future[None] | None = None
            if isinstance(item, tuple):
                item, receipt = item
            try:
                await self._demultiplexer._write_frame(
                    item, confirmed=receipt is not None
                )
            except asyncio.CancelledError:
                if receipt is not None and not receipt.done():
                    receipt.cancel()
                raise
            except DemuxConnectionError as exc:
                if receipt is not None and not receipt.done():
                    receipt.set_exception(exc)
                return
            else:
                if receipt is not None and not receipt.done():
                    receipt.set_result(None)

    def _terminate(self, error: RuntimeError) -> int:
        self._terminal = error
        self._accepting = False
        discarded = self._discard_queued()
        if self._writer_task is not None:
            self._queue.put_nowait(_STOP)
        self._closed.set()
        return discarded

    def _discard_queued(self) -> int:
        discarded = 0
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if item is not _STOP:
                discarded += 1
                if isinstance(item, tuple):
                    receipt = item[1]
                    if not receipt.done():
                        receipt.cancel()
        self._discarded += discarded
        return discarded

    def _connection_failed(self, error: DemuxConnectionError) -> None:
        if self._terminal is None:
            self._terminate(error)

    async def _close_gracefully(self) -> None:
        if not self._accepting:
            await self._closed.wait()
            return
        self._accepting = False
        if self._writer_task is None:
            self._terminal = DemuxClosedError(self._stream_id, "demultiplexer closed")
            self._closed.set()
            return
        try:
            self._queue.put_nowait(_STOP)
        except asyncio.QueueFull:
            await self._queue.put(_STOP)
        try:
            await self._writer_task
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        if self._terminal is None:
            self._terminal = DemuxClosedError(self._stream_id, "demultiplexer closed")
        self._closed.set()

    def _reject_error(self) -> RuntimeError:
        if self._terminal is not None:
            return self._terminal
        return DemuxClosedError(self._stream_id, "demultiplexer closed")


class StreamDemultiplexer:
    """Per-stream bounded FIFOs over one shared connection writer.

    ``DemuxStream.abort()`` is demultiplexer-scoped: the aborted stream's
    queued frames are discarded while the shared connection and the other
    streams continue. ``close()`` is the graceful counterpart: it drains every
    live stream in FIFO order and leaves the connection itself open.
    """

    def __init__(self, transport: FrameSink, *, stream_queue_size: int = 64) -> None:
        if stream_queue_size < 1:
            raise ValueError("stream_queue_size must be positive")
        self._transport = transport
        self._stream_queue_size = stream_queue_size
        self._streams: dict[str, DemuxStream] = {}
        self._failure: DemuxConnectionError | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def connection_error(self) -> DemuxConnectionError | None:
        return self._failure

    @property
    def streams(self) -> tuple[DemuxStream, ...]:
        return tuple(self._streams.values())

    def open_stream(self, stream_id: str) -> DemuxStream:
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if self._closed:
            raise DemuxClosedError(stream_id, "demultiplexer is closed")
        if self._failure is not None:
            raise self._failure
        current = self._streams.get(stream_id)
        if current is not None and current.terminal_error is None:
            raise ValueError(f"stream {stream_id!r} is already open")
        stream = DemuxStream(self, stream_id, queue_size=self._stream_queue_size)
        self._streams[stream_id] = stream
        return stream

    def stream(self, stream_id: str) -> DemuxStream | None:
        return self._streams.get(stream_id)

    async def close(self) -> None:
        """Drain every live stream in order, then mark the demux closed."""
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        await asyncio.shield(self._close_task)

    async def _finish_close(self) -> None:
        await asyncio.gather(
            *(stream._close_gracefully() for stream in tuple(self._streams.values()))
        )

    async def abortive_close(self) -> None:
        self._fail_connection(RuntimeError("shared connection aborted"))
        abortive = getattr(self._transport, "abortive_close", None)
        if abortive is None:
            raise DemuxConnectionError("transport does not support abortive close")
        await abortive()

    async def _write_frame(
        self, message: dict[str, Any], *, confirmed: bool = False
    ) -> None:
        failure = self._failure
        if failure is not None:
            raise failure
        try:
            if confirmed:
                send_confirmed = getattr(self._transport, "send_confirmed", None)
                if send_confirmed is None:
                    raise RuntimeError("transport does not support confirmed writes")
                receipt = await send_confirmed(message)
                await receipt
            else:
                await self._transport.send(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise self._fail_connection(exc) from exc

    def _fail_connection(self, cause: BaseException) -> DemuxConnectionError:
        if self._failure is not None:
            return self._failure
        reason = str(cause) or type(cause).__name__
        self._failure = DemuxConnectionError(reason)
        for stream in tuple(self._streams.values()):
            stream._connection_failed(self._failure)
        return self._failure
