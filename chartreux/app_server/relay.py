"""Relay foundation: closed frame schema, byte accounting, fair stream FIFOs.

A relay hop admits frames from many logical producer streams onto one shared
connection. Three properties are load-bearing here:

* The wire schema is CLOSED. Only explicitly declared presentation deltas are
  droppable; every other frame is control-class, and a notification that is not
  part of the declared registry fails typed instead of being guessed at.
* Retention is byte-bounded. Presentation frames draw on a per-stream budget
  and on a per-hop aggregate, while control frames draw on a reserved capacity
  that presentation can never consume — a saturated presentation stream cannot
  delay an approval prompt or a snapshot page.
* Ordering is FIFO within a stream and fair across streams: one stream's
  backlog cannot starve another's, and no frame overtakes another on its own
  stream.

Per-stream write isolation (abortive stream close versus a failed shared
connection) belongs to :mod:`chartreux.app_server.demux`;
:meth:`RelayHop.over_demux` composes the two rather than reimplementing it.

Terminal propagation is delivery-confirmed. A stream-terminal close sends a
control-class notice to the peer as that stream's last frame, and the hop
reports the propagation confirmed only once the scheduler's write of the frame
completed — queue admission is not delivery. If the notice is not confirmed
within the per-hop bound, the hop aborts the shared connection instead, so EOF
carries the failure to the peer within the bound; every stream on that
connection then ends as a connection failure, not as an isolated stream abort.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum, auto
from functools import cache
import json
import math
import time
from types import MappingProxyType
from typing import Any, Protocol

from chartreux.app_server.demux import (
    DemuxConnectionError,
    DemuxStream,
    StreamDemultiplexer,
    StreamTerminatedError,
)
from chartreux.app_server.protocol import server_notification_registry
from chartreux.app_server.transport import JsonRpcTransport

# Numeric bounds from the v0.7.7 plan's table. They are validation candidates,
# not constants: a hop may be built with tighter limits for one deployment.
PRESENTATION_BYTES_PER_STREAM = 4 * 1024 * 1024
PRESENTATION_BYTES_PER_HOP = 16 * 1024 * 1024
CONTROL_BYTES_PER_HOP = 1024 * 1024

# Plan bound: control (including terminal) propagation per hop. A notice the
# scheduler has not written within it forces the connection fallback.
CONTROL_PROPAGATION_TIMEOUT = 2.0

# The relay's own control methods. Like ``relay/retire`` they are outside the
# protocol's notification registry. Snapshot pages are control-class too: an
# initial snapshot is never droppable, and the consumer installs it atomically.
RETIRE_METHOD = "relay/retire"
TERMINAL_METHOD = "relay/terminal"
SNAPSHOT_PAGE_METHOD = "relay/snapshotPage"
RELAY_CONTROL_METHODS = frozenset({
    RETIRE_METHOD,
    TERMINAL_METHOD,
    SNAPSHOT_PAGE_METHOD,
})

# The only notification methods the schema makes droppable: patch-style
# presentation deltas a consumer re-derives by re-registering. Terminal turn
# state, snapshot pages, approvals, settlements, heartbeats, and revalidation
# are control-class and are never shed.
PRESENTATION_DELTA_METHODS = frozenset({
    "agents/update",
    "history/entryAdded",
    "history/entryUpdated",
    "session/statsUpdated",
    "session/updated",
    "turn/started",
    "turn/updated",
    "turn_queue_updated",
})


class FrameClass(StrEnum):
    """How a frame is retained when a queue is under pressure."""

    PRESENTATION = auto()
    CONTROL = auto()


class FrameDisposition(StrEnum):
    """What a hop did with one offered frame."""

    QUEUED = auto()
    SHED = auto()


class TerminalDisposition(StrEnum):
    """Why a stream ended, as carried to the peer hop."""

    STREAM_ABORTED = auto()
    CONNECTION_FAILED = auto()


class RelayError(RuntimeError):
    """Base class for relay failures."""


class FrameSchemaError(RelayError):
    """A frame is not part of the closed relay wire schema."""


class ControlCapacityExhausted(RelayError):
    """The hop's reserved control capacity cannot take one more frame."""


class RelayClosed(RelayError):
    """The hop no longer accepts frames."""


class RelayStreamTerminated(RelayError):
    """A logical stream reached a terminal disposition."""

    def __init__(self, stream_id: str, reason: str) -> None:
        super().__init__(f"relay stream {stream_id!r} is terminally closed: {reason}")
        self.stream_id = stream_id
        self.reason = reason


class RelayStreamAborted(RelayStreamTerminated):
    """The stream was closed abortively and its queued frames were discarded."""


class RelayConnectionFailed(RelayError):
    """The shared sink failed, which is terminal for every stream on the hop."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"relay connection failed: {reason}")
        self.reason = reason


@dataclass(frozen=True, slots=True)
class EnqueueOutcome:
    """The disposition of one offered frame, with the reason for a shed."""

    frame_class: FrameClass
    disposition: FrameDisposition
    shed_reason: str | None = None


@dataclass(frozen=True, slots=True)
class QueuedFrame:
    """One frame retained in a stream's FIFO, with its accounted size."""

    message: dict[str, Any]
    size: int
    frame_class: FrameClass
    delivery: DeliveryConfirmation | None = None


class RelaySink(Protocol):
    """Where one stream's frames go once the scheduler admits them."""

    async def send(self, message: dict[str, Any]) -> None: ...

    def abort(self, reason: str) -> int: ...


ConnectionAbort = Callable[[str], Awaitable[None]]


class DeliveryConfirmation:
    """Whether one frame's write actually completed.

    Relay queue admission is not delivery: a frame the scheduler holds may
    still fail, or never be written at all. Anything that must genuinely reach
    the peer — terminal teardown above all — waits on this handle, which the
    scheduler settles only after its sink write returns.
    """

    def __init__(self, *, deadline: float, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._deadline = deadline
        self._confirmed = False
        self._error: RelayError | None = None
        self._settled = asyncio.Event()

    @property
    def deadline(self) -> float:
        """The monotonic time by which confirmation was expected."""
        return self._deadline

    @property
    def confirmed(self) -> bool:
        """Whether the frame's write completed."""
        return self._confirmed

    @property
    def error(self) -> RelayError | None:
        """Why the delivery failed, or ``None``."""
        return self._error

    @property
    def settled(self) -> bool:
        """Whether confirmation or failure has already been recorded."""
        return self._settled.is_set()

    def remaining(self, now: float) -> float:
        """Seconds left before the delivery deadline, never negative."""
        return max(self._deadline - now, 0.0)

    def expired(self) -> bool:
        """Whether the delivery deadline has passed."""
        return self._clock() >= self._deadline

    def confirm(self) -> bool:
        """Record a completed write; the first outcome wins."""
        if self._settled.is_set():
            return False
        self._confirmed = True
        self._settled.set()
        return True

    def fail(self, error: RelayError) -> bool:
        """Record that the frame will not be delivered; first outcome wins."""
        if self._settled.is_set():
            return False
        self._error = error
        self._settled.set()
        return True

    async def wait(self) -> bool:
        """Wait for the outcome; report whether the write completed."""
        await self._settled.wait()
        return self._confirmed


class TerminalPropagation(DeliveryConfirmation):
    """A stream's terminal notice, awaiting its peer-facing write."""

    def __init__(
        self,
        stream_id: str,
        reason: str,
        *,
        deadline: float,
        clock: Callable[[], float],
    ) -> None:
        super().__init__(deadline=deadline, clock=clock)
        self._stream_id = stream_id
        self._reason = reason

    @property
    def stream_id(self) -> str:
        """The stream whose disposition this notice carries."""
        return self._stream_id

    @property
    def reason(self) -> str:
        """The disposition reason as sent on the wire."""
        return self._reason


@cache
def frame_classification() -> Mapping[str, FrameClass]:
    """Return the closed schema: every declared notification and its class.

    The schema is fixed by the protocol's notification registry, which is
    closed at import time, so it is built once and needs no invalidation. A
    declared droppable delta that no longer names a registered notification is
    a schema error rather than a silently dangling entry.
    """
    registered = tuple(server_notification_registry())
    undeclared = PRESENTATION_DELTA_METHODS - set(registered)
    if undeclared:
        raise RelayError(
            "presentation deltas are not declared notifications: "
            + ", ".join(sorted(undeclared))
        )
    return MappingProxyType({
        method: (
            FrameClass.PRESENTATION
            if method in PRESENTATION_DELTA_METHODS
            else FrameClass.CONTROL
        )
        for method in registered
    })


def classify_frame(message: Mapping[str, Any]) -> FrameClass:
    """Classify one wire frame, or fail typed when it is outside the schema.

    Requests and responses are always control-class: they carry approvals,
    settlements, revalidation, and snapshot acknowledgements, so they are never
    droppable. Notification methods are closed against the declared registry.
    """
    if not isinstance(message, Mapping):
        raise FrameSchemaError("relay frames must be JSON-RPC objects")
    if message.get("jsonrpc") != "2.0":
        raise FrameSchemaError("relay frames must be JSON-RPC 2.0 envelopes")
    if message.get("method") in RELAY_CONTROL_METHODS and "id" not in message:
        return FrameClass.CONTROL
    method = message.get("method")
    if isinstance(method, str):
        if "id" in message:
            return FrameClass.CONTROL
        frame_class = frame_classification().get(method)
        if frame_class is None:
            raise FrameSchemaError(f"undeclared notification method {method!r}")
        return frame_class
    if "id" in message and ("result" in message or "error" in message):
        return FrameClass.CONTROL
    raise FrameSchemaError(
        "relay frames must be a declared request, response, or notification"
    )


def terminal_disposition(message: Mapping[str, Any]) -> TerminalDisposition | None:
    """Read a terminal notice's disposition, or ``None`` for another frame.

    A notice that claims the terminal method but does not carry a disposition
    and a reason is outside the closed wire schema and fails typed.
    """
    if message.get("method") != TERMINAL_METHOD:
        return None
    params = message.get("params")
    if not isinstance(params, Mapping):
        raise FrameSchemaError("terminal notices must carry params")
    disposition = params.get("disposition")
    reason = params.get("reason")
    if not isinstance(reason, str) or not reason:
        raise FrameSchemaError("terminal notices must carry a reason")
    if not isinstance(disposition, str):
        raise FrameSchemaError("terminal notices must carry a disposition")
    try:
        return TerminalDisposition(disposition)
    except ValueError as exc:
        raise FrameSchemaError(f"unknown terminal disposition {disposition!r}") from exc


def encoded_frame_size(message: Mapping[str, Any]) -> int:
    """Return one frame's wire size: its compact JSON encoding in bytes."""
    try:
        encoded = json.dumps(message, separators=(",", ":")).encode()
    except (TypeError, ValueError) as exc:
        raise FrameSchemaError("relay frames must be JSON-encodable") from exc
    return len(encoded)


class ByteBudget:
    """Monotonic byte accounting against one retention bound."""

    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError("byte budget must be positive")
        self._limit = limit
        self._reserved = 0

    @property
    def limit(self) -> int:
        """The bound this budget enforces, in bytes."""
        return self._limit

    @property
    def reserved(self) -> int:
        """Bytes currently held against the bound."""
        return self._reserved

    @property
    def available(self) -> int:
        """Bytes that can still be reserved."""
        return self._limit - self._reserved

    def try_reserve(self, size: int) -> bool:
        """Reserve ``size`` bytes, or report that the bound would be exceeded."""
        if size < 0:
            raise ValueError("cannot reserve a negative size")
        if self._reserved + size > self._limit:
            return False
        self._reserved += size
        return True

    def release(self, size: int) -> None:
        """Release bytes that were previously reserved."""
        if size < 0 or size > self._reserved:
            raise ValueError("cannot release bytes that were not reserved")
        self._reserved -= size


class IncarnationState:
    """Track announced incarnations and their current idle deadline."""

    def __init__(
        self,
        incarnation: str,
        *,
        heartbeat_interval: float = 2.0,
        idle_timeout: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not incarnation:
            raise ValueError("incarnation must be non-empty")
        if not all(
            math.isfinite(value) and value > 0
            for value in (heartbeat_interval, idle_timeout)
        ):
            raise ValueError("heartbeat and idle intervals must be finite and positive")
        self.heartbeat_interval = heartbeat_interval
        self.idle_timeout = idle_timeout
        self._clock = clock
        self.current: str | None = incarnation
        self._retired: set[str] = set()
        self.deadline: float | None = clock() + idle_timeout

    def retire(self, incarnation: str) -> bool:
        if incarnation != self.current:
            return False
        self._retired.add(incarnation)
        self.current = None
        self.deadline = None
        return True

    def replace(self, incarnation: str, replaces: str) -> bool:
        if not incarnation or incarnation == replaces or incarnation in self._retired:
            raise FrameSchemaError("replacement incarnation must be fresh")
        if self.current == incarnation:
            return False
        if self.current == replaces:
            self.retire(replaces)
        elif self.current is not None or replaces not in self._retired:
            raise FrameSchemaError("replacement does not name the current incarnation")
        self.current = incarnation
        self.deadline = self._clock() + self.idle_timeout
        return True

    def receive(self, incarnation: str) -> bool:
        if incarnation in self._retired:
            return False
        if incarnation != self.current:
            raise FrameSchemaError("frame has an unannounced incarnation")
        self.deadline = self._clock() + self.idle_timeout
        return True

    def expired(self) -> bool:
        return self.deadline is not None and self._clock() >= self.deadline


class RelayStream:
    """One logical stream's byte-accounted FIFO on a relay hop."""

    def __init__(
        self, hop: RelayHop, stream_id: str, sink: RelaySink, *, incarnation: bool
    ) -> None:
        self._hop = hop
        self._incarnation_enabled = incarnation
        self._generation = 0
        self._deadline = hop._clock() + hop.idle_timeout
        self._invalidated = 0
        self._stream_id = stream_id
        self._sink = sink
        self._queued: deque[QueuedFrame] = deque()
        self._presentation_bytes = 0
        self._control_bytes = 0
        self._shed = 0
        self._terminal: RelayError | None = None

    @property
    def stream_id(self) -> str:
        """This stream's identity on the hop."""
        return self._stream_id

    @property
    def terminal_error(self) -> RelayError | None:
        """The disposition that ended this stream, or ``None`` while it lives."""
        return self._terminal

    @property
    def queued_frames(self) -> int:
        """Frames retained for this stream, not yet handed to the sink."""
        return len(self._queued)

    @property
    def queued_bytes(self) -> int:
        """Bytes retained for this stream across both classes."""
        return self._presentation_bytes + self._control_bytes

    @property
    def presentation_bytes(self) -> int:
        """Droppable bytes retained for this stream."""
        return self._presentation_bytes

    @property
    def control_bytes(self) -> int:
        """Control bytes retained for this stream."""
        return self._control_bytes

    @property
    def shed_frames(self) -> int:
        """Presentation frames this stream had to shed for lack of capacity."""
        return self._shed

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def invalidated_frames(self) -> int:
        return self._invalidated

    def idle_expired(self) -> bool:
        return self._hop._clock() >= self._deadline

    def receive(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        if not self._incarnation_enabled:
            raise RelayError("incarnation tracking is not enabled")
        generation = message.get("relay_incarnation")
        if type(generation) is not int or generation < 0:
            raise FrameSchemaError("frame must carry an incarnation generation")
        if generation != self._generation:
            self._invalidated += 1
            return None
        if message.get("method") == "relay/retire":
            self._invalidate_queue()
            self._generation += 1
            self._deadline = self._hop._clock() + self._hop.idle_timeout
            return None
        self._deadline = self._hop._clock() + self._hop.idle_timeout
        return {
            key: value for key, value in message.items() if key != "relay_incarnation"
        }

    def retire_incarnation(self) -> int:
        if not self._incarnation_enabled:
            raise RelayError("incarnation tracking is not enabled")
        if self._terminal is not None:
            raise self._terminal
        if self._hop.closed:
            raise RelayClosed("relay hop is closed")
        notice = {
            "jsonrpc": "2.0",
            "method": "relay/retire",
            "relay_incarnation": self._generation,
        }
        size = encoded_frame_size(notice)
        reclaimable = sum(
            frame.size
            for frame in self._queued
            if frame.frame_class is FrameClass.CONTROL
            and frame.message.get("method") != "relay/retire"
        )
        if size > self._hop._control.available + reclaimable:
            raise ControlCapacityExhausted(
                "retirement exceeds reserved control capacity"
            )
        invalidated = self._invalidate_queue()
        reserved = self._hop._control.try_reserve(size)
        assert reserved
        self._hop._admit(self, QueuedFrame(notice, size, FrameClass.CONTROL))
        self._generation += 1
        self._deadline = self._hop._clock() + self._hop.idle_timeout
        return invalidated

    def _invalidate_queue(self) -> int:
        retained: deque[QueuedFrame] = deque()
        invalidated = 0
        while self._queued:
            frame = self._queued.popleft()
            if frame.message.get("method") in {"relay/retire", TERMINAL_METHOD}:
                retained.append(frame)
            else:
                self._abandon(
                    frame, RelayStreamAborted(self._stream_id, "incarnation retired")
                )
                invalidated += 1
        self._queued = retained
        self._invalidated += invalidated
        if not retained:
            self._hop._stream_terminated(self)
        return invalidated

    def enqueue(self, message: Mapping[str, Any]) -> EnqueueOutcome:
        """Admit one frame for this stream without blocking."""
        return self._hop.enqueue(self._stream_id, message)

    def discard(self, reason: str) -> int:
        """Abortively close the stream and return the discarded frame count.

        A composed sink is told to abort too, so a demultiplexer-backed stream
        stops writing without draining what it still holds.
        """
        if self._terminal is not None:
            return self._terminate(self._terminal)
        self._sink.abort(reason)
        return self._terminate(RelayStreamAborted(self._stream_id, reason))

    def terminate(self, reason: str) -> TerminalPropagation:
        """Close this stream abortively and hand the peer the same disposition.

        Unlike :meth:`discard`, which stops at this hop's edge, this propagates
        the teardown: the notice is the stream's last frame, and the returned
        handle reports whether the scheduler actually wrote it.
        """
        if self._terminal is not None:
            raise self._terminal
        if self._hop.closed:
            raise RelayClosed("relay hop is closed")
        notice: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": TERMINAL_METHOD,
            "params": {
                "disposition": TerminalDisposition.STREAM_ABORTED.value,
                "reason": reason,
            },
        }
        if self._incarnation_enabled:
            notice["relay_incarnation"] = self._generation
        size = encoded_frame_size(notice)
        if not self._hop._control.try_reserve(size):
            raise ControlCapacityExhausted(
                f"{self._stream_id!r} terminal notice of {size} bytes exceeds the "
                f"{self._hop._control.limit}-byte reserved control capacity"
            )
        propagation = TerminalPropagation(
            self._stream_id,
            reason,
            deadline=self._hop._clock() + self._hop.control_propagation_timeout,
            clock=self._hop._clock,
        )
        self._terminate(RelayStreamAborted(self._stream_id, reason))
        self._hop._admit(
            self, QueuedFrame(notice, size, FrameClass.CONTROL, delivery=propagation)
        )
        self._hop._watch_terminal(propagation)
        return propagation

    def accept_inbound(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Accept one inbound frame, applying staleness, then terminal teardown.

        Returns the frame for the consumer, or ``None`` when relay control
        consumed it. A frame for a terminally closed stream is stale rather
        than a protocol failure, and the same staleness rule drops a teardown
        raced by its stream's retirement.
        """
        if self._terminal is not None:
            return None
        frame = self.receive(message) if self._incarnation_enabled else dict(message)
        if frame is None:
            return None
        disposition = terminal_disposition(frame)
        if disposition is None:
            return frame
        self._apply_terminal(disposition, frame)
        return None

    def _apply_terminal(
        self, disposition: TerminalDisposition, frame: Mapping[str, Any]
    ) -> None:
        params = frame.get("params")
        reason = params.get("reason") if isinstance(params, Mapping) else None
        detail = reason if isinstance(reason, str) else "terminal notice"
        if disposition is TerminalDisposition.CONNECTION_FAILED:
            self._hop._fail_connection(f"peer connection failed: {detail}")
            return
        self._terminate(RelayStreamAborted(self._stream_id, detail))

    def _append(self, queued: QueuedFrame) -> None:
        self._queued.append(queued)
        if queued.frame_class is FrameClass.CONTROL:
            self._control_bytes += queued.size
            return
        self._presentation_bytes += queued.size

    def _popleft(self) -> QueuedFrame | None:
        if not self._queued:
            return None
        queued = self._queued.popleft()
        self._release(queued)
        return queued

    def _release(self, queued: QueuedFrame) -> None:
        if queued.frame_class is FrameClass.CONTROL:
            self._control_bytes -= queued.size
            self._hop._release_control(queued.size)
            return
        self._presentation_bytes -= queued.size
        self._hop._release_presentation(queued.size)

    def _abandon(self, queued: QueuedFrame, error: RelayError) -> None:
        """Release a frame that will never be written, failing its delivery."""
        self._release(queued)
        if queued.delivery is not None:
            queued.delivery.fail(error)

    def _note_shed(self) -> None:
        self._shed += 1

    def _terminate(self, error: RelayError) -> int:
        self._terminal = error
        discarded = 0
        while self._queued:
            self._abandon(self._queued.popleft(), error)
            discarded += 1
        self._hop._stream_terminated(self)
        return discarded


class RelayHop:
    """Byte-bounded per-stream FIFOs drained fairly onto one shared sink."""

    def __init__(
        self,
        sink_factory: Callable[[str], RelaySink],
        *,
        presentation_per_stream: int = PRESENTATION_BYTES_PER_STREAM,
        presentation_per_hop: int = PRESENTATION_BYTES_PER_HOP,
        control_capacity: int = CONTROL_BYTES_PER_HOP,
        heartbeat_interval: float = 2.0,
        idle_timeout: float = 10.0,
        control_propagation_timeout: float = CONTROL_PROPAGATION_TIMEOUT,
        connection_abort: ConnectionAbort | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not all(
            math.isfinite(value) and value > 0
            for value in (heartbeat_interval, idle_timeout)
        ):
            raise ValueError("heartbeat and idle intervals must be finite and positive")
        if not (
            math.isfinite(control_propagation_timeout)
            and control_propagation_timeout > 0
        ):
            raise ValueError("control_propagation_timeout must be finite and positive")
        self.heartbeat_interval = heartbeat_interval
        self.idle_timeout = idle_timeout
        self.control_propagation_timeout = control_propagation_timeout
        self._clock = clock
        self._connection_abort = connection_abort
        if presentation_per_stream < 1:
            raise ValueError("presentation_per_stream must be positive")
        self._sink_factory = sink_factory
        self._presentation_per_stream = presentation_per_stream
        self._presentation = ByteBudget(presentation_per_hop)
        self._control = ByteBudget(control_capacity)
        self._streams: dict[str, RelayStream] = {}
        self._ready: deque[str] = deque()
        self._failure: RelayConnectionFailed | None = None
        self._scheduler: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._wakeup = asyncio.Event()
        self._closed = False
        self._shed_frames = 0
        self._terminal_tasks: dict[TerminalPropagation, asyncio.Task[bool]] = {}

    @classmethod
    def over_transport(cls, transport: JsonRpcTransport, **limits: Any) -> RelayHop:
        """Build a hop that writes straight to one shared connection.

        Every stream shares the transport's writer, so this hop's scheduler is
        what keeps the streams fair and the transport's outbox supplies the
        backpressure.
        """
        limits.setdefault("connection_abort", transport_abort(transport))
        return cls(lambda _stream_id: _TransportSink(transport), **limits)

    @classmethod
    def over_demux(cls, demultiplexer: StreamDemultiplexer, **limits: Any) -> RelayHop:
        """Build a hop whose per-stream writes go through a demultiplexer.

        The demultiplexer supplies abortive per-stream close and
        shared-connection failure semantics; the hop supplies admission,
        byte accounting, and fair rotation.
        """
        limits.setdefault("connection_abort", transport_abort(demultiplexer))
        return cls(_DemuxSinkFactory(demultiplexer), **limits)

    @property
    def closed(self) -> bool:
        """Whether the hop stopped accepting frames."""
        return self._closed

    @property
    def failure(self) -> RelayError | None:
        """The shared failure that ended every stream, or ``None``."""
        return self._failure

    @property
    def presentation_bytes(self) -> int:
        """Presentation bytes retained across the whole hop."""
        return self._presentation.reserved

    @property
    def control_bytes(self) -> int:
        """Control bytes retained against the reserved capacity."""
        return self._control.reserved

    @property
    def presentation_capacity(self) -> int:
        """The hop's aggregate presentation bound, in bytes."""
        return self._presentation.limit

    @property
    def control_capacity(self) -> int:
        """The hop's reserved control capacity, in bytes."""
        return self._control.limit

    @property
    def shed_frames(self) -> int:
        """Presentation frames the hop shed across every stream."""
        return self._shed_frames

    @property
    def streams(self) -> tuple[RelayStream, ...]:
        """Every stream registered on the hop, live or terminated."""
        return tuple(self._streams.values())

    def open_stream(self, stream_id: str, *, incarnation: bool = False) -> RelayStream:
        """Register one logical stream and return its relay state."""
        if not isinstance(stream_id, str) or not stream_id:
            raise ValueError("stream_id must be a non-empty string")
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RelayClosed("relay hop is closed")
        current = self._streams.get(stream_id)
        if current is not None and current.terminal_error is None:
            raise ValueError(f"relay stream {stream_id!r} is already open")
        stream = RelayStream(
            self, stream_id, self._sink_factory(stream_id), incarnation=incarnation
        )
        self._streams[stream_id] = stream
        return stream

    def stream(self, stream_id: str) -> RelayStream | None:
        """Return a stream's relay state, or ``None`` when it is not open."""
        return self._streams.get(stream_id)

    def enqueue(self, stream_id: str, message: Mapping[str, Any]) -> EnqueueOutcome:
        """Admit one frame for ``stream_id`` without blocking.

        Presentation frames beyond their budget are shed and reported, so the
        caller can invalidate that stream's generation. Control frames are
        never shed: when the reserved capacity cannot take one, the enqueue
        fails typed and the caller must treat the stream as overloaded.
        """
        stream = self._streams.get(stream_id)
        if stream is None:
            raise ValueError(f"relay stream {stream_id!r} is not open")
        terminal = stream.terminal_error
        if terminal is not None:
            raise terminal
        if self._closed:
            raise RelayClosed("relay hop is closed")
        frame_class = classify_frame(message)
        wire_message = dict(message)
        if stream._incarnation_enabled:
            if (
                "relay_incarnation" in wire_message
                or wire_message.get("method") == "relay/retire"
            ):
                raise FrameSchemaError("reserved relay incarnation metadata")
            wire_message["relay_incarnation"] = stream.generation
        queued = QueuedFrame(
            wire_message, encoded_frame_size(wire_message), frame_class
        )
        if frame_class is FrameClass.PRESENTATION:
            return self._admit_presentation(stream, queued)
        if not self._control.try_reserve(queued.size):
            raise ControlCapacityExhausted(
                f"{stream_id!r} control frame of {queued.size} bytes exceeds the "
                f"{self._control.limit}-byte reserved control capacity"
            )
        self._admit(stream, queued)
        return EnqueueOutcome(FrameClass.CONTROL, FrameDisposition.QUEUED)

    def dequeue(self) -> tuple[RelayStream, QueuedFrame] | None:
        """Take the next frame in fair rotation, or ``None`` when idle."""
        while self._ready:
            stream = self._streams[self._ready.popleft()]
            queued = stream._popleft()
            if queued is None:
                continue
            if stream.queued_frames:
                self._ready.append(stream.stream_id)
            return stream, queued
        return None

    def start(self) -> None:
        """Start the fair scheduler that writes queued frames to the sink."""
        if self._scheduler is None and not self._closed:
            self._scheduler = asyncio.create_task(self._schedule())

    async def close(self, *, drain: bool = True) -> None:
        """Stop accepting frames; ``drain`` writes what is already queued.

        The first caller decides whether the close drains or aborts; later
        callers await the same shutdown.
        """
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close(drain))
        await asyncio.shield(self._close_task)

    def _watch_terminal(self, propagation: TerminalPropagation) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        task = loop.create_task(self._settle_terminal(propagation))
        task.add_done_callback(
            lambda done: None if done.cancelled() else done.exception()
        )
        self._terminal_tasks[propagation] = task

    async def settle_terminal(self, propagation: TerminalPropagation) -> bool:
        task = self._terminal_tasks.get(propagation)
        if task is None:
            self._watch_terminal(propagation)
            task = self._terminal_tasks[propagation]
        return await asyncio.shield(task)

    async def _settle_terminal(self, propagation: TerminalPropagation) -> bool:
        if not propagation.settled:
            remaining = propagation.remaining(self._clock())
            if remaining > 0:
                with suppress(TimeoutError):
                    await asyncio.wait_for(propagation.wait(), timeout=remaining)
        if propagation.confirmed:
            return True
        reason = (
            f"terminal propagation for stream {propagation.stream_id!r} was not "
            f"confirmed within {self.control_propagation_timeout}s"
        )
        failure = self._fail_connection(reason)
        propagation.fail(failure)
        abort = self._connection_abort
        if abort is not None:
            try:
                await abort(reason)
            except Exception as exc:
                raise RelayConnectionFailed(
                    f"connection abort failed: {str(exc) or type(exc).__name__}"
                ) from exc
        return False

    def _admit_presentation(
        self, stream: RelayStream, queued: QueuedFrame
    ) -> EnqueueOutcome:
        reason: str | None = None
        if stream.presentation_bytes + queued.size > self._presentation_per_stream:
            reason = "stream presentation budget exhausted"
        elif not self._presentation.try_reserve(queued.size):
            reason = "hop presentation budget exhausted"
        else:
            self._admit(stream, queued)
            return EnqueueOutcome(FrameClass.PRESENTATION, FrameDisposition.QUEUED)
        self._shed_frames += 1
        stream._note_shed()
        return EnqueueOutcome(FrameClass.PRESENTATION, FrameDisposition.SHED, reason)

    def _admit(self, stream: RelayStream, queued: QueuedFrame) -> None:
        started = stream.queued_frames == 0
        stream._append(queued)
        if started:
            self._ready.append(stream.stream_id)
        self._wakeup.set()

    def _release_presentation(self, size: int) -> None:
        self._presentation.release(size)

    def _release_control(self, size: int) -> None:
        self._control.release(size)

    def _stream_terminated(self, stream: RelayStream) -> None:
        with suppress(ValueError):
            self._ready.remove(stream.stream_id)

    def _fail_connection(self, reason: str) -> RelayConnectionFailed:
        if self._failure is not None:
            return self._failure
        self._failure = RelayConnectionFailed(reason)
        self._closed = True
        for stream in tuple(self._streams.values()):
            stream._terminate(self._failure)
        self._ready.clear()
        self._wakeup.set()
        return self._failure

    async def _finish_close(self, drain: bool) -> None:
        scheduler = self._scheduler
        if not drain:
            for stream in tuple(self._streams.values()):
                stream.discard("relay hop closed abortively")
            self._ready.clear()
            if scheduler is not None and not scheduler.done():
                scheduler.cancel()
                with suppress(asyncio.CancelledError):
                    await scheduler
            return
        if scheduler is None or scheduler.done():
            for stream in tuple(self._streams.values()):
                stream.discard("relay hop closed without a writer")
            return
        self._wakeup.set()
        await asyncio.gather(scheduler)

    async def _schedule(self) -> None:
        while True:
            item = self.dequeue()
            if item is None:
                if self._closed:
                    return
                self._wakeup.clear()
                if not self._ready:
                    await self._wakeup.wait()
                continue
            stream, queued = item
            await self._write(stream, queued)

    async def _write(self, stream: RelayStream, queued: QueuedFrame) -> None:
        try:
            if queued.delivery is None:
                await stream._sink.send(queued.message)
            else:
                confirmed = getattr(stream._sink, "send_confirmed", None)
                if confirmed is None:
                    raise RelayConnectionFailed(
                        "sink does not support confirmed writes"
                    )
                await confirmed(queued.message)
        except asyncio.CancelledError:
            raise
        except DemuxConnectionError as exc:
            self._fail_delivery(queued, self._fail_connection(exc.reason))
        except StreamTerminatedError as exc:
            error = RelayStreamAborted(stream.stream_id, exc.reason)
            self._fail_delivery(queued, error)
            stream._terminate(error)
        except Exception as exc:
            self._fail_delivery(
                queued, self._fail_connection(str(exc) or type(exc).__name__)
            )
        else:
            if queued.delivery is not None and not queued.delivery.expired():
                queued.delivery.confirm()

    @staticmethod
    def _fail_delivery(queued: QueuedFrame, error: RelayError) -> None:
        if queued.delivery is not None:
            queued.delivery.fail(error)


def transport_abort(transport: Any) -> ConnectionAbort:
    """Build an abort hook without falling back to a draining close."""

    async def abort(_reason: str) -> None:
        abortive = getattr(transport, "abortive_close", None)
        if abortive is None:
            raise RelayConnectionFailed("transport does not support abortive close")
        # A kernel-blocked write is not unblocked by close; EOF requires the
        # abortive close and termination of the child holding the pipe.
        await abortive()

    return abort


class _TransportSink:
    """A stream sink over one shared transport writer."""

    def __init__(self, transport: JsonRpcTransport) -> None:
        self._transport = transport

    async def send(self, message: dict[str, Any]) -> None:
        await self._transport.send(message)

    async def send_confirmed(self, message: dict[str, Any]) -> None:
        send_confirmed = getattr(self._transport, "send_confirmed", None)
        if send_confirmed is None:
            raise RelayConnectionFailed("transport does not support confirmed writes")
        receipt = await send_confirmed(message)
        await receipt

    def abort(self, reason: str) -> int:
        # A dedicated sink holds no per-stream state to abandon; the shared
        # connection's lifecycle belongs to its owner.
        return 0


class _DemuxSinkFactory:
    """Open one demultiplexer stream per relay stream."""

    def __init__(self, demultiplexer: StreamDemultiplexer) -> None:
        self._demultiplexer = demultiplexer

    def __call__(self, stream_id: str) -> RelaySink:
        try:
            stream = self._demultiplexer.open_stream(stream_id)
        except StreamTerminatedError as exc:
            raise RelayConnectionFailed(str(exc)) from exc
        return _DemuxStreamSink(stream)


class _DemuxStreamSink:
    """Adapt a demultiplexer stream to the relay's sink surface."""

    def __init__(self, stream: DemuxStream) -> None:
        self._stream = stream

    async def send(self, message: dict[str, Any]) -> None:
        # The demultiplexer queues synchronously and owns the per-stream writer,
        # so admission never blocks the hop's fair rotation.
        self._stream.send(message)

    async def send_confirmed(self, message: dict[str, Any]) -> None:
        await self._stream.send_confirmed(message)

    def abort(self, reason: str) -> int:
        return self._stream.abort(reason)
