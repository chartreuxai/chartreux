"""WP6 [6C]: the presentation component the differential traces host.

A trace drives the same production stack in both topologies:

* the producer side of a relay hop (:class:`~chartreux.app_server.relay.RelayHop`
  with an incarnation-enabled stream) — byte accounting, shedding, retirement,
  and terminal propagation;
* the receiving hop's inbound path (:meth:`RelayStream.accept_inbound`) —
  generation staleness, liveness reset, and peer terminal mapping;
* the TUI side (:class:`~chartreux.app_server.events.PresentationConsumer`) —
  announced incarnations, atomic paged snapshot installs, and the
  generation-scoped registration timers;
* the ordinary projection
  (:class:`~chartreux.app_server.events.ClientProjection`), which every
  presentation frame the relay consumer passes through is applied to, so a
  sequence gap is detected by production code.

The in-process topology removes the process boundary; the spawned topology runs
this identical function inside the egress-blocked replay child. Observation
frames carry only fake-clock values, so the two topologies normalize to the
same digest.

The fake clock is advanced by command, never by sleeping: every deadline in a
trace fires at an injected delay, which is what makes the registration,
backoff, idle, and propagation schedules assertable at their worst case.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
import hashlib
import os
from typing import Any

from chartreux.app_server.demux import StreamDemultiplexer
from chartreux.app_server.events import (
    DEFAULT_PRESENTATION_STREAM_ID,
    REGISTRATION_AWAIT_TIMEOUT,
    REGISTRATION_BACKOFF_SECONDS,
    REGISTRATION_MAX_ATTEMPTS,
    REGISTRATION_TO_INSTALL_TIMEOUT,
    RELAY_INCARNATION_FIELD,
    RETIRE_METHOD,
    SNAPSHOT_MAX_BYTES,
    SNAPSHOT_MAX_PAGES,
    SNAPSHOT_PAGE_MAX_BYTES,
    SNAPSHOT_PAGE_METHOD,
    TERMINAL_METHOD,
    ClientProjection,
    PresentationConsumer,
    PresentationProtocolError,
    projection_notification_methods,
)
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    PublicSession,
    PublicSessionState,
    TextContentBlock,
)
from chartreux.app_server.protocol import HistoryEntryAddedParams, Notification
from chartreux.app_server.relay import (
    CONTROL_PROPAGATION_TIMEOUT,
    RelayError,
    RelayHop,
    RelayStream,
    TerminalDisposition,
    TerminalPropagation,
)
from tests.differential.replay import (
    Channel,
    ReplayError,
    build_replay_script,
    canonical_json,
)

SESSION_ID = "differential-session"
DEFAULT_CLOCK = 100.0
#: The seeded producer history stays small enough that a snapshot fits one page.
DEFAULT_HISTORY = 6


class _ManualClock:
    """The deterministic monotonic clock both topologies advance by command."""

    def __init__(self, value: float = DEFAULT_CLOCK) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _RecordingSink:
    """A relay sink that retains what the scheduler wrote."""

    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []
        self.aborted: list[str] = []

    async def send(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    async def abortive_close(self) -> None:
        self.aborted.append("abortive close")

    def abort(self, reason: str) -> int:
        self.aborted.append(reason)
        return 0


def _number(source: Mapping[str, Any], key: str, default: float) -> float:
    value = source.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _integer(source: Mapping[str, Any], key: str, default: int) -> int:
    value = source.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return value


def _floats(
    source: Mapping[str, Any], key: str, default: tuple[float, ...]
) -> tuple[float, ...]:
    value = source.get(key)
    if not isinstance(value, (list, tuple)):
        return default
    entries = [
        float(item)
        for item in value
        if isinstance(item, (int, float)) and not isinstance(item, bool)
    ]
    return tuple(entries) if entries else default


def _initial_state() -> PublicSessionState:
    """The empty projection both topologies start from."""
    return PublicSessionState(
        event_id=0,
        session=PublicSession(
            id=SESSION_ID, status=IdleSessionStatus(), created_at=1, updated_at=1
        ),
        history=[],
        turns=[],
        active_callbacks=[],
    )


def _message_entry(entry_id: str, text: str) -> PublicMessageEntry:
    return PublicMessageEntry(
        id=entry_id,
        session_id=SESSION_ID,
        turn_id="turn-1",
        role="assistant",
        content=[TextContentBlock(text=text)],
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        created_at=1,
        updated_at=1,
    )


def build_history(
    seed: int, count: int = DEFAULT_HISTORY
) -> tuple[tuple[str, str], ...]:
    """The seeded ``(entry id, text)`` history every trace draws from."""
    words = build_replay_script(seed, turns=count, words=1).turns
    return tuple(
        (f"entry-{index}", f"{word}-{index}") for index, word in enumerate(words)
    )


def history_entries(history: tuple[tuple[str, str], ...]) -> list[PublicMessageEntry]:
    return [_message_entry(entry_id, text) for entry_id, text in history]


def history_payload(history: tuple[tuple[str, str], ...]) -> str:
    """The canonical snapshot payload for one prefix of the seeded history."""
    return canonical_json({
        "entries": [{"id": entry_id, "text": text} for entry_id, text in history]
    })


def page_frame(
    index: int, count: int, payload: str, boundary: int | None = None
) -> dict[str, Any]:
    """One bounded snapshot page, as the producer puts it on the wire."""
    return {
        "jsonrpc": "2.0",
        "method": SNAPSHOT_PAGE_METHOD,
        "params": {
            "pageIndex": index,
            "pageCount": count,
            "payload": payload,
            "sequenceBoundary": boundary,
        },
    }


def snapshot_frames(history: tuple[tuple[str, str], ...]) -> list[dict[str, Any]]:
    """The seeded snapshot: one boundary-bearing page over the whole history."""
    return [page_frame(0, 1, history_payload(history), len(history))]


def entry_added_frame(entry: PublicMessageEntry, event_id: int) -> dict[str, Any]:
    """One history delta, sequenced the way the producer numbers it."""
    params = HistoryEntryAddedParams(
        event_id=event_id, session_id=SESSION_ID, emitted_at=event_id, entry=entry
    )
    return {
        "jsonrpc": "2.0",
        "method": "history/entryAdded",
        "params": params.model_dump(mode="json", by_alias=True),
    }


def heartbeat_frame() -> dict[str, Any]:
    """A sequenced-free control notification: registered, never event-numbered."""
    return {"jsonrpc": "2.0", "method": "usage/updated", "params": {}}


def delta_frame(method: str = "session/updated") -> dict[str, Any]:
    """A droppable presentation delta with no projection effect."""
    return {"jsonrpc": "2.0", "method": method, "params": {}}


def padded_delta_frame(pad: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "session/updated", "params": {"pad": "x" * pad}}


def control_request_frame(request_id: str) -> dict[str, Any]:
    """An approval-class request: control lane, never shed."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "session/approval",
        "params": {},
    }


def control_result_frame(request_id: str) -> dict[str, Any]:
    """The settlement for that request: control lane, never shed."""
    return {"jsonrpc": "2.0", "id": request_id, "result": {"decision": "deny"}}


def terminal_notice(
    generation: int,
    reason: str,
    disposition: TerminalDisposition = TerminalDisposition.STREAM_ABORTED,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": TERMINAL_METHOD,
        "relay_incarnation": generation,
        "params": {"disposition": disposition.value, "reason": reason},
    }


def retire_frame(generation: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": RETIRE_METHOD, "relay_incarnation": generation}


def _digest_payloads(pages: tuple[str, ...]) -> str:
    return hashlib.sha256(canonical_json(list(pages)).encode("utf-8")).hexdigest()


def _digest_entries(entries: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        canonical_json({"entries": entries}).encode("utf-8")
    ).hexdigest()


class PresentationHost:
    """The component: relay producer hop, receiving hop, consumer, projection."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel
        self._run = f"run-{os.getpid()}"
        self._sequence = 0
        self._clock = _ManualClock()
        self._stream_ids: tuple[str, ...] = (DEFAULT_PRESENTATION_STREAM_ID,)
        self._multiplexed = False
        self._producer_hops: dict[str, RelayHop] = {}
        self._producer_streams: dict[str, RelayStream] = {}
        self._peer_hops: dict[str, RelayHop] = {}
        self._peer_streams: dict[str, RelayStream] = {}
        self._demux: StreamDemultiplexer | None = None
        self._sinks: dict[str, _RecordingSink] = {}
        self._consumer: PresentationConsumer | None = None
        self._projection: ClientProjection | None = None
        self._events: list[dict[str, object]] = []
        self._transitions: list[dict[str, object]] = []
        self._aborts: list[str] = []
        self._hold_method: str | None = None
        self._held: list[tuple[str, int, dict[str, Any]]] = []
        self._propagations: dict[str, TerminalPropagation] = {}
        self._terminal_counts: dict[str, int] = {}
        self._last_phase: dict[str, str] = {}

    # -- instruction stream ----------------------------------------------
    async def serve(self) -> None:
        while (message := self._channel.receive()) is not None:
            kind = message.get("type")
            if kind == "init":
                self._initialize(message)
                self._observe("init", -1)
            elif kind == "op":
                index = message.get("index")
                op = message.get("op")
                if type(index) is not int or not isinstance(op, dict):
                    raise ReplayError(f"malformed presentation op: {message!r}")
                await self._run_op(index, op)
            else:
                raise ReplayError(f"unknown presentation message: {message!r}")

    def _initialize(self, message: Mapping[str, Any]) -> None:
        self._clock = _ManualClock(_number(message, "clock", DEFAULT_CLOCK))
        streams = message.get("streams")
        if isinstance(streams, list) and streams:
            self._stream_ids = tuple(str(stream) for stream in streams)
        self._multiplexed = bool(message.get("multiplexed", False))
        hop_bounds = message.get("hop")
        consumer_bounds = message.get("consumer")
        hop: dict[str, Any] = (
            {str(key): value for key, value in hop_bounds.items()}
            if isinstance(hop_bounds, Mapping)
            else {}
        )
        limits: dict[str, Any] = (
            {str(key): value for key, value in consumer_bounds.items()}
            if isinstance(consumer_bounds, Mapping)
            else {}
        )
        record_abort = bool(message.get("record_abort", False))

        async def _record_abort(reason: str) -> None:
            self._aborts.append(reason)

        consumer = PresentationConsumer(
            registration_await_timeout=_number(
                limits, "registration_await_timeout", REGISTRATION_AWAIT_TIMEOUT
            ),
            registration_to_install_timeout=_number(
                limits,
                "registration_to_install_timeout",
                REGISTRATION_TO_INSTALL_TIMEOUT,
            ),
            max_attempts=_integer(limits, "max_attempts", REGISTRATION_MAX_ATTEMPTS),
            backoff_schedule=_floats(
                limits, "backoff_schedule", REGISTRATION_BACKOFF_SECONDS
            ),
            page_max_bytes=_integer(limits, "page_max_bytes", SNAPSHOT_PAGE_MAX_BYTES),
            snapshot_max_pages=_integer(
                limits, "snapshot_max_pages", SNAPSHOT_MAX_PAGES
            ),
            snapshot_max_bytes=_integer(
                limits, "snapshot_max_bytes", SNAPSHOT_MAX_BYTES
            ),
            clock=self._clock,
            listener=self._listener,
        )
        self._consumer = consumer
        self._projection = ClientProjection(_initial_state())

        hop_limits: dict[str, Any] = {
            "presentation_per_stream": _integer(
                hop, "presentation_per_stream", 4 * 1024 * 1024
            ),
            "presentation_per_hop": _integer(
                hop, "presentation_per_hop", 16 * 1024 * 1024
            ),
            "control_capacity": _integer(hop, "control_capacity", 1024 * 1024),
            "heartbeat_interval": _number(hop, "heartbeat_interval", 2.0),
            "idle_timeout": _number(hop, "idle_timeout", 10.0),
            "control_propagation_timeout": _number(
                hop, "control_propagation_timeout", CONTROL_PROPAGATION_TIMEOUT
            ),
            "clock": self._clock,
        }
        if record_abort:
            hop_limits["connection_abort"] = _record_abort

        shared_sink = _RecordingSink()
        if self._multiplexed and bool(message.get("demux", False)):
            self._demux = StreamDemultiplexer(shared_sink)
        for stream_id in self._stream_ids:
            self._sinks[stream_id] = (
                shared_sink if self._multiplexed else _RecordingSink()
            )
            self._consumer.stream(stream_id)

        if self._multiplexed:
            if self._demux is not None:
                producer_hop = RelayHop.over_demux(self._demux, **hop_limits)
            else:
                producer_hop = RelayHop(lambda _stream_id: shared_sink, **hop_limits)
            peer_hop = RelayHop(lambda _stream_id: _RecordingSink(), **hop_limits)
            for stream_id in self._stream_ids:
                self._producer_hops[stream_id] = producer_hop
                self._producer_streams[stream_id] = producer_hop.open_stream(
                    stream_id, incarnation=True
                )
                self._peer_hops[stream_id] = peer_hop
                self._peer_streams[stream_id] = peer_hop.open_stream(
                    stream_id, incarnation=True
                )
            return

        for stream_id in self._stream_ids:
            sink = self._sinks[stream_id]
            producer_hop = RelayHop(lambda _stream_id, sink=sink: sink, **hop_limits)
            peer_hop = RelayHop(lambda _stream_id: _RecordingSink(), **hop_limits)
            self._producer_hops[stream_id] = producer_hop
            self._producer_streams[stream_id] = producer_hop.open_stream(
                stream_id, incarnation=True
            )
            self._peer_hops[stream_id] = peer_hop
            self._peer_streams[stream_id] = peer_hop.open_stream(
                stream_id, incarnation=True
            )

    def _listener(self, stream_id: str, state: Any) -> None:
        install = state.last_install
        terminal = state.terminal
        phase = str(state.phase)
        self._terminal_counts.setdefault(stream_id, 0)
        if terminal is not None and self._last_phase.get(stream_id) != phase:
            self._terminal_counts[stream_id] += 1
        self._last_phase[stream_id] = phase
        self._transitions.append({
            "stream": stream_id,
            "phase": str(state.phase),
            "announced": state.current,
            "attempts": state.attempts,
            "install": None if install is None else install.generation,
            "terminal": None if terminal is None else terminal.reason,
        })

    # -- ops --------------------------------------------------------------
    async def _run_op(self, index: int, op: dict[str, Any]) -> None:
        self._events = []
        self._transitions = []
        kind = str(op.get("kind", ""))
        stream_id = str(op.get("stream", self._stream_ids[0]))
        if stream_id not in self._producer_streams:
            raise ReplayError(f"unknown presentation stream: {stream_id!r}")
        if kind == "snapshot":
            self._hold(op)
            self._enqueue_frames(stream_id, self._snapshot_op_frames(op))
            self._drain(stream_id)
        elif kind == "delta":
            self._hold(op)
            frame = op.get("frame")
            if not isinstance(frame, dict):
                raise ReplayError(f"delta op carries no frame: {op!r}")
            self._enqueue_frames(stream_id, [frame])
            self._drain(stream_id)
        elif kind == "batch":
            frames = op.get("frames")
            if not isinstance(frames, list):
                raise ReplayError(f"batch op carries no frames: {op!r}")
            self._enqueue_frames(
                stream_id, [frame for frame in frames if isinstance(frame, dict)]
            )
            self._drain(stream_id)
        elif kind == "enqueue":
            # Admit without draining: what the caller needs is the admission
            # disposition while the hop still holds the retained bytes.
            frames = op.get("frames")
            if not isinstance(frames, list):
                raise ReplayError(f"enqueue op carries no frames: {op!r}")
            self._enqueue_frames(
                stream_id, [frame for frame in frames if isinstance(frame, dict)]
            )
        elif kind == "retire":
            self._hold(op)
            stream = self._producer_streams[stream_id]
            try:
                invalidated = stream.retire_incarnation()
            except RelayError as exc:
                self._events.append({
                    "kind": "retire",
                    "stream": stream_id,
                    "rejected": type(exc).__name__,
                })
            else:
                self._events.append({
                    "kind": "retire",
                    "stream": stream_id,
                    "invalidated": invalidated,
                })
            self._drain(stream_id)
        elif kind == "terminate":
            stream = self._producer_streams[stream_id]
            try:
                propagation = stream.terminate(str(op.get("reason", "terminated")))
            except RelayError as exc:
                self._events.append({
                    "kind": "terminate",
                    "stream": stream_id,
                    "rejected": type(exc).__name__,
                })
            else:
                self._propagations[stream_id] = propagation
                self._events.append({
                    "kind": "terminate",
                    "stream": stream_id,
                    "deadline": round(propagation.deadline, 6),
                })
        elif kind == "write":
            self._drain(stream_id)
            propagation = self._propagations.get(stream_id)
            if propagation is not None:
                self._events.append({
                    "kind": "write",
                    "stream": stream_id,
                    "confirmed": propagation.confirmed,
                    "settled": propagation.settled,
                })
        elif kind == "settle":
            propagation = self._propagations.get(stream_id)
            if propagation is None:
                raise ReplayError(f"no pending terminal propagation for {stream_id!r}")
            hop = self._producer_hops[stream_id]
            confirmed = await hop.settle_terminal(propagation)
            self._events.append({
                "kind": "settle",
                "stream": stream_id,
                "confirmed": confirmed,
                "propagation_error": None
                if propagation.error is None
                else str(propagation.error),
                "hop_failure": None if hop.failure is None else str(hop.failure),
            })
            if not confirmed:
                # The notice missed its bound, so the hop aborted the shared
                # connection: every stream on it sees the same downstream
                # disposition (design's "a failed relay path closes the
                # downstream connection").
                for target in self._stream_ids:
                    if self._producer_hops[target] is not hop:
                        continue
                    self._eof(
                        target,
                        "connection failed within the propagation bound",
                        TerminalDisposition.CONNECTION_FAILED,
                    )
        elif kind == "eof":
            for target in self._stream_ids:
                if self._producer_hops[target] is not self._producer_hops[stream_id]:
                    continue
                self._eof(
                    target,
                    str(op.get("reason", "upstream pipe closed")),
                    TerminalDisposition(
                        str(
                            op.get(
                                "disposition",
                                TerminalDisposition.CONNECTION_FAILED.value,
                            )
                        )
                    ),
                )
            self._events.append({"kind": "eof", "stream": stream_id})
        elif kind == "inject":
            wire = op.get("wire")
            if not isinstance(wire, dict):
                raise ReplayError(f"inject op carries no wire frame: {op!r}")
            self._deliver(stream_id, wire)
        elif kind == "demux_abortive_close":
            if self._demux is None:
                raise ReplayError("demux_abortive_close needs a demultiplexed hop")
            await self._demux.abortive_close()
            self._events.append({"kind": "demux_abortive_close"})
            for target in self._stream_ids:
                self._eof(
                    target,
                    "shared connection aborted",
                    TerminalDisposition.CONNECTION_FAILED,
                )
        elif kind == "discard":
            stream = self._producer_streams[stream_id]
            discarded = stream.discard(str(op.get("reason", "abortive close")))
            self._events.append({
                "kind": "discard",
                "stream": stream_id,
                "discarded": discarded,
            })
        elif kind == "tick":
            seconds = _number(op, "seconds", 0.0)
            self._clock.advance(seconds)
            fired = self._consumer_fire_due()
            self._events.append({"kind": "tick", "seconds": seconds, "fired": fired})
        elif kind == "peer_notice":
            generation = int(self._producer_streams[stream_id].generation)
            disposition = str(
                op.get("disposition", TerminalDisposition.STREAM_ABORTED.value)
            )
            notice = terminal_notice(
                generation,
                str(op.get("reason", "peer notice")),
                TerminalDisposition(disposition),
            )
            self._deliver(stream_id, notice)
        elif kind == "connection_failed":
            reason = str(op.get("reason", "shared connection failed"))
            for target in self._stream_ids:
                generation = int(self._producer_streams[target].generation)
                self._deliver(
                    target,
                    terminal_notice(
                        generation, reason, TerminalDisposition.CONNECTION_FAILED
                    ),
                )
            self._events.append({"kind": "connection_failed", "reason": reason})
        elif kind == "close_hop":
            drain = bool(op.get("drain", True))
            hops = {sid: self._producer_hops[sid] for sid in self._stream_ids}
            for hop in dict.fromkeys(hops.values()):
                await hop.close(drain=drain)
            self._events.append({"kind": "close_hop", "drain": drain})
        elif kind == "release_held":
            held, self._held = self._held, []
            for held_stream, held_generation, held_message in held:
                event: dict[str, Any] = {
                    "kind": "wire",
                    "stream": held_stream,
                    "method": held_message.get("method"),
                    "generation": held_generation,
                    "relay": "release",
                }
                self._events.append(event)
                self._offer_consumer(held_stream, held_generation, held_message, event)
            self._events.append({"kind": "release_held", "released": len(held)})
        elif kind == "peer_enqueue":
            frame = op.get("frame")
            if not isinstance(frame, dict):
                raise ReplayError(f"peer_enqueue op carries no frame: {op!r}")
            hop = self._peer_hops[stream_id]
            try:
                outcome = hop.enqueue(stream_id, frame)
            except RelayError as exc:
                self._events.append({
                    "kind": "peer_enqueue",
                    "stream": stream_id,
                    "rejected": type(exc).__name__,
                })
            else:
                self._events.append({
                    "kind": "peer_enqueue",
                    "stream": stream_id,
                    "disposition": str(outcome.disposition),
                })
        elif kind == "observe":
            pass
        else:
            raise ReplayError(f"unknown presentation op: {kind!r}")
        self._hold_method = None
        self._observe(kind, index)

    def _hold(self, op: Mapping[str, Any]) -> None:
        method = op.get("hold")
        self._hold_method = str(method) if isinstance(method, str) else None

    def _snapshot_op_frames(self, op: Mapping[str, Any]) -> list[dict[str, Any]]:
        frames = op.get("frames")
        if isinstance(frames, list):
            return [frame for frame in frames if isinstance(frame, dict)]
        payloads = op.get("payloads")
        boundary = op.get("boundary")
        if not isinstance(payloads, list) or not payloads:
            raise ReplayError(f"snapshot op carries no pages: {op!r}")
        count = len(payloads)
        return [
            page_frame(
                index,
                count,
                str(payload),
                int(boundary) if isinstance(boundary, int) else None,
            )
            for index, payload in enumerate(payloads)
        ]

    def _enqueue_frames(self, stream_id: str, frames: list[dict[str, Any]]) -> None:
        hop = self._producer_hops[stream_id]
        for frame in frames:
            try:
                outcome = hop.enqueue(stream_id, frame)
            except RelayError as exc:
                self._events.append({
                    "kind": "enqueue",
                    "stream": stream_id,
                    "method": frame.get("method"),
                    "rejected": type(exc).__name__,
                })
                continue
            self._events.append({
                "kind": "enqueue",
                "stream": stream_id,
                "method": frame.get("method"),
                "class": str(outcome.frame_class),
                "disposition": str(outcome.disposition),
                "shed_reason": outcome.shed_reason,
            })

    def _drain(self, stream_id: str) -> None:
        if self._multiplexed:
            hops = {id(hop): hop for hop in self._producer_hops.values()}
        else:
            hops = {id(self._producer_hops[stream_id]): self._producer_hops[stream_id]}
        for hop in hops.values():
            while (item := hop.dequeue()) is not None:
                stream, queued = item
                self._deliver(stream.stream_id, queued.message)
                if queued.delivery is not None and not queued.delivery.expired():
                    queued.delivery.confirm()

    def _deliver(self, stream_id: str, wire: dict[str, Any]) -> None:
        held = self._hold_method is not None and wire.get("method") == self._hold_method
        self._feed(stream_id, wire, offer=not held)

    def _eof(
        self, stream_id: str, reason: str, disposition: TerminalDisposition
    ) -> None:
        """Carry a local pipe/connection close to the endpoint as a disposition.

        A downstream EOF is the same terminal notice the relay's closed hop
        would carry; the endpoint maps it to a typed presentation failure.
        """
        generation = int(self._producer_streams[stream_id].generation)
        self._feed(stream_id, terminal_notice(generation, reason, disposition))

    def _feed(
        self, stream_id: str, wire: dict[str, Any], *, offer: bool = True
    ) -> None:
        generation = wire.get(RELAY_INCARNATION_FIELD)
        if type(generation) is not int or generation < 0:
            raise ReplayError(f"wire frame carries no incarnation: {wire!r}")
        peer_stream = self._peer_streams[stream_id]
        peer_frame = peer_stream.accept_inbound(dict(wire))
        event: dict[str, Any] = {
            "kind": "wire",
            "stream": stream_id,
            "method": wire.get("method"),
            "generation": generation,
            "relay": "forwarded" if peer_frame is not None else "consumed",
            "peer_terminal": (
                None
                if peer_stream.terminal_error is None
                else str(peer_stream.terminal_error)
            ),
        }
        stripped = {
            key: value for key, value in wire.items() if key != RELAY_INCARNATION_FIELD
        }
        self._events.append(event)
        if not offer:
            # The relay saw the frame; the endpoint has not. A lost or delayed
            # control frame is exactly the announcement-before-notice case.
            event["held"] = True
            self._held.append((stream_id, generation, stripped))
            return
        self._offer_consumer(stream_id, generation, stripped, event)

    def _offer_consumer(
        self,
        stream_id: str,
        generation: int,
        message: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        consumer = self._consumer
        if consumer is None:
            raise ReplayError("presentation consumer is not initialized")
        try:
            consumed = consumer.receive_frame(stream_id, generation, message)
        except PresentationProtocolError as exc:
            event["protocol_error"] = str(exc)
            return
        event["consumed"] = consumed
        if consumed:
            return
        self._apply_projection(message, event)

    def _apply_projection(
        self, message: Mapping[str, Any], event: dict[str, Any]
    ) -> None:
        method = message.get("method")
        if (
            not isinstance(method, str)
            or method not in projection_notification_methods()
        ):
            event["projection"] = "not_a_projection_event"
            return
        projection = self._projection
        if projection is None:
            raise ReplayError("client projection is not initialized")
        params = message.get("params")
        try:
            notification = Notification(
                method=method, params=params if isinstance(params, dict) else {}
            )
            applied = projection.consume(notification)
        except Exception as exc:
            event["gap"] = f"{type(exc).__name__}: {exc}"
            return
        event["event_id"] = projection.last_event_id
        event["projection_event"] = None if applied is None else type(applied).__name__

    def _consumer_fire_due(self) -> list[dict[str, object]]:
        consumer = self._consumer
        if consumer is None:
            raise ReplayError("presentation consumer is not initialized")
        return [
            {
                "kind": str(timer.kind),
                "generation": timer.generation,
                "attempt": timer.attempt,
            }
            for timer in consumer.fire_due()
        ]

    # -- observation -------------------------------------------------------
    def _observe(self, op: str, index: int) -> None:
        consumer = self._consumer
        projection = self._projection
        if consumer is None or projection is None:
            raise ReplayError("presentation host is not initialized")
        streams: dict[str, object] = {}
        for stream_id in self._stream_ids:
            state = consumer.stream(stream_id)
            install = state.last_install
            terminal = state.terminal
            producer = self._producer_streams[stream_id]
            peer = self._peer_streams[stream_id]
            producer_error = producer.terminal_error
            peer_error = peer.terminal_error
            streams[stream_id] = {
                "announced": state.current,
                "phase": str(state.phase),
                "attempts": state.attempts,
                "terminal": None if terminal is None else terminal.reason,
                "install": None
                if install is None
                else {
                    "generation": install.generation,
                    "boundary": install.sequence_boundary,
                    "pages": len(install.pages),
                    "payloads": list(install.pages),
                    "digest": _digest_payloads(install.pages),
                },
                "terminal_transitions": self._terminal_counts.get(stream_id, 0),
                "timers": [
                    {
                        "kind": kind,
                        "generation": generation,
                        "deadline": deadline,
                        "attempt": attempt,
                    }
                    for kind, generation, deadline, attempt in sorted(
                        (
                            str(timer.kind),
                            timer.generation,
                            round(timer.deadline, 6),
                            timer.attempt,
                        )
                        for timer in state.timers
                    )
                ],
                "idle_expired": peer.idle_expired(),
                "producer": {
                    "generation": producer.generation,
                    "terminal": None if producer_error is None else str(producer_error),
                    "queued": producer.queued_frames,
                    "presentation_bytes": producer.presentation_bytes,
                    "control_bytes": producer.control_bytes,
                    "shed": producer.shed_frames,
                    "invalidated": producer.invalidated_frames,
                },
                "peer": {
                    "terminal": None if peer_error is None else str(peer_error),
                    "invalidated": peer.invalidated_frames,
                },
                "hop": self._hop_summary(self._producer_hops[stream_id]),
                "peer_hop": self._hop_summary(self._peer_hops[stream_id]),
            }
        history = [
            {"id": entry.id, "text": entry.text}
            for entry in (projection.state.history or [])
            if isinstance(entry, PublicMessageEntry)
        ]
        frame: dict[str, object] = {
            "type": "obs",
            "run": self._run,
            "sequence": self._sequence,
            "op": op,
            "op_index": index,
            "fake_clock": round(self._clock.value, 6),
            "events": list(self._events),
            "transitions": list(self._transitions),
            "streams": streams,
            "projection": {
                "event_id": projection.last_event_id,
                "digest": _digest_entries(history),
                "entries": history,
            },
            "aborts": list(self._aborts),
            "demux": None
            if self._demux is None
            else {
                "closed": self._demux.closed,
                "connection_error": None
                if self._demux.connection_error is None
                else str(self._demux.connection_error),
                "streams": {
                    stream.stream_id: None
                    if stream.terminal_error is None
                    else str(stream.terminal_error)
                    for stream in self._demux.streams
                },
            },
        }
        self._channel.send(frame)
        self._sequence += 1

    @staticmethod
    def _hop_summary(hop: RelayHop) -> dict[str, object]:
        return {
            "closed": hop.closed,
            "failure": None if hop.failure is None else str(hop.failure),
            "presentation_bytes": hop.presentation_bytes,
            "control_bytes": hop.control_bytes,
            "shed": hop.shed_frames,
        }


def run_presentation_loop(channel: Channel) -> None:
    """Component entry point shared by both topologies."""
    asyncio.run(PresentationHost(channel).serve())


class PresentationDriver:
    """Drive one presentation trace over any channel."""

    def __init__(self, channel: Channel) -> None:
        self._channel = channel

    def initialize(self, init: Mapping[str, Any]) -> dict[str, Any]:
        self._channel.send({"type": "init", **init})
        return self._next_obs(expect=-1)

    def run(self, ops: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
        frames: list[dict[str, Any]] = []
        for index, op in enumerate(ops):
            self._channel.send({"type": "op", "index": index, "op": op})
            frames.append(self._next_obs(expect=index))
        return tuple(frames)

    def _next_obs(self, *, expect: int) -> dict[str, Any]:
        while True:
            frame = self._channel.receive()
            if frame is None:
                raise ReplayError("presentation host closed mid-trace")
            if frame.get("type") == "obs" and frame.get("op_index") == expect:
                return frame
