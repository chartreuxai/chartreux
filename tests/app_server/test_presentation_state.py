from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from chartreux.app_server.client import AppServerClient, AppServerConnectionClosed
from chartreux.app_server.events import (
    RETIRE_METHOD,
    SNAPSHOT_PAGE_MAX_BYTES,
    SNAPSHOT_PAGE_METHOD,
    TERMINAL_METHOD,
    PresentationConsumer,
    PresentationProtocolError,
    PresentationStreamState,
    StreamPhase,
    TimerKind,
    parse_snapshot_page,
)
from chartreux.app_server.protocol import Notification
from chartreux.app_server.transport import memory_transport_pair


class _FakeClock:
    """Manual monotonic clock; deterministic without real sleeps."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _consumer(clock: _FakeClock, **kwargs: Any) -> PresentationConsumer:
    return PresentationConsumer(clock=clock, **kwargs)


def _page_frame(
    generation: int,
    page_index: int,
    page_count: int,
    payload: str,
    boundary: int | None = None,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": SNAPSHOT_PAGE_METHOD,
        "relay_incarnation": generation,
        "params": {
            "pageIndex": page_index,
            "pageCount": page_count,
            "payload": payload,
            "sequenceBoundary": boundary,
        },
    }


def _delta_frame(
    generation: int, method: str = "history/entryUpdated"
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": method,
        "relay_incarnation": generation,
        "params": {},
    }


def _retire_frame(generation: int) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": RETIRE_METHOD, "relay_incarnation": generation}


def _terminal_frame(generation: int, reason: str = "aborted") -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "method": TERMINAL_METHOD,
        "relay_incarnation": generation,
        "params": {"disposition": "stream_aborted", "reason": reason},
    }


def _receive(
    consumer: PresentationConsumer, stream_id: str, frame: dict[str, Any]
) -> bool:
    generation = frame.pop("relay_incarnation")
    return consumer.receive_frame(stream_id, generation, frame)


def _install_snapshot(
    consumer: PresentationConsumer,
    stream_id: str,
    generation: int,
    payloads: list[str],
    boundary: int,
) -> None:
    count = len(payloads)
    for index, payload in enumerate(payloads):
        final = index == count - 1
        assert _receive(
            consumer,
            stream_id,
            _page_frame(generation, index, count, payload, boundary if final else None),
        )


def test_initial_incarnation_is_announced_and_awaits_its_snapshot() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    assert stream.current == 0
    assert stream.phase is StreamPhase.INSTALLING
    assert stream.terminal is None
    (timer,) = stream.timers
    assert timer.kind is TimerKind.REGISTRATION_TO_INSTALL
    assert timer.generation == 0
    assert timer.deadline == 8.0


def test_snapshot_installs_atomically_only_at_the_final_boundary_page() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    assert _receive(consumer, "s", _page_frame(0, 0, 3, "a"))
    assert _receive(consumer, "s", _page_frame(0, 1, 3, "b"))
    # No install before the final page: nothing partial is visible.
    assert stream.last_install is None
    assert stream.phase is StreamPhase.INSTALLING
    assert _receive(consumer, "s", _page_frame(0, 2, 3, "c", 42))
    install = stream.last_install
    assert install is not None
    assert install.generation == 0
    assert install.sequence_boundary == 42
    assert install.pages == ("a", "b", "c")
    assert stream.phase is StreamPhase.INSTALLED
    assert stream.attempts == 0
    assert stream.timers == ()
    # A current-generation delta now passes through to the projection.
    assert _receive(consumer, "s", _delta_frame(0)) is False


def test_empty_snapshot_installs_from_its_single_boundary_page() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    assert _receive(consumer, "s", _page_frame(0, 0, 1, "", 7))
    install = stream.last_install
    assert install is not None
    assert install.pages == ("",)
    assert install.sequence_boundary == 7
    assert stream.phase is StreamPhase.INSTALLED


def test_delta_before_the_snapshot_installs_is_a_typed_protocol_failure() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    with pytest.raises(PresentationProtocolError, match="before the incarnation"):
        _receive(consumer, "s", _delta_frame(0))


def test_unannounced_incarnation_frame_fails_typed() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    with pytest.raises(PresentationProtocolError, match="unannounced incarnation 7"):
        _receive(consumer, "s", _delta_frame(7))


def test_malformed_incarnation_stamp_fails_typed() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    with pytest.raises(PresentationProtocolError, match="non-negative"):
        consumer.receive_frame("s", None, _delta_frame(0))
    with pytest.raises(PresentationProtocolError, match="non-negative"):
        consumer.receive_frame("s", -1, _delta_frame(0))


def test_disordered_page_invalidates_the_attempt_without_any_install() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    # Page 1 before page 0: disorder invalidates the whole attempt.
    assert _receive(consumer, "s", _page_frame(0, 1, 3, "b"))
    assert stream.last_install is None
    assert stream.phase is StreamPhase.AWAITING
    assert stream.current is None
    assert stream.attempts == 1
    (timer,) = stream.timers
    assert timer.kind is TimerKind.BACKOFF
    assert timer.deadline == 0.5
    # The retired generation's frames are now stale and discarded.
    assert _receive(consumer, "s", _page_frame(0, 0, 3, "a")) is True
    assert stream.last_install is None


def test_oversized_page_invalidates_the_attempt() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    payload = "x" * (SNAPSHOT_PAGE_MAX_BYTES + 1)
    assert _receive(consumer, "s", _page_frame(0, 0, 1, payload, 1))
    assert stream.last_install is None
    assert stream.attempts == 1


def test_page_size_bound_is_a_constructor_parameter() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock, page_max_bytes=100)
    stream = consumer.stream("s")
    assert _receive(consumer, "s", _page_frame(0, 0, 1, "x" * 200, 1))
    assert stream.last_install is None
    assert stream.attempts == 1


def test_page_count_bound_invalidates_the_attempt() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock, snapshot_max_pages=2)
    stream = consumer.stream("s")
    assert _receive(consumer, "s", _page_frame(0, 0, 3, "a"))
    assert stream.last_install is None
    assert stream.attempts == 1


def test_total_snapshot_byte_bound_invalidates_the_attempt() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock, page_max_bytes=600, snapshot_max_bytes=1000)
    stream = consumer.stream("s")
    assert _receive(consumer, "s", _page_frame(0, 0, 2, "x" * 500))
    assert stream.last_install is None
    assert stream.phase is StreamPhase.INSTALLING
    # Second page pushes the total encoded size past 1000 bytes: all-or-nothing.
    assert _receive(consumer, "s", _page_frame(0, 1, 2, "y" * 500, 1))
    assert stream.last_install is None
    assert stream.attempts == 1


def test_final_page_without_boundary_invalidates_the_attempt() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    assert _receive(consumer, "s", _page_frame(0, 0, 1, "x", None))
    assert stream.last_install is None
    assert stream.attempts == 1


def test_parse_snapshot_page_rejects_malformed_params() -> None:
    with pytest.raises(PresentationProtocolError, match="object"):
        parse_snapshot_page(["not", "a", "mapping"])
    base: dict[str, Any] = {
        "pageIndex": 0,
        "pageCount": 1,
        "payload": "x",
        "sequenceBoundary": 1,
    }
    for key, value in (
        ("pageIndex", -1),
        ("pageIndex", "0"),
        ("pageCount", 0),
        ("payload", 5),
        ("sequenceBoundary", "1"),
        ("sequenceBoundary", -1),
    ):
        with pytest.raises(PresentationProtocolError):
            parse_snapshot_page({**base, key: value})


def test_retire_notice_retires_exactly_once_and_arms_the_await() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    assert _receive(consumer, "s", _retire_frame(0))
    assert stream.current is None
    assert stream.phase is StreamPhase.AWAITING
    (timer,) = stream.timers
    assert timer.kind is TimerKind.REGISTRATION_AWAIT
    assert timer.generation == 0
    assert timer.deadline == 30.0
    # A duplicate notice for the retired incarnation is a stale no-op.
    assert _receive(consumer, "s", _retire_frame(0))
    assert stream.timers == (timer,)
    assert clock.value < 30.0
    assert stream.fire_due() == []
    clock.advance(30.0)
    fired = stream.fire_due()
    assert [timer.kind for timer in fired] == [TimerKind.REGISTRATION_AWAIT]
    assert stream.phase is StreamPhase.TERMINAL
    assert stream.terminal is not None
    assert "await expired" in stream.terminal.reason
    # Frames on a terminally failed stream are consumed as stale.
    assert _receive(consumer, "s", _delta_frame(1)) is True


def test_announcement_acceptance_retires_await_and_arms_one_install() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    assert _receive(consumer, "s", _retire_frame(0))
    clock.advance(1.0)
    # The replacement's first frame announces generation 1.
    assert _receive(consumer, "s", _page_frame(1, 0, 2, "a"))
    assert stream.current == 1
    assert stream.phase is StreamPhase.INSTALLING
    (timer,) = stream.timers
    assert timer.kind is TimerKind.REGISTRATION_TO_INSTALL
    assert timer.generation == 1
    assert timer.deadline == 1.0 + 8.0
    assert _receive(consumer, "s", _page_frame(1, 1, 2, "b", 9))
    assert stream.phase is StreamPhase.INSTALLED
    assert stream.last_install is not None
    assert stream.last_install.sequence_boundary == 9
    assert stream.attempts == 0
    # The retired await deadline (due at t=30) can never fire into gen 1.
    clock.advance(29.5)
    assert stream.fire_due() == []
    assert stream.terminal is None
    clock.advance(1.0)
    assert stream.fire_due() == []
    assert stream.terminal is None


def test_announcement_before_notice_retires_the_old_incarnation_once() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    # Announcement wins the race: gen 1 is accepted while gen 0 is current.
    assert _receive(consumer, "s", _page_frame(1, 0, 1, "z", 5))
    assert stream.current == 1
    # The late notice for the already-retired gen 0 is stale: no-op.
    assert _receive(consumer, "s", _retire_frame(0))
    assert stream.current == 1
    assert stream.phase is StreamPhase.INSTALLED
    assert stream.last_install is not None
    assert stream.last_install.generation == 1


def test_retired_incarnations_timers_never_fire_into_the_replacement() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    assert _receive(consumer, "s", _retire_frame(0))
    clock.advance(1.0)
    # Gen 1 announces with a partial snapshot, so its install deadline governs.
    assert _receive(consumer, "s", _page_frame(1, 0, 2, "z"))
    # Gen 1's install deadline fires at t=9; gen 0's await (due t=30) must not.
    clock.advance(8.0)
    fired = stream.fire_due()
    assert [timer.kind for timer in fired] == [TimerKind.REGISTRATION_TO_INSTALL]
    assert fired[0].generation == 1
    assert stream.attempts == 1
    assert stream.phase is StreamPhase.AWAITING
    # Backoff (0.5s) then the await; the old gen 0 timers stay dead.
    clock.advance(0.5)
    assert [t.kind for t in stream.fire_due()] == [TimerKind.BACKOFF]
    (await_timer,) = stream.timers
    assert await_timer.kind is TimerKind.REGISTRATION_AWAIT
    assert await_timer.generation == 1
    clock.advance(29.0)
    assert stream.fire_due() == []
    assert stream.terminal is None


def test_install_deadline_exhaustion_ends_in_persistent_typed_failure() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    generation = 0
    for attempt in range(3):
        clock.advance(8.0)
        fired = stream.fire_due()
        assert [t.kind for t in fired] == [TimerKind.REGISTRATION_TO_INSTALL]
        assert stream.attempts == attempt + 1
        if attempt < 2:
            assert stream.phase is StreamPhase.AWAITING
            (backoff,) = stream.timers
            assert backoff.kind is TimerKind.BACKOFF
            assert backoff.deadline == clock.value + (0.5, 1.0)[attempt]
            # The failed attempt is retired before a fresh incarnation; the
            # new incarnation's snapshot stays partial so its install
            # deadline is what fires next.
            generation += 1
            assert _receive(consumer, "s", _page_frame(generation, 0, 2, "z"))
            assert stream.current == generation
            assert stream.phase is StreamPhase.INSTALLING
    assert stream.phase is StreamPhase.TERMINAL
    assert stream.terminal is not None
    assert "attempts exhausted" in stream.terminal.reason
    assert stream.fire_due() == []
    assert _receive(consumer, "s", _delta_frame(9)) is True


def test_backoff_schedule_comes_from_the_constructor() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock, backoff_schedule=(0.25, 2.0))
    stream = consumer.stream("s")
    clock.advance(8.0)
    stream.fire_due()
    (backoff,) = stream.timers
    assert backoff.deadline == 8.0 + 0.25
    assert _receive(consumer, "s", _page_frame(1, 0, 2, "z"))
    clock.advance(8.0)
    stream.fire_due()
    (backoff,) = stream.timers
    assert backoff.deadline == 16.0 + 2.0


def test_partial_snapshot_hits_the_install_deadline_then_retry_installs() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    assert _receive(consumer, "s", _retire_frame(0))
    clock.advance(1.0)
    # Partial snapshot for gen 1: one of two pages, no boundary yet.
    assert _receive(consumer, "s", _page_frame(1, 0, 2, "a"))
    assert stream.last_install is not None  # still the gen 0 install
    clock.advance(8.0)
    assert [t.kind for t in stream.fire_due()] == [TimerKind.REGISTRATION_TO_INSTALL]
    assert stream.attempts == 1
    clock.advance(0.5)
    assert [t.kind for t in stream.fire_due()] == [TimerKind.BACKOFF]
    # The producer re-registers as gen 2 and completes the snapshot.
    clock.advance(1.0)
    assert _receive(consumer, "s", _page_frame(2, 0, 2, "p"))
    assert _receive(consumer, "s", _page_frame(2, 1, 2, "q", 11))
    assert stream.phase is StreamPhase.INSTALLED
    assert stream.attempts == 0
    install = stream.last_install
    assert install is not None
    assert install.generation == 2
    assert install.pages == ("p", "q")
    assert install.sequence_boundary == 11


def test_await_expiry_terminates_the_stream_with_a_persistent_typed_failure() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    assert _receive(consumer, "s", _retire_frame(0))
    clock.advance(30.0)
    assert [t.kind for t in stream.fire_due()] == [TimerKind.REGISTRATION_AWAIT]
    assert stream.phase is StreamPhase.TERMINAL
    assert stream.terminal is not None
    assert "await expired" in stream.terminal.reason


def test_terminal_notice_terminates_only_the_matching_incarnation() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    stream = consumer.stream("s")
    _install_snapshot(consumer, "s", 0, ["a"], 1)
    assert _receive(consumer, "s", _retire_frame(0))
    # A terminal notice for the retired generation is stale, not fatal.
    assert _receive(consumer, "s", _terminal_frame(0, "old"))
    assert stream.terminal is None
    assert _receive(consumer, "s", _page_frame(1, 0, 1, "z", 2))
    assert _receive(consumer, "s", _terminal_frame(1, "gone"))
    assert stream.terminal is not None
    assert "gone" in stream.terminal.reason
    assert stream.phase is StreamPhase.TERMINAL


def test_streams_are_tracked_independently() -> None:
    clock = _FakeClock()
    consumer = _consumer(clock)
    first = consumer.stream("root")
    second = consumer.stream("child")
    _install_snapshot(consumer, "root", 0, ["a"], 1)
    assert _receive(consumer, "root", _retire_frame(0))
    assert first.phase is StreamPhase.AWAITING
    assert second.phase is StreamPhase.INSTALLING
    assert second.current == 0
    assert consumer.fire_due("root") == []
    assert consumer.fire_due("child") == []


def test_constructor_rejects_unusable_bounds() -> None:
    clock = _FakeClock()
    with pytest.raises(ValueError, match="finite and positive"):
        PresentationConsumer(clock=clock, registration_await_timeout=0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        PresentationConsumer(clock=clock, registration_to_install_timeout=float("inf"))
    with pytest.raises(ValueError, match="max_attempts"):
        PresentationConsumer(clock=clock, max_attempts=0)
    with pytest.raises(ValueError, match="backoff"):
        PresentationConsumer(clock=clock, backoff_schedule=())
    with pytest.raises(ValueError, match="backoff"):
        PresentationConsumer(clock=clock, backoff_schedule=(0.5, -1.0))
    with pytest.raises(ValueError, match="stream_id"):
        PresentationStreamState("", clock=clock)


@pytest.mark.asyncio
async def test_relay_frame_routes_to_the_presentation_handler_and_is_consumed() -> None:
    client_transport, server_transport = memory_transport_pair()
    calls: list[tuple[str, int | None, dict[str, Any]]] = []

    def handler(
        stream_id: str, generation: int | None, message: dict[str, Any]
    ) -> bool:
        calls.append((stream_id, generation, message))
        return message.get("method") == RETIRE_METHOD

    client = AppServerClient(client_transport, presentation_handler=handler)
    await client.start()
    await server_transport.send(_retire_frame(0))
    await server_transport.send({
        "jsonrpc": "2.0",
        "method": "history/entryAdded",
        "params": {"entry": {}},
    })
    incoming = client.incoming()
    message = await asyncio.wait_for(anext(incoming), timeout=5.0)
    assert isinstance(message, Notification)
    assert message.method == "history/entryAdded"
    assert calls == [("session", 0, {"jsonrpc": "2.0", "method": RETIRE_METHOD})]
    await incoming.aclose()
    await client.close()
    await server_transport.close()


@pytest.mark.asyncio
async def test_unconsumed_relay_frame_passes_through_with_the_stamp_stripped() -> None:
    client_transport, server_transport = memory_transport_pair()
    calls: list[tuple[str, int | None, dict[str, Any]]] = []

    def handler(
        stream_id: str, generation: int | None, message: dict[str, Any]
    ) -> bool:
        calls.append((stream_id, generation, message))
        return False

    client = AppServerClient(client_transport, presentation_handler=handler)
    await client.start()
    await server_transport.send(_delta_frame(0))
    incoming = client.incoming()
    message = await asyncio.wait_for(anext(incoming), timeout=5.0)
    assert isinstance(message, Notification)
    assert message.method == "history/entryUpdated"
    assert "relay_incarnation" not in message.model_dump()
    assert calls == [
        (
            "session",
            0,
            {"jsonrpc": "2.0", "method": "history/entryUpdated", "params": {}},
        )
    ]
    await incoming.aclose()
    await client.close()
    await server_transport.close()


@pytest.mark.asyncio
async def test_plain_frames_never_reach_the_presentation_handler() -> None:
    client_transport, server_transport = memory_transport_pair()
    calls: list[tuple[str, int | None, dict[str, Any]]] = []

    def handler(
        stream_id: str, generation: int | None, message: dict[str, Any]
    ) -> bool:
        calls.append((stream_id, generation, message))
        return True

    client = AppServerClient(client_transport, presentation_handler=handler)
    await client.start()
    await server_transport.send({
        "jsonrpc": "2.0",
        "method": "session/updated",
        "params": {},
    })
    incoming = client.incoming()
    message = await asyncio.wait_for(anext(incoming), timeout=5.0)
    assert isinstance(message, Notification)
    assert calls == []
    await incoming.aclose()
    await client.close()
    await server_transport.close()


@pytest.mark.asyncio
async def test_malformed_relay_incarnation_stamp_fails_the_connection_typed() -> None:
    client_transport, server_transport = memory_transport_pair()
    client = AppServerClient(client_transport)
    await client.start()
    await server_transport.send({
        "jsonrpc": "2.0",
        "method": "session/updated",
        "relay_incarnation": -1,
        "params": {},
    })
    incoming = client.incoming()
    with pytest.raises(AppServerConnectionClosed):
        await asyncio.wait_for(anext(incoming), timeout=5.0)
    await incoming.aclose()
    await client.close()
    await server_transport.close()


@pytest.mark.asyncio
async def test_attach_presentation_routes_frames_after_construction() -> None:
    client_transport, server_transport = memory_transport_pair()
    calls: list[tuple[str, int | None, dict[str, Any]]] = []

    def handler(
        stream_id: str, generation: int | None, message: dict[str, Any]
    ) -> bool:
        calls.append((stream_id, generation, message))
        return True

    client = AppServerClient(client_transport)
    await client.start()
    client.attach_presentation(handler, stream_id="root")
    await server_transport.send(_retire_frame(3))
    await server_transport.send({
        "jsonrpc": "2.0",
        "method": "session/updated",
        "params": {},
    })
    incoming = client.incoming()
    message = await asyncio.wait_for(anext(incoming), timeout=5.0)
    assert isinstance(message, Notification)
    assert message.method == "session/updated"
    assert calls == [("root", 3, {"jsonrpc": "2.0", "method": RETIRE_METHOD})]
    await incoming.aclose()
    await client.close()
    await server_transport.close()


# --- WP4 [4B]: session-side driving of the consumer --------------------------
#
# The session starts exactly one timer driver on the first relay frame, so
# reconnects never arm a duplicate loop; on resume it re-announces by retiring
# the current incarnation, and connection loss/resume is observable through
# the connection listeners the TUI uses for its typed failure state.


def _bare_session(clock: Any = None, **consumer_kwargs: Any) -> Any:
    from chartreux.app_server.session import AppServerSession

    session = object.__new__(AppServerSession)
    session.presentation = _consumer(
        clock if clock is not None else _FakeClock(), **consumer_kwargs
    )
    session._presentation_stream_id = "s"
    session._presentation_timer_task = None
    session._presentation_timer_sleep = asyncio.sleep
    session._connection_listeners = []
    session._connection_lost = False
    session._closing = False
    return session


@pytest.mark.asyncio
async def test_first_relay_frame_starts_exactly_one_timer_driver() -> None:
    session = _bare_session()

    assert session._presentation_timer_task is None
    assert session._on_presentation_frame("s", 0, _page_frame(0, 0, 1, "x", 0)) is True
    first = session._presentation_timer_task
    assert first is not None
    # A current-generation delta passes through; the driver task is unchanged.
    assert session._on_presentation_frame("s", 0, _delta_frame(0)) is False
    # Further frames reuse the same driver task: no duplicate timer loop.
    assert session._presentation_timer_task is first
    session._stop_presentation_timer_driver()
    assert session._presentation_timer_task is None
    await asyncio.gather(first, return_exceptions=True)
    assert first.cancelled() or first.done()


def test_reannounce_retires_the_current_incarnation_exactly_once() -> None:
    session = _bare_session()
    stream_id = "s"
    _install_snapshot(session.presentation, stream_id, 0, ["a"], 7)
    stream = session.presentation.stream(stream_id)
    assert stream.phase is StreamPhase.INSTALLED
    assert stream.current == 0

    session._reannounce_presentation()

    assert stream.phase is StreamPhase.AWAITING
    assert stream.current is None
    (timer,) = stream.timers
    assert timer.kind is TimerKind.REGISTRATION_AWAIT
    assert timer.generation == 0
    # A second resume re-announce is a no-op: no second await deadline.
    session._reannounce_presentation()
    assert stream.phase is StreamPhase.AWAITING
    assert len(stream.timers) == 1


def test_reannounce_after_terminal_failure_is_a_no_op() -> None:
    session = _bare_session()
    stream_id = "s"
    _install_snapshot(session.presentation, stream_id, 0, ["a"], 7)
    stream = session.presentation.stream(stream_id)
    _receive(session.presentation, stream_id, _terminal_frame(0, "aborted"))
    assert stream.terminal is not None

    session._reannounce_presentation()

    assert stream.terminal is not None
    assert stream.phase is StreamPhase.TERMINAL
    assert stream.timers == ()


def test_connection_listeners_observe_loss_and_resume() -> None:
    session = _bare_session()
    events: list[tuple[bool, str]] = []
    listener = lambda lost, reason: events.append((lost, reason))
    session.add_connection_listener(listener)

    session._notify_connection(True, "connection reset")
    assert session.connection_lost is True
    session._notify_connection(False, "")
    assert session.connection_lost is False
    assert events == [(True, "connection reset"), (False, "")]

    session.remove_connection_listener(listener)
    session._notify_connection(True, "again")
    assert events == [(True, "connection reset"), (False, "")]


@pytest.mark.asyncio
async def test_timer_driver_fires_due_generation_scoped_timers() -> None:
    # A consumer with a short install deadline on a real clock proves the
    # single driver task actually advances fire_due.
    session = _bare_session(clock=time.monotonic, registration_to_install_timeout=0.05)
    stream_id = "s"
    session._on_presentation_frame(stream_id, 0, _page_frame(0, 0, 2, "a"))
    task = session._presentation_timer_task
    assert task is not None
    await asyncio.sleep(0.5)
    stream = session.presentation.stream(stream_id)
    assert stream.attempts == 1
    assert stream.phase is StreamPhase.AWAITING
    session._closing = True
    session._stop_presentation_timer_driver()
    await asyncio.gather(task, return_exceptions=True)


def test_consumer_listener_observes_frames_and_timer_fires() -> None:
    clock = _FakeClock()
    seen: list[tuple[str, StreamPhase]] = []
    consumer = PresentationConsumer(
        clock=clock, listener=lambda sid, stream: seen.append((sid, stream.phase))
    )
    _install_snapshot(consumer, "s", 0, ["a"], 3)
    assert seen[-1] == ("s", StreamPhase.INSTALLED)
    _receive(consumer, "s", _retire_frame(0))
    assert seen[-1] == ("s", StreamPhase.AWAITING)
    clock.advance(31.0)
    consumer.fire_due("s")
    stream = consumer.stream("s")
    assert stream.terminal is not None
    assert seen[-1] == ("s", StreamPhase.TERMINAL)
