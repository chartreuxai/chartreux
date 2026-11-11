"""WP6 [6C]: the nineteen presentation fault traces in both topologies.

Every trace runs the same seeded script through the in-process escape and the
spawned, egress-blocked replay child, compares the normalized observation
streams (they must be identical), and then asserts the trace's own oracle on
the common stream. The traces are the design's presentation fault set
(design.md:474-555); the production components they drive live in
``tests/differential/presentation.py``.

Trace 16 additionally validates the registration schedule against its
published worst-case per-attempt timeline at worst-case injected delays.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from chartreux.app_server.events import (
    REGISTRATION_AWAIT_TIMEOUT,
    REGISTRATION_BACKOFF_SECONDS,
    REGISTRATION_TO_INSTALL_TIMEOUT,
)
from chartreux.app_server.relay import CONTROL_PROPAGATION_TIMEOUT, encoded_frame_size
from tests.differential import ledger
from tests.differential.presentation import (
    PresentationDriver,
    build_history,
    control_request_frame,
    control_result_frame,
    entry_added_frame,
    heartbeat_frame,
    history_entries,
    history_payload,
    page_frame,
    retire_frame,
)
from tests.differential.replay import (
    COMPONENT_PRESENTATION,
    canonical_json,
    digest,
    in_process_component,
    normalize_frames,
    subprocess_component,
)

IN_PROCESS = "in_process"
SPAWNED = "spawned"
TOPOLOGIES = (IN_PROCESS, SPAWNED)
_TRACE_TIMEOUT = 90.0
_CLOCK = 100.0
_SESSION = "session"


def _host(topology: str):
    if topology == IN_PROCESS:
        return in_process_component(
            component=COMPONENT_PRESENTATION, timeout=_TRACE_TIMEOUT
        )
    return subprocess_component(
        component=COMPONENT_PRESENTATION, timeout=_TRACE_TIMEOUT
    )


def _run(
    topology: str, init: dict[str, Any], ops: list[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    with _host(topology) as channel:
        driver = PresentationDriver(channel)
        ready = driver.initialize(init)
        assert ready["op"] == "init"
        return driver.run(ops)


def _both(
    init: dict[str, Any], ops: list[dict[str, Any]]
) -> tuple[dict[str, Any], ...]:
    """Run one trace in both topologies and return the common observation stream."""
    runs = {topology: _run(topology, init, ops) for topology in TOPOLOGIES}
    normalized = {topology: normalize_frames(runs[topology]) for topology in TOPOLOGIES}
    assert normalized[IN_PROCESS] == normalized[SPAWNED]
    assert digest(normalized[IN_PROCESS]) == digest(normalized[SPAWNED])
    ledger.record(
        ledger.FAMILY_PRESENTATION, {"init": init, "ops": ops}, normalized[IN_PROCESS]
    )
    return normalized[IN_PROCESS]


def _stream(obs: dict[str, Any], stream_id: str = _SESSION) -> dict[str, Any]:
    streams = obs["streams"]
    assert isinstance(streams, dict)
    stream = streams[stream_id]
    assert isinstance(stream, dict)
    return stream


def _events(obs: dict[str, Any]) -> list[dict[str, Any]]:
    events = obs["events"]
    assert isinstance(events, list)
    return [event for event in events if isinstance(event, dict)]


def _of_kind(obs: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [event for event in _events(obs) if event.get("kind") == kind]


def _wire(obs: dict[str, Any]) -> list[dict[str, Any]]:
    return _of_kind(obs, "wire")


def _install(obs: dict[str, Any], stream_id: str = _SESSION) -> dict[str, Any] | None:
    install = _stream(obs, stream_id)["install"]
    assert install is None or isinstance(install, dict)
    return install


def _installed(obs: dict[str, Any], stream_id: str = _SESSION) -> dict[str, Any]:
    install = _install(obs, stream_id)
    assert install is not None
    return install


def _install_entries(
    obs: dict[str, Any], stream_id: str = _SESSION
) -> list[dict[str, str]] | None:
    install = _install(obs, stream_id)
    if install is None:
        return None
    entries: list[dict[str, str]] = []
    for payload in install["payloads"]:
        assert isinstance(payload, str)
        entries.extend(json.loads(payload)["entries"])
    return entries


def _expected(history: tuple[tuple[str, str], ...]) -> list[dict[str, str]]:
    return [{"id": entry_id, "text": text} for entry_id, text in history]


def _content_digest(entries: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        canonical_json({"entries": entries}).encode("utf-8")
    ).hexdigest()


def _projection_entries(obs: dict[str, Any]) -> list[dict[str, str]]:
    projection = obs["projection"]
    assert isinstance(projection, dict)
    entries = projection["entries"]
    assert isinstance(entries, list)
    return entries


def _init(streams: tuple[str, ...] = (_SESSION,), **overrides: Any) -> dict[str, Any]:
    init: dict[str, Any] = {"clock": _CLOCK, "streams": list(streams)}
    init.update(overrides)
    return init


def _snapshot(
    payloads: list[str], boundary: int | None, stream_id: str = _SESSION, **extra: Any
) -> dict[str, Any]:
    op: dict[str, Any] = {
        "kind": "snapshot",
        "stream": stream_id,
        "payloads": payloads,
        "boundary": boundary,
    }
    op.update(extra)
    return op


def _delta(
    frame: dict[str, Any], stream_id: str = _SESSION, **extra: Any
) -> dict[str, Any]:
    op: dict[str, Any] = {"kind": "delta", "stream": stream_id, "frame": frame}
    op.update(extra)
    return op


def _tick(seconds: float, stream_id: str = _SESSION) -> dict[str, Any]:
    return {"kind": "tick", "stream": stream_id, "seconds": seconds}


def _retire(stream_id: str = _SESSION, **extra: Any) -> dict[str, Any]:
    op: dict[str, Any] = {"kind": "retire", "stream": stream_id}
    op.update(extra)
    return op


def _op(kind: str, stream_id: str = _SESSION, **fields: Any) -> dict[str, Any]:
    op: dict[str, Any] = {"kind": kind, "stream": stream_id}
    op.update(fields)
    return op


def _stamped(frame: dict[str, Any], generation: int) -> dict[str, Any]:
    return {**frame, "relay_incarnation": generation}


def _big_history(count: int = 3, size: int = 900) -> tuple[tuple[str, str], ...]:
    return tuple((f"entry-{index}", f"{index}" + "x" * size) for index in range(count))


@pytest.mark.timeout(240)
def test_trace_01_no_loss_replay_and_patch_vs_full_parity(replay_seed: int) -> None:
    history = build_history(replay_seed, 4)
    entries = history_entries(history)

    full = _both(_init(), [_snapshot([history_payload(history)], len(history))])
    assert _install_entries(full[-1]) == _expected(history)

    patch = _both(
        _init(),
        [
            _snapshot([history_payload(())], 0),
            *[
                _delta(entry_added_frame(entries[index], index + 1))
                for index in range(len(entries))
            ],
        ],
    )
    final = patch[-1]
    assert _projection_entries(final) == _expected(history)
    assert final["projection"]["event_id"] == len(entries)
    # No-loss replay: every event is applied exactly once, contiguously.
    applied = [
        event["event_id"]
        for frame in patch
        for event in _wire(frame)
        if "event_id" in event
    ]
    assert applied == list(range(1, len(entries) + 1))
    # Patch-vs-full parity: the same history, from either path.
    assert _content_digest(_install_entries(full[-1]) or []) == _content_digest(
        _projection_entries(final)
    )


@pytest.mark.timeout(240)
def test_trace_02_producer_outbox_shed_through_invalidate_and_register(
    replay_seed: int,
) -> None:
    history = _big_history(3)
    entries = history_entries(history)
    frames_to_send = [
        entry_added_frame(entries[index], index + 1) for index in range(3)
    ]
    largest = max(encoded_frame_size(_stamped(frame, 0)) for frame in frames_to_send)
    # Exactly two deltas fit the stream budget; the final one is shed.
    init = _init(hop={"presentation_per_stream": 2 * largest + 1})
    ops = [
        _snapshot([history_payload(())], 0),
        # One op, no drain in between: the last delta is shed, then the
        # producer quietens and must invalidate and re-register.
        _op("batch", frames=frames_to_send),
        _retire(),
        _snapshot([history_payload(history)], 3),
    ]
    frames = _both(init, ops)

    dispositions = [
        (event["disposition"], event.get("shed_reason"))
        for event in _of_kind(frames[1], "enqueue")
    ]
    assert dispositions == [
        ("queued", None),
        ("queued", None),
        ("shed", "stream presentation budget exhausted"),
    ]
    assert _projection_entries(frames[1]) == _expected(history[:2])
    assert _of_kind(frames[2], "retire")[0]["invalidated"] == 0
    assert _install(frames[3]) is not None
    assert _install_entries(frames[3]) == _expected(history)
    assert _stream(frames[3])["announced"] == 1
    assert _stream(frames[3])["phase"] == "installed"


@pytest.mark.timeout(240)
def test_trace_03_consumer_detected_sequence_gap_invalidates_and_registers(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 3)
    entries = history_entries(history)
    ops = [
        _snapshot([history_payload(())], 0),
        _delta(entry_added_frame(entries[0], 1)),
        _delta(entry_added_frame(entries[2], 3)),
        _retire(),
        _snapshot([history_payload(history)], 3),
        _delta(entry_added_frame(entries[1], 2)),
    ]
    frames = _both(_init(), ops)

    gaps = [event for event in _wire(frames[2]) if "gap" in event]
    assert len(gaps) == 1
    assert "App-server event gap: expected 2, received 3" in str(gaps[0]["gap"])
    assert frames[2]["projection"]["event_id"] == 1
    # Invalidate and register: the replacement snapshot installs atomically.
    assert _install_entries(frames[4]) == _expected(history)
    assert _stream(frames[4])["announced"] == 1
    # Contiguity is restored: the formerly missing event applies once, late,
    # and the earlier applied event is not replayed or duplicated.
    assert frames[5]["projection"]["event_id"] == 2
    assert _projection_entries(frames[5]) == [
        {"id": "entry-0", "text": history[0][1]},
        {"id": "entry-1", "text": history[1][1]},
    ]


@pytest.mark.timeout(240)
def test_trace_04_hop_presentation_overflow_invalidates_and_registers(
    replay_seed: int,
) -> None:
    history = _big_history(2)
    entries = history_entries(history)
    frames_to_send = [
        entry_added_frame(entries[index], index + 1) for index in range(2)
    ]
    largest = max(encoded_frame_size(_stamped(frame, 0)) for frame in frames_to_send)
    # The hop's aggregate budget fits exactly one of the two deltas.
    init = _init(hop={"presentation_per_hop": largest + 1})
    ops = [
        _snapshot([history_payload(())], 0),
        _op("batch", frames=frames_to_send),
        _retire(),
        _snapshot([history_payload(history)], 2),
    ]
    frames = _both(init, ops)

    dispositions = [
        (event["disposition"], event.get("shed_reason"))
        for event in _of_kind(frames[1], "enqueue")
    ]
    assert dispositions == [
        ("queued", None),
        ("shed", "hop presentation budget exhausted"),
    ]
    assert _install_entries(frames[3]) == _expected(history)


@pytest.mark.timeout(240)
def test_trace_05_snapshot_capture_overflow_retries_then_exhausts(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 2)
    oversized = "x" * 600
    init = _init(streams=("retry", "exhaust"), consumer={"page_max_bytes": 400})
    ops = [
        _snapshot([oversized], 1, stream_id="retry"),
        _tick(0.5, stream_id="retry"),
        _retire("retry"),
        _snapshot([history_payload(history)], 2, stream_id="retry"),
        _snapshot([oversized], 1, stream_id="exhaust"),
        _tick(0.5, stream_id="exhaust"),
        _retire("exhaust"),
        _snapshot([oversized], 1, stream_id="exhaust"),
        _tick(1.0, stream_id="exhaust"),
        _retire("exhaust"),
        _snapshot([oversized], 1, stream_id="exhaust"),
        _tick(100.0, stream_id="exhaust"),
    ]
    frames = _both(init, ops)

    # A bounded failed attempt leaves a backoff timer, then the retry installs.
    assert _stream(frames[0], "retry")["attempts"] == 1
    assert [timer["kind"] for timer in _stream(frames[0], "retry")["timers"]] == [
        "backoff"
    ]
    assert _stream(frames[1], "retry")["attempts"] == 1
    assert [timer["kind"] for timer in _stream(frames[1], "retry")["timers"]] == [
        "registration_await"
    ]
    assert _stream(frames[3], "retry")["attempts"] == 0
    assert _install(frames[3], "retry") is not None

    # Sustained failure exhausts the bounded attempts, with no restart storm.
    fired = [
        (event["kind"], timer["kind"], timer["attempt"])
        for event in _of_kind(frames[5], "tick")
        for timer in event["fired"]
    ]
    assert ("tick", "backoff", 1) in fired
    exhausted = _stream(frames[-1], "exhaust")
    assert exhausted["attempts"] == 3
    assert exhausted["phase"] == "terminal"
    assert "registration attempts exhausted" in str(exhausted["terminal"])
    assert exhausted["timers"] == []
    assert _of_kind(frames[-1], "tick")[0]["fired"] == []


@pytest.mark.timeout(240)
def test_trace_06_snapshot_enqueue_exhaustion_and_partial_delivery(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 2)
    filler = control_request_frame("filler")
    partial_pages = [
        page_frame(0, 3, history_payload(history[:1])),
        page_frame(1, 3, history_payload(history[1:2])),
    ]

    # Enqueue exhaustion: the page cannot take control capacity at all.
    exhausted = _both(
        _init(streams=("exhausted",), hop={"control_capacity": 100}),
        [
            _op("enqueue", "exhausted", frames=[filler]),
            _snapshot([history_payload(history)], 2, stream_id="exhausted"),
            _tick(8.0, stream_id="exhausted"),
        ],
    )
    rejected = _of_kind(exhausted[1], "enqueue")
    assert rejected[0]["rejected"] == "ControlCapacityExhausted"
    assert _install(exhausted[1], "exhausted") is None
    assert _stream(exhausted[1], "exhausted")["phase"] == "installing"
    after_deadline = _stream(exhausted[2], "exhausted")
    assert after_deadline["attempts"] == 1
    assert after_deadline["phase"] == "awaiting"

    # Delivery failure after a partial page set: nothing partial installs.
    partial = _both(
        _init(streams=("partial",)),
        [
            _op("snapshot", "partial", frames=partial_pages),
            _op("terminate", "partial", reason="snapshot delivery failed"),
            _op("write", "partial"),
        ],
    )
    assert _stream(partial[0], "partial")["phase"] == "installing"
    assert _install(partial[0], "partial") is None
    assert _install(partial[2], "partial") is None
    assert "relay terminal notice: snapshot delivery failed" in str(
        _stream(partial[2], "partial")["terminal"]
    )


@pytest.mark.timeout(240)
def test_trace_07_empty_missing_and_partial_snapshots(replay_seed: int) -> None:
    history = build_history(replay_seed, 3)
    init = _init(streams=("empty", "missing", "partial"))
    partial_pages = [
        page_frame(0, 3, history_payload(history[:1])),
        page_frame(1, 3, history_payload(history[1:2])),
    ]
    ops = [
        _snapshot([history_payload(())], 0, stream_id="empty"),
        _op("snapshot", "partial", frames=partial_pages),
        _tick(8.0, "missing"),
        _tick(0.5, "missing"),
        _tick(30.0, "missing"),
    ]
    frames = _both(init, ops)

    empty = _stream(frames[0], "empty")
    assert empty["phase"] == "installed"
    assert empty["install"] is not None
    assert empty["install"]["boundary"] == 0
    assert empty["install"]["pages"] == 1
    assert empty["attempts"] == 0

    after_deadline = _stream(frames[2], "missing")
    assert after_deadline["attempts"] == 1
    assert after_deadline["phase"] == "awaiting"
    backoff = [timer["kind"] for timer in after_deadline["timers"]]
    assert backoff == ["backoff"]
    final_missing = _stream(frames[4], "missing")
    assert final_missing["phase"] == "terminal"
    assert "registration await expired" in str(final_missing["terminal"])

    assert _install(frames[-1], "partial") is None
    assert _stream(frames[-1], "partial")["phase"] == "terminal"
    assert _stream(frames[-1], "empty")["terminal_transitions"] == 0


@pytest.mark.timeout(240)
def test_trace_08_mutation_racing_snapshot_capture_is_never_lost(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 3)
    entries = history_entries(history)
    ops = [
        _snapshot([history_payload(history)], 3, stream_id="in_snapshot"),
        _snapshot(
            [history_payload(history[:1]), history_payload(history[1:2])],
            2,
            stream_id="after_boundary",
        ),
        _delta(entry_added_frame(entries[2], 1), stream_id="after_boundary"),
    ]
    frames = _both(_init(streams=("in_snapshot", "after_boundary")), ops)

    assert _install_entries(frames[0], "in_snapshot") == _expected(history)

    # Page ordering and atomic final-page installation: nothing installs
    # before the boundary-bearing final page.
    transitions = frames[1]["transitions"]
    assert [transition["phase"] for transition in transitions] == [
        "installing",
        "installed",
    ]
    assert transitions[0]["install"] is None
    assert _installed(frames[1], "after_boundary")["pages"] == 2
    assert _install_entries(frames[1], "after_boundary") == _expected(history[:2])

    # The racing mutation appears in the later delta, exactly once.
    assert _projection_entries(frames[2]) == [{"id": "entry-2", "text": history[2][1]}]
    installed = _install_entries(frames[2], "after_boundary") or []
    assert {"id": "entry-2", "text": history[2][1]} not in installed
    combined = installed + _projection_entries(frames[2])
    assert [entry["id"] for entry in combined] == ["entry-0", "entry-1", "entry-2"]


@pytest.mark.timeout(240)
def test_trace_09_re_registration_race_keeps_the_first_winner(replay_seed: int) -> None:
    history = build_history(replay_seed, 2)
    entries = history_entries(history)
    ops = [
        _snapshot([history_payload(())], 0),
        _retire(),
        _tick(1.0),
        _op("inject", wire=_stamped(entry_added_frame(entries[0], 1), 0)),
        _op("inject", wire=retire_frame(0)),
        _snapshot([history_payload(history)], 2),
        _tick(30.0),
        _op("inject", wire=_stamped(page_frame(0, 1, history_payload(()), 0), 0)),
    ]
    frames = _both(_init(), ops)

    stale = _wire(frames[3])
    assert stale[0]["consumed"] is True
    assert stale[0]["generation"] == 0
    assert frames[3]["projection"]["event_id"] == 0

    duplicates = _wire(frames[4])
    assert duplicates[0]["consumed"] is True
    assert _stream(frames[4])["phase"] == "awaiting"

    accepted = _stream(frames[5])
    assert accepted["announced"] == 1
    assert accepted["phase"] == "installed"
    assert [timer["kind"] for timer in accepted["timers"]] == []
    assert _install_entries(frames[5]) == _expected(history)

    # The old await deadline passed long ago; it was disarmed at acceptance.
    assert _of_kind(frames[6], "tick")[0]["fired"] == []
    assert _stream(frames[6])["phase"] == "installed"
    assert _stream(frames[6])["terminal_transitions"] == 0

    late = _wire(frames[7])
    assert late[0]["consumed"] is True
    assert _install(frames[7]) == _install(frames[5])


@pytest.mark.timeout(240)
def test_trace_10_producer_eof_reaches_the_consumer_within_the_bound(
    replay_seed: int,
) -> None:
    init = _init(streams=("confirmed", "unconfirmed"))
    ops = [
        _snapshot([history_payload(())], 0, stream_id="confirmed"),
        _snapshot([history_payload(())], 0, stream_id="unconfirmed"),
        _op("terminate", "confirmed", reason="producer eof"),
        _op("write", "confirmed"),
        _op("terminate", "unconfirmed", reason="producer eof"),
        _tick(2.0, stream_id="unconfirmed"),
        _op("settle", "unconfirmed"),
    ]
    frames = _both(init, ops)

    written = _of_kind(frames[3], "write")[0]
    assert written["confirmed"] is True
    confirmed = _stream(frames[3], "confirmed")
    assert confirmed["phase"] == "terminal"
    assert "producer eof" in str(confirmed["terminal"])
    assert confirmed["peer"]["terminal"] is not None
    assert frames[3]["fake_clock"] == _CLOCK

    failed = _of_kind(frames[6], "settle")[0]
    assert failed["confirmed"] is False
    assert "was not confirmed within 2.0s" in str(failed["hop_failure"])
    unconfirmed = _stream(frames[6], "unconfirmed")
    assert unconfirmed["phase"] == "terminal"
    assert "connection failed within the propagation bound" in str(
        unconfirmed["terminal"]
    )
    # Bounded: the failure lands exactly at the control-propagation deadline.
    assert frames[6]["fake_clock"] == _CLOCK + CONTROL_PROPAGATION_TIMEOUT


@pytest.mark.timeout(240)
def test_trace_11_heartbeat_enqueue_failure_closes_downstream(replay_seed: int) -> None:
    filler = control_request_frame("heartbeat-filler")
    heartbeat = heartbeat_frame()
    capacity = (
        encoded_frame_size(_stamped(filler, 0))
        + encoded_frame_size(_stamped(heartbeat, 0))
        - 1
    )
    init = _init(
        streams=("producer_fail", "peer_fail"), hop={"control_capacity": capacity}
    )
    ops = [
        _snapshot([history_payload(())], 0, stream_id="producer_fail"),
        _snapshot([history_payload(())], 0, stream_id="peer_fail"),
        _op("enqueue", "producer_fail", frames=[filler]),
        _op("enqueue", "producer_fail", frames=[heartbeat]),
        _op("discard", "producer_fail", reason="heartbeat enqueue failed"),
        _op("eof", "producer_fail", reason="heartbeat enqueue failed"),
        _op("peer_enqueue", "peer_fail", frame=filler),
        _op("peer_enqueue", "peer_fail", frame=heartbeat),
        _op("eof", "peer_fail", reason="heartbeat enqueue failed"),
    ]
    frames = _both(init, ops)

    assert _of_kind(frames[3], "enqueue")[0]["rejected"] == "ControlCapacityExhausted"
    discarded = _of_kind(frames[4], "discard")[0]
    assert discarded["discarded"] >= 0
    assert _stream(frames[4], "producer_fail")["producer"]["terminal"] is not None
    producer = _stream(frames[5], "producer_fail")
    assert producer["phase"] == "terminal"
    assert "heartbeat enqueue failed" in str(producer["terminal"])

    assert (
        _of_kind(frames[7], "peer_enqueue")[0]["rejected"] == "ControlCapacityExhausted"
    )
    peer = _stream(frames[8], "peer_fail")
    assert peer["phase"] == "terminal"
    assert peer["terminal_transitions"] == 1
    assert peer["producer"]["generation"] == 0


@pytest.mark.timeout(240)
def test_trace_12_idle_eof_and_registration_deadline_races_settle_once(
    replay_seed: int,
) -> None:
    init = _init(streams=("idle", "deadline", "eof_first"), hop={"idle_timeout": 10.0})
    ops = [
        _snapshot([history_payload(())], 0, stream_id="idle"),
        _snapshot([history_payload(())], 0, stream_id="eof_first"),
        # The first tick is already past the uninstalled stream's install
        # deadline, so that deadline fires while idle is being kept alive.
        _tick(9.9, "idle"),
        _delta(heartbeat_frame(), "idle"),
        _tick(0.1, "idle"),
        _tick(9.8, "idle"),
        _tick(0.2, "idle"),
        _op("terminate", "idle", reason="idle expired"),
        _op("write", "idle"),
        _op("eof", "deadline", reason="producer eof"),
        _op("eof", "eof_first", reason="producer eof"),
        _tick(100.0, "idle"),
    ]
    frames = _both(init, ops)

    # A current-incarnation frame resets idle; the deadline is then enforced.
    assert _stream(frames[3], "idle")["idle_expired"] is False
    assert _stream(frames[4], "idle")["idle_expired"] is False
    assert _stream(frames[5], "idle")["idle_expired"] is False
    assert _stream(frames[6], "idle")["idle_expired"] is True

    # The registration deadline fired exactly once, into a bounded retry.
    expired_attempt = _stream(frames[2], "deadline")
    assert expired_attempt["attempts"] == 1
    assert expired_attempt["phase"] == "awaiting"
    # Then the backoff released it into the registration await, and the pipe
    # EOF raced that await.
    assert _stream(frames[5], "deadline")["timers"][0]["kind"] == "registration_await"

    # The EOF notice carries the retired incarnation, and the consumer is
    # awaiting its replacement: the retired generation's teardown is stale, so
    # it is discarded and cannot pre-empt the deadline that governs the wait.
    race = _of_kind(frames[9], "wire")
    assert [event["stream"] for event in race] == ["deadline"]
    assert race[0]["method"] == "relay/terminal"
    assert race[0]["consumed"] is True
    assert _stream(frames[9], "deadline")["phase"] == "awaiting"
    assert _stream(frames[9], "deadline")["terminal"] is None
    # The installed stream's EOF terminates it directly.
    assert _stream(frames[10], "eof_first")["terminal"] is not None

    for stream_id in ("idle", "deadline", "eof_first"):
        final = _stream(frames[-1], stream_id)
        assert final["phase"] == "terminal"
        assert final["terminal_transitions"] == 1
        assert final["timers"] == []
    # One terminal disposition each: the idle deadline, the await deadline
    # that the stale EOF could not pre-empt, and the direct EOF notice.
    assert _stream(frames[-1], "idle")["terminal"] == (
        "relay terminal notice: idle expired"
    )
    assert _stream(frames[-1], "deadline")["terminal"] == (
        "registration await expired for incarnation 0"
    )
    assert _of_kind(frames[-1], "tick")[0]["fired"] == [
        {"kind": "registration_await", "generation": 0, "attempt": 1}
    ]


@pytest.mark.timeout(240)
def test_trace_13_idle_reset_stale_frames_and_sequence_free_heartbeats(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 1)
    ops = [
        _snapshot([history_payload(())], 0),
        _tick(9.9),
        _delta(heartbeat_frame()),
        _tick(9.9),
        _retire(),
        _tick(9.9),
        _op("inject", wire=_stamped(heartbeat_frame(), 0)),
        _tick(0.3),
        _snapshot([history_payload(history)], 1),
        _delta(heartbeat_frame()),
    ]
    frames = _both(_init(hop={"idle_timeout": 10.0}), ops)

    # A current-incarnation frame resets the idle deadline.
    assert _stream(frames[2])["idle_expired"] is False
    assert _stream(frames[3])["idle_expired"] is False
    # A stale frame does not reset it.
    stale = _wire(frames[6])
    assert stale[0]["relay"] == "consumed"
    assert stale[0]["generation"] == 0
    assert _stream(frames[6])["peer"]["invalidated"] == 1
    assert _stream(frames[7])["idle_expired"] is True

    # Heartbeats carry no sequence and create no false gap.
    for index in (2, 9):
        heartbeats = [
            event
            for event in _wire(frames[index])
            if event["method"] == "usage/updated"
        ]
        assert heartbeats
        assert heartbeats[0]["projection"] == "not_a_projection_event"
    for index in (2, 9):
        assert frames[index]["projection"]["event_id"] == 0
        assert not any("gap" in event for event in _wire(frames[index]))
    assert _stream(frames[8])["phase"] == "installed"


@pytest.mark.timeout(240)
def test_trace_14_sustained_overload_exhausts_bounded_attempts(
    replay_seed: int,
) -> None:
    bad = _snapshot(["x" * 20], None)
    ops = [
        bad,
        _tick(0.5),
        _retire(),
        _snapshot(["x" * 20], None),
        _tick(1.0),
        _retire(),
        _snapshot(["x" * 20], None),
        _tick(100.0),
    ]
    init = _init(consumer={"max_attempts": 3, "backoff_schedule": [0.5, 1.0]})
    frames = _both(init, ops)

    breakpoints = [
        (event["fired"][0]["kind"], event["fired"][0]["attempt"])
        for event in (_of_kind(frames[1], "tick")[0], _of_kind(frames[4], "tick")[0])
    ]
    assert breakpoints == [("backoff", 1), ("backoff", 2)]

    final = _stream(frames[-1])
    assert final["attempts"] == 3
    assert final["phase"] == "terminal"
    assert "registration attempts exhausted" in str(final["terminal"])
    assert final["timers"] == []
    assert final["producer"]["generation"] == 2
    assert _of_kind(frames[-1], "tick")[0]["fired"] == []
    assert final["terminal_transitions"] == 1


@pytest.mark.timeout(240)
def test_trace_15_approvals_and_results_survive_a_presentation_refresh(
    replay_seed: int,
) -> None:
    history = _big_history(3)
    entries = history_entries(history)
    request = control_request_frame("approval-1")
    result = control_result_frame("approval-1")
    presentation_budget = 1500
    init = _init(hop={"presentation_per_stream": presentation_budget})
    ops = [
        _snapshot([history_payload(())], 0),
        _op(
            "batch",
            frames=[
                entry_added_frame(entries[0], 1),
                entry_added_frame(entries[1], 2),
                entry_added_frame(entries[2], 3),
            ],
        ),
        _retire(),
        _op("batch", frames=[request, result]),
        _snapshot([history_payload(history)], 3),
    ]
    frames = _both(init, ops)

    presentation = _of_kind(frames[1], "enqueue")
    assert any(event["disposition"] == "shed" for event in presentation)

    control = _of_kind(frames[3], "enqueue")
    assert [(event["class"], event["disposition"]) for event in control] == [
        ("control", "queued"),
        ("control", "queued"),
    ]
    control_wire = _wire(frames[3])
    assert [event["consumed"] for event in control_wire] == [False, False]
    assert not any("protocol_error" in event for event in control_wire)
    # The lane carried them while the replacement snapshot was not installed.
    assert [transition["phase"] for transition in frames[3]["transitions"]] == [
        "installing",
        "installing",
    ]
    assert _install_entries(frames[4]) == _expected(history)


def _worst_case_per_attempt_seconds() -> float:
    """The published worst-case per-attempt registration timeline.

    Invalidation propagation (2 s per hop) + backoff + snapshot capture +
    return hops + the install deadline. This is the timeline the consumer's
    registration-await bound must cover — not the producer budget alone.
    """
    hop_count = 2.0
    invalidation = hop_count * CONTROL_PROPAGATION_TIMEOUT
    backoff = max(REGISTRATION_BACKOFF_SECONDS)
    capture = REGISTRATION_TO_INSTALL_TIMEOUT
    return_hops = hop_count * CONTROL_PROPAGATION_TIMEOUT
    install = REGISTRATION_TO_INSTALL_TIMEOUT
    return invalidation + backoff + capture + return_hops + install


@pytest.mark.timeout(240)
def test_trace_16_invalidation_propagation_and_schedule_bounds(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 2)
    worst_case = _worst_case_per_attempt_seconds()
    assert worst_case == 25.0
    assert REGISTRATION_AWAIT_TIMEOUT >= worst_case
    # The producer budget alone (3 x 8 + 0.5 + 1) is not the derivation.
    assert worst_case != REGISTRATION_TO_INSTALL_TIMEOUT * 3 + 0.5 + 1.0

    streams = (
        "notice_first",
        "announcement_first",
        "await_expiry",
        "acceptance_race",
        "unannounced",
        "schedule",
        "missing_announcement",
    )
    empty = [history_payload(())]
    full = [history_payload(history)]
    ops = [
        # Notice before the replacement announcement.
        _snapshot(empty, 0, stream_id="notice_first"),
        _retire("notice_first"),
        _snapshot(full, 2, stream_id="notice_first"),
        # Replacement announcement before the (delayed) notice.
        _snapshot(empty, 0, stream_id="announcement_first"),
        _retire("announcement_first", hold="relay/retire"),
        _snapshot(full, 2, stream_id="announcement_first"),
        _op("release_held", "announcement_first"),
        # Await expiry wins over a late replacement.
        _snapshot(empty, 0, stream_id="await_expiry"),
        _retire("await_expiry"),
        _tick(30.0, "await_expiry"),
        _snapshot(full, 2, stream_id="await_expiry"),
        # Acceptance racing await expiry, from the safe side.
        _snapshot(empty, 0, stream_id="acceptance_race"),
        _retire("acceptance_race"),
        _tick(29.9, "acceptance_race"),
        _snapshot(full, 2, stream_id="acceptance_race"),
        _tick(0.1, "acceptance_race"),
        # An incarnation the protocol never announced fails typed.
        _op(
            "inject",
            "unannounced",
            wire=_stamped(page_frame(0, 1, history_payload(()), 0), 5),
        ),
        # Worst-case injected delays: invalidation 4 s + capture 8 s + return
        # 4 s, then a fresh install inside the 8 s install deadline.
        _snapshot(empty, 0, stream_id="schedule"),
        _retire("schedule"),
        _tick(16.0, "schedule"),
        _op(
            "snapshot",
            "schedule",
            frames=[page_frame(0, 2, history_payload(history[:1]))],
        ),
        _tick(7.999, "schedule"),
        _op(
            "snapshot",
            "schedule",
            frames=[page_frame(1, 2, history_payload(history[1:]), 2)],
        ),
        # A missing replacement announcement expires the await deadline.
        _snapshot(empty, 0, stream_id="missing_announcement"),
        _retire("missing_announcement"),
        _tick(30.0, "missing_announcement"),
        _tick(100.0, "missing_announcement"),
    ]
    frames = _both(_init(streams=streams), ops)

    notice_first = _stream(frames[2], "notice_first")
    assert notice_first["announced"] == 1
    assert notice_first["phase"] == "installed"
    assert _install_entries(frames[2], "notice_first") == _expected(history)

    announcement_first = _stream(frames[5], "announcement_first")
    assert announcement_first["announced"] == 1
    assert announcement_first["phase"] == "installed"
    late = _stream(frames[6], "announcement_first")
    assert late["phase"] == "installed"
    assert late["terminal_transitions"] == 0
    assert _install(frames[6], "announcement_first") == _install(
        frames[5], "announcement_first"
    )
    released = _wire(frames[6])
    assert released[0]["consumed"] is True

    expired = _stream(frames[9], "await_expiry")
    assert expired["phase"] == "terminal"
    assert "registration await expired for incarnation 0" in str(expired["terminal"])
    assert frames[9]["fake_clock"] == _CLOCK + REGISTRATION_AWAIT_TIMEOUT
    # The late replacement installs nothing new: generation 0 stays last.
    assert _installed(frames[10], "await_expiry")["generation"] == 0
    assert _stream(frames[10], "await_expiry")["terminal"] == expired["terminal"]

    raced = _stream(frames[14], "acceptance_race")
    assert raced["phase"] == "installed"
    assert raced["announced"] == 1
    assert _of_kind(frames[15], "tick")[0]["fired"] == []
    assert _stream(frames[15], "acceptance_race")["terminal_transitions"] == 0

    unannounced = _wire(frames[16])
    assert "frame for unannounced incarnation 5" in str(
        unannounced[0]["protocol_error"]
    )
    assert _stream(frames[16], "unannounced")["install"] is None

    # Schedule: exactly one install deadline armed at acceptance, fired never.
    accepted = _stream(frames[20], "schedule")
    assert accepted["announced"] == 1
    assert accepted["phase"] == "installing"
    assert [
        (timer["kind"], timer["generation"], timer["deadline"])
        for timer in accepted["timers"]
    ] == [
        (
            "registration_to_install",
            1,
            frames[20]["fake_clock"] + REGISTRATION_TO_INSTALL_TIMEOUT,
        )
    ]
    assert _of_kind(frames[21], "tick")[0]["fired"] == []
    scheduled = _stream(frames[22], "schedule")
    assert scheduled["phase"] == "installed"
    assert scheduled["attempts"] == 0
    assert _install_entries(frames[22], "schedule") == _expected(history)
    elapsed = frames[22]["fake_clock"] - frames[19]["fake_clock"]
    assert elapsed <= worst_case
    assert _stream(frames[24], "schedule")["terminal_transitions"] == 0

    missing = _stream(frames[25], "missing_announcement")
    assert missing["phase"] == "terminal"
    assert "registration await expired" in str(missing["terminal"])
    assert (
        frames[25]["fake_clock"] - frames[24]["fake_clock"]
        == REGISTRATION_AWAIT_TIMEOUT
    )
    assert _stream(frames[26], "missing_announcement")["terminal_transitions"] == 1


@pytest.mark.timeout(240)
def test_trace_17_pipe_eof_and_relay_forwarding_failure(replay_seed: int) -> None:
    init = _init(streams=("eof", "forward_failure"), record_abort=True)
    ops = [
        _snapshot([history_payload(())], 0, stream_id="eof"),
        _snapshot([history_payload(())], 0, stream_id="forward_failure"),
        _op("terminate", "eof", reason="pipe eof"),
        _op("write", "eof"),
        _op("terminate", "forward_failure", reason="pipe eof"),
        _tick(2.0, "forward_failure"),
        _op("settle", "forward_failure"),
    ]
    frames = _both(init, ops)

    assert _of_kind(frames[3], "write")[0]["confirmed"] is True
    assert _stream(frames[3], "eof")["phase"] == "terminal"

    settle = _of_kind(frames[6], "settle")[0]
    assert settle["confirmed"] is False
    assert "was not confirmed within 2.0s" in str(settle["hop_failure"])
    failed = _stream(frames[6], "forward_failure")
    assert failed["phase"] == "terminal"
    assert failed["hop"]["closed"] is True
    assert failed["hop"]["failure"] is not None
    assert failed["peer_hop"]["failure"] is not None
    assert frames[6]["aborts"]
    assert "was not confirmed within 2.0s" in frames[6]["aborts"][0]
    assert frames[6]["fake_clock"] == _CLOCK + CONTROL_PROPAGATION_TIMEOUT


@pytest.mark.timeout(240)
def test_trace_18_abortive_stream_close_and_connection_blast_radius(
    replay_seed: int,
) -> None:
    history = build_history(replay_seed, 1)
    entries = history_entries(history)
    init = _init(
        streams=("session", "agent"), multiplexed=True, demux=True, record_abort=True
    )
    ops = [
        _snapshot([history_payload(())], 0, stream_id="session"),
        _snapshot([history_payload(())], 0, stream_id="agent"),
        _delta(entry_added_frame(entries[0], 1), stream_id="session"),
        _op("discard", "session", reason="abortive close"),
        _delta(entry_added_frame(entries[0], 1), stream_id="agent"),
        _op("batch", "session", frames=[control_request_frame("approval-1")]),
        _op("observe", "agent"),
        _op("demux_abortive_close", "agent"),
    ]
    frames = _both(init, ops)

    aborted = _stream(frames[3], "session")["producer"]
    assert aborted["terminal"] is not None
    assert _stream(frames[3], "session")["producer"]["terminal"].startswith(
        "relay stream 'session' is terminally closed: abortive close"
    )
    survivors = frames[6]["demux"]
    assert survivors == {
        "closed": False,
        "connection_error": None,
        "streams": {
            "session": "stream 'session' is terminally closed: abortive close",
            "agent": None,
        },
    }
    # The surviving stream keeps working, presentation and control alike.
    assert frames[4]["projection"]["event_id"] == 1
    rejected = _of_kind(frames[5], "enqueue")[0]
    assert rejected["rejected"] == "RelayStreamAborted"

    final = frames[7]
    assert final["demux"]["closed"] is False
    assert "shared connection aborted" in str(final["demux"]["connection_error"])
    assert final["demux"]["streams"]["agent"] is not None
    for stream_id in ("session", "agent"):
        stream = _stream(final, stream_id)
        assert stream["phase"] == "terminal"
        assert stream["terminal_transitions"] == 1
        assert stream["timers"] == []
        assert "shared connection aborted" in str(stream["terminal"])


@pytest.mark.timeout(240)
def test_trace_19_upstream_teardown_discards_later_frames(replay_seed: int) -> None:
    history = build_history(replay_seed, 1)
    entries = history_entries(history)
    init = _init(streams=("terminal", "teardown_lost"))
    ops = [
        _snapshot([history_payload(())], 0, stream_id="terminal"),
        _snapshot([history_payload(())], 0, stream_id="teardown_lost"),
        _op("terminate", "terminal", reason="upstream terminal"),
        _op("write", "terminal"),
        _delta(entry_added_frame(entries[0], 1), stream_id="terminal"),
        _op(
            "inject",
            "terminal",
            wire=_stamped(page_frame(0, 1, history_payload(history), 1), 1),
        ),
        _op("terminate", "teardown_lost", reason="upstream terminal"),
        _tick(2.0, "teardown_lost"),
        _op("settle", "teardown_lost"),
    ]
    frames = _both(init, ops)

    terminal = _stream(frames[3], "terminal")
    assert terminal["phase"] == "terminal"
    assert "upstream terminal" in str(terminal["terminal"])
    assert terminal["producer"]["generation"] == 0
    assert terminal["producer"]["terminal"] is not None
    assert terminal["peer"]["terminal"] is not None
    assert terminal["timers"] == []

    rejected = _of_kind(frames[4], "enqueue")[0]
    assert rejected["rejected"] == "RelayStreamAborted"
    late = _wire(frames[5])
    assert late[0]["consumed"] is True
    # Nothing new installs after the teardown: generation 0 stays last.
    assert _installed(frames[5], "terminal")["generation"] == 0
    assert _stream(frames[5], "terminal")["terminal_transitions"] == 1

    lost = _of_kind(frames[8], "settle")[0]
    assert lost["confirmed"] is False
    assert "was not confirmed within 2.0s" in str(lost["hop_failure"])
    bounded = _stream(frames[8], "teardown_lost")
    assert bounded["phase"] == "terminal"
    assert bounded["terminal_transitions"] == 1
    assert frames[8]["fake_clock"] == _CLOCK + CONTROL_PROPAGATION_TIMEOUT
