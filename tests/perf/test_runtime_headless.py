"""WP3b contract node: the real headless transport/projector scenario harness.

The node proves the harness the runtime cells measure through, not a stand-in:
the projector is live for the whole root turn, the server-to-client
notifications cross a real memory transport pair into a real client
projection, the backend receives the gateway's memoized request context, and
the P1 provenance pins hold on the real projector/client delta path. It also
pins the committed workload definitions (seed, message sizes, deterministic
response shape) that travel with the WP3a fixture hash.

In-suite contract nodes run shortened fixtures (plan §5); the full-scale cell
definitions are asserted here and executed only by the opt-in external
campaign runner. Negative controls keep the node from passing vacuously: the
egress guard must block a non-loopback connect, a mutated workload definition
must change its digest, and the projector snapshot must come from a live
projector.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import sys
from time import perf_counter

import pytest

from chartreux.app_server.models import PublicMessageEntry
from tests.perf._runtime_fixture import (
    FIXTURE_APPROX_TOKENS,
    FIXTURE_HASH,
    FIXTURE_MESSAGE_CHARS,
    FIXTURE_MESSAGE_COUNT,
    FIXTURE_RESPONSE_CHARS,
    FIXTURE_SEED,
    FIXTURE_TURNS,
    fixture_spec,
    response_content,
)
from tests.perf._runtime_scenario import (
    CELL_FRESH_HEADLESS,
    CELL_LOADED_HEADLESS,
    CONFIG_HOME_ENV,
    FULL_SCALES,
    LOADED_CHILD_AGENTS,
    LOADED_TOTAL_AGENTS,
    PROVIDER_CREDENTIAL_NAMES,
    ROOT_TURNS,
    SCENARIO_CELLS,
    SCENARIO_METRIC_VERSION,
    SHORTENED_SCALE,
    WARMUP_TURNS,
    WORKLOAD_HASHES,
    DeltaDumpLog,
    EgressGuard,
    HeadlessScenario,
    assert_fingerprint_equality,
    build_headless_scenario,
    instrument_delta_dumps,
    projected_history_fingerprint,
    response_chunks,
    root_user_content,
    workload_spec,
)

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the headless runtime cells are Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_SCENARIO_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_ENTRY_UPDATED = "history/entryUpdated"
_PER_SIDE_DUMPS = 2
_EgressProbe = "192.0.2.1"


def _assert_workload_definitions() -> None:
    """The committed cell definitions are deterministic and hash-pinned."""
    assert set(SCENARIO_CELLS) == {CELL_FRESH_HEADLESS, CELL_LOADED_HEADLESS}

    fresh = workload_spec(CELL_FRESH_HEADLESS)
    loaded = workload_spec(CELL_LOADED_HEADLESS)
    fixture = fixture_spec()
    for spec in (fresh, loaded):
        assert spec.fixture_hash == FIXTURE_HASH
        assert spec.fixture_spec == fixture
        assert spec.fixture_spec["seed"] == FIXTURE_SEED
        assert spec.fixture_spec["message_count"] == FIXTURE_MESSAGE_COUNT
        assert spec.fixture_spec["message_chars"] == FIXTURE_MESSAGE_CHARS
        assert spec.history_tokens == FIXTURE_APPROX_TOKENS
        assert spec.root_turns == ROOT_TURNS == FIXTURE_TURNS
        assert spec.warmup_turns == WARMUP_TURNS
        assert spec.checkpoint_turns == (0, 100, 200, 300, 400)
        assert spec.response_chars == FIXTURE_RESPONSE_CHARS
        assert spec.response_chunks == len(response_chunks(0))
        # The committed digest binds the seed, sizes and response shape to the
        # fixture hash; drift in any of them changes it.
        assert spec.digest() == WORKLOAD_HASHES[spec.cell]

    # The loaded cell is the binding 8-agent dimension (root + 7 children).
    assert loaded.total_agents == LOADED_TOTAL_AGENTS == 8
    assert loaded.child_agents == LOADED_CHILD_AGENTS == 7
    assert fresh.total_agents == 1
    assert fresh.child_agents == 0
    assert FULL_SCALES[CELL_LOADED_HEADLESS].child_agents == LOADED_CHILD_AGENTS
    assert FULL_SCALES[CELL_LOADED_HEADLESS].root_turns == ROOT_TURNS

    # No injected delay is charged in the loaded cell: the load is the fixture.
    assert fresh.injected_turn_delay_ms == 30.0
    assert loaded.injected_turn_delay_ms == 0.0

    # The deterministic response shape is fixed-size, seeded, and chunk-splits
    # back to exactly the committed content.
    for turn in (0, 1, 7):
        chunks = response_chunks(turn)
        assert "".join(chunks) == response_content(turn)
        assert all(chunks)
        assert len(chunks) == fresh.response_chunks
    assert response_chunks(0) != response_chunks(1)

    prompts = [root_user_content(index) for index in range(3)]
    assert all(len(prompt) == fresh.root_user_chars for prompt in prompts)
    assert prompts[0].startswith("perf-root-0000 ")
    assert len(set(prompts)) == len(prompts)

    # Negative control: the digest must move when the committed seed moves.
    drifted = workload_spec(CELL_FRESH_HEADLESS)
    object.__setattr__(drifted, "seed", drifted.seed + 1)
    assert drifted.digest() != WORKLOAD_HASHES[CELL_FRESH_HEADLESS]


def _connect(host: str) -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _assert_egress_guard(tmp_path: Path) -> EgressGuard:
    """The guard blocks egress, strips credentials and redirects every home."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert guard.config_home.is_dir()
    assert Path(os.environ[CONFIG_HOME_ENV]) == guard.config_home
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    # Negative control: the block is address-selective, not a blanket kill.
    with pytest.raises(OSError, match="blocked"):
        _connect(_EgressProbe)
    return guard


def _assert_notification_delivery(scenario: HeadlessScenario) -> None:
    """Every serialized notification arrived at the real client end."""
    delivered = scenario.delivered_notifications
    assert len(delivered) == len(scenario.sink.serialized)
    assert [message["method"] for message in delivered] == [
        record.method for record in scenario.sink.serialized
    ]
    assert all(message["jsonrpc"] == "2.0" for message in delivered)


def _delta_update_ids(scenario: HeadlessScenario) -> list[str]:
    """Collect the entry ids of every delta update in the turn's window."""
    entry_ids: list[str] = []
    for message in scenario.delivered_notifications:
        params = message["params"]
        if message["method"] != _ENTRY_UPDATED:
            continue
        entry_ids.append(params["entryId"])
    return entry_ids


def _assert_delta_serialization(dump_log: DeltaDumpLog, updated_ids: list[str]) -> None:
    """P1: one model_dump per side per delta, and the predicates share it."""
    assert updated_ids, "the turn produced no delta updates to check"
    for entry_id in set(updated_ids):
        expected = _PER_SIDE_DUMPS * updated_ids.count(entry_id)
        assert dump_log.dumps(entry_id) == expected, entry_id
    assert dump_log.shared_raw_calls > 0
    assert dump_log.unshared_raw_calls == 0


def _assert_projector_and_indexes(scenario: HeadlessScenario) -> None:
    """The projector was live, and both sides keep a consistent id index."""
    snapshots = scenario.projector_snapshots
    assert len(snapshots) == len(scenario.turns)
    for snapshot in snapshots:
        assert snapshot.live
        assert snapshot.index_consistent
        assert snapshot.index_size == len(snapshot.history_ids)
        assert len(set(snapshot.history_ids)) == len(snapshot.history_ids)

    projection = scenario.projection
    history = projection.history
    assert history
    assert projection._entry_positions == {
        entry.id: position for position, entry in enumerate(history)
    }
    for entry in history:
        assert projection._entries[entry.id] is entry
    projected = {entry.id for entry in history}
    assert set(snapshots[0].history_ids) <= projected
    assert projected_history_fingerprint(history) == snapshots[0].history_fingerprint


def _assert_request_context(scenario: HeadlessScenario) -> None:
    """The backend received the gateway's memoized request for the seeded history."""
    context = scenario.backend_request_context()
    assert context.request_count >= 1
    assert context.memo_present
    assert context.memo_matches_request
    assert context.memo_message_count == context.message_count
    seeded_chars = scenario.scale.history_messages * FIXTURE_MESSAGE_CHARS
    assert context.context_chars >= seeded_chars + len(root_user_content(0))
    assert context.message_count >= scenario.scale.history_messages

    # P2: re-preparing the same live snapshot is an identity hit, and the live
    # version re-read invalidates it after the transcript mutates.
    first, hit, mutated = scenario.prepared_messages_identity_probe()
    assert first == hit
    assert mutated != first


async def _exercise_scenario(scenario: HeadlessScenario) -> None:
    """Drive the shortened loaded workload through the real machinery."""
    seeded = len(scenario.root_loop.messages)
    with instrument_delta_dumps(PublicMessageEntry) as dump_log:
        first_turn = await scenario.run_root_turn(0)
    assert first_turn.status == "completed"
    assert first_turn.window.count > 0
    assert first_turn.window.wire_bytes > 0
    assert not first_turn.window.open
    assert first_turn.injected_delay_ms == 0.0
    assert first_turn.history_messages == seeded + 2

    await scenario.settle_transport()
    _assert_notification_delivery(scenario)
    _assert_delta_serialization(dump_log, _delta_update_ids(scenario))
    _assert_projector_and_indexes(scenario)
    _assert_request_context(scenario)

    await scenario.run_child_burst(0)
    assert scenario.retained_child_sessions == scenario.scale.burst_agents
    scenario.assert_child_session_records()
    assert not scenario.registry._children
    scenario.assert_child_history_fidelity()

    second = await scenario.run_root_turn(1)
    third = await scenario.run_root_turn(2)
    # History grows by the turn's own user/assistant pair; a synchronized
    # child completion adds its own injected completion turn on top.
    assert second.history_messages >= first_turn.history_messages + 2
    assert third.history_messages >= second.history_messages + 2
    assert first_turn.wall_ms > 0.0


async def _build_and_exercise(session_root: Path) -> None:
    first = await build_headless_scenario(
        CELL_LOADED_HEADLESS, scale=SHORTENED_SCALE, session_root=session_root
    )
    second = await build_headless_scenario(CELL_LOADED_HEADLESS, scale=SHORTENED_SCALE)
    try:
        # Topology fingerprint equality across two independent constructions.
        assert_fingerprint_equality(
            first.topology(projector_live=True), second.topology(projector_live=True)
        )
        assert first.topology(projector_live=True)["projector_live"] is True
        # Between turns the runtime holds no projector at all; the live one is
        # asserted inside the turn, never faked here.
        assert first.projector is None

        # The transport is the real memory pair: two distinct ends.
        client, server = first.transports
        assert client is not server
        assert first.projector_snapshots == []
        await _exercise_scenario(first)

        coverage = first.chunks_coverage()
        assert coverage["complete"] is True
        assert coverage["missing"] == []
        assert coverage["root_turns"] == 3
        assert coverage["notification_windows"] == 3
        assert coverage["measured_root_turns"] == 0  # all three are warmup turns
        assert coverage["warmup_root_turns"] == 3
        assert coverage["notifications"] > 0
        assert coverage["notification_bytes"] > 0
        assert len(coverage["notifications_per_root_turn"]) == 3
        assert all(count > 0 for count in coverage["notifications_per_root_turn"])
        assert coverage["child_bursts"] == 1
        assert coverage["child_turns"] == first.scale.burst_agents

        record = first.run_record()
        assert record["metric_version"] == SCENARIO_METRIC_VERSION
        assert record["fixture_hash"] == FIXTURE_HASH
        assert record["workload_hash"] == WORKLOAD_HASHES[CELL_LOADED_HEADLESS]
        assert record["topology"]["projector_live"] is True
        assert record["scale"]["mode"] == "contract"
    finally:
        await first.close()
        await second.close()

    # Cleanup: no child runtime or detached cohort outlives the run.
    first.assert_cleanup()
    second.assert_cleanup()
    assert first.registry._children == {}


@pytest.mark.timeout(_SCENARIO_TIMEOUT_S)
def test_headless_scenario_contract(tmp_path: Path) -> None:
    """The harness holds its transport, projector, workload and P1/P2 contracts."""
    started = perf_counter()
    _assert_workload_definitions()
    guard = _assert_egress_guard(tmp_path)
    try:
        asyncio.run(_build_and_exercise(tmp_path / "sessions"))
        guard.assert_closed()
    finally:
        guard.restore()
    # After restore the process is the caller's again: no hooks, no forced home.
    assert not guard._installed
    assert socket.socket.connect is guard._original_connect
    assert os.environ.get(CONFIG_HOME_ENV) != str(guard.config_home)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"headless scenario contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
