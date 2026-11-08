"""X8a contract node: the compaction / pre-seal execution-cost cell.

Plan §3's X8a row asks for a compaction cost cell that models the landed P3
delta — retain-2 pre-seal releases and boundary-None descriptors — and plan §4
places that cost in the OBSERVATION-ONLY set: fully reported with its recorded
reasons, never gated. This node pins the cell's shape and the reason it is not
gated, and it also pins the production rules the cell models, on the production
functions themselves.

The node stays inside the in-suite contract-node envelope (plan §1, §5): it
drives a shortened deterministic transcript, never the full 200k fixture, and
asserts one measured wall inside the 2 s hard window. The single new node is
``test_runtime_compaction_contract``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import sys
from time import perf_counter
from typing import Any

import pytest

from chartreux.core.llm_models import Role
from chartreux.core.session.transcript_archive import (
    PRE_SEAL_RETAIN_RECENT_USER_ROUNDS,
    eligible_pre_seal_prefix,
    trim_attempt_key,
)
from tests.perf._runtime_compaction import (
    BASE_ROUNDS,
    BLOB_LINE_PREFIX,
    BLOB_SECTION,
    COMPACTION_BOUNDARY_ID,
    COMPACTION_CELL,
    COMPACTION_REASONS,
    COMPACTION_SCHEMA,
    EXCLUDED_SCOPE,
    MEASURED_ATTEMPTS,
    NO_REPEATABLE_UNIT_REASON,
    OBSERVATION_ONLY_REASON,
    POST_BOUNDARY_ROUNDS,
    ROUND_WIDTH,
    TIMED_SCOPE,
    WARMUP_ATTEMPTS,
    CompactionObservation,
    build_compaction_blob,
    build_transcript,
    emit_compaction_blob,
    observe_compaction_cell,
    pre_seal_attempt,
    transcript_fingerprint,
)
from tests.perf._runtime_metrics import (
    REPORT_ENV,
    RUN_DIR_ENV,
    PerfPathError,
    repository_root,
)
from tests.perf._runtime_report import METRIC_VERSION, SCHEMA_ID
from tests.perf._runtime_scenario import PROVIDER_CREDENTIAL_NAMES, EgressGuard

pytestmark = [
    pytest.mark.perf,
    pytest.mark.skipif(
        sys.platform != "linux", reason="the runtime compaction cost cell is Linux-only"
    ),
]

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_CONTRACT_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_EGRESS_PROBE_HOST = "192.0.2.1"


def _assert_retain_rule() -> None:
    """The pre-seal rule the cell models is the landed retain-2 rule.

    Everything strictly before the older of the newest two user-started rounds
    is eligible; those two rounds are retained in full with their payloads. The
    negative controls keep the assertion from passing vacuously: too few user
    rounds, and a prefix with nothing to release, are both refused.
    """
    assert PRE_SEAL_RETAIN_RECENT_USER_ROUNDS == 2
    for rounds in (BASE_ROUNDS, BASE_ROUNDS + 1, BASE_ROUNDS + 5):
        messages = build_transcript(rounds)
        assert len(messages) == rounds * ROUND_WIDTH
        prefix = eligible_pre_seal_prefix(messages)
        assert prefix is not None, rounds
        assert len(prefix.messages) == (rounds - 2) * ROUND_WIDTH
        # The prefix stops at the answer of the round before the retained pair.
        assert prefix.messages[-1].message_id == f"compaction-answer-{rounds - 3:04d}"
        assert any(message.reasoning_payloads for message in prefix.messages)

    # Fewer than two user-started rounds: nothing is eligible to release.
    assert eligible_pre_seal_prefix(build_transcript(2)) is None
    assert eligible_pre_seal_prefix(build_transcript(1)) is None
    # A closed prefix with no reasoning payload has nothing to release.
    assert (
        eligible_pre_seal_prefix(build_transcript(BASE_ROUNDS, payloads=False)) is None
    )


def _assert_boundary_none_key() -> None:
    """A boundary-None attempt keys as pre-seal; a bare None key is refused."""
    key = trim_attempt_key("x8a-contract", 0, None, "8:compaction-answer-0001")
    assert key == ("pre-seal", "x8a-contract", 0, "8:compaction-answer-0001")
    with pytest.raises(ValueError):
        trim_attempt_key("x8a-contract", 0, None, None)


def _assert_release_is_read_only() -> None:
    """One attempt prepares replacements and mutates nothing in the transcript."""
    messages = build_transcript(BASE_ROUNDS + 1)
    before = transcript_fingerprint(messages)
    attempt = pre_seal_attempt(messages, published=set())
    assert attempt.suppressed is False
    assert attempt.descriptor is not None
    assert attempt.replaced_messages == len(messages) - 2 * ROUND_WIDTH
    # Every payload is still live: nothing was published, which is why the cost
    # is not a repeatable gated unit.
    assert transcript_fingerprint(messages) == before
    assert all(
        message.reasoning_payloads
        for message in messages
        if message.role is Role.assistant
    )


def _connect(host: str) -> None:
    """Attempt one TCP connect to ``host`` (the guard must refuse it)."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.connect((host, 80))
    finally:
        probe.close()


def _assert_egress_guard(tmp_path: Path) -> EgressGuard:
    """The cell runs fail-closed: egress blocked, credentials stripped."""
    guard = EgressGuard(tmp_path / "egress")
    guard.install()
    guard.assert_closed()
    assert not [name for name in PROVIDER_CREDENTIAL_NAMES if name in os.environ]
    assert os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
    with pytest.raises(OSError, match="blocked"):
        _connect(_EGRESS_PROBE_HOST)
    return guard


def _assert_pre_seal_observation(observation: CompactionObservation) -> None:
    """The cell models the retain-2, boundary-None pre-seal release."""
    attempt = observation.pre_seal
    assert attempt.suppressed is False
    assert attempt.retain_recent_user_rounds == PRE_SEAL_RETAIN_RECENT_USER_ROUNDS == 2
    assert attempt.retained_round_count == 2
    assert attempt.retained_message_count == 2 * ROUND_WIDTH
    assert (
        attempt.eligible_message_count + attempt.retained_message_count
        == attempt.user_round_count * ROUND_WIDTH
    )
    # Boundary-None descriptor: no compaction boundary exists yet.
    assert attempt.pending.boundary_message_id is None
    assert attempt.pending.session_id == attempt.attempt_key[1]
    assert attempt.descriptor is not None
    assert attempt.descriptor.boundary_message_id is None
    assert attempt.descriptor.trim_mode == "payloads"
    assert attempt.descriptor.archived_count == attempt.eligible_message_count
    assert attempt.descriptor.prefix_end_byte > 0
    assert attempt.attempt_key[0] == "pre-seal"
    assert attempt.attempt_key[3] == attempt.endpoint
    assert attempt.replaced_messages == attempt.eligible_message_count
    assert attempt.eligible_payload_messages > 0
    # Two assistant messages per retained round still carry their payloads.
    assert attempt.retained_payload_messages == 2 * 2


def _assert_suppression(observation: CompactionObservation) -> None:
    """A duplicate attempt at the same state is suppressed, not re-prepared."""
    suppressed = observation.suppressed_samples[0]
    assert suppressed.suppressed is True
    assert suppressed.attempt_key == observation.pre_seal.attempt_key
    assert suppressed.descriptor is None
    assert suppressed.replaced_messages == 0
    assert all(sample.suppressed for sample in observation.suppressed_samples)


def _assert_boundary_observation(observation: CompactionObservation) -> None:
    """A real boundary keys the attempt and its descriptor as boundary."""
    attempt = observation.boundary_trim
    assert attempt.suppressed is False
    assert attempt.boundary_message_id == COMPACTION_BOUNDARY_ID
    assert attempt.pending.boundary_message_id == COMPACTION_BOUNDARY_ID
    assert attempt.attempt_key == (
        "boundary",
        attempt.attempt_key[1],
        attempt.attempt_key[2],
        COMPACTION_BOUNDARY_ID,
    )
    assert attempt.descriptor is not None
    assert attempt.descriptor.boundary_message_id == COMPACTION_BOUNDARY_ID
    assert attempt.eligible_message_count == BASE_ROUNDS * ROUND_WIDTH
    assert attempt.post_boundary_message_count == POST_BOUNDARY_ROUNDS * ROUND_WIDTH
    assert attempt.replaced_messages == attempt.eligible_message_count
    # The trailing duplicate of the same boundary epoch is suppressed.
    assert observation.boundary_samples[-1].suppressed is True
    assert observation.archive_mutated is False


def _assert_sample_series(observation: CompactionObservation) -> None:
    """Warmups are discarded and the reported series are the measured ones."""
    assert len(observation.pre_seal_samples) == MEASURED_ATTEMPTS
    assert len(observation.suppressed_samples) == MEASURED_ATTEMPTS
    assert len(observation.boundary_samples) == MEASURED_ATTEMPTS + 1
    every_attempt = (
        *observation.pre_seal_samples,
        *observation.suppressed_samples,
        *observation.boundary_samples,
    )
    assert all(sample.duration_ms > 0.0 for sample in every_attempt)
    assert not any(sample.suppressed for sample in observation.pre_seal_samples)


def _assert_record(observation: CompactionObservation) -> dict[str, Any]:
    """The record is fully reported and explicitly not gated (plan §4)."""
    record = observation.record()
    assert record["cell"] == COMPACTION_CELL
    assert record["schema"] == COMPACTION_SCHEMA
    assert record["metric_version"] == METRIC_VERSION
    assert record["mode"] == "contract"
    assert record["gated"] is False
    assert record["observation_only"] is True
    assert record["reasons"] == list(COMPACTION_REASONS)
    assert OBSERVATION_ONLY_REASON in record["reasons"]
    assert NO_REPEATABLE_UNIT_REASON in record["reasons"]
    assert record["timed_scope"] == list(TIMED_SCOPE)
    assert record["excluded_scope"] == list(EXCLUDED_SCOPE)
    assert any("publication" in entry for entry in record["excluded_scope"])
    assert record["archive_mutated"] is False

    pre_seal = record["pre_seal"]
    assert pre_seal["retain_recent_user_rounds"] == 2
    assert pre_seal["boundary_message_id"] is None
    assert pre_seal["descriptor_boundary_message_id"] is None
    assert pre_seal["attempt_key_kind"] == "pre-seal"
    assert pre_seal["descriptor_trim_mode"] == "payloads"
    assert pre_seal["suppressed_attempt"] is False

    boundary = record["boundary_trim"]
    assert boundary["boundary_message_id"] == COMPACTION_BOUNDARY_ID
    assert boundary["attempt_key_kind"] == "boundary"
    assert boundary["descriptor_boundary_message_id"] == COMPACTION_BOUNDARY_ID
    assert len(str(boundary["descriptor_prefix_sha256"])) == 64

    expected_counts = {
        "pre_seal_advance": MEASURED_ATTEMPTS,
        "pre_seal_suppressed": MEASURED_ATTEMPTS,
        "boundary_trim": MEASURED_ATTEMPTS + 1,
    }
    for kind, expected in expected_counts.items():
        series = record["samples"][kind]
        assert series["count"] == expected, kind
        assert series["unit"] == "ms"
        assert series["warmup_attempts"] == WARMUP_ATTEMPTS
        assert series["min_ms"] > 0.0
        assert series["max_ms"] >= series["min_ms"]
        assert len(series["durations_ms"]) == expected
    assert record["suppressed_attempts"] == {
        "pre_seal": MEASURED_ATTEMPTS,
        "boundary": 1,
    }

    transcript = record["workload"]["transcript"]
    assert transcript["shortened"] is True
    assert transcript["round_width"] == ROUND_WIDTH
    assert record["workload"]["attempts"] == {
        "warmup": WARMUP_ATTEMPTS,
        "measured": MEASURED_ATTEMPTS,
    }
    json.dumps(record)
    return record


def _assert_blob(observation: CompactionObservation) -> dict[str, Any]:
    """The observation rides the WP6 report as an additive section."""
    blob = build_compaction_blob(observation)
    assert blob["schema"] == SCHEMA_ID
    assert blob["metric_version"] == METRIC_VERSION
    assert blob["runs"] == []
    assert blob[BLOB_SECTION] == observation.record()
    return blob


def _assert_guarded_emission(blob: dict[str, Any], tmp_path: Path) -> None:
    """The observation is written only through the WP1 path guard."""
    scratchpad = tmp_path / "chartreux-scratchpad-x8a"
    run_dir = scratchpad / "run"
    run_dir.mkdir(parents=True)
    report = run_dir / "report.jsonl"
    line = emit_compaction_blob(
        blob, environ={RUN_DIR_ENV: str(run_dir), REPORT_ENV: str(report)}
    )
    assert line.startswith(BLOB_LINE_PREFIX)
    assert json.loads(line[len(BLOB_LINE_PREFIX) :]) == blob
    assert report.read_text(encoding="utf-8") == f"{line}\n"

    escaping_report = repository_root() / "perf-compaction-never-written.jsonl"
    with pytest.raises(PerfPathError) as escaping:
        emit_compaction_blob(
            blob, environ={RUN_DIR_ENV: str(run_dir), REPORT_ENV: str(escaping_report)}
        )
    assert escaping.value.reason == "report_under_repo_root"
    assert not escaping_report.exists()

    with pytest.raises(PerfPathError) as orphaned:
        emit_compaction_blob(blob, environ={REPORT_ENV: str(report)})
    assert orphaned.value.reason == "report_without_run_dir"


@pytest.mark.timeout(_CONTRACT_TIMEOUT_S)
def test_runtime_compaction_contract(tmp_path: Path) -> None:
    """Compaction cost is observation-only, modelling retain-2 and boundary-None."""
    started = perf_counter()
    _assert_retain_rule()
    _assert_boundary_none_key()
    _assert_release_is_read_only()
    guard = _assert_egress_guard(tmp_path)
    try:
        observation = observe_compaction_cell()
        guard.assert_closed()
    finally:
        guard.restore()
    _assert_pre_seal_observation(observation)
    _assert_suppression(observation)
    _assert_boundary_observation(observation)
    _assert_sample_series(observation)
    _assert_record(observation)
    blob = _assert_blob(observation)
    _assert_guarded_emission(blob, tmp_path)
    elapsed = perf_counter() - started
    assert elapsed <= _CONTRACT_WINDOW_S, (
        f"the compaction contract took {elapsed:.3f}s, window {_CONTRACT_WINDOW_S}s"
    )
