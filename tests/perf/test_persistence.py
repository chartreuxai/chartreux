from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc
from types import TracebackType
from unittest.mock import MagicMock

import pytest

from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.session.session_logger import (
    TRANSCRIPT_VERIFY_INTERVAL,
    SessionLogger,
    _TranscriptCursor,
)
from chartreux.core.session_types import AgentStats, SessionMetadata
from chartreux.core.tools.manager import ToolManager
from chartreux.utils.durable_io import durable_replace
from chartreux.utils.io import read_safe
from tests.conftest import build_test_vibe_config
from tests.perf._metrics import (
    alloc_peak,
    machine_context,
    percentiles,
    record,
    rss_mib,
    rss_peak_mib,
)
from tests.perf._synthetic import synthetic_messages

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)


@dataclass
class _Measurements:
    async_pre_lock_ms: list[float] = field(default_factory=list)
    sync_worker_ms: list[float] = field(default_factory=list)
    awaited_save_ms: list[float] = field(default_factory=list)
    full_verify_calls: int = 0
    measure_next_worker: bool = False
    worker_peak_mb: float | None = None


class _TimedLock:
    def __init__(
        self,
        lock: asyncio.Lock,
        pending_start_times: list[float],
        measurements: _Measurements,
    ) -> None:
        self._lock = lock
        self._pending_start_times = pending_start_times
        self._measurements = measurements

    async def __aenter__(self) -> None:
        if self._pending_start_times:
            started_at = self._pending_start_times.pop(0)
            self._measurements.async_pre_lock_ms.append(
                (perf_counter() - started_at) * 1_000
            )
        await self._lock.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        self._lock.release()
        return False


def _instrument_logger(
    logger: SessionLogger, monkeypatch: pytest.MonkeyPatch
) -> _Measurements:
    measurements = _Measurements()
    pending_start_times: list[float] = []

    get_session_info = logger._get_session_info

    def timed_get_session_info() -> tuple[Path, SessionMetadata] | None:
        session_info = get_session_info()
        if session_info is not None:
            pending_start_times.append(perf_counter())
        return session_info

    monkeypatch.setattr(logger, "_get_session_info", timed_get_session_info)
    monkeypatch.setattr(
        logger,
        "_save_lock",
        _TimedLock(logger._save_lock, pending_start_times, measurements),
    )

    save_interaction_sync = logger._save_interaction_sync

    def timed_save_interaction_sync(
        messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_dir: Path,
        session_metadata: SessionMetadata,
        allow_empty: bool,
        launch_config_dirty: bool,
        cursor: _TranscriptCursor | None,
        saves_since_verify: int,
    ) -> tuple[_TranscriptCursor | None, int]:
        measure_memory = measurements.measure_next_worker
        measurements.measure_next_worker = False
        was_tracing = tracemalloc.is_tracing()
        starting_memory = 0
        if measure_memory:
            if not was_tracing:
                tracemalloc.start()
            tracemalloc.reset_peak()
            starting_memory, _ = tracemalloc.get_traced_memory()

        started_at = perf_counter()
        try:
            return save_interaction_sync(
                messages,
                stats,
                tool_manager,
                session_dir,
                session_metadata,
                allow_empty,
                launch_config_dirty,
                cursor,
                saves_since_verify,
            )
        finally:
            measurements.sync_worker_ms.append((perf_counter() - started_at) * 1_000)
            if measure_memory:
                _, peak_memory = tracemalloc.get_traced_memory()
                measurements.worker_peak_mb = max(0, peak_memory - starting_memory) / (
                    1024 * 1024
                )
                if not was_tracing:
                    tracemalloc.stop()

    monkeypatch.setattr(logger, "_save_interaction_sync", timed_save_interaction_sync)

    save_full_verify = logger._save_full_verify

    def counted_save_full_verify(
        messages: list[LLMMessage],
        stats: AgentStats,
        tool_manager: ToolManager,
        session_dir: Path,
        session_metadata: SessionMetadata,
        launch_config_dirty: bool,
    ) -> _TranscriptCursor:
        measurements.full_verify_calls += 1
        return save_full_verify(
            messages,
            stats,
            tool_manager,
            session_dir,
            session_metadata,
            launch_config_dirty,
        )

    monkeypatch.setattr(logger, "_save_full_verify", counted_save_full_verify)
    return measurements


def _save_timing_record(
    async_pre_lock_ms: list[float],
    sync_worker_ms: list[float],
    awaited_save_ms: list[float],
) -> dict[str, object]:
    return {
        "async_pre_lock_ms": percentiles(async_pre_lock_ms),
        "sync_worker_ms": percentiles(sync_worker_ms),
        "awaited_save_ms": percentiles(awaited_save_ms),
    }


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_persistence_micro_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: ChartreuxConfigSchema = build_test_vibe_config()
    stats = AgentStats()
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    tool_manager.available_tool_specs.return_value = []

    message_counts = (5_000, 20_000)
    approx_tokens = 20_000
    for message_count in message_counts:
        messages = synthetic_messages(approx_tokens, message_count)
        assert len(messages) == message_count
        session_config = SessionLoggingConfig(
            save_dir=str(tmp_path / f"sessions-{message_count}"),
            session_prefix="perf",
            enabled=True,
        )
        session_id = f"persistence-{message_count}"
        logger = SessionLogger(session_config, session_id)

        # The first full save establishes the transcript cursor; the measured
        # saves below then take the cursor-backed warm path.
        await logger.save_interaction(
            messages, stats=stats, config=config, tool_manager=tool_manager
        )
        assert logger._transcript_cursor is not None

        warm_measurements = _instrument_logger(logger, monkeypatch)
        warm_save_count = 0
        for _ in range(20):
            started_at = perf_counter()
            await logger.save_interaction(
                messages, stats=stats, config=config, tool_manager=tool_manager
            )
            warm_measurements.awaited_save_ms.append(
                (perf_counter() - started_at) * 1_000
            )
            warm_save_count += 1

        assert logger.session_dir is not None
        cold_logger = SessionLogger(
            session_config, session_id, session_dir=logger.session_dir
        )
        assert cold_logger._transcript_cursor is None
        cold_measurements = _instrument_logger(cold_logger, monkeypatch)
        started_at = perf_counter()
        await cold_logger.save_interaction(
            messages, stats=stats, config=config, tool_manager=tool_manager
        )
        cold_measurements.awaited_save_ms.append((perf_counter() - started_at) * 1_000)
        cold_save_count = 1

        memory_measurements: _Measurements | None = None
        memory_save_count = 0
        if message_count == 20_000:
            traced_session_config = SessionLoggingConfig(
                save_dir=str(tmp_path / f"sessions-{message_count}-traced"),
                session_prefix="perf",
                enabled=True,
            )
            traced_logger = SessionLogger(traced_session_config, f"{session_id}-traced")
            memory_measurements = _instrument_logger(traced_logger, monkeypatch)
            memory_measurements.measure_next_worker = True
            await traced_logger.save_interaction(
                messages, stats=stats, config=config, tool_manager=tool_manager
            )
            memory_save_count = 1
            assert traced_logger._transcript_cursor is not None
            assert memory_measurements.worker_peak_mb is not None

        transcript_path = cold_logger.messages_filepath
        assert transcript_path.exists()
        file_size_bytes = transcript_path.stat().st_size
        assert file_size_bytes > 0
        assert warm_save_count == 20
        assert cold_save_count == 1
        assert len(warm_measurements.async_pre_lock_ms) == warm_save_count
        assert len(warm_measurements.sync_worker_ms) == warm_save_count
        assert len(warm_measurements.awaited_save_ms) == warm_save_count
        assert len(cold_measurements.async_pre_lock_ms) == cold_save_count
        assert len(cold_measurements.sync_worker_ms) == cold_save_count
        assert len(cold_measurements.awaited_save_ms) == cold_save_count
        assert warm_measurements.full_verify_calls == 0
        assert cold_measurements.full_verify_calls == 1
        assert cold_logger._transcript_cursor is not None
        assert (
            len(warm_measurements.awaited_save_ms)
            + len(cold_measurements.awaited_save_ms)
            == 21
        )

        warm_save_timings = [
            {
                "save": index,
                "async_pre_lock_ms": round(
                    warm_measurements.async_pre_lock_ms[index], 3
                ),
                "sync_worker_ms": round(warm_measurements.sync_worker_ms[index], 3),
                "awaited_save_ms": round(warm_measurements.awaited_save_ms[index], 3),
            }
            for index in range(warm_save_count)
        ]
        cold_save_timings = {
            "async_pre_lock_ms": round(cold_measurements.async_pre_lock_ms[0], 3),
            "sync_worker_ms": round(cold_measurements.sync_worker_ms[0], 3),
            "awaited_save_ms": round(cold_measurements.awaited_save_ms[0], 3),
        }
        record(
            f"persistence_{message_count}",
            {
                "message_count": message_count,
                "approx_tokens_chars_over_4": approx_tokens,
                "warm_save_count": warm_save_count,
                "cold_save_count": cold_save_count,
                "memory_measurement_save_count": memory_save_count,
                "warm": _save_timing_record(
                    warm_measurements.async_pre_lock_ms,
                    warm_measurements.sync_worker_ms,
                    warm_measurements.awaited_save_ms,
                ),
                "cold": {
                    "timings": cold_save_timings,
                    "async_pre_lock_ms": percentiles(
                        cold_measurements.async_pre_lock_ms
                    ),
                    "sync_worker_ms": percentiles(cold_measurements.sync_worker_ms),
                    "awaited_save_ms": percentiles(cold_measurements.awaited_save_ms),
                },
                "warm_saves": warm_save_timings,
                "worker_peak_mb": (
                    memory_measurements.worker_peak_mb
                    if memory_measurements is not None
                    else None
                ),
                "transcript_file_bytes": file_size_bytes,
                "machine": machine_context(),
            },
        )
        if message_count == 20_000:
            assert memory_save_count == 1
            assert memory_measurements is not None
            assert memory_measurements.worker_peak_mb is not None
        else:
            assert memory_save_count == 0
            assert memory_measurements is None
        assert len(messages) == message_count


# FROZEN CEILINGS (P0b). Non-regression upper bounds for the full-verify worker
# at 20k records: exceeding one is a blocker, never a threshold change.
# Baseline: the perf-baselines/p0b-persistence-baseline.jsonl run at 5a1e121f
# (see the artifact's .meta.json for machine context): duration-invalidated
# populated-verify awaited p50 131.2 ms, populated-verify worker peak
# 20.1161 MiB, initial full-save (overwrite) worker peak 11.8049 MiB — the
# last reproduces the round-1 review's 11.7997 MiB at 57217e5f. Values are 4x
# (latency) or 6x (memory) the recorded baseline, so host load cannot mask a
# real regression.
# MEMORY CEILINGS ARE FIRST-OBSERVATION BASELINES: no earlier recorded values
# exist for them, and they are valid on the baseline machine only.
_P0B_DURATION_VERIFY_P50_CEILING_MS = 525.0
_P0B_VERIFY_WORKER_PEAK_CEILING_MIB = 120.7
_P0B_INITIAL_SAVE_WORKER_PEAK_CEILING_MIB = 70.8

# P4b freezes its own bound from the P0b floor input rather than scaling it:
# the corruption-repair save (quarantine + full rewrite) peaked at 18.0078 MiB
# at 5a1e121f, when the rewrite payload and the original-transcript copies were
# each held whole. P4b streams all of them, so the recorded baseline is the
# upper bound and any value at or above it means the streaming regressed.
_P4B_REPAIR_WORKER_PEAK_CEILING_MIB = 18.0078


def _perf_tool_manager() -> MagicMock:
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    tool_manager.available_tool_specs.return_value = []
    return tool_manager


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_persistence_full_verify_repeats_and_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: ChartreuxConfigSchema = build_test_vibe_config()
    stats = AgentStats()
    tool_manager = _perf_tool_manager()

    message_count = 20_000
    messages = synthetic_messages(20_000, message_count)
    session_config = SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions-verify"), session_prefix="perf", enabled=True
    )
    logger = SessionLogger(session_config, "persistence-verify")

    # The initial full save establishes the cursor and the populated transcript.
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    assert logger._transcript_cursor is not None
    assert logger.session_dir is not None
    messages_path = logger.messages_filepath
    transcript_file_bytes = messages_path.stat().st_size

    measurements = _instrument_logger(logger, monkeypatch)

    # Repeated duration-invalidated full verifies: an unchanged populated
    # transcript verified because the save counter reached the interval, not
    # because the cursor was invalidated. The metadata-cache short-circuit
    # means each one is read + parse + compare with no transcript rewrite.
    repeat_count = 5
    repeat_awaited_ms: list[float] = []
    verify_calls = 0
    for _ in range(repeat_count):
        logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
        started_at = perf_counter()
        await logger.save_interaction(
            messages, stats=stats, config=config, tool_manager=tool_manager
        )
        repeat_awaited_ms.append((perf_counter() - started_at) * 1_000)
    verify_calls += repeat_count
    assert measurements.full_verify_calls == verify_calls
    assert percentiles(repeat_awaited_ms)["p50"] <= _P0B_DURATION_VERIFY_P50_CEILING_MS

    # One traced repeat for the populated-verify worker peak; its timing is
    # excluded from the percentiles (tracemalloc inflates it).
    logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
    measurements.measure_next_worker = True
    started_at = perf_counter()
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    verify_calls += 1
    traced_awaited_ms = (perf_counter() - started_at) * 1_000
    assert measurements.full_verify_calls == verify_calls
    assert measurements.worker_peak_mb is not None
    assert measurements.worker_peak_mb <= _P0B_VERIFY_WORKER_PEAK_CEILING_MIB
    verify_worker_peak_mb = measurements.worker_peak_mb

    # File-size mismatch branch: the cursor no longer matches the transcript
    # on disk, so the next save must full-verify even with a fresh counter.
    cursor = logger._transcript_cursor
    assert cursor is not None
    logger._transcript_saves_since_verify = 0
    logger._transcript_cursor = replace(cursor, file_size=cursor.file_size + 1)
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    verify_calls += 1
    assert measurements.full_verify_calls == verify_calls
    repaired_cursor = logger._transcript_cursor
    assert repaired_cursor is not None
    assert repaired_cursor.file_size == messages_path.stat().st_size

    # Boundary-digest mismatch branch: same size, wrong boundary digest.
    cursor = logger._transcript_cursor
    assert cursor is not None
    logger._transcript_saves_since_verify = 0
    logger._transcript_cursor = replace(cursor, boundary_digest="0" * 64)
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    verify_calls += 1
    assert measurements.full_verify_calls == verify_calls
    repaired_cursor = logger._transcript_cursor
    assert repaired_cursor is not None
    assert repaired_cursor.boundary_digest != "0" * 64

    # Corruption branch: an unreadable line quarantines the transcript and the
    # verify repairs it in place from the live messages.
    with messages_path.open("a", encoding="utf-8") as transcript_file:
        transcript_file.write('{"corrupt": true\n')
    logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    verify_calls += 1
    assert measurements.full_verify_calls == verify_calls
    quarantine_files = list(logger.session_dir.glob("*.corrupt-*"))
    assert len(quarantine_files) == 1
    assert quarantine_files[0].stat().st_size > transcript_file_bytes
    repaired_lines = messages_path.read_text(encoding="utf-8").splitlines()
    assert len(repaired_lines) == message_count
    for line in repaired_lines:
        json.loads(line)
    assert logger._transcript_cursor is not None

    record(
        "persistence_p0b_full_verify_repeats",
        {
            "message_count": message_count,
            "transcript_file_bytes": transcript_file_bytes,
            "duration_verify_repeats": repeat_count,
            "full_verify_calls": verify_calls,
            "duration_verify_awaited_ms": percentiles(repeat_awaited_ms),
            "duration_verify_sync_worker_ms": percentiles(
                measurements.sync_worker_ms[:repeat_count]
            ),
            "traced_repeat_awaited_ms": round(traced_awaited_ms, 3),
            "verify_worker_peak_mb": verify_worker_peak_mb,
            "verify_worker_peak_note": (
                "tracemalloc peak of one populated duration-invalidated full "
                "verify (read + parse + compare, no transcript rewrite); "
                "first-observation baseline, valid on this machine only"
            ),
            "quarantine_file_count": len(quarantine_files),
            "repaired_line_count": len(repaired_lines),
            "machine": machine_context(),
        },
    )


@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.asyncio
async def test_persistence_full_verify_memory_breakdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: ChartreuxConfigSchema = build_test_vibe_config()
    stats = AgentStats()
    tool_manager = _perf_tool_manager()

    message_count = 20_000
    messages = synthetic_messages(20_000, message_count)
    session_config = SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions-breakdown"),
        session_prefix="perf",
        enabled=True,
    )
    logger = SessionLogger(session_config, "persistence-breakdown")
    measurements = _instrument_logger(logger, monkeypatch)

    rss_before_mib = rss_mib()
    rss_hwm_before_mib = rss_peak_mib()

    # Denominator branch 1: initial full save at 20k (cursor absent, transcript
    # absent -> full verify with a full-file overwrite). The branch the round-1
    # execution review measured at 11.7997 MiB worker_peak_mb at 57217e5f.
    measurements.measure_next_worker = True
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    assert measurements.worker_peak_mb is not None
    initial_save_worker_peak_mb = measurements.worker_peak_mb
    assert initial_save_worker_peak_mb <= _P0B_INITIAL_SAVE_WORKER_PEAK_CEILING_MIB

    # Denominator branch 2: duration-invalidated populated-transcript full
    # verify (read + parse + compare; metadata-cache short-circuit, no rewrite).
    logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
    measurements.measure_next_worker = True
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    assert measurements.worker_peak_mb is not None
    verify_worker_peak_mb = measurements.worker_peak_mb
    assert verify_worker_peak_mb <= _P0B_VERIFY_WORKER_PEAK_CEILING_MIB

    rss_after_mib = rss_mib()
    rss_hwm_after_mib = rss_peak_mib()

    # Four-bucket breakdown, phase-scoped on the same live data. Buckets are
    # phase peaks, not disjoint parts: the real verify holds several phases
    # co-resident, which is what the worker peaks above capture.
    messages_path = logger.messages_filepath
    transcript_file_bytes = messages_path.stat().st_size

    # Bucket 2: raw-text buffer (session_logger.py:774 —
    # read_safe(messages_path).text.splitlines()).
    with alloc_peak() as raw_trace:
        raw_text = read_safe(messages_path).text
        raw_lines = raw_text.splitlines()
    raw_text_peak_mib = raw_trace.peak_mib
    raw_line_count = len(raw_lines)
    assert raw_line_count == message_count

    # Bucket 1: parsed-record residency. Measured through the real reader: the
    # peak includes the transient raw text co-resident during the read, and
    # the net is the residency the comparison keeps alive.
    with alloc_peak() as persisted_trace:
        persisted_records, transcript_valid = SessionLogger._read_persisted_messages(
            messages_path
        )
    assert transcript_valid
    assert len(persisted_records) == message_count
    read_phase_peak_mib = persisted_trace.peak_mib
    persisted_records_net_mib = persisted_trace.net_mib
    del raw_text, raw_lines

    # Bucket 1 (current side): the live records the verify compares against.
    with alloc_peak() as current_trace:
        current_data = [
            SessionLogger._message_record(message)
            for message in messages
            if message.role != Role.system
        ]
    current_records_net_mib = current_trace.net_mib

    # Bucket 3: serialization (the full-file payload the write path publishes).
    with alloc_peak() as serialize_trace:
        payload = SessionLogger._transcript_bytes(current_data)
    serialization_peak_mib = serialize_trace.peak_mib
    payload_bytes = len(payload)
    assert payload_bytes == transcript_file_bytes

    # Bucket 4: atomic-write cost — durable_replace on a scratch copy with the
    # payload prebuilt, so this measures the write machinery's own allocations
    # on top of the payload residency counted in bucket 3.
    scratch_path = tmp_path / "scratch-replace.jsonl"
    scratch_path.write_bytes(b"")
    with alloc_peak() as write_trace:
        durable_replace(scratch_path, payload)
    atomic_write_peak_mib = write_trace.peak_mib
    del payload, persisted_records, current_data

    # P4b floor input: the corruption-repair path holds the full rewrite plus
    # the quarantined original (and its read-back), so it peaks above the
    # ordinary branches. Measured here so P4b can set its floor from this
    # breakdown without re-measuring; recorded, not ceiled.
    with messages_path.open("a", encoding="utf-8") as transcript_file:
        transcript_file.write('{"corrupt": true\n')
    logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
    measurements.measure_next_worker = True
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    assert measurements.worker_peak_mb is not None
    repair_worker_peak_mb = measurements.worker_peak_mb
    assert repair_worker_peak_mb <= _P4B_REPAIR_WORKER_PEAK_CEILING_MIB
    assert logger.session_dir is not None
    assert len(list(logger.session_dir.glob("*.corrupt-*"))) == 1

    record(
        "persistence_p0b_memory_breakdown",
        {
            "message_count": message_count,
            "baseline_kind": (
                "first-observation baseline (no earlier recorded values; "
                "valid on this machine only)"
            ),
            "worker_peak_mb": {
                "initial_full_save_overwrite": initial_save_worker_peak_mb,
                "populated_duration_verify": verify_worker_peak_mb,
                "corruption_repair": repair_worker_peak_mb,
                "round1_reference": 11.7997,
                "round1_reference_commit": "57217e5f",
            },
            "rss": {
                "before_mib": rss_before_mib,
                "after_mib": rss_after_mib,
                "process_hwm_before_mib": rss_hwm_before_mib,
                "process_hwm_after_mib": rss_hwm_after_mib,
                "note": (
                    "ru_maxrss is a process-wide monotonic high-water mark, so "
                    "the delta is a lower bound attributable to the measured "
                    "saves"
                ),
            },
            "transcript_file_bytes": transcript_file_bytes,
            "bucket_1_parsed_record_residency_mib": {
                "persisted_side_net": persisted_records_net_mib,
                "current_side_net": current_records_net_mib,
                "comparison_coresident_net": round(
                    persisted_records_net_mib + current_records_net_mib, 4
                ),
                "read_phase_peak_including_raw_text": read_phase_peak_mib,
            },
            "bucket_2_raw_text_buffer_mib": {
                "source": (
                    "chartreux/core/session/session_logger.py:774 "
                    "read_safe(...).text.splitlines()"
                ),
                "peak": raw_text_peak_mib,
                "line_count": raw_line_count,
            },
            "bucket_3_serialization_mib": {
                "peak": serialization_peak_mib,
                "payload_bytes": payload_bytes,
            },
            "bucket_4_atomic_write_cost_mib": {
                "peak": atomic_write_peak_mib,
                "note": (
                    "durable_replace on a scratch copy with the payload "
                    "prebuilt; write-machinery allocations on top of the "
                    "bucket-3 payload residency"
                ),
            },
            "machine": machine_context(),
        },
    )
