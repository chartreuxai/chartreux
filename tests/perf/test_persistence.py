from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
import sys
from time import perf_counter
import tracemalloc
from types import TracebackType
from unittest.mock import MagicMock

import pytest

from chartreux.agents import AgentSafety
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMMessage
from chartreux.core.session.session_logger import SessionLogger, _TranscriptCursor
from chartreux.core.session_types import AgentStats, SessionMetadata
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config
from tests.perf._metrics import machine_context, percentiles, record
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
        agent_profile: AgentProfile | None,
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
                agent_profile,
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
        agent_profile: AgentProfile | None,
        session_dir: Path,
        session_metadata: SessionMetadata,
        launch_config_dirty: bool,
    ) -> _TranscriptCursor:
        measurements.full_verify_calls += 1
        return save_full_verify(
            messages,
            stats,
            tool_manager,
            agent_profile,
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
    agent_profile = AgentProfile(
        name="perf-profile",
        display_name="Performance profile",
        description="Persistence benchmark profile",
        safety=AgentSafety.NEUTRAL,
        overrides={},
    )

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
            messages,
            stats=stats,
            config=config,
            tool_manager=tool_manager,
            agent_profile=agent_profile,
        )
        assert logger._transcript_cursor is not None

        warm_measurements = _instrument_logger(logger, monkeypatch)
        warm_save_count = 0
        for _ in range(20):
            started_at = perf_counter()
            await logger.save_interaction(
                messages,
                stats=stats,
                config=config,
                tool_manager=tool_manager,
                agent_profile=agent_profile,
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
            messages,
            stats=stats,
            config=config,
            tool_manager=tool_manager,
            agent_profile=agent_profile,
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
                messages,
                stats=stats,
                config=config,
                tool_manager=tool_manager,
                agent_profile=agent_profile,
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
