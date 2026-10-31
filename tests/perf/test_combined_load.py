from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
import time
from time import perf_counter
import tracemalloc
from typing import Any
from unittest.mock import MagicMock

import pytest

from chartreux.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
)
from chartreux.app_server.models import PublicMessageEntry
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.launch import LaunchCandidate
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.llm_models import LLMChunk, LLMMessage
from chartreux.core.session.session_logger import (
    TRANSCRIPT_VERIFY_INTERVAL,
    SessionLogger,
    _TranscriptCursor,
)
from chartreux.core.session_types import AgentStats, SessionMetadata
from chartreux.core.subagents import TaskResult, WaitStatus
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.perf._harness import (
    GatedSequenceBackend,
    SubagentHarness,
    create_subagent_harness,
    launch_explicit_tasks,
)
from tests.perf._metrics import machine_context, percentiles, record
from tests.perf._synthetic import synthetic_messages
from tests.perf.test_streaming_render import (
    ASSISTANT_CHUNK_COUNT,
    YieldingStreamingBackend,
    _build_streaming_app,
    _wait_until,
)
from tests.stubs.fake_backend import FakeBackend

_AGENT_COUNT = 8
_CONTEXT_APPROX_TOKENS = 100_000
_CONTEXT_MESSAGE_COUNT = 100
_MIDSTREAM_CHUNK_COUNT = ASSISTANT_CHUNK_COUNT // 2
_HEARTBEAT_INTERVAL_SECONDS = 0.005

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)


@dataclass(slots=True)
class StreamMeasurement:
    flush_delay_ms: list[float]
    chunk_count: int
    total_bytes: int
    turn_wall_ms: float


class MidStreamGatedBackend(YieldingStreamingBackend):
    """WP4-paced stream which can pause at its midpoint for child release."""

    def __init__(self, chunks: list[LLMChunk], *, pause_at_midpoint: bool) -> None:
        super().__init__(chunks)
        self.pause_at_midpoint = pause_at_midpoint
        self.midpoint_reached = asyncio.Event()
        self.resume_stream = asyncio.Event()

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk
            if self.pause_at_midpoint and self.chunks_yielded == _MIDSTREAM_CHUNK_COUNT:
                self.midpoint_reached.set()
                await self.resume_stream.wait()


class ContextGatedBackend(GatedSequenceBackend):
    """WP3 gates which also attest to each in-flight request's context size."""

    def __init__(self) -> None:
        super().__init__()
        self.context_chars: list[int] = []

    async def complete(self, *, messages: list[Any], **kwargs: Any) -> Any:
        self.context_chars.append(
            sum(
                len(message.content)
                for message in messages
                if isinstance(message, LLMMessage) and isinstance(message.content, str)
            )
        )
        return await super().complete(messages=messages, **kwargs)


def _chunk_texts() -> list[str]:
    return [f"token-{index:03d} " for index in range(ASSISTANT_CHUNK_COUNT)]


def _chunks() -> list[LLMChunk]:
    texts = _chunk_texts()
    return [
        mock_llm_chunk(
            content=text,
            stop_reason="stop" if index == ASSISTANT_CHUNK_COUNT - 1 else None,
        )
        for index, text in enumerate(texts)
    ]


def _rss_mb() -> float:
    """Read current resident memory (not process lifetime high-water RSS)."""
    resident_pages = int(
        Path("/proc/self/statm").read_text(encoding="utf-8").split()[1]
    )
    page_size = os.sysconf("SC_PAGE_SIZE")
    return resident_pages * page_size / (1024 * 1024)


async def _heartbeat(stop: asyncio.Event, loop_lag_ms: list[float]) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected_wake = loop.time() + _HEARTBEAT_INTERVAL_SECONDS
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        loop_lag_ms.append(max(0.0, loop.time() - expected_wake) * 1000)


async def _measure_stream(
    backend: MidStreamGatedBackend,
    monkeypatch: pytest.MonkeyPatch,
    *,
    during_midstream: Callable[[], Awaitable[None]] | None = None,
) -> StreamMeasurement:
    chunk_texts = _chunk_texts()
    chunk_end_offsets: list[int] = []
    total_chars = 0
    for text in chunk_texts:
        total_chars += len(text)
        chunk_end_offsets.append(total_chars)

    app = _build_streaming_app(backend)
    flush_delay_ms: list[float] = []
    chunk_arrival_times: list[float | None] = [None] * ASSISTANT_CHUNK_COUNT
    flushed_chunk_count = [0]
    received_chars = 0
    submitted_at: float | None = None

    def observe_visible_content(message: AssistantMessage) -> None:
        if submitted_at is None or message._markdown is None:
            return
        visible_chars = len(message._markdown._markdown)
        visible_at = time.perf_counter()
        while (
            flushed_chunk_count[0] < len(chunk_end_offsets)
            and chunk_end_offsets[flushed_chunk_count[0]] <= visible_chars
        ):
            arrival_at = chunk_arrival_times[flushed_chunk_count[0]]
            if arrival_at is None:
                raise AssertionError("visible assistant chunk has no arrival timestamp")
            flush_delay_ms.append((visible_at - arrival_at) * 1000)
            flushed_chunk_count[0] += 1

    # This per-chunk event-arrival → visible-flush capture mirrors WP4's
    # test_streaming_render.py pattern. Measuring from turn submission would
    # incorrectly include the deliberate child-launch pause in cumulative latency.
    original_flush = AssistantMessage._flush_write_buffer
    original_initial_write = AssistantMessage.write_initial_content
    original_handler = app._handle_turn_event

    async def observe_timer_flush(message: AssistantMessage) -> None:
        await original_flush(message)
        observe_visible_content(message)

    async def observe_initial_write(message: AssistantMessage) -> None:
        await original_initial_write(message)
        observe_visible_content(message)

    async def time_bridge_handler(event: AppServerEvent) -> None:
        nonlocal received_chars
        arrival_at = time.perf_counter()
        assistant_delta = ""
        match event:
            case HistoryEntryAdded(entry=PublicMessageEntry(role="assistant") as entry):
                assistant_delta = entry.text
            case HistoryEntryUpdated(entry=PublicMessageEntry(role="assistant")):
                assistant_delta = "".join(
                    operation.value
                    for operation in event.patch
                    if operation.op == "append"
                    and operation.path == "/content/0/text"
                    and isinstance(operation.value, str)
                )
        if assistant_delta:
            previous_received_chars = received_chars
            received_chars += len(assistant_delta)
            for index, end_offset in enumerate(chunk_end_offsets):
                if previous_received_chars < end_offset <= received_chars:
                    chunk_arrival_times[index] = arrival_at
        await original_handler(event)

    with monkeypatch.context() as scoped_monkeypatch:
        scoped_monkeypatch.setattr(
            AssistantMessage, "_flush_write_buffer", observe_timer_flush
        )
        scoped_monkeypatch.setattr(
            AssistantMessage, "write_initial_content", observe_initial_write
        )
        app._handle_turn_event = time_bridge_handler

        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            chat_input = app.query_one(ChatInputContainer)
            submitted_at = time.perf_counter()
            chat_input.post_message(
                ChatInputContainer.Submitted("combined-load streaming pilot")
            )

            if during_midstream is not None:
                await asyncio.wait_for(backend.midpoint_reached.wait(), timeout=60)
                try:
                    await during_midstream()
                finally:
                    backend.resume_stream.set()

            await _wait_until(
                pilot, lambda: backend.chunks_yielded == ASSISTANT_CHUNK_COUNT
            )
            await _wait_until(pilot, lambda: not app._agent_job_active(), timeout=120.0)
            turn_wall_ms = (time.perf_counter() - submitted_at) * 1000

            assistant = app.query_one(AssistantMessage)
            expected_content = "".join(chunk_texts)
            assert assistant.get_content() == expected_content
            assert len(assistant.get_content().encode("utf-8")) == len(
                expected_content.encode("utf-8")
            )
            assert backend.chunks_yielded == ASSISTANT_CHUNK_COUNT
            assert received_chars == total_chars
            assert all(arrival is not None for arrival in chunk_arrival_times)
            assert flushed_chunk_count[0] == ASSISTANT_CHUNK_COUNT
            assert len(flush_delay_ms) == ASSISTANT_CHUNK_COUNT

    return StreamMeasurement(
        flush_delay_ms=flush_delay_ms,
        chunk_count=backend.chunks_yielded,
        total_bytes=len("".join(chunk_texts).encode("utf-8")),
        turn_wall_ms=turn_wall_ms,
    )


def _install_large_child_contexts(
    harness: SubagentHarness,
    backend: ContextGatedBackend,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    factory = harness.registry._runtime_factory
    original_create_child = factory.create_child

    async def create_child_with_large_history(
        parent: AgentLoop,
        candidate: LaunchCandidate | str,
        *,
        session_id: str | None = None,
        session_dir: Path | None = None,
    ) -> AgentLoop:
        child = await original_create_child(
            parent, candidate, session_id=session_id, session_dir=session_dir
        )
        history = synthetic_messages(_CONTEXT_APPROX_TOKENS, _CONTEXT_MESSAGE_COUNT)
        child.messages.extend(history)
        assert sum(len(message.content or "") for message in history) == (
            _CONTEXT_APPROX_TOKENS * 4
        )
        return child

    monkeypatch.setattr(factory, "create_child", create_child_with_large_history)

    def create_gated_backend(**_kwargs: object) -> FakeBackend:
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_gated_backend
    )


async def _start_children_and_release(
    harness: SubagentHarness,
    backend: ContextGatedBackend,
    *,
    rss_at_gate_mb: list[float],
    results: list[TaskResult],
) -> None:
    launches = await asyncio.wait_for(
        launch_explicit_tasks(harness.registry, harness.context, harness.model_names),
        timeout=120,
    )
    assert len(launches) == _AGENT_COUNT
    assert all(
        result.status == "launched"
        and result.agent_id is not None
        and result.run_id is not None
        for result in launches
    )

    await asyncio.wait_for(
        asyncio.gather(*(started.wait() for started in backend.started)), timeout=120
    )
    assert len(backend.started) == _AGENT_COUNT
    assert all(started.is_set() for started in backend.started), (
        f"only {sum(started.is_set() for started in backend.started)} of "
        f"{_AGENT_COUNT} gated children started"
    )
    assert len(backend.context_chars) == _AGENT_COUNT
    assert all(
        chars // 4 >= _CONTEXT_APPROX_TOKENS for chars in backend.context_chars
    ), (
        f"child requests did not carry the full synthetic context: {backend.context_chars}"
    )
    rss_at_gate_mb.append(_rss_mb())
    results.extend(launches)

    for release in backend.releases:
        release.set()


async def _await_children(harness: SubagentHarness, results: list[TaskResult]) -> None:
    members: list[tuple[str, str]] = []
    for result in results:
        if result.agent_id and result.run_id:
            members.append((result.agent_id, result.run_id))
    assert len(members) == _AGENT_COUNT
    completions = await asyncio.wait_for(
        asyncio.gather(
            *(
                harness.registry.wait_for_agent(agent_id, run_id, timeout=120)
                for agent_id, run_id in members
            )
        ),
        timeout=120,
    )
    assert all(
        completion.status is WaitStatus.COMPLETED
        and completion.result is not None
        and completion.result.completed
        for completion in completions
    )


async def _iteration(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    streaming_only_first = iteration % 2 == 0
    measurement_order = (
        "streaming_only_then_combined"
        if streaming_only_first
        else "combined_then_streaming_only"
    )
    streaming_only: StreamMeasurement | None = None
    if streaming_only_first:
        baseline_backend = MidStreamGatedBackend(_chunks(), pause_at_midpoint=False)
        streaming_only = await _measure_stream(baseline_backend, monkeypatch)

    model_names = [f"perf-combined-{index}" for index in range(_AGENT_COUNT)]
    child_backend = ContextGatedBackend()
    for _ in range(_AGENT_COUNT):
        child_backend.add_gate()

    combined_backend = MidStreamGatedBackend(_chunks(), pause_at_midpoint=True)
    harness: SubagentHarness | None = None
    heartbeat: asyncio.Task[None] | None = None
    heartbeat_results: tuple[None | BaseException, ...] = ()
    stop_heartbeat = asyncio.Event()
    loop_lag_ms: list[float] = []
    rss_at_gate_mb: list[float] = []
    task_results: list[TaskResult] = []
    rss_before_mb: float | None = None
    combined: StreamMeasurement | None = None
    stream_completed = False
    try:
        harness = await create_subagent_harness(model_names)
        _install_large_child_contexts(harness, child_backend, monkeypatch)
        rss_before_mb = _rss_mb()

        async def launch_and_release() -> None:
            assert harness is not None
            await _start_children_and_release(
                harness,
                child_backend,
                rss_at_gate_mb=rss_at_gate_mb,
                results=task_results,
            )

        heartbeat = asyncio.create_task(
            _heartbeat(stop_heartbeat, loop_lag_ms),
            name=f"perf-combined-loop-heartbeat-{iteration}",
        )
        combined = await _measure_stream(
            combined_backend, monkeypatch, during_midstream=launch_and_release
        )
        stream_completed = True
        assert len(task_results) == _AGENT_COUNT
        await _await_children(harness, task_results)
        assert len(child_backend.started) == _AGENT_COUNT
        assert combined.chunk_count == ASSISTANT_CHUNK_COUNT
        assert rss_at_gate_mb
        assert loop_lag_ms
    finally:
        for release in child_backend.releases:
            release.set()
        combined_backend.resume_stream.set()
        stop_heartbeat.set()
        if heartbeat is not None:
            heartbeat_results = await asyncio.gather(heartbeat, return_exceptions=True)
        if harness is not None:
            await harness.close()
        assert all(
            result is None or isinstance(result, asyncio.CancelledError)
            for result in heartbeat_results
        ), f"heartbeat failed: {heartbeat_results}"

    if streaming_only is None:
        baseline_backend = MidStreamGatedBackend(_chunks(), pause_at_midpoint=False)
        streaming_only = await _measure_stream(baseline_backend, monkeypatch)

    assert combined is not None
    assert rss_before_mb is not None
    assert combined.chunk_count == streaming_only.chunk_count == ASSISTANT_CHUNK_COUNT
    assert combined.total_bytes == streaming_only.total_bytes

    stream_only_flush = percentiles(streaming_only.flush_delay_ms)
    combined_flush = percentiles(combined.flush_delay_ms)
    assert stream_only_flush["p95"] > 0
    degradation_pct = (
        (combined_flush["p95"] - stream_only_flush["p95"])
        / stream_only_flush["p95"]
        * 100
    )
    return {
        "iteration": iteration,
        "measurement_order": measurement_order,
        "stream_only_flush_delay_p50_ms": stream_only_flush["p50"],
        "stream_only_flush_delay_p95_ms": stream_only_flush["p95"],
        "combined_flush_delay_p50_ms": combined_flush["p50"],
        "combined_flush_delay_p95_ms": combined_flush["p95"],
        "flush_delay_p95_degradation_pct": round(degradation_pct, 3),
        "loop_lag_p99_ms": percentiles(loop_lag_ms)["p99"],
        "rss_before_mb": round(rss_before_mb, 3),
        "rss_at_in_flight_context_mb": round(rss_at_gate_mb[0], 3),
        "rss_delta_mb": round(rss_at_gate_mb[0] - rss_before_mb, 3),
        "children_started": len(child_backend.started),
        "child_context_tokens_approx": [
            chars // 4 for chars in child_backend.context_chars
        ],
        "stream_chunk_count": combined.chunk_count,
        "stream_total_bytes": combined.total_bytes,
        "stream_only_turn_wall_ms": round(streaming_only.turn_wall_ms, 3),
        "combined_turn_wall_ms": round(combined.turn_wall_ms, 3),
        "stream_completed": stream_completed,
    }


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(600)
async def test_combined_load(monkeypatch: pytest.MonkeyPatch) -> None:
    iterations = [await _iteration(index, monkeypatch) for index in range(4)]
    assert len(iterations) == 4
    assert [iteration["measurement_order"] for iteration in iterations] == [
        "streaming_only_then_combined",
        "combined_then_streaming_only",
        "streaming_only_then_combined",
        "combined_then_streaming_only",
    ]
    assert all(iteration["stream_completed"] for iteration in iterations)
    assert all(
        iteration["children_started"] == _AGENT_COUNT for iteration in iterations
    )
    assert all(
        iteration["stream_chunk_count"] == ASSISTANT_CHUNK_COUNT
        for iteration in iterations
    )

    record(
        "combined_load_n8_100k_context",
        {
            "directional": True,
            "label": "DIRECTIONAL ONLY",
            "iterations": iterations,
            "machine": machine_context(),
        },
    )


# ---------------------------------------------------------------------------
# P0d: long-history streaming + repeated duration-invalidated full verify
# ---------------------------------------------------------------------------

# Sizing: 20k records matches the P0b persistence fixture denominators, so the
# verify cost recorded here is directly comparable to P0b's isolated
# duration-verify numbers (awaited p50 131.2 ms at 5a1e121f). The verifies run
# back-to-back for as long as the streaming turn lasts, capped at
# _P0D_VERIFY_REPEAT_CAP (worst case roughly 10-20 s of added verify wall time
# at P0b's p50 ceiling); a single comparison pair (stream-only, then combined)
# keeps the whole scenario inside a couple of minutes.
_P0D_VERIFY_RECORD_COUNT = 20_000
_P0D_VERIFY_REPEAT_CAP = 40

# FROZEN CEILINGS (P0d). Non-regression upper bounds for the combined
# streaming + duration-invalidated-verify load at 20k records: exceeding one is
# a blocker, never a threshold change. Baseline: the perf-baselines/
# p0d-combined-load-baseline.jsonl run at 441467b7 (see the artifact's
# .meta.json for machine context): stream-only flush p95 50.965 ms,
# duration-verify awaited p50 238.2 ms under the stream, verify sync-worker p50
# 182.0 ms, verify worker peak 20.1170 MiB (matches P0b's isolated 20.1161 MiB),
# combined loop-lag p50 4.798 ms. Values are 4x (latency) or 6x (memory) the
# recorded baseline, so host load cannot mask a real regression; loop-lag p50
# is frozen at 4x, above the P0c 10 ms floor convention.
# THE MEMORY CEILING IS A FIRST-OBSERVATION BASELINE: no earlier recorded
# combined-load value exists for it, and it is valid on the baseline machine
# only. The combined-phase flush p95 (81.4 ms, +59.6% over stream-only) is
# recorded as a comparison metric, not ceiled — the stream-only p95 ceiling
# guards the render baseline.
_P0D_STREAM_FLUSH_P95_CEILING_MS = 204.0
_P0D_VERIFY_AWAITED_P50_CEILING_MS = 953.0
_P0D_VERIFY_WORKER_P50_CEILING_MS = 728.0
_P0D_VERIFY_WORKER_PEAK_CEILING_MIB = 120.7
_P0D_COMBINED_LOOP_LAG_P50_CEILING_MS = 19.2


@dataclass
class _VerifyMeasurements:
    sync_worker_ms: list[float] = field(default_factory=list)
    full_verify_calls: int = 0
    measure_next_worker: bool = False
    worker_peak_mb: float | None = None


def _perf_tool_manager() -> MagicMock:
    tool_manager = MagicMock(spec=ToolManager)
    tool_manager.available_tools = {}
    tool_manager.available_tool_specs.return_value = []
    return tool_manager


def _instrument_verify_logger(
    logger: SessionLogger, monkeypatch: pytest.MonkeyPatch
) -> _VerifyMeasurements:
    """Count full verifies, time the sync save worker, and trace one peak.

    Duplicated (reduced) from tests/perf/test_persistence.py on purpose: the
    Wave-0 perf files stay file-disjoint, and duplicating the small helper is
    cheaper than coupling perf modules across packages.
    """
    measurements = _VerifyMeasurements()

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


async def _run_duration_invalidated_verifies(
    logger: SessionLogger,
    messages: list[LLMMessage],
    config: ChartreuxConfigSchema,
    tool_manager: ToolManager,
    stats: AgentStats,
    stop: asyncio.Event,
    awaited_ms: list[float],
) -> int:
    """Issue duration-invalidated full verifies until asked to stop.

    Each save pins the save counter one below the verify interval, so the save
    full-verifies because the counter reached the interval, not because the
    cursor was invalidated: the populated read + parse + compare branch.
    """
    count = 0
    while not stop.is_set() and count < _P0D_VERIFY_REPEAT_CAP:
        logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
        started_at = perf_counter()
        await logger.save_interaction(
            messages, stats=stats, config=config, tool_manager=tool_manager
        )
        awaited_ms.append((perf_counter() - started_at) * 1_000)
        count += 1
        await asyncio.sleep(0)
    return count


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(600)
async def test_combined_load_streaming_plus_full_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: ChartreuxConfigSchema = build_test_vibe_config()
    stats = AgentStats()
    tool_manager = _perf_tool_manager()

    message_count = _P0D_VERIFY_RECORD_COUNT
    messages = synthetic_messages(20_000, message_count)
    assert len(messages) == message_count
    session_config = SessionLoggingConfig(
        save_dir=str(tmp_path / "sessions-combined-verify"),
        session_prefix="perf",
        enabled=True,
    )
    logger = SessionLogger(session_config, "combined-load-verify")

    # The initial full save establishes the cursor and the populated transcript;
    # it sits outside the instrumented phases and is not part of the load.
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    assert logger._transcript_cursor is not None
    assert logger.session_dir is not None
    messages_path = logger.messages_filepath
    transcript_file_bytes = messages_path.stat().st_size

    measurements = _instrument_verify_logger(logger, monkeypatch)
    rss_before_mb = _rss_mb()

    # Phase A: the streaming turn alone (no persistence load) — comparison base.
    only_lag_ms: list[float] = []
    stop_only_heartbeat = asyncio.Event()
    only_heartbeat = asyncio.create_task(_heartbeat(stop_only_heartbeat, only_lag_ms))
    try:
        stream_only = await _measure_stream(
            MidStreamGatedBackend(_chunks(), pause_at_midpoint=False), monkeypatch
        )
    finally:
        stop_only_heartbeat.set()
        heartbeat_results = await asyncio.gather(only_heartbeat, return_exceptions=True)
    assert all(
        result is None or isinstance(result, asyncio.CancelledError)
        for result in heartbeat_results
    ), f"stream-only heartbeat failed: {heartbeat_results}"

    # Phase B: the same streaming turn plus repeated duration-invalidated full
    # verifies issued concurrently on the same event loop. The save's sync
    # worker runs in a thread, so this contends on CPU, GIL, and disk with the
    # render loop — the real combined-load shape.
    verify_awaited_ms: list[float] = []
    combined_lag_ms: list[float] = []
    stop_verifies = asyncio.Event()
    stop_combined_heartbeat = asyncio.Event()
    verify_task = asyncio.create_task(
        _run_duration_invalidated_verifies(
            logger,
            messages,
            config,
            tool_manager,
            stats,
            stop_verifies,
            verify_awaited_ms,
        )
    )
    combined_heartbeat = asyncio.create_task(
        _heartbeat(stop_combined_heartbeat, combined_lag_ms)
    )
    try:
        combined = await _measure_stream(
            MidStreamGatedBackend(_chunks(), pause_at_midpoint=False), monkeypatch
        )
    finally:
        stop_verifies.set()
        stop_combined_heartbeat.set()
        verify_count = await verify_task
        heartbeat_results = await asyncio.gather(
            combined_heartbeat, return_exceptions=True
        )
    assert all(
        result is None or isinstance(result, asyncio.CancelledError)
        for result in heartbeat_results
    ), f"combined heartbeat failed: {heartbeat_results}"
    assert verify_count >= 3, f"only {verify_count} verifies ran under the stream"
    assert len(verify_awaited_ms) == verify_count
    assert len(measurements.sync_worker_ms) == verify_count
    assert measurements.full_verify_calls == verify_count

    # One traced repeat for the verify worker peak, outside both streaming
    # phases so tracemalloc cannot inflate the latency samples.
    logger._transcript_saves_since_verify = TRANSCRIPT_VERIFY_INTERVAL - 1
    measurements.measure_next_worker = True
    started_at = perf_counter()
    await logger.save_interaction(
        messages, stats=stats, config=config, tool_manager=tool_manager
    )
    traced_awaited_ms = (perf_counter() - started_at) * 1_000
    assert measurements.full_verify_calls == verify_count + 1
    assert measurements.worker_peak_mb is not None
    verify_worker_peak_mb = measurements.worker_peak_mb

    # Preserved behavior: the repeated verifies never rewrote or appended to
    # the transcript, and the cursor still matches the file on disk.
    cursor = logger._transcript_cursor
    assert cursor is not None
    assert cursor.file_size == messages_path.stat().st_size == transcript_file_bytes
    assert len(messages_path.read_text(encoding="utf-8").splitlines()) == message_count

    assert combined.chunk_count == stream_only.chunk_count == ASSISTANT_CHUNK_COUNT
    assert combined.total_bytes == stream_only.total_bytes

    stream_only_flush = percentiles(stream_only.flush_delay_ms)
    combined_flush = percentiles(combined.flush_delay_ms)
    verify_awaited = percentiles(verify_awaited_ms)
    verify_worker = percentiles(measurements.sync_worker_ms[:verify_count])
    assert stream_only_flush["p95"] <= _P0D_STREAM_FLUSH_P95_CEILING_MS
    assert verify_awaited["p50"] <= _P0D_VERIFY_AWAITED_P50_CEILING_MS
    assert verify_worker["p50"] <= _P0D_VERIFY_WORKER_P50_CEILING_MS
    assert verify_worker_peak_mb <= _P0D_VERIFY_WORKER_PEAK_CEILING_MIB
    combined_lag = percentiles(combined_lag_ms)
    assert combined_lag["p50"] <= _P0D_COMBINED_LOOP_LAG_P50_CEILING_MS

    degradation_pct = (
        (combined_flush["p95"] - stream_only_flush["p95"])
        / stream_only_flush["p95"]
        * 100
    )
    record(
        "combined_load_streaming_plus_full_verify",
        {
            "package": "P0d",
            "record_count": message_count,
            "transcript_file_bytes": transcript_file_bytes,
            "verify_repeats_under_stream": verify_count,
            "full_verify_calls": measurements.full_verify_calls,
            "stream_only_flush_delay_p50_ms": stream_only_flush["p50"],
            "stream_only_flush_delay_p95_ms": stream_only_flush["p95"],
            "combined_flush_delay_p50_ms": combined_flush["p50"],
            "combined_flush_delay_p95_ms": combined_flush["p95"],
            "flush_delay_p95_degradation_pct": round(degradation_pct, 3),
            "loop_lag_p50_ms": {
                "stream_only": percentiles(only_lag_ms)["p50"],
                "combined": combined_lag["p50"],
            },
            "loop_lag_p99_ms": {
                "stream_only": percentiles(only_lag_ms)["p99"],
                "combined": combined_lag["p99"],
            },
            "verify_awaited_ms": verify_awaited,
            "verify_sync_worker_ms": verify_worker,
            "traced_repeat_awaited_ms": round(traced_awaited_ms, 3),
            "verify_worker_peak_mb": verify_worker_peak_mb,
            "verify_worker_peak_note": (
                "tracemalloc peak of one populated duration-invalidated full "
                "verify (read + parse + compare, no transcript rewrite); "
                "first-observation baseline, valid on this machine only"
            ),
            "p0b_isolated_reference_awaited_p50_ms": 131.2,
            "turn_wall_ms": {
                "stream_only": round(stream_only.turn_wall_ms, 3),
                "combined": round(combined.turn_wall_ms, 3),
            },
            "rss_before_mib": round(rss_before_mb, 3),
            "rss_at_end_mb": round(_rss_mb(), 3),
            "stream_chunk_count": combined.chunk_count,
            "stream_total_bytes": combined.total_bytes,
            "memory_baseline_kind": (
                "first-observation baseline (no earlier recorded values; "
                "valid on this machine only)"
            ),
            "machine": machine_context(),
        },
    )
