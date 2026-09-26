from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import time
from typing import Any

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
from chartreux.core.llm_models import LLMChunk, LLMMessage
from chartreux.core.subagents import TaskResult
from tests.mock.utils import mock_llm_chunk
from tests.perf._harness import (
    FanOutHarness,
    GatedSequenceBackend,
    create_fan_out_harness,
    launch_fan_out,
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
    harness: FanOutHarness,
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
    harness: FanOutHarness,
    backend: ContextGatedBackend,
    *,
    rss_at_gate_mb: list[float],
    results: list[TaskResult],
) -> None:
    result = await asyncio.wait_for(
        launch_fan_out(harness.registry, harness.context, harness.role), timeout=120
    )
    assert result.members is not None
    assert len(result.members) == _AGENT_COUNT
    assert all(member.status == "running" for member in result.members)

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
    results.append(result)

    for release in backend.releases:
        release.set()


async def _await_children(harness: FanOutHarness, result: TaskResult) -> None:
    assert result.members is not None
    members: list[tuple[str, str]] = []
    for member in result.members:
        if member.agent_id and member.run_id:
            members.append((member.agent_id, member.run_id))
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
    assert all(completion.completed for completion in completions)


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

    role_members = [f"perf-combined-{index}" for index in range(_AGENT_COUNT)]
    child_backend = ContextGatedBackend()
    for _ in range(_AGENT_COUNT):
        child_backend.add_gate()

    combined_backend = MidStreamGatedBackend(_chunks(), pause_at_midpoint=True)
    harness: FanOutHarness | None = None
    heartbeat: asyncio.Task[None] | None = None
    heartbeat_results: tuple[None | BaseException, ...] = ()
    stop_heartbeat = asyncio.Event()
    loop_lag_ms: list[float] = []
    rss_at_gate_mb: list[float] = []
    fan_out_results: list[TaskResult] = []
    rss_before_mb: float | None = None
    combined: StreamMeasurement | None = None
    stream_completed = False
    try:
        harness = await create_fan_out_harness(role_members)
        _install_large_child_contexts(harness, child_backend, monkeypatch)
        rss_before_mb = _rss_mb()

        async def launch_and_release() -> None:
            assert harness is not None
            await _start_children_and_release(
                harness,
                child_backend,
                rss_at_gate_mb=rss_at_gate_mb,
                results=fan_out_results,
            )

        heartbeat = asyncio.create_task(
            _heartbeat(stop_heartbeat, loop_lag_ms),
            name=f"perf-combined-loop-heartbeat-{iteration}",
        )
        combined = await _measure_stream(
            combined_backend, monkeypatch, during_midstream=launch_and_release
        )
        stream_completed = True
        assert len(fan_out_results) == 1
        await _await_children(harness, fan_out_results[0])
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
