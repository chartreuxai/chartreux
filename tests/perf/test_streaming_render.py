from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
import sys
from time import perf_counter, perf_counter_ns
from typing import Any

import pytest
from textual.geometry import Size
from textual.pilot import Pilot
from textual.screen import Screen

from chartreux.app_server.events import (
    AppServerEvent,
    HistoryEntryAdded,
    HistoryEntryUpdated,
)
from chartreux.app_server.models import (
    CompletedEffectState,
    PublicEffectEntry,
    PublicMessageEntry,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import (
    STREAM_WRITE_FRAME_SECONDS_MAX,
    AssistantMessage,
    ReasoningMessage,
    StreamingMessageBase,
)
from chartreux.cli.textual_ui.widgets.tool_widgets import (
    EFFECT_WIDGETS,
    BashResultWidget,
)
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm_models import LLMChunk
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.perf._metrics import (
    machine_context,
    measure_loop_lag,
    percentiles,
    record,
    rss_mib,
    rss_peak_mib,
)
from tests.stubs.fake_backend import FakeBackend

ASSISTANT_CHUNK_COUNT = 500
SHELL_LINE_COUNT = 20_000

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)


class YieldingStreamingBackend(FakeBackend):
    """Fake backend paced to allow Textual's 50ms stream flush to run."""

    def __init__(
        self, chunks: list[LLMChunk], *, total_chunks: int | None = None
    ) -> None:
        super().__init__(chunks)
        self.chunks_yielded = 0
        self._chunk_total = len(chunks) if total_chunks is None else total_chunks

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        async for chunk in super().complete_streaming(**kwargs):
            self.chunks_yielded += 1
            yield chunk
            # Keep the 50ms timer observable across the stream and guarantee a
            # final timer flush before the completion event finalizes the widget.
            await asyncio.sleep(
                0.001 if self.chunks_yielded < self._chunk_total else 0.075
            )


def _build_streaming_app(backend: BackendLike, *, config: Any = None) -> ChartreuxApp:
    """Build the real Textual app with the requested streaming backend.

    ``config`` is passed to BOTH the agent loop and the app so content-shape
    fixtures can flip UI-affecting settings (e.g. show_thinking_nodes) on the
    same config object. WP5 should reuse this helper/signature for its
    combined-load scenario.
    """
    agent_loop = build_test_agent_loop(
        backend=backend, enable_streaming=True, config=config
    )
    return build_test_chartreux_app(config=config, agent_loop=agent_loop)


async def _wait_until(
    pilot: Pilot[ChartreuxApp], predicate: Callable[[], bool], *, timeout: float = 30.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("performance pilot condition did not become true")
        await pilot.pause(0.01)


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.parametrize("iteration", range(5))
async def test_streaming_assistant_markdown(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk_texts = [f"token-{index:03d} " for index in range(ASSISTANT_CHUNK_COUNT)]
    chunks = [
        mock_llm_chunk(
            content=text,
            stop_reason="stop" if index == ASSISTANT_CHUNK_COUNT - 1 else None,
        )
        for index, text in enumerate(chunk_texts)
    ]
    backend = YieldingStreamingBackend(chunks)
    app = _build_streaming_app(backend)
    chunk_end_offsets: list[int] = []
    total_chars = 0
    for text in chunk_texts:
        total_chars += len(text)
        chunk_end_offsets.append(total_chars)

    flush_delay_ms: list[float] = []
    handler_latency_ms: list[float] = []
    chunk_arrival_times: list[float | None] = [None] * ASSISTANT_CHUNK_COUNT
    flushed_chunk_count = [0]
    timer_flush_count = [0]
    received_chars = 0
    submitted_at: float | None = None
    first_visible_ms: list[float | None] = [None]

    def observe_visible_content(message: AssistantMessage) -> None:
        if submitted_at is None or message._markdown is None:
            return
        visible_chars = len(message._markdown._markdown)
        visible_at = perf_counter()
        if visible_chars > 0 and first_visible_ms[0] is None:
            first_visible_ms[0] = (visible_at - submitted_at) * 1000
        while (
            flushed_chunk_count[0] < len(chunk_end_offsets)
            and chunk_end_offsets[flushed_chunk_count[0]] <= visible_chars
        ):
            arrival_at = chunk_arrival_times[flushed_chunk_count[0]]
            if arrival_at is None:
                raise AssertionError("visible assistant chunk has no arrival timestamp")
            flush_delay_ms.append((visible_at - arrival_at) * 1000)
            flushed_chunk_count[0] += 1

    original_flush = AssistantMessage._flush_write_buffer
    original_initial_write = AssistantMessage.write_initial_content

    async def observe_timer_flush(message: AssistantMessage) -> None:
        await original_flush(message)
        timer_flush_count[0] += 1
        observe_visible_content(message)

    async def observe_initial_write(message: AssistantMessage) -> None:
        await original_initial_write(message)
        observe_visible_content(message)

    monkeypatch.setattr(AssistantMessage, "_flush_write_buffer", observe_timer_flush)
    monkeypatch.setattr(
        AssistantMessage, "write_initial_content", observe_initial_write
    )

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        session = app.app_server
        queue_max_events = [0]
        queue_max_unsolicited = [0]
        sampling_stopped = asyncio.Event()

        async def sample_event_queues() -> None:
            while not sampling_stopped.is_set():
                queue_max_events[0] = max(queue_max_events[0], session._events.qsize())
                queue_max_unsolicited[0] = max(
                    queue_max_unsolicited[0], session._unsolicited_events.qsize()
                )
                await asyncio.sleep(0.005)
            queue_max_events[0] = max(queue_max_events[0], session._events.qsize())
            queue_max_unsolicited[0] = max(
                queue_max_unsolicited[0], session._unsolicited_events.qsize()
            )

        original_handler = app._handle_turn_event

        async def time_bridge_handler(event: AppServerEvent) -> None:
            nonlocal received_chars
            started = perf_counter_ns()
            arrival_at = perf_counter()
            queue_max_events[0] = max(queue_max_events[0], session._events.qsize())
            queue_max_unsolicited[0] = max(
                queue_max_unsolicited[0], session._unsolicited_events.qsize()
            )
            assistant_delta = ""
            match event:
                case HistoryEntryAdded(
                    entry=PublicMessageEntry(role="assistant") as entry
                ):
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
            try:
                await original_handler(event)
            finally:
                handler_latency_ms.append((perf_counter_ns() - started) / 1_000_000)

        app._handle_turn_event = time_bridge_handler
        sampler = asyncio.create_task(sample_event_queues())
        try:
            chat_input = app.query_one(ChatInputContainer)
            submitted_at = perf_counter()
            chat_input.post_message(
                ChatInputContainer.Submitted("streaming perf pilot")
            )
            await _wait_until(
                pilot, lambda: backend.chunks_yielded == ASSISTANT_CHUNK_COUNT
            )
            await _wait_until(pilot, lambda: not app._agent_job_active(), timeout=120.0)
            turn_wall_ms = (perf_counter() - submitted_at) * 1000
            assistant = app.query_one(AssistantMessage)
            assert assistant.get_content() == "".join(chunk_texts)
            assert len(assistant.get_content().encode("utf-8")) == len(
                "".join(chunk_texts).encode("utf-8")
            )
            assert backend.chunks_yielded == ASSISTANT_CHUNK_COUNT
            assert received_chars == total_chars
            assert all(arrival is not None for arrival in chunk_arrival_times)
            assert flushed_chunk_count[0] == ASSISTANT_CHUNK_COUNT
            assert timer_flush_count[0] > 0
            assert len(flush_delay_ms) == ASSISTANT_CHUNK_COUNT
            assert first_visible_ms[0] is not None
            assert handler_latency_ms
        finally:
            sampling_stopped.set()
            await sampler

    output_bytes = len("".join(chunk_texts).encode("utf-8"))
    flush_delay_percentiles = percentiles(flush_delay_ms)
    handler_percentiles = percentiles(handler_latency_ms)
    first_visible_latency = first_visible_ms[0]
    assert first_visible_latency is not None
    record(
        "streaming_assistant_markdown",
        {
            "iteration": iteration,
            "chunk_count": backend.chunks_yielded,
            "total_bytes": output_bytes,
            "flush_delay_p50": flush_delay_percentiles["p50"],
            "flush_delay_p95": flush_delay_percentiles["p95"],
            "first_visible_ms": round(first_visible_latency, 3),
            "handler_p50": handler_percentiles["p50"],
            "queue_max_events": queue_max_events[0],
            "queue_max_unsolicited": queue_max_unsolicited[0],
            "timer_flush_count": timer_flush_count[0],
            "turn_wall": round(turn_wall_ms, 3),
            "machine": machine_context(),
        },
    )


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(300)
@pytest.mark.parametrize("iteration", range(5))
async def test_streaming_shell_output(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Live-path trace: manual shell output is emitted as appended effect-output
    # patches. EventHandler._handle_entry_updated forwards each delta to
    # ToolCallMessage.set_stream_message; updates are coalesced into 50ms visual
    # frames, while the completed result maps Shell to BashResultWidget.
    append_latency_ms: list[float] = []
    output_chunk_bytes: list[int] = []
    original_set_stream_message = ToolCallMessage.set_stream_message

    def time_live_output_refresh(call: ToolCallMessage, text: str) -> None:
        started = perf_counter_ns()
        try:
            original_set_stream_message(call, text)
        finally:
            append_latency_ms.append((perf_counter_ns() - started) / 1_000_000)
            output_chunk_bytes.append(len(text.encode("utf-8")))

    monkeypatch.setattr(ToolCallMessage, "set_stream_message", time_live_output_refresh)
    app = build_test_chartreux_app()
    refresh_passes = [0]
    measure_refresh_passes = [False]
    original_refresh_layout = Screen._refresh_layout

    def count_refresh_passes(
        screen: Screen, size: Size | None = None, scroll: bool = False
    ) -> None:
        if measure_refresh_passes[0] and screen.app is app:
            refresh_passes[0] += 1
        original_refresh_layout(screen, size, scroll)

    monkeypatch.setattr(Screen, "_refresh_layout", count_refresh_passes)
    output_command = "python -c \"for i in range(20000): print(f'{i:05d} ' + 'x'*74)\""
    expected_output = "".join(
        f"{line:05d} {'x' * 74}\n" for line in range(SHELL_LINE_COUNT)
    )
    expected_bytes = len(expected_output.encode("utf-8"))

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        submitted_at = perf_counter()
        measure_refresh_passes[0] = True
        chat_input = app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted(f"!{output_command}"))
        await _wait_until(pilot, lambda: app._bash_task is not None, timeout=10.0)
        await _wait_until(pilot, lambda: app._bash_task is None, timeout=240.0)
        turn_wall_ms = (perf_counter() - submitted_at) * 1000
        measure_refresh_passes[0] = False
        await pilot.pause(0.05)

        shell_entries = [
            entry
            for entry in app.app_server.resources.sessions.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SHELL
        ]
        assert len(shell_entries) == 1
        shell_entry = shell_entries[0]
        assert isinstance(shell_entry.state, CompletedEffectState)
        assert shell_entry.state.output_text == expected_output
        assert len(shell_entry.state.output_text.encode("utf-8")) == expected_bytes
        assert shell_entry.state.output_text.count("\n") == SHELL_LINE_COUNT
        assert len(output_chunk_bytes) == len(append_latency_ms)
        assert len(append_latency_ms) > 1
        assert refresh_passes[0] < len(append_latency_ms) // 3
        assert sum(output_chunk_bytes) == expected_bytes
        assert EFFECT_WIDGETS[ToolEffectKind.SHELL].result is BashResultWidget
        result_message = app.query_one(ToolResultMessage)
        assert result_message._entry.id == shell_entry.id

    append_first_ms = append_latency_ms[0]
    append_last_ms = append_latency_ms[-1]
    append_cost_by_decile_ms: list[dict[str, float | int]] = []
    decile_count = min(10, len(append_latency_ms))
    for decile in range(decile_count):
        chunk_start = decile * len(append_latency_ms) // decile_count
        chunk_end = (decile + 1) * len(append_latency_ms) // decile_count
        decile_percentiles = percentiles(append_latency_ms[chunk_start:chunk_end])
        append_cost_by_decile_ms.append({
            "decile": decile + 1,
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
            "p50": decile_percentiles["p50"],
            "p95": decile_percentiles["p95"],
        })
    record(
        "streaming_shell_output",
        {
            "iteration": iteration,
            "chunk_count": len(output_chunk_bytes),
            "total_bytes": sum(output_chunk_bytes),
            "append_first_ms": round(append_first_ms, 3),
            "append_last_ms": round(append_last_ms, 3),
            "append_last_vs_first_ratio": round(
                append_last_ms / append_first_ms if append_first_ms else 0.0, 3
            ),
            "append_latency_ms": percentiles(append_latency_ms),
            "append_cost_by_decile_ms": append_cost_by_decile_ms,
            "refresh_passes": refresh_passes[0],
            "turn_wall_ms": round(turn_wall_ms, 3),
            "live_stream_widget": "ToolCallMessage.set_stream_message",
            "final_result_widget": "BashResultWidget",
            "machine": machine_context(),
        },
    )


# ---------------------------------------------------------------------------
# P0c — streaming content-shape baselines. Two DISTINCT kinds of frozen values
# live below:
#
# CEILINGS (non-regression upper bounds, asserted in the tests): exceeding one
# is a blocker. Frozen at 4x the recorded baseline for latencies (P0a
# convention) and 6x for RSS (P0b convention), so host load cannot mask a real
# regression.
#
# The LONG-STREAM OPTIMIZATION TARGET (_P0C_LONG_STREAM_FRAME_P95_TARGET_MS) was
# frozen as the P5 trigger and is NOT a P0c ceiling. P5 is gated ON (its
# recorded baseline exceeded the target), so the target is asserted from P5
# onward: the long-stream fixture must meet or beat it. The 150ms figure below
# is P5's implementation cap on its own change, a different value entirely.
#
# All memory metrics below are FIRST-OBSERVATION BASELINES: no earlier recorded
# values exist, and they are valid on the baseline machine only.
# ---------------------------------------------------------------------------

_P0C_STREAM_ITERATIONS = 3
_P0C_LONG_STREAM_CHUNK_COUNT = 2000
_P0C_SHAPE_CHUNK_COUNT = 400
# 400 streamed bullets cost a 176s turn wall (frame p50 ~640ms) in the
# measurement run — the pathological long-list shape P5 targets — too heavy
# for a bounded baseline fixture; frozen at 120 items instead, with the 400-item
# finding recorded in the P0c baseline meta.
_P0C_LIST_ITEM_COUNT = 120
_P0C_REASONING_CHUNK_COUNT = 600

# CEILINGS frozen from the 2026-10-29 measurement run at commit 4e0d579c on the
# baseline machine (AMD Ryzen 9 5900X, 24 cores): 4x the recorded baseline for
# latencies, 6x for RSS, with a 10ms floor on loop-lag p50 so tiny baselines do
# not produce flake-prone bounds. See perf-baselines/p0c-streaming-baseline.meta.json.
_P0C_FRAME_P95_CEILINGS_MS: dict[str, float] = {
    "long_paragraph": 692.0,
    "fenced_code": 659.0,
    "list": 1992.0,
    "expanded_reasoning": 545.0,
}
# fenced_code/expanded_reasoning flush-cost p95 ceilings removed at freeze calibration (heavy-tail jitter, user-approved 2026-10-28).
_P0C_FLUSH_P95_CEILINGS_MS: dict[str, float] = {"long_paragraph": 17.0, "list": 35.4}
_P0C_LOOP_LAG_P50_CEILINGS_MS: dict[str, float] = {
    "long_paragraph": 11.2,
    "fenced_code": 10.0,
    "list": 89.0,
    "expanded_reasoning": 10.0,
}
_P0C_RSS_PEAK_CEILING_MIB = 1954.0

# LONG-STREAM OPTIMIZATION TARGET — frozen as the P5 trigger from the recorded
# long-paragraph baseline (median frame-interval p95 of the measurement run's
# three iterations: 160.989 / 172.886 / 167.681 -> 168.0) against the 176.677 ms
# recorded P0c denominator. P5 asserts it on the long-paragraph fixture: the
# adaptive cadence must meet or beat the target. The 150ms figure below is P5's
# implementation cap on the scheduled interval, a different value entirely.
_P0C_LONG_STREAM_FRAME_P95_TARGET_MS = 168.0
_P0C_LONG_STREAM_FRAME_P95_BASELINE_MS = 176.677

# P5 implementation cap: the scheduled batching interval never exceeds 150ms.
_P5_SCHEDULED_INTERVAL_CAP_SECONDS = 0.15


def _stream_chunks(texts: list[str]) -> list[LLMChunk]:
    return [
        mock_llm_chunk(
            content=text, stop_reason="stop" if index == len(texts) - 1 else None
        )
        for index, text in enumerate(texts)
    ]


def _long_paragraph_texts() -> list[str]:
    texts = []
    for index in range(_P0C_LONG_STREAM_CHUNK_COUNT):
        text = (
            f"Clause {index:05d} feeds the renderer another substantive piece of "
            "prose, so each flushed frame must reflow a paragraph that keeps "
            "growing instead of appending isolated short tokens"
        )
        text += ".\n\n" if index % 20 == 19 else ". "
        texts.append(text)
    return texts


def _fenced_code_texts() -> list[str]:
    texts = ["Streaming a module into a live fence:\n\n```python\n"]
    for index in range(_P0C_SHAPE_CHUNK_COUNT - 2):
        if index % 4 == 0:
            texts.append(f"def handler_{index:03d}(payload: dict) -> bool:\n")
        elif index % 4 == 1:
            texts.append(f"    return payload.get('kind_{index:03d}') is not None\n")
        elif index % 4 == 2:
            texts.append("\n\n")
        else:
            texts.append(f"# tail comment {index:03d} keeps the fence open\n")
    texts.append("```\n")
    texts.append("The fence closed after the streamed module.")
    return texts


def _list_texts() -> list[str]:
    texts = ["Plan under construction:\n\n"]
    for index in range(_P0C_LIST_ITEM_COUNT):
        texts.append(
            f"- Item {index:03d}: a streamed bullet whose text is long enough to "
            "wrap and force list-aware markdown layout on each flush\n"
        )
    return texts


def _reasoning_chunks() -> list[LLMChunk]:
    chunks = [
        mock_llm_chunk(
            content="",
            reasoning_content=(
                f"Step {index:04d} weighs the streaming cadence against the flush "
                "budget, since expanded reasoning renders live while the spinner "
                "spins and every delta extends the reasoning markdown. "
            ),
            stop_reason=None,
        )
        for index in range(_P0C_REASONING_CHUNK_COUNT)
    ]
    chunks.append(
        mock_llm_chunk(content="Done - reasoning complete.", stop_reason="stop")
    )
    return chunks


class StreamingFrameObserver:
    """Records flush frame cadence and cost across all streaming widgets.

    Frames are start-to-start intervals between non-empty write-buffer flushes,
    so they capture the visible update cadence for both AssistantMessage and
    ReasoningMessage (both flush through StreamingMessageBase).
    """

    def __init__(self) -> None:
        self.frame_intervals_ms: list[float] = []
        self.flush_costs_ms: list[float] = []
        self.scheduled_delays_s: list[float] = []
        self.flush_count = 0
        self._last_flush_started: float | None = None

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = StreamingMessageBase._flush_write_buffer

        async def timed_flush(stream_widget: StreamingMessageBase) -> None:
            if not stream_widget._to_write_buffer:
                await original(stream_widget)
                return
            # The delay that scheduled this frame; the adaptive cadence must keep
            # every one of them under the P5 implementation cap.
            self.scheduled_delays_s.append(stream_widget._frame_delay)
            started = perf_counter()
            if self._last_flush_started is not None:
                self.frame_intervals_ms.append(
                    (started - self._last_flush_started) * 1000
                )
            self._last_flush_started = started
            started_ns = perf_counter_ns()
            try:
                await original(stream_widget)
            finally:
                self.flush_costs_ms.append((perf_counter_ns() - started_ns) / 1_000_000)
                self.flush_count += 1

        monkeypatch.setattr(StreamingMessageBase, "_flush_write_buffer", timed_flush)


async def _stream_fixture_turn(
    monkeypatch: pytest.MonkeyPatch,
    chunks: list[LLMChunk],
    *,
    assert_final: Callable[[ChartreuxApp], None],
    config: Any = None,
    expand_reasoning: bool = False,
) -> dict[str, Any]:
    """Stream ``chunks`` through the real app; measure frames and loop lag."""
    backend = YieldingStreamingBackend(chunks)
    app = _build_streaming_app(backend, config=config)
    observer = StreamingFrameObserver()
    observer.install(monkeypatch)
    turn_walls_ms: list[float] = []

    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        async def run_turn() -> None:
            submitted_at = perf_counter()
            chat_input = app.query_one(ChatInputContainer)
            chat_input.post_message(
                ChatInputContainer.Submitted("streaming perf pilot")
            )
            if expand_reasoning:
                # Reasoning nodes always mount collapsed (expansion_state
                # register default); expanding mid-stream is the real user
                # flow that turns on live reasoning rendering.
                await _wait_until(pilot, lambda: len(app.query(ReasoningMessage)) > 0)
                await app.query_one(ReasoningMessage).set_collapsed(False)
            await _wait_until(
                pilot, lambda: backend.chunks_yielded == len(chunks), timeout=240.0
            )
            await _wait_until(pilot, lambda: not app._agent_job_active(), timeout=180.0)
            turn_walls_ms.append((perf_counter() - submitted_at) * 1000)
            assert_final(app)

        lag = await measure_loop_lag(run_turn(), interval_s=0.005)

    assert observer.flush_count > 0
    assert observer.frame_intervals_ms
    return {
        "chunk_count": backend.chunks_yielded,
        "flush_count": observer.flush_count,
        "frame_interval_ms": percentiles(observer.frame_intervals_ms),
        "flush_cost_ms": percentiles(observer.flush_costs_ms),
        "max_scheduled_delay_s": max(observer.scheduled_delays_s),
        "loop_lag_ms": lag,
        "turn_wall_ms": round(turn_walls_ms[0], 3),
        "rss_current_mib": rss_mib(),
        # Process-wide monotonic high-water mark: includes prior tests in the
        # same worker process, so this is a lower-bound attribution.
        "rss_peak_mib": rss_peak_mib(),
    }


def _assert_p5_scheduled_interval_cap(metrics: dict[str, Any]) -> None:
    """P5: no scheduled batching interval may exceed the 150ms implementation cap."""
    assert STREAM_WRITE_FRAME_SECONDS_MAX <= _P5_SCHEDULED_INTERVAL_CAP_SECONDS, (
        f"configured frame-delay cap {STREAM_WRITE_FRAME_SECONDS_MAX}s exceeds "
        f"the P5 implementation cap {_P5_SCHEDULED_INTERVAL_CAP_SECONDS}s"
    )
    assert metrics["max_scheduled_delay_s"] <= _P5_SCHEDULED_INTERVAL_CAP_SECONDS, (
        f"scheduled frame delay {metrics['max_scheduled_delay_s']}s exceeds the "
        f"P5 implementation cap {_P5_SCHEDULED_INTERVAL_CAP_SECONDS}s"
    )


def _assert_p5_long_stream_target(metrics: dict[str, Any]) -> None:
    """P5: the long-stream frame interval must meet or beat the frozen target."""
    p95 = metrics["frame_interval_ms"]["p95"]
    assert p95 <= _P0C_LONG_STREAM_FRAME_P95_TARGET_MS, (
        f"long-stream frame-interval p95 {p95}ms misses the P5 target "
        f"{_P0C_LONG_STREAM_FRAME_P95_TARGET_MS}ms (P0c baseline "
        f"{_P0C_LONG_STREAM_FRAME_P95_BASELINE_MS}ms)"
    )


def _assert_p0c_ceilings(fixture: str, metrics: dict[str, Any]) -> None:
    assert metrics["frame_interval_ms"]["p95"] <= _P0C_FRAME_P95_CEILINGS_MS[fixture], (
        f"frame-interval p95 {metrics['frame_interval_ms']['p95']} exceeds ceiling "
        f"{_P0C_FRAME_P95_CEILINGS_MS[fixture]} for {fixture}"
    )
    flush_p95_ceiling = _P0C_FLUSH_P95_CEILINGS_MS.get(fixture)
    if flush_p95_ceiling is not None:
        assert metrics["flush_cost_ms"]["p95"] <= flush_p95_ceiling, (
            f"flush-cost p95 {metrics['flush_cost_ms']['p95']} exceeds ceiling "
            f"{flush_p95_ceiling} for {fixture}"
        )
    assert metrics["loop_lag_ms"]["p50"] <= _P0C_LOOP_LAG_P50_CEILINGS_MS[fixture], (
        f"loop-lag p50 {metrics['loop_lag_ms']['p50']} exceeds ceiling "
        f"{_P0C_LOOP_LAG_P50_CEILINGS_MS[fixture]} for {fixture}"
    )
    assert metrics["rss_peak_mib"] <= _P0C_RSS_PEAK_CEILING_MIB, (
        f"rss peak {metrics['rss_peak_mib']} exceeds ceiling "
        f"{_P0C_RSS_PEAK_CEILING_MIB}"
    )
    _assert_p5_scheduled_interval_cap(metrics)


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(600)
@pytest.mark.parametrize("iteration", range(_P0C_STREAM_ITERATIONS))
async def test_streaming_long_paragraph_frames(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = _long_paragraph_texts()

    def assert_final(app: ChartreuxApp) -> None:
        assert app.query_one(AssistantMessage).get_content() == "".join(texts)

    metrics = await _stream_fixture_turn(
        monkeypatch, _stream_chunks(texts), assert_final=assert_final
    )
    _assert_p0c_ceilings("long_paragraph", metrics)
    _assert_p5_long_stream_target(metrics)
    record(
        "streaming_long_paragraph_frames",
        {
            "iteration": iteration,
            "fixture": "long-paragraph",
            "total_bytes": len("".join(texts).encode()),
            "p5_long_stream_frame_p95_target_ms": _P0C_LONG_STREAM_FRAME_P95_TARGET_MS,
            "p5_long_stream_frame_p95_baseline_ms": (
                _P0C_LONG_STREAM_FRAME_P95_BASELINE_MS
            ),
            "p5_scheduled_interval_cap_s": _P5_SCHEDULED_INTERVAL_CAP_SECONDS,
            **metrics,
            "machine": machine_context(),
        },
    )


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(600)
@pytest.mark.parametrize("iteration", range(_P0C_STREAM_ITERATIONS))
async def test_streaming_fenced_code_frames(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = _fenced_code_texts()

    def assert_final(app: ChartreuxApp) -> None:
        content = app.query_one(AssistantMessage).get_content()
        assert content == "".join(texts)
        assert content.count("```") == 2

    metrics = await _stream_fixture_turn(
        monkeypatch, _stream_chunks(texts), assert_final=assert_final
    )
    _assert_p0c_ceilings("fenced_code", metrics)
    record(
        "streaming_fenced_code_frames",
        {
            "iteration": iteration,
            "fixture": "fenced-code",
            "total_bytes": len("".join(texts).encode()),
            **metrics,
            "machine": machine_context(),
        },
    )


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(600)
@pytest.mark.parametrize("iteration", range(_P0C_STREAM_ITERATIONS))
async def test_streaming_list_frames(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts = _list_texts()

    def assert_final(app: ChartreuxApp) -> None:
        content = app.query_one(AssistantMessage).get_content()
        assert content == "".join(texts)
        assert content.count("\n- ") == _P0C_LIST_ITEM_COUNT

    metrics = await _stream_fixture_turn(
        monkeypatch, _stream_chunks(texts), assert_final=assert_final
    )
    _assert_p0c_ceilings("list", metrics)
    record(
        "streaming_list_frames",
        {
            "iteration": iteration,
            "fixture": "list",
            "total_bytes": len("".join(texts).encode()),
            **metrics,
            "machine": machine_context(),
        },
    )


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(600)
@pytest.mark.parametrize("iteration", range(_P0C_STREAM_ITERATIONS))
async def test_streaming_expanded_reasoning_frames(
    iteration: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunks = _reasoning_chunks()
    reasoning_text = "".join(
        chunk.message.reasoning_content or "" for chunk in chunks[:-1]
    )
    final_text = chunks[-1].message.content or ""
    config = build_test_vibe_config(show_thinking_nodes=True)

    def assert_final(app: ChartreuxApp) -> None:
        reasoning = app.query_one(ReasoningMessage)
        assert reasoning.get_content() == reasoning_text
        assistant = app.query_one(AssistantMessage)
        assert assistant.get_content() == final_text

    metrics = await _stream_fixture_turn(
        monkeypatch,
        chunks,
        assert_final=assert_final,
        config=config,
        expand_reasoning=True,
    )
    _assert_p0c_ceilings("expanded_reasoning", metrics)
    record(
        "streaming_expanded_reasoning_frames",
        {
            "iteration": iteration,
            "fixture": "expanded-reasoning",
            "total_bytes": len((reasoning_text + final_text).encode()),
            **metrics,
            "machine": machine_context(),
        },
    )
