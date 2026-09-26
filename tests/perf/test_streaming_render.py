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
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage
from chartreux.cli.textual_ui.widgets.tool_widgets import (
    EFFECT_WIDGETS,
    BashResultWidget,
)
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from chartreux.core.llm.types import BackendLike
from chartreux.core.llm_models import LLMChunk
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.mock.utils import mock_llm_chunk
from tests.perf._metrics import machine_context, percentiles, record
from tests.stubs.fake_backend import FakeBackend

ASSISTANT_CHUNK_COUNT = 500
SHELL_LINE_COUNT = 20_000

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)


class YieldingStreamingBackend(FakeBackend):
    """Fake backend paced to allow Textual's 50ms stream flush to run."""

    def __init__(self, chunks: list[LLMChunk]) -> None:
        super().__init__(chunks)
        self.chunks_yielded = 0

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        async for chunk in super().complete_streaming(**kwargs):
            self.chunks_yielded += 1
            yield chunk
            # Keep the 50ms timer observable across the stream and guarantee a
            # final timer flush before the completion event finalizes the widget.
            await asyncio.sleep(
                0.001 if self.chunks_yielded < ASSISTANT_CHUNK_COUNT else 0.075
            )


def _build_streaming_app(backend: BackendLike) -> ChartreuxApp:
    """Build the real Textual app with the requested streaming backend.

    WP5 should reuse this helper/signature for its combined-load stream pilot.
    """
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    return build_test_chartreux_app(agent_loop=agent_loop)


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
