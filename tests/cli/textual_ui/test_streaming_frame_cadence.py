"""P5 adaptive streaming cadence: frame scheduling and the stream-write paths.

The perf suites measure the cadence end to end but never drive the individual
write-path branches (terminal flush, cancelled frame timer, scroll-away/back,
expanded reasoning), so those live here. The cadence tests run on a fake clock
so the correction law is deterministic.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import monotonic as _real_monotonic
from typing import cast

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.pilot import Pilot
from textual.timer import Timer, TimerCallback

import chartreux.cli.textual_ui.widgets.messages as messages_module
from chartreux.cli.textual_ui.widgets.messages import (
    STREAM_WRITE_FRAME_DEADBAND_SECONDS,
    STREAM_WRITE_FRAME_GAP_RESET_SECONDS,
    STREAM_WRITE_FRAME_SECONDS,
    STREAM_WRITE_FRAME_SECONDS_MAX,
    STREAM_WRITE_FRAME_SECONDS_MIN,
    AssistantMessage,
    ReasoningMessage,
    StreamingMessageBase,
)

_P5_SCHEDULED_INTERVAL_CAP_SECONDS = 0.15


class FakeStream:
    def __init__(self) -> None:
        self.written: list[str] = []
        self.stopped = False

    async def write(self, content: str) -> None:
        self.written.append(content)

    async def stop(self) -> None:
        self.stopped = True

    @property
    def all_written(self) -> str:
        return "".join(self.written)


class FakeTimer:
    def __init__(self, callback: Callable[[], Awaitable[None]]) -> None:
        self.callback = callback
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    async def fire(self) -> None:
        await self.callback()


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class CadenceMessage(StreamingMessageBase):
    """Minimal StreamingMessageBase double that records the scheduled delays."""

    def __init__(self, *, at_bottom: bool = True) -> None:
        self._content = ""
        self._content_initialized = False
        self._to_write_buffer = ""
        self._write_timer = None
        self._stream = None
        self._markdown = None
        self._frame_delay = STREAM_WRITE_FRAME_SECONDS
        self._last_frame_started = None
        self._at_bottom = at_bottom
        self._fake_stream = FakeStream()
        self._fake_timer: FakeTimer | None = None
        self.scheduled_delays: list[float] = []

    def set_timer(
        self,
        delay: float,
        callback: TimerCallback | None = None,
        *,
        name: str | None = None,
        pause: bool = False,
    ) -> Timer:
        assert callback is not None
        self.scheduled_delays.append(delay)
        self._fake_timer = FakeTimer(cast(Callable[[], Awaitable[None]], callback))
        return cast(Timer, self._fake_timer)

    async def fire_timer(self) -> None:
        assert self._fake_timer is not None
        await self._fake_timer.fire()

    def _ensure_stream(self) -> FakeStream:  # type: ignore[override]
        if self._stream is None:
            self._stream = self._fake_stream  # type: ignore[assignment]
        return self._fake_stream

    def _is_chat_at_bottom(self) -> bool:
        return self._at_bottom


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    fake = Clock()
    monkeypatch.setattr(messages_module, "monotonic", fake)
    return fake


async def _drive_frames(
    msg: CadenceMessage, clock: Clock, *, count: int, spacing: float
) -> None:
    for _ in range(count):
        await msg.append_content("tick ")
        clock.advance(spacing)
        await msg.fire_timer()


class TestFrameCadence:
    @pytest.mark.asyncio
    async def test_first_frame_uses_the_base_cadence(self, clock: Clock) -> None:
        msg = CadenceMessage()

        await msg.append_content("hello")

        assert msg.scheduled_delays == [STREAM_WRITE_FRAME_SECONDS]

    @pytest.mark.asyncio
    async def test_late_frames_tighten_toward_the_floor(self, clock: Clock) -> None:
        msg = CadenceMessage()

        # The pump hands each frame its turn 80ms after the previous one, well
        # past the 50ms base cadence: the delay must shrink, never below MIN.
        await _drive_frames(msg, clock, count=6, spacing=0.08)

        assert msg._frame_delay == STREAM_WRITE_FRAME_SECONDS_MIN
        assert min(msg.scheduled_delays) >= STREAM_WRITE_FRAME_SECONDS_MIN
        assert max(msg.scheduled_delays) <= STREAM_WRITE_FRAME_SECONDS_MAX

    @pytest.mark.asyncio
    async def test_prompt_frames_relax_back_up_to_the_cap(self, clock: Clock) -> None:
        msg = CadenceMessage()
        msg._frame_delay = STREAM_WRITE_FRAME_SECONDS_MIN

        # Frames land far earlier than the target period, so batching may grow —
        # the scheduled interval still never exceeds the 150ms implementation cap.
        await _drive_frames(msg, clock, count=40, spacing=0.004)

        assert msg._frame_delay == STREAM_WRITE_FRAME_SECONDS_MAX
        assert max(msg.scheduled_delays) <= _P5_SCHEDULED_INTERVAL_CAP_SECONDS
        assert STREAM_WRITE_FRAME_SECONDS_MAX <= _P5_SCHEDULED_INTERVAL_CAP_SECONDS

    @pytest.mark.asyncio
    async def test_on_target_frames_leave_the_delay_untouched(
        self, clock: Clock
    ) -> None:
        msg = CadenceMessage()
        msg._frame_delay = 0.04

        # Within the deadband of the 50ms target: no correction, no jitter.
        await _drive_frames(msg, clock, count=5, spacing=0.046)

        assert msg._frame_delay == 0.04

    @pytest.mark.asyncio
    async def test_deadband_absorbs_tiny_overshoot(self, clock: Clock) -> None:
        msg = CadenceMessage()
        target_overshoot = (
            STREAM_WRITE_FRAME_SECONDS + STREAM_WRITE_FRAME_DEADBAND_SECONDS
        )

        await _drive_frames(msg, clock, count=5, spacing=target_overshoot)

        assert msg._frame_delay == STREAM_WRITE_FRAME_SECONDS

    @pytest.mark.asyncio
    async def test_idle_gap_restarts_tracking_without_correction(
        self, clock: Clock
    ) -> None:
        msg = CadenceMessage()
        msg._frame_delay = 0.03
        await _drive_frames(msg, clock, count=1, spacing=0.03)

        clock.advance(STREAM_WRITE_FRAME_GAP_RESET_SECONDS + 1.0)
        await msg.append_content("after the pause")
        await msg.fire_timer()

        assert msg._frame_delay == 0.03
        assert msg._last_frame_started == pytest.approx(clock.now, abs=0.001)


class TestScrollAwayAndBack:
    @pytest.mark.asyncio
    async def test_scrolled_away_streams_arm_no_frame_timer(self, clock: Clock) -> None:
        msg = CadenceMessage(at_bottom=False)

        await msg.append_content("one ")
        await msg.append_content("two")

        assert msg.scheduled_delays == []
        assert msg._write_timer is None
        assert msg._last_frame_started is None
        assert msg._to_write_buffer == "one two"
        assert msg._fake_stream.all_written == ""

    @pytest.mark.asyncio
    async def test_scroll_back_resumes_the_same_cadence(self, clock: Clock) -> None:
        msg = CadenceMessage(at_bottom=False)
        msg._frame_delay = STREAM_WRITE_FRAME_SECONDS_MIN
        await msg.append_content("buffered ")

        msg._at_bottom = True
        await msg.append_content("live")
        await msg.fire_timer()

        assert msg.scheduled_delays == [STREAM_WRITE_FRAME_SECONDS_MIN]
        assert msg._fake_stream.written == ["buffered live"]
        assert msg._to_write_buffer == ""


class _StreamApp(App[None]):
    CSS_PATH = "../../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield Vertical(id="history")


async def _wait_until(pilot: Pilot[None], predicate: Callable[[], bool]) -> None:
    deadline = _real_monotonic() + 2.0
    while not predicate():
        if _real_monotonic() >= deadline:
            raise TimeoutError("stream write did not reach the markdown widget")
        await pilot.pause(0.01)


def _markdown_source(message: StreamingMessageBase) -> str:
    markdown = message._markdown
    assert markdown is not None
    return markdown._markdown


@pytest.mark.asyncio
async def test_terminal_flush_writes_the_pending_frame_once() -> None:
    app = _StreamApp()
    async with app.run_test() as pilot:
        history = app.query_one("#history", Vertical)
        message = AssistantMessage("")
        await history.mount(message)
        await pilot.pause()

        await message.append_content("half a sentence ")
        await message.append_content("and the rest")
        assert message._write_timer is not None

        await message.stop_stream()

        assert message._to_write_buffer == ""
        await _wait_until(
            pilot, lambda: _markdown_source(message) == "half a sentence and the rest"
        )


@pytest.mark.asyncio
async def test_cancelled_frame_timer_defers_to_the_terminal_flush() -> None:
    app = _StreamApp()
    async with app.run_test() as pilot:
        history = app.query_one("#history", Vertical)
        message = AssistantMessage("")
        await history.mount(message)
        await pilot.pause()

        await message.append_content("buffered while the timer was cancelled")
        message._cancel_write_timer()
        assert message._write_timer is None

        # Longer than the implementation cap: a live timer would have fired.
        await pilot.pause(_P5_SCHEDULED_INTERVAL_CAP_SECONDS + 0.05)
        assert _markdown_source(message) == ""
        assert message._to_write_buffer == "buffered while the timer was cancelled"

        await message.stop_stream()

        await _wait_until(
            pilot, lambda: _markdown_source(message).startswith("buffered")
        )
        assert _markdown_source(message) == "buffered while the timer was cancelled"


@pytest.mark.asyncio
async def test_expanded_reasoning_streams_live_frames() -> None:
    app = _StreamApp()
    async with app.run_test() as pilot:
        history = app.query_one("#history", Vertical)
        reasoning = ReasoningMessage("", collapsed=True, completed=True)
        await history.mount(reasoning)
        await pilot.pause()

        await reasoning.append_content("first thought ")
        await pilot.pause(0.2)
        # Collapsed nodes discard frames: nothing renders and no timer is armed.
        assert reasoning._write_timer is None
        assert _markdown_source(reasoning) == ""
        assert reasoning.get_content() == "first thought "

        await reasoning.set_collapsed(False)
        # Expanding replays the accumulated content into a fresh stream.
        await _wait_until(
            pilot, lambda: _markdown_source(reasoning) == "first thought "
        )
        assert reasoning._to_write_buffer == ""

        await reasoning.append_content("second thought")
        assert reasoning._write_timer is not None
        assert reasoning._frame_delay <= _P5_SCHEDULED_INTERVAL_CAP_SECONDS
        await _wait_until(
            pilot, lambda: _markdown_source(reasoning) == "first thought second thought"
        )

        await reasoning.set_collapsed(True)
        await reasoning.append_content(" hidden tail")
        await pilot.pause(0.2)
        assert reasoning._write_timer is None
        assert "hidden tail" not in _markdown_source(reasoning)
        assert reasoning.get_content() == "first thought second thought hidden tail"
