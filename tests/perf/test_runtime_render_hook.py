"""Contract node for the D1 render-flush observer (v0.7.5 stage 2, WP2).

The observer is exercised against the real streaming render path: the
production ``AssistantMessage`` is mounted in a real Textual app under the real
stylesheet and streams its frames through the production flush boundary, so the
node checks sampled timings, the empty-flush classification, the D1 floor
verdict, and the chunks-coverage fields against live behavior instead of a stub.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import sys
from time import perf_counter

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.messages import AssistantMessage
from tests.perf._runtime_render import (
    D1_CELL,
    D1_METRIC,
    D1_METRIC_VERSION,
    D1_MIN_COMPLETED_TURNS,
    D1_MIN_MEASURED_FLUSHES,
    D1_WARMUP_FLUSHES,
    RenderTimingObserver,
    build_run_record,
    evaluate_d1_floor,
    histogram_ms,
    nearest_rank,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="performance scenarios require Linux /proc readers"
)

# Plan §1 in-suite envelope: a contract node plans for a 1s budget (measured
# target <= 0.5s) inside a 2s hard window, and its timeout marker is 2x budget.
_CONTRACT_BUDGET_S = 1.0
_CONTRACT_WINDOW_S = 2.0
_HOOK_TIMEOUT_S = 2 * _CONTRACT_BUDGET_S

_CONTRACT_WARMUP_FLUSHES = 2
_MIN_MEASURED_FLUSHES = 2


class _RenderApp(App[None]):
    """The production stylesheet over an empty history container."""

    CSS_PATH = "../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield Vertical(id="history")


def _rendered_markdown(message: AssistantMessage) -> str:
    markdown = message._markdown
    assert markdown is not None
    return markdown._markdown


async def _wait_until(
    pilot: Pilot[None], predicate: Callable[[], bool], *, timeout: float = 5.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("render-hook fixture condition did not become true")
        await pilot.pause(0.01)


async def _drive_real_flushes(
    pilot: Pilot[None], message: AssistantMessage, *, flushes: int
) -> str:
    """Stream one frame per iteration through the production flush boundary."""
    rendered = ""
    for index in range(flushes):
        chunk = f"flush-{index} "
        rendered += chunk
        await message.append_content(chunk)
        await _wait_until(
            pilot, lambda expected=rendered: _rendered_markdown(message) == expected
        )
    return rendered


@pytest.mark.asyncio
@pytest.mark.timeout(_HOOK_TIMEOUT_S)
async def test_render_timing_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    started = perf_counter()
    observer = RenderTimingObserver(warmup_flushes=_CONTRACT_WARMUP_FLUSHES)
    observer.install(monkeypatch)

    app = _RenderApp()
    async with app.run_test() as pilot:
        history = app.query_one("#history", Vertical)
        message = AssistantMessage("")
        await history.mount(message)
        await pilot.pause()
        rendered = await _drive_real_flushes(
            pilot, message, flushes=_CONTRACT_WARMUP_FLUSHES + _MIN_MEASURED_FLUSHES
        )
        # Instrumentation only: the live render path still shows the full text.
        assert _rendered_markdown(message) == rendered
        assert message.get_content() == rendered
        # A flush with an empty write buffer renders nothing: the observer must
        # count it as empty and never as a D1 sample.
        await message._flush_write_buffer()

    assert observer.log.non_empty_flushes >= (
        _CONTRACT_WARMUP_FLUSHES + _MIN_MEASURED_FLUSHES
    )
    assert len(observer.log.warmup_ms) == _CONTRACT_WARMUP_FLUSHES
    assert len(observer.log.measured_ms) >= _MIN_MEASURED_FLUSHES
    assert all(sample > 0 for sample in observer.log.measured_ms)
    assert observer.log.empty_flushes >= 1
    measured = list(observer.log.measured_ms)

    run_record = build_run_record(observer.log, completed_turns=1, fixture="fixture")
    coverage = run_record["chunks_coverage"]
    assert isinstance(coverage, dict)
    assert coverage["unit"] == "render flushes"
    assert coverage["completed_turns"] == 1
    assert coverage["warmup_flushes"] == _CONTRACT_WARMUP_FLUSHES
    assert coverage["measured_flushes"] == len(measured)
    assert coverage["empty_flushes"] >= 1
    assert run_record["warmup_flushes_required"] == _CONTRACT_WARMUP_FLUSHES
    assert run_record["campaign_id"] == "D1"
    assert run_record["metric"] == D1_METRIC
    assert run_record["cell"] == D1_CELL
    assert run_record["metric_version"] == D1_METRIC_VERSION

    samples = run_record["samples_ms"]
    assert isinstance(samples, list)
    assert len(samples) == len(measured)
    # p50 is reported alongside the gated p95, both by nearest rank.
    assert run_record["render_flush_p95_ms"] == nearest_rank(samples, fraction=0.95)
    assert run_record["render_flush_p50_ms"] == nearest_rank(samples, fraction=0.5)
    assert run_record["render_flush_max_ms"] == max(samples)
    histogram = run_record["histogram_ms"]
    assert isinstance(histogram, dict)
    assert sum(histogram.values()) == len(samples)
    # A short in-suite run fails the D1 floor honestly: the record states why.
    assert run_record["eligibility"] == {
        "eligible": False,
        "reasons": ["ops_floor_completed_turns", "ops_floor_measured_flushes"],
    }

    # The pinned D1 floor is checked directly too, so the barrier semantics are
    # not inherited from this short run's accidental sample counts.
    assert (D1_WARMUP_FLUSHES, D1_MIN_COMPLETED_TURNS, D1_MIN_MEASURED_FLUSHES) == (
        10,
        100,
        100,
    )
    at_floor = evaluate_d1_floor(
        completed_turns=D1_MIN_COMPLETED_TURNS, measured_flushes=D1_MIN_MEASURED_FLUSHES
    )
    assert at_floor.eligible is True
    assert at_floor.reasons == ()
    assert evaluate_d1_floor(
        completed_turns=D1_MIN_COMPLETED_TURNS - 1,
        measured_flushes=D1_MIN_MEASURED_FLUSHES,
    ).reasons == ("ops_floor_completed_turns",)
    assert evaluate_d1_floor(
        completed_turns=D1_MIN_COMPLETED_TURNS,
        measured_flushes=D1_MIN_MEASURED_FLUSHES - 1,
    ).reasons == ("ops_floor_measured_flushes",)

    # Nearest rank: k = ceil(0.95 * n), never interpolated.
    assert nearest_rank([float(value) for value in range(1, 21)], fraction=0.95) == 19.0
    assert nearest_rank([3.0], fraction=0.95) == 3.0
    assert nearest_rank([4.0, 1.0, 9.0], fraction=0.95) == 9.0
    with pytest.raises(ValueError):
        nearest_rank([], fraction=0.95)
    assert histogram_ms([])["<=0.5"] == 0
    assert histogram_ms([1500.0])[">1000"] == 1

    assert perf_counter() - started <= _CONTRACT_WINDOW_S
