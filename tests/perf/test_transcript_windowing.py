from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Iterator, Sequence
import os
import sys
from time import perf_counter_ns
from types import SimpleNamespace
from typing import Any, TypedDict, cast
from weakref import WeakKeyDictionary

import pytest
from textual.geometry import Region, Size
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button

from chartreux.app_server.models import (
    CompletedEffectState,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    TextContentBlock,
)
from chartreux.app_server.session import AppServerSession
from chartreux.cli.textual_ui.app import _TRANSCRIPT_HIGH_MARK, ChartreuxApp, ChatScroll
from chartreux.cli.textual_ui.transcript_presenter import TranscriptPresenter
from chartreux.cli.textual_ui.widgets.collapsible import CollapsibleSection
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.load_more import HistoryLoadMoreMessage
from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.widgets.tools import ToolGroup, ToolResultMessage
from chartreux.cli.textual_ui.windowing.history import _build_history_widgets_raw
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.perf._metrics import (
    alloc_peak,
    machine_context,
    measure_loop_lag,
    percentiles,
    record,
    repeat_timed_ms,
    rss_mib,
    rss_peak_mib,
)
from tests.perf._transcript import (
    OVERSIZED_GROUP_LINE_COUNT,
    SHELL_LINE_COUNT,
    TRANSCRIPT_HEIGHT,
    TRANSCRIPT_WIDTH,
    TranscriptFixture,
    build_transcript_fixture,
)
from tests.stubs.app_server import create_test_app_server_session

# Keep the two-repetition 400-entry smoke; opt in to scaling and five reps.
_FULL_SCOPE = os.environ.get("CHARTREUX_PERF_TRANSCRIPT_FULL") == "1"
_REPETITIONS = 5 if _FULL_SCOPE else 2
_RESUME_TAIL_SIZE = 20
_TRAVERSAL_STEP_COUNT = 8
_RECONCILE_PASS_CEILING = 3 * 414
_ENTRY_COUNTS = (300, 400, 500) if _FULL_SCOPE else (400,)
# Wall-clock guard only: leave headroom above the smoke's ~410s runtime.
_SCENARIO_TIMEOUT = 1800 if _FULL_SCOPE else 600


class _AnchorSample(TypedDict):
    anchor_id: str
    screen_row_before: int
    screen_row_at_80: int
    screen_row_at_120: int
    screen_row_error_at_80: int
    screen_row_error_roundtrip: int


@pytest.fixture
def reconstruction_baseline_ms() -> Callable[[list[PublicHistoryEntry]], float]:
    """Test-only reconstruction timer for the pre-C windowing baseline."""

    def measure(entries: list[PublicHistoryEntry]) -> float:
        indices: WeakKeyDictionary[Widget, int] = WeakKeyDictionary()
        started = perf_counter_ns()
        widgets = _build_history_widgets_raw(
            entries,
            start_index=0,
            history_widget_indices=indices,
            tools_collapsed=True,
            expansion_state=ToolGroupExpansionState(default_collapsed=True),
            entry_expansion_state=EntryExpansionState(default_collapsed=True),
        )
        elapsed_ms = (perf_counter_ns() - started) / 1_000_000
        assert widgets
        del widgets
        return elapsed_ms

    return measure


def _walk_widgets(root: Widget) -> list[Widget]:
    widgets: list[Widget] = []
    pending = list(root.children)
    while pending:
        widget = pending.pop()
        widgets.append(widget)
        pending.extend(widget.children)
    return widgets


def _mounted_history_indices(app: ChartreuxApp) -> set[int]:
    # Test-only metric extraction: correlate mounted DOM widgets to fixture rows;
    # this reads the index map without consulting or mutating windowing state.
    messages_area = app.query_one("#messages")
    return {
        index
        for widget in _walk_widgets(messages_area)
        if (index := app._history_widget_indices.get(widget)) is not None
    }


def _widget_descendant_count(widget: Widget) -> int:
    return 1 + sum(_widget_descendant_count(child) for child in widget.children)


def _registry_snapshot(app: ChartreuxApp) -> dict[str, int]:
    window = app._transcript
    units = list(window.units.values())
    resident = [unit for unit in units if unit.mounted_roots and not unit.placeholders]
    placeholders = [p for unit in units for p in unit.placeholders]
    mounted_rows = sum(
        unit.mounted_roots[-1].region.bottom - unit.mounted_roots[0].region.y
        for unit in resident
    )
    budgeted_rows = sum(
        unit.mounted_roots[-1].region.bottom - unit.mounted_roots[0].region.y
        for unit in resident
        if not unit.pin_reasons
        and unit.mounted_roots[-1].region.bottom - unit.mounted_roots[0].region.y
        <= 1500
    )
    virtual_rows = sum(p.region.height for p in placeholders)
    return {
        "history_entries_admitted": window.admitted_end_index
        - window.admitted_start_index,
        "known_units": len(units),
        "resident_units": len(resident),
        "resident_descendants": sum(
            _widget_descendant_count(root)
            for unit in resident
            for root in unit.mounted_roots
        ),
        "placeholder_count": len(placeholders),
        "mounted_rows": mounted_rows,
        "budgeted_resident_rows": budgeted_rows,
        "virtual_rows": virtual_rows,
        "mounted_plus_virtual_rows": mounted_rows + virtual_rows,
        "visible_rows": app.query_one("#chat", ChatScroll).size.height,
    }


def _find_unit_for_entry(
    app: ChartreuxApp, entry_index: int
) -> tuple[ToolGroup, ToolResultMessage]:
    # Test-only DOM selection: map a mounted result back to the fixture entry.
    for group in app.query(ToolGroup):
        for child in group.content_container.children:
            if app._history_widget_indices.get(child) == entry_index and isinstance(
                child, ToolResultMessage
            ):
                return group, child
    raise AssertionError(f"No mounted tool group for history entry {entry_index}")


def _entry_index(fixture: TranscriptFixture, entry_id: str) -> int:
    return next(
        index for index, entry in enumerate(fixture.entries) if entry.id == entry_id
    )


async def _wait_until(
    pilot: Pilot[ChartreuxApp], predicate: Callable[[], bool], *, timeout: float = 15.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("transcript-windowing pilot condition did not settle")
        await pilot.pause(0.01)


async def _wait_for_resume(
    app: ChartreuxApp, pilot: Pilot[ChartreuxApp], entry_count: int
) -> None:
    expected_tail = min(_RESUME_TAIL_SIZE, entry_count)
    await _wait_until(
        pilot,
        lambda: (
            len(_mounted_history_indices(app)) == expected_tail
            and len(app.query(HistoryLoadMoreMessage)) == 1
        ),
    )


async def _wait_for_load_more_state(
    app: ChartreuxApp, expected_start_index: int, *, timeout: float = 120.0
) -> None:
    """Wait for a stable, completed Load More state without Pilot's 30s wait."""
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        load_more_messages = list(app.query(HistoryLoadMoreMessage))
        if (
            expected_start_index == 0
            and app._transcript.admitted_start_index == expected_start_index
            and not load_more_messages
        ):
            return
        if (
            app._transcript.admitted_start_index == expected_start_index
            and len(load_more_messages) == 1
            # The message can enter the DOM before its Button is composed.
            and len(load_more_messages[0].query(Button)) == 1
        ):
            button = load_more_messages[0].query_one(Button)
            remaining = app._history_backfill_remaining
            expected_label = (
                "Load more messages"
                if remaining is None
                else f"Load more messages ({remaining})"
            )
            if not button.disabled and str(button.label) == expected_label:
                return
        if asyncio.get_running_loop().time() >= deadline:
            button_state = [
                (str(button.label), button.disabled)
                for message in load_more_messages
                for button in message.query(Button)
            ]
            raise TimeoutError(
                "transcript Load More did not settle at admitted start index "
                f"{expected_start_index} within {timeout:.0f}s; "
                f"actual={app._transcript.admitted_start_index}, "
                f"remaining={app._windowing.remaining}, buttons={button_state}"
            )
        # Pilot.pause() waits for pending widget messages with a fixed 30s cap.
        # Polling operation state avoids mistaking initial label propagation as
        # batch completion and lets the per-batch timeout accommodate load noise.
        await asyncio.sleep(0.01)


async def _load_all_more(
    app: ChartreuxApp,
    pilot: Pilot[ChartreuxApp],
    prepend_anchors: list[dict[str, int | str | bool]] | None = None,
) -> tuple[list[float], float]:
    batch_latency_ms: list[float] = []
    total_started = perf_counter_ns()
    while app.query(HistoryLoadMoreMessage):
        previous_start_index = app._transcript.admitted_start_index
        await _wait_for_load_more_state(app, previous_start_index)
        chat = app.query_one("#chat", ChatScroll)
        chat.scroll_to(y=0, animate=False, force=True, immediate=True)
        await pilot.pause(0)
        button = app.query_one(HistoryLoadMoreMessage).query_one(Button)
        anchor = None
        if prepend_anchors is not None:
            try:
                anchor = _visible_anchor(app)
            except AssertionError:
                pass  # Load More button can occupy the entire reading viewport.
        expected_start_index = max(
            previous_start_index - app._windowing.load_more_batch_size, 0
        )
        started = perf_counter_ns()
        await pilot.click(button)
        await _wait_for_load_more_state(app, expected_start_index)
        await pilot.pause(0)
        if anchor is not None and prepend_anchors is not None:
            anchor_id, widget, before_row = anchor
            mounted = widget.is_mounted
            after_row = _anchor_row(widget, chat) if mounted else before_row
            prepend_anchors.append({
                "anchor_id": anchor_id,
                "screen_row_before": before_row,
                "screen_row_after": after_row,
                "screen_row_error": after_row - before_row,
                "anchor_lost": not mounted,
            })
        batch_latency_ms.append((perf_counter_ns() - started) / 1_000_000)
    await pilot.pause(0.02)
    return batch_latency_ms, (perf_counter_ns() - total_started) / 1_000_000


async def _expand_effect_group(
    app: ChartreuxApp, pilot: Pilot[ChartreuxApp], entry_index: int
) -> tuple[ToolGroup, ToolResultMessage, CollapsibleSection, float]:
    group, result = _find_unit_for_entry(app, entry_index)
    started = perf_counter_ns()
    group.set_collapsed(False)
    sections = list(result.query(CollapsibleSection))
    assert len(sections) == 1
    section = sections[0]
    section.set_collapsed(False)
    await pilot.pause(0)
    return group, result, section, (perf_counter_ns() - started) / 1_000_000


def _visible_anchor(app: ChartreuxApp) -> tuple[str, Widget, int]:
    chat = app.query_one("#chat", ChatScroll)
    candidates: list[tuple[int, int, Widget]] = []
    # Test-only anchor metric extraction needs the mounted widget-to-history map.
    for widget, history_index in app._history_widget_indices.items():
        region = widget.region
        if (
            region.height > 0
            and region.bottom > chat.region.y
            and region.y < chat.region.bottom
        ):
            candidates.append((region.y, history_index, widget))
    if not candidates:
        raise AssertionError(
            "no mounted history widget intersects the reading viewport"
        )
    _, index, widget = min(candidates, key=lambda candidate: candidate[0])
    return app.app_server.history[index].id, widget, widget.region.y - chat.region.y


def _anchor_row(widget: Widget, chat: ChatScroll) -> int:
    return widget.region.y - chat.region.y


async def _measure_eviction_remount_anchor(
    app: ChartreuxApp, pilot: Pilot[ChartreuxApp]
) -> dict[str, int | str | bool]:
    chat = app.query_one("#chat", ChatScroll)
    window = app._transcript
    candidates: list[tuple[int, int, Widget, str]] = []
    for widget, entry_index in app._history_widget_indices.items():
        region = widget.region
        if (
            region.height <= 0
            or region.bottom <= chat.region.y
            or region.y >= chat.region.bottom
        ):
            continue
        entry_id = app.app_server.history[entry_index].id
        unit_id = window._entry_to_unit.get(entry_id)
        unit = window.units.get(unit_id) if unit_id is not None else None
        if (
            unit is None
            or not unit.mounted_roots
            or unit.placeholders
            or unit.pin_reasons
        ):
            continue
        unit_height = (
            unit.mounted_roots[-1].region.bottom - unit.mounted_roots[0].region.y
        )
        if unit_height <= 1500:
            candidates.append((region.y, entry_index, widget, unit.id))
    if not candidates:
        raise AssertionError(
            "no visible, evictable unit available for anchor measurement"
        )

    _, entry_index, anchor_widget, unit_id = min(
        candidates, key=lambda candidate: candidate[0]
    )
    anchor_entry_id = app.app_server.history[entry_index].id
    before_row = _anchor_row(anchor_widget, chat)
    await window.evict_unit(unit_id)
    restored = await window.restore_unit(
        unit_id,
        app.query_one("#messages"),
        app._history_widget_indices,
        follow_bottom=False,
    )
    assert restored
    await pilot.pause(0)
    # The retained reference keeps the evicted widget in the weak index map.
    # Textual's is_mounted remains true after removal; only the attached remount
    # has meaningful geometry. The reading anchor must still stay at the same row.
    restored_anchor = next(
        (
            widget
            for widget, index in app._history_widget_indices.items()
            if index == entry_index and widget.is_attached
        ),
        None,
    )
    assert restored_anchor is not None
    assert restored_anchor is not anchor_widget
    after_row = _anchor_row(restored_anchor, chat)
    delta = after_row - before_row
    assert delta == 0, f"eviction/remount moved retained anchor by {delta} rows"
    return {
        "anchor_unit_id": unit_id,
        "anchor_entry_id": anchor_entry_id,
        "anchor_row_offset": before_row,
        "screen_row_before": before_row,
        "screen_row_after": after_row,
        "screen_row_delta": delta,
        "screen_row_exact": delta == 0,
    }


async def _resize_reading_anchor(
    app: ChartreuxApp, pilot: Pilot[ChartreuxApp]
) -> _AnchorSample:
    chat = app.query_one("#chat", ChatScroll)
    target_scroll = min(max(1, chat.size.height * 3), int(chat.max_scroll_y))
    chat.scroll_to(y=target_scroll, animate=False, force=True, immediate=True)
    await pilot.pause(0)
    chat.preserve_scroll_position()
    anchor_id, anchor_widget, before_row = _visible_anchor(app)

    await pilot.resize_terminal(80, TRANSCRIPT_HEIGHT)
    await pilot.pause(0.03)
    at_80_row = _anchor_row(anchor_widget, chat)
    await pilot.resize_terminal(TRANSCRIPT_WIDTH, TRANSCRIPT_HEIGHT)
    await pilot.pause(0.03)
    at_120_row = _anchor_row(anchor_widget, chat)
    return {
        "anchor_id": anchor_id,
        "screen_row_before": before_row,
        "screen_row_at_80": at_80_row,
        "screen_row_at_120": at_120_row,
        "screen_row_error_at_80": at_80_row - before_row,
        "screen_row_error_roundtrip": at_120_row - before_row,
    }


async def _set_scroll_target(
    app: ChartreuxApp,
    pilot: Pilot[ChartreuxApp],
    target: int,
    *,
    phase: str,
    scroll_samples: dict[str, list[float]],
    mount_changes: dict[str, dict[str, int]],
    track_mount_changes: bool,
    residency_samples: dict[str, list[dict[str, int]]] | None = None,
    traversal_anchors: dict[str, list[dict[str, object]]] | None = None,
    remount_windows: dict[str, list[dict[str, float | int]]] | None = None,
    per_step_reflow_residency: dict[str, list[dict[str, object]]] | None = None,
    refresh_pass_ms_by_phase: dict[str, list[float]] | None = None,
    eviction_counts: Counter[str] | None = None,
) -> None:
    chat = app.query_one("#chat", ChatScroll)
    messages_area = app.query_one("#messages")
    window = app._transcript
    pending_before = {unit.id for unit in window.units.values() if unit.placeholders}
    reflow_start = (
        len(refresh_pass_ms_by_phase.get(phase, []))
        if refresh_pass_ms_by_phase is not None
        else 0
    )
    evictions_before = (
        eviction_counts.get(phase, 0) if eviction_counts is not None else 0
    )
    before_roots = {id(child) for child in messages_area.children}
    started = perf_counter_ns()
    chat.scroll_to(y=target, animate=False, force=True, immediate=True)
    await pilot.pause(0)
    restore_top = chat.region.y - chat.size.height
    restore_bottom = chat.region.y + 2 * chat.size.height
    requested_restore_ids = {
        unit_id
        for unit_id in pending_before
        if (unit := window.units.get(unit_id)) is not None
        and (placement := window._placement(unit)) is not None
        and placement.region.y < restore_bottom
        and placement.region.bottom > restore_top
    }
    if requested_restore_ids:
        await _wait_until(
            pilot,
            lambda: all(
                unit_id in window.units
                and window.units[unit_id].mounted_roots
                and not window.units[unit_id].placeholders
                for unit_id in requested_restore_ids
            ),
        )
    elapsed_ms = (perf_counter_ns() - started) / 1_000_000
    scroll_samples.setdefault(phase, []).append(elapsed_ms)
    if requested_restore_ids and remount_windows is not None:
        remount_windows.setdefault(phase, []).append({
            "units": len(requested_restore_ids),
            "scroll_to_ready_ms": elapsed_ms,
        })
    if residency_samples is not None:
        residency = _registry_snapshot(app)
        evictions_this_step = (
            eviction_counts.get(phase, 0) - evictions_before
            if eviction_counts is not None
            else 0
        )
        residency["evictions_this_step"] = evictions_this_step
        residency_samples.setdefault(phase, []).append(residency)
    else:
        residency = None
    if per_step_reflow_residency is not None:
        phase_reflows = (
            refresh_pass_ms_by_phase.get(phase, [])[reflow_start:]
            if refresh_pass_ms_by_phase is not None
            else []
        )
        per_step_reflow_residency.setdefault(phase, []).append({
            "scroll_y": int(chat.scroll_y),
            "resident_units": residency["resident_units"] if residency else 0,
            "budgeted_resident_rows": (
                residency["budgeted_resident_rows"] if residency else 0
            ),
            "evictions_this_step": (
                residency["evictions_this_step"] if residency else 0
            ),
            "refresh_layout_pass_count": len(phase_reflows),
            "refresh_layout_pass_ms": [round(value, 3) for value in phase_reflows],
        })
    if traversal_anchors is not None:
        anchor = window._reading_anchor(messages_area)
        traversal_anchors.setdefault(phase, []).append({
            "scroll_y": int(chat.scroll_y),
            "anchor_id": anchor[0] if anchor else None,
            "anchor_screen_row_error": anchor[2] - chat.region.y if anchor else None,
            "blank_region": anchor is None or anchor[2] >= chat.region.bottom,
        })
    if track_mount_changes:
        after_roots = {id(child) for child in messages_area.children}
        changes = mount_changes.setdefault(phase, {"mounts": 0, "evictions": 0})
        changes["mounts"] += len(after_roots - before_roots)
        changes["evictions"] += len(before_roots - after_roots)


async def _walk_to_boundary(
    app: ChartreuxApp,
    pilot: Pilot[ChartreuxApp],
    boundary: int,
    *,
    phase: str,
    scroll_samples: dict[str, list[float]],
    mount_changes: dict[str, dict[str, int]],
    track_mount_changes: bool = False,
    residency_samples: dict[str, list[dict[str, int]]] | None = None,
    traversal_anchors: dict[str, list[dict[str, object]]] | None = None,
    remount_windows: dict[str, list[dict[str, float | int]]] | None = None,
    per_step_reflow_residency: dict[str, list[dict[str, object]]] | None = None,
    refresh_pass_ms_by_phase: dict[str, list[float]] | None = None,
    eviction_counts: Counter[str] | None = None,
) -> None:
    chat = app.query_one("#chat", ChatScroll)
    step = max(1, chat.size.height // 2)
    current = int(chat.scroll_y)
    boundary = max(0, min(boundary, int(chat.max_scroll_y)))
    while current != boundary:
        target = (
            min(boundary, current + step)
            if current < boundary
            else max(boundary, current - step)
        )
        await _set_scroll_target(
            app,
            pilot,
            target,
            phase=phase,
            scroll_samples=scroll_samples,
            mount_changes=mount_changes,
            track_mount_changes=track_mount_changes,
            residency_samples=residency_samples,
            traversal_anchors=traversal_anchors,
            remount_windows=remount_windows,
            per_step_reflow_residency=per_step_reflow_residency,
            refresh_pass_ms_by_phase=refresh_pass_ms_by_phase,
            eviction_counts=eviction_counts,
        )
        updated = int(chat.scroll_y)
        if updated != target:
            raise AssertionError(
                f"transcript scroll targeted {target} but settled at {updated}"
            )
        current = updated


async def _bidirectional_traverse_window(
    app: ChartreuxApp,
    pilot: Pilot[ChartreuxApp],
    *,
    scroll_samples: dict[str, list[float]],
    mount_changes: dict[str, dict[str, int]],
    residency_samples: dict[str, list[dict[str, int]]] | None = None,
    traversal_anchors: dict[str, list[dict[str, object]]] | None = None,
    remount_windows: dict[str, list[dict[str, float | int]]] | None = None,
    per_step_reflow_residency: dict[str, list[dict[str, object]]] | None = None,
    refresh_pass_ms_by_phase: dict[str, list[float]] | None = None,
    eviction_counts: Counter[str] | None = None,
) -> None:
    """Measure three half-viewport passes over a bounded, representative span."""
    chat = app.query_one("#chat", ChatScroll)
    step = max(1, chat.size.height // 2)
    start = int(chat.max_scroll_y) // 2
    end = min(int(chat.max_scroll_y), start + step * _TRAVERSAL_STEP_COUNT)
    chat.scroll_to(y=start, animate=False, force=True, immediate=True)
    await pilot.pause(0)
    assert int(chat.scroll_y) == start
    if not any(unit.placeholders for unit in app._transcript.units.values()):
        # The 400-entry smoke fits the frozen 1500-row budget. Exercise the
        # same production eviction/restore path without changing its budgets.
        target_y = chat.region.bottom + chat.size.height
        candidate = next(
            (
                unit
                for unit in app._transcript.units.values()
                if unit.mounted_roots
                and not unit.pin_reasons
                and unit.mounted_roots[-1].region.bottom
                - unit.mounted_roots[0].region.y
                <= _TRANSCRIPT_HIGH_MARK
                and unit.mounted_roots[0].region.y >= target_y
                and unit.mounted_roots[0].region.y < target_y + chat.size.height
            ),
            None,
        )
        if candidate is not None:
            await app._transcript.evict_unit(candidate.id)
            await pilot.pause(0)
    for boundary, track_mount_changes in ((end, False), (start, True), (end, True)):
        await _walk_to_boundary(
            app,
            pilot,
            boundary,
            phase="bidirectional_traverse",
            scroll_samples=scroll_samples,
            mount_changes=mount_changes,
            track_mount_changes=track_mount_changes,
            residency_samples=residency_samples,
            traversal_anchors=traversal_anchors,
            remount_windows=remount_windows,
            per_step_reflow_residency=per_step_reflow_residency,
            refresh_pass_ms_by_phase=refresh_pass_ms_by_phase,
            eviction_counts=eviction_counts,
        )


async def _traverse_oversized_group(
    app: ChartreuxApp,
    pilot: Pilot[ChartreuxApp],
    group: ToolGroup,
    *,
    scroll_samples: dict[str, list[float]],
    mount_changes: dict[str, dict[str, int]],
    residency_samples: dict[str, list[dict[str, int]]] | None = None,
    traversal_anchors: dict[str, list[dict[str, object]]] | None = None,
    remount_windows: dict[str, list[dict[str, float | int]]] | None = None,
    per_step_reflow_residency: dict[str, list[dict[str, object]]] | None = None,
    refresh_pass_ms_by_phase: dict[str, list[float]] | None = None,
    eviction_counts: Counter[str] | None = None,
) -> int:
    chat = app.query_one("#chat", ChatScroll)
    content_top = max(0, group.region.y - chat.region.y + int(chat.scroll_y))
    content_bottom = content_top + max(1, group.size.height)
    start = max(0, content_top - chat.size.height // 2)
    end = min(int(chat.max_scroll_y), content_bottom + chat.size.height // 2)
    chat.scroll_to(y=start, animate=False, force=True, immediate=True)
    await pilot.pause(0)
    assert int(chat.scroll_y) == start
    await _walk_to_boundary(
        app,
        pilot,
        end,
        phase="oversized_group_traverse",
        scroll_samples=scroll_samples,
        mount_changes=mount_changes,
        residency_samples=residency_samples,
        traversal_anchors=traversal_anchors,
        remount_windows=remount_windows,
        per_step_reflow_residency=per_step_reflow_residency,
        refresh_pass_ms_by_phase=refresh_pass_ms_by_phase,
        eviction_counts=eviction_counts,
    )
    await _walk_to_boundary(
        app,
        pilot,
        start,
        phase="oversized_group_traverse",
        scroll_samples=scroll_samples,
        mount_changes=mount_changes,
        residency_samples=residency_samples,
        traversal_anchors=traversal_anchors,
        remount_windows=remount_windows,
        per_step_reflow_residency=per_step_reflow_residency,
        refresh_pass_ms_by_phase=refresh_pass_ms_by_phase,
        eviction_counts=eviction_counts,
    )
    return group.size.height


async def _measure_large_shell_expansion(
    app: ChartreuxApp, pilot: Pilot[ChartreuxApp], entry_index: int
) -> float:
    group, _, section, elapsed_ms = await _expand_effect_group(app, pilot, entry_index)
    # The section mount/layout is the shell-body cost; collapse it again so the
    # bounded transcript traversal remains focused on the oversized group.
    section.set_collapsed(True)
    group.set_collapsed(True)
    await pilot.pause(0.03)
    return elapsed_ms


def _assert_fixture_composition(fixture: TranscriptFixture) -> None:
    assert fixture.entry_count == len(fixture.entries)
    assert (
        fixture.turn_count * 6
        + fixture.checkpoint_count
        + fixture.ordinary_message_count
        == (fixture.entry_count)
    )
    assert fixture.checkpoint_count == fixture.ordinary_message_count
    assert sum(
        isinstance(entry, PublicCheckpointEntry) for entry in fixture.entries
    ) == (fixture.checkpoint_count)
    assert sum(isinstance(entry, PublicMessageEntry) for entry in fixture.entries) == (
        fixture.turn_count * 3 + fixture.ordinary_message_count
    )
    shell_entries = [
        entry
        for entry in fixture.entries
        if isinstance(entry, PublicEffectEntry)
        and entry.detail.kind.value == "shell"
        and isinstance(entry.state, CompletedEffectState)
        and entry.state.output_text.count("\n") >= SHELL_LINE_COUNT
    ]
    assert len(shell_entries) == 1
    oversized = next(
        entry
        for entry in fixture.entries
        if entry.id == fixture.oversized_group_entry_id
    )
    assert isinstance(oversized, PublicEffectEntry)
    assert isinstance(oversized.state, CompletedEffectState)
    oversized_output = oversized.state.output
    assert isinstance(oversized_output, dict)
    content = oversized_output.get("content")
    assert isinstance(content, str)
    assert len(content.splitlines()) == OVERSIZED_GROUP_LINE_COUNT
    assert OVERSIZED_GROUP_LINE_COUNT > 1_500


def _reconstruction_fixture_timing(
    measure: Callable[[list[PublicHistoryEntry]], float], fixture: TranscriptFixture
) -> float:
    return measure(fixture.entries)


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(_SCENARIO_TIMEOUT)
@pytest.mark.parametrize("entry_count", _ENTRY_COUNTS)
async def test_transcript_windowing_baseline(
    entry_count: int,
    monkeypatch: pytest.MonkeyPatch,
    reconstruction_baseline_ms: Callable[[list[PublicHistoryEntry]], float],
) -> None:
    fixture = build_transcript_fixture(entry_count)
    _assert_fixture_composition(fixture)
    agent_loop = build_test_agent_loop(enable_streaming=False)

    async def start_synthetic_session() -> AppServerSession:
        session = await create_test_app_server_session(agent_loop)
        session._state.projection.state.history = fixture.entries
        return session

    app = build_test_chartreux_app(
        agent_loop=agent_loop, app_server=start_synthetic_session
    )
    target_app: list[ChartreuxApp | None] = [app]
    active_phase: list[str | None] = ["resume"]
    refresh_pass_ms: list[float] = []
    refresh_pass_counts: Counter[str] = Counter()
    refresh_pass_ms_by_phase: dict[str, list[float]] = {}
    remount_ready_ms: list[float] = []
    remount_ready_ms_by_phase: dict[str, list[float]] = {}
    oversized_eviction_geometry: dict[str, bool] = {}
    unknown_restore_classifications = [0]
    oversized_reconcile_passes_current = [0]
    oversized_reconcile_passes_by_repetition: list[int] = []
    reconcile_counts: Counter[str] = Counter()
    registry_mount_counts: Counter[str] = Counter()
    registry_evict_counts: Counter[str] = Counter()
    original_restore_unit = TranscriptWindow.restore_unit
    original_evict_unit = TranscriptWindow.evict_unit
    original_reconcile_once = TranscriptWindow._reconcile_once

    async def measure_restore(
        window: TranscriptWindow, *args: object, **kwargs: object
    ) -> bool:
        unit_id = args[0] if args and isinstance(args[0], str) else None
        oversized = (
            oversized_eviction_geometry.get(unit_id) if unit_id is not None else None
        )
        started = perf_counter_ns()
        completed = await original_restore_unit(window, *args, **kwargs)  # type: ignore[arg-type]
        if completed and active_phase[0] is not None:
            phase = active_phase[0]
            registry_mount_counts[phase] += 1
            if oversized is False:
                elapsed_ms = (perf_counter_ns() - started) / 1_000_000
                remount_ready_ms.append(elapsed_ms)
                remount_ready_ms_by_phase.setdefault(phase, []).append(elapsed_ms)
            elif oversized is None:
                unknown_restore_classifications[0] += 1
        return completed

    async def measure_evict(window: TranscriptWindow, unit_id: str):
        unit = window.units.get(unit_id)
        if unit is not None and unit.mounted_roots:
            roots = unit.mounted_roots
            height = roots[-1].region.bottom - roots[0].region.y
            oversized_eviction_geometry[unit_id] = height > 1500
        placeholders = await original_evict_unit(window, unit_id)
        if active_phase[0] is not None:
            registry_evict_counts[active_phase[0]] += 1
        return placeholders

    async def measure_reconcile(
        window: TranscriptWindow, *args: object, **kwargs: object
    ) -> None:
        phase = active_phase[0]
        if phase is not None:
            reconcile_counts[phase] += 1
            if phase == "oversized_group_traverse":
                oversized_reconcile_passes_current[0] += 1
                if oversized_reconcile_passes_current[0] > _RECONCILE_PASS_CEILING:
                    raise AssertionError(
                        "oversized traversal exceeded reconciliation pass ceiling "
                        f"{_RECONCILE_PASS_CEILING}"
                    )
        await original_reconcile_once(window, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(TranscriptWindow, "restore_unit", measure_restore)
    monkeypatch.setattr(TranscriptWindow, "evict_unit", measure_evict)
    monkeypatch.setattr(TranscriptWindow, "_reconcile_once", measure_reconcile)
    original_refresh_layout = Screen._refresh_layout

    def measure_refresh_layout(
        screen: Screen, size: Size | None = None, scroll: bool = False
    ) -> None:
        phase = active_phase[0]
        if phase is not None and screen.app is target_app[0]:
            started = perf_counter_ns()
            try:
                original_refresh_layout(screen, size, scroll)
            finally:
                elapsed_ms = (perf_counter_ns() - started) / 1_000_000
                refresh_pass_ms.append(elapsed_ms)
                refresh_pass_ms_by_phase.setdefault(phase, []).append(elapsed_ms)
                refresh_pass_counts[phase] += 1
            return
        original_refresh_layout(screen, size, scroll)

    monkeypatch.setattr(Screen, "_refresh_layout", measure_refresh_layout)

    async def resume_fixture_session(
        session: AppServerSession,
        session_id: str,
        *,
        on_adopt: Callable[[], None] | None = None,
    ) -> None:
        assert session_id == session.session_id
        if on_adopt is not None:
            on_adopt()

    # The fixture has no persisted session log; retain its client-side history
    # while exercising the same app event path as a picker selection.
    monkeypatch.setattr(AppServerSession, "resume", resume_fixture_session)

    warmup_reconstruction = _reconstruction_fixture_timing(
        reconstruction_baseline_ms, fixture
    )
    resume_samples: list[float] = []
    load_more_samples: list[float] = []
    load_more_counts: list[int] = []
    load_more_total_samples: list[float] = []
    reconstruction_samples: list[float] = []
    bidirectional_traverse_samples: dict[str, list[float]] = {}
    reversal_changes: dict[str, dict[str, int]] = {}
    resize_anchors: list[_AnchorSample] = []
    prepend_anchors: list[dict[str, int | str | bool]] = []
    oversized_expansion_samples: list[float] = []
    oversized_height_samples: list[int] = []
    shell_expansion_samples: list[float] = []
    residency_samples: dict[str, list[dict[str, int]]] = {}
    remount_windows: dict[str, list[dict[str, float | int]]] = {}
    traversal_anchors: dict[str, list[dict[str, object]]] = {}
    per_step_reflow_residency: dict[str, list[dict[str, object]]] = {}
    eviction_remount_anchor_samples: list[dict[str, int | str | bool]] = []
    registry_samples: dict[str, list[dict[str, int]]] = {
        "after_resume": [],
        "after_load_more": [],
        "oversized_group_expanded": [],
        "after_traversal": [],
    }

    warmup_started = perf_counter_ns()
    async with app.run_test(size=(TRANSCRIPT_WIDTH, TRANSCRIPT_HEIGHT)) as pilot:
        await _wait_for_resume(app, pilot, entry_count)
        warmup_resume_ms = (perf_counter_ns() - warmup_started) / 1_000_000
        active_phase[0] = None
        warmup_resume_registry = _registry_snapshot(app)

        active_phase[0] = "load_more"
        warmup_batches, warmup_load_more_total = await _load_all_more(app, pilot)
        active_phase[0] = None
        warmup_load_more_registry = _registry_snapshot(app)
        assert app._transcript.admitted_end_index == entry_count

        warmup_oversized_index = _entry_index(fixture, fixture.oversized_group_entry_id)
        warmup_group, _, _, warmup_oversized_ms = await _expand_effect_group(
            app, pilot, warmup_oversized_index
        )
        assert warmup_group.size.height > 1_500
        app.query_one("#chat", ChatScroll).preserve_scroll_position()
        warmup_scroll_samples: dict[str, list[float]] = {}
        warmup_mount_changes: dict[str, dict[str, int]] = {}
        await _bidirectional_traverse_window(
            app,
            pilot,
            scroll_samples=warmup_scroll_samples,
            mount_changes=warmup_mount_changes,
        )
        warmup_anchor = await _resize_reading_anchor(app, pilot)
        await _traverse_oversized_group(
            app,
            pilot,
            warmup_group,
            scroll_samples=warmup_scroll_samples,
            mount_changes=warmup_mount_changes,
        )
        warmup_shell_index = _entry_index(fixture, fixture.large_shell_entry_id)
        warmup_shell_expansion_ms = await _measure_large_shell_expansion(
            app, pilot, warmup_shell_index
        )
        warmup_expanded_registry = _registry_snapshot(app)
        warmup_metrics = {
            "resume_ms": round(warmup_resume_ms, 3),
            "load_more_batches": len(warmup_batches),
            "load_more_total_ms": round(warmup_load_more_total, 3),
            "reconstruction_baseline_ms": round(warmup_reconstruction, 3),
            "oversized_group_expansion_ms": round(warmup_oversized_ms, 3),
            "oversized_group_rows": warmup_group.size.height,
            "large_shell_expansion_ms": round(warmup_shell_expansion_ms, 3),
            "anchor_id": warmup_anchor["anchor_id"],
            "registry_after_resume": warmup_resume_registry,
            "registry_after_load_more": warmup_load_more_registry,
            "registry_after_oversized_expansion": warmup_expanded_registry,
        }

        # The automatic resume + first Load More pass above is the one warmup.
        # Rebuild through the real app resume path for each measured sample.
        refresh_pass_ms.clear()
        refresh_pass_counts.clear()
        refresh_pass_ms_by_phase.clear()
        remount_ready_ms.clear()
        remount_ready_ms_by_phase.clear()
        oversized_eviction_geometry.clear()
        unknown_restore_classifications[0] = 0
        oversized_reconcile_passes_current[0] = 0
        oversized_reconcile_passes_by_repetition.clear()
        reconcile_counts.clear()
        registry_mount_counts.clear()
        registry_evict_counts.clear()
        active_phase[0] = None

        for _ in range(_REPETITIONS):
            active_phase[0] = "resume"
            session_id = app.app_server.session_id
            started = perf_counter_ns()
            # Dispatch through the same public app event handler used by the
            # SessionPickerApp; the fixture stubs only the unavailable disk resume.
            await app.on_session_picker_app_session_selected(
                SessionPickerApp.SessionSelected(
                    option_id=session_id, session_id=session_id
                )
            )
            await _wait_for_resume(app, pilot, entry_count)
            resume_samples.append((perf_counter_ns() - started) / 1_000_000)
            active_phase[0] = None
            registry_samples["after_resume"].append(_registry_snapshot(app))

            active_phase[0] = "load_more"
            batches, load_more_total = await _load_all_more(app, pilot, prepend_anchors)
            load_more_samples.extend(batches)
            load_more_counts.append(len(batches))
            load_more_total_samples.append(load_more_total)
            active_phase[0] = None
            registry_samples["after_load_more"].append(_registry_snapshot(app))
            assert app._transcript.admitted_end_index == entry_count

            reconstruction_samples.append(
                _reconstruction_fixture_timing(reconstruction_baseline_ms, fixture)
            )
            oversized_index = _entry_index(fixture, fixture.oversized_group_entry_id)
            active_phase[0] = "oversized_group_expand"
            oversized_group, _, _, oversized_ms = await _expand_effect_group(
                app, pilot, oversized_index
            )
            oversized_expansion_samples.append(oversized_ms)
            oversized_height_samples.append(oversized_group.size.height)
            assert oversized_group.size.height > 1_500
            registry_samples["oversized_group_expanded"].append(_registry_snapshot(app))

            app.query_one("#chat", ChatScroll).preserve_scroll_position()
            active_phase[0] = "bidirectional_traverse"
            await _bidirectional_traverse_window(
                app,
                pilot,
                scroll_samples=bidirectional_traverse_samples,
                mount_changes=reversal_changes,
                residency_samples=residency_samples,
                traversal_anchors=traversal_anchors,
                remount_windows=remount_windows,
                per_step_reflow_residency=per_step_reflow_residency,
                refresh_pass_ms_by_phase=refresh_pass_ms_by_phase,
                eviction_counts=registry_evict_counts,
            )
            active_phase[0] = "resize_anchor"
            resize_anchors.append(await _resize_reading_anchor(app, pilot))

            active_phase[0] = "oversized_group_traverse"
            oversized_reconcile_passes_current[0] = 0
            actual_group_height = await _traverse_oversized_group(
                app,
                pilot,
                oversized_group,
                scroll_samples=bidirectional_traverse_samples,
                mount_changes=reversal_changes,
                residency_samples=residency_samples,
                traversal_anchors=traversal_anchors,
                remount_windows=remount_windows,
                per_step_reflow_residency=per_step_reflow_residency,
                refresh_pass_ms_by_phase=refresh_pass_ms_by_phase,
                eviction_counts=registry_evict_counts,
            )
            oversized_reconcile_passes_by_repetition.append(
                oversized_reconcile_passes_current[0]
            )
            assert actual_group_height > 1_500

            active_phase[0] = "eviction_remount_anchor"
            eviction_remount_anchor_samples.append(
                await _measure_eviction_remount_anchor(app, pilot)
            )
            active_phase[0] = "large_shell_expand"
            shell_index = _entry_index(fixture, fixture.large_shell_entry_id)
            shell_expansion_samples.append(
                await _measure_large_shell_expansion(app, pilot, shell_index)
            )
            active_phase[0] = None
            registry_samples["after_traversal"].append(_registry_snapshot(app))

    assert len(resume_samples) == _REPETITIONS
    assert len(reconstruction_samples) == _REPETITIONS
    assert len(resize_anchors) == _REPETITIONS
    assert all(sample["anchor_id"] for sample in resize_anchors)
    assert len(eviction_remount_anchor_samples) == _REPETITIONS
    assert all(sample["screen_row_exact"] for sample in eviction_remount_anchor_samples)
    assert len(oversized_reconcile_passes_by_repetition) == _REPETITIONS
    assert (
        max(oversized_reconcile_passes_by_repetition, default=0)
        <= _RECONCILE_PASS_CEILING
    )
    assert len(shell_expansion_samples) == _REPETITIONS
    eviction_was_expected = (
        entry_count >= 500 or registry_evict_counts["bidirectional_traverse"] > 0
    )
    if eviction_was_expected:
        assert remount_ready_ms, (
            "expected an eviction phase to produce non-oversized restores"
        )

    scroll_latency_summary = {
        phase: percentiles(samples)
        for phase, samples in bidirectional_traverse_samples.items()
    }
    refresh_distribution = percentiles(refresh_pass_ms) if refresh_pass_ms else {}
    anchor_errors = [
        float(sample["screen_row_error_roundtrip"]) for sample in resize_anchors
    ]
    traversal_budget_envelope = {
        phase: {
            "sample_count": len(samples),
            "max_budgeted_rows": max(
                (sample["budgeted_resident_rows"] for sample in samples), default=0
            ),
            "high_mark": 1500,
            "within_high_mark": bool(samples)
            and all(sample["budgeted_resident_rows"] <= 1500 for sample in samples),
            "low_mark": 1000,
            "post_eviction_sample_count": sum(
                sample["evictions_this_step"] > 0 for sample in samples
            ),
            "post_eviction_max_budgeted_rows": max(
                (
                    sample["budgeted_resident_rows"]
                    for sample in samples
                    if sample["evictions_this_step"] > 0
                ),
                default=0,
            ),
            "post_eviction_max_resident_units": max(
                (
                    sample["resident_units"]
                    for sample in samples
                    if sample["evictions_this_step"] > 0
                ),
                default=0,
            ),
            "reached_low_mark_after_eviction": bool(
                any(sample["evictions_this_step"] > 0 for sample in samples)
            )
            and all(
                sample["budgeted_resident_rows"] <= 1000
                for sample in samples
                if sample["evictions_this_step"] > 0
            ),
        }
        for phase, samples in residency_samples.items()
        if samples
    }
    scroll_to_ready_samples = {
        phase: [float(sample["scroll_to_ready_ms"]) for sample in samples]
        for phase, samples in remount_windows.items()
    }
    reversal_mounts = sum(changes["mounts"] for changes in reversal_changes.values())
    reversal_evictions = sum(
        changes["evictions"] for changes in reversal_changes.values()
    )

    record(
        f"transcript_windowing.{entry_count}",
        {
            "entry_count": entry_count,
            "terminal_size": {"width": TRANSCRIPT_WIDTH, "height": TRANSCRIPT_HEIGHT},
            "fixture_composition": {
                "turns": fixture.turn_count,
                "entries_per_turn": 6,
                "compaction_checkpoints": fixture.checkpoint_count,
                "ordinary_messages": fixture.ordinary_message_count,
                "small_shell_results": fixture.turn_count - 1,
                "large_shell_results": 1,
                "large_shell_lines": SHELL_LINE_COUNT,
                "oversized_non_shell_group_lines": fixture.oversized_group_line_count,
            },
            "warmups": 1,
            "repetitions": _REPETITIONS,
            "repetition_scope": (
                "resume and Load More are rebuilt through the real app path; "
                f"one warmup is excluded from {_REPETITIONS} measured samples"
            ),
            "scenario_budget_note": (
                "post-activation full scope: 300/400/500 entries, five repetitions"
                if _FULL_SCOPE
                else "post-activation smoke: 400 entries, two repetitions"
            ),
            "bidirectional_traversal_scope": (
                f"three passes over {_TRAVERSAL_STEP_COUNT} half-viewport steps "
                "with two reversals; not a full-transcript sweep"
            ),
            "resume_ms": percentiles(resume_samples),
            "load_more_batch_latency_ms": percentiles(load_more_samples),
            "load_more_total_ms": percentiles(load_more_total_samples),
            "load_more_batches_per_repetition": load_more_counts,
            "reconstruction_baseline_ms": percentiles(reconstruction_samples),
            "reconstruction_baseline_label": (
                "pre-C baseline: test-only _build_history_widgets_raw fixture; no DOM mount"
            ),
            "registry_snapshots": registry_samples,
            "traversal_residency_samples": residency_samples,
            "traversal_budget_envelope": traversal_budget_envelope,
            "remount_ready_latency_ms": percentiles(remount_ready_ms)
            if remount_ready_ms
            else {},
            "remount_ready_sample_count": len(remount_ready_ms),
            "remount_ready_sample_count_by_phase": {
                phase: len(samples)
                for phase, samples in remount_ready_ms_by_phase.items()
            },
            "remount_ready_samples_ms": [round(ms, 3) for ms in remount_ready_ms],
            "remount_ready_classification_unknown_count": unknown_restore_classifications[
                0
            ],
            "scroll_to_ready_latency_ms": {
                phase: percentiles(samples)
                for phase, samples in scroll_to_ready_samples.items()
            },
            "scroll_to_ready_sample_count_by_phase": {
                phase: len(samples) for phase, samples in remount_windows.items()
            },
            "remount_window_samples": remount_windows,
            "remount_latency_scope": (
                "per-unit placeholder to settled resident via restore_unit; "
                "per-scroll request to all in-range pending restores settled; "
                "oversized units excluded from per-unit distribution"
            ),
            "reconcile_pass_count_by_phase": dict(reconcile_counts),
            "oversized_traversal_reconcile_passes_by_repetition": (
                oversized_reconcile_passes_by_repetition
            ),
            "oversized_traversal_reconcile_pass_ceiling": _RECONCILE_PASS_CEILING,
            "registry_remount_count_by_phase": dict(registry_mount_counts),
            "registry_eviction_count_by_phase": dict(registry_evict_counts),
            "oversized_traversal_anchor_samples": traversal_anchors.get(
                "oversized_group_traverse", []
            ),
            "oversized_blank_region_count": sum(
                bool(sample["blank_region"])
                for sample in traversal_anchors.get("oversized_group_traverse", [])
            ),
            "refresh_layout_pass_count": len(refresh_pass_ms),
            "refresh_layout_pass_count_by_phase": dict(refresh_pass_counts),
            "refresh_layout_pass_ms": [round(value, 3) for value in refresh_pass_ms],
            "refresh_layout_distribution_ms": refresh_distribution,
            "refresh_layout_distribution_by_phase_ms": {
                phase: percentiles(samples)
                for phase, samples in refresh_pass_ms_by_phase.items()
                if samples
            },
            "per_step_reflow_and_residency": per_step_reflow_residency,
            "scroll_step_latency_ms": scroll_latency_summary,
            "scroll_step_counts": {
                phase: len(samples)
                for phase, samples in bidirectional_traverse_samples.items()
            },
            "anchor_samples": resize_anchors,
            "eviction_remount_anchor_samples": eviction_remount_anchor_samples,
            "prepend_anchor_samples": prepend_anchors,
            "traversal_anchor_samples": traversal_anchors,
            "repetition_phase_order": [
                "resume",
                "load_more",
                "reconstruction_baseline",
                "oversized_group_expand",
                "bidirectional_traverse",
                "resize_anchor",
                "oversized_group_traverse",
                "eviction_remount_anchor",
                "large_shell_expand",
            ],
            "anchor_phase_order": (
                "resume → Load More → traversal → resize → oversized traversal → "
                "retained-anchor eviction/remount → shell expansion"
            ),
            "anchor_screen_row_error_roundtrip": percentiles(anchor_errors),
            "oversized_group_expansion_cost_ms": percentiles(
                oversized_expansion_samples
            ),
            "oversized_group_rows": percentiles([
                float(value) for value in oversized_height_samples
            ]),
            "oversized_group_traversal_scope": (
                "expanded group traversed in both directions; pinned/oversized exempt"
            ),
            "large_shell_expansion_cost_ms": percentiles(shell_expansion_samples),
            "large_shell_expansion_scope": (
                "expanded once per repetition for body mount/layout cost, then "
                "collapsed before transcript traversal"
            ),
            "reversal_dom_mount_count": reversal_mounts,
            "reversal_dom_evict_count": reversal_evictions,
            "reversal_dom_changes_per_phase": reversal_changes,
            "warmup_sample": warmup_metrics,
            "machine": machine_context(),
        },
    )


# --- P0a: bounded registry-scale baseline ----------------------------------
#
# Measures the reconcile scans as a function of REGISTRY size at a FIXED
# residency, with no Textual app and no DOM: a bare TranscriptWindow for the
# engine scans, plus the real TranscriptPresenter tick (the gate
# ``TranscriptPresenter.request_transcript_reconcile``, where the
# registry-wide scans lived pre-P3b2b) driven against a stub host. The
# existing 400-entry scenario above measures bounded residency through the
# app; this fixture measures the denominators that bound registry scale, and
# it is scope-independent (it does not follow _ENTRY_COUNTS
# or _REPETITIONS).
#
# This fixture RECORDS the pre-optimization baseline and FREEZES the values
# later packages compare against. The flat-scaling, unchanged-tick, and
# allocation-plateau assertions below were gated off at P0a and are TURNED ON
# by P3b2b's gate cutover; P3b1 turned the anchor-scaling assertion on with
# its ordered placement index. Exceeding a frozen ceiling is a blocker;
# meeting a frozen improvement bound is the named later package's
# acceptance.
#
# FROZEN FIXTURE SIZE (decided here, before any threshold): 50k, not the 20k
# fallback. Measured on the baseline machine while deciding: admission 1.30 s
# untraced / 3.88 s traced, presenter tick 39.7 ms p50 at 50k, whole three-size
# fixture ~11 s against the package's 60 s budget. Runtime therefore never
# argues for the fallback, while 20k would halve the 1k->top span that makes the
# P3b2b flat-scaling bound discriminating (the recorded baseline grows 82.7x
# from 1k to 50k; the frozen bound is 2.0x).
#
# Entries are deliberately ungrouped (one registry unit per entry) so registry
# size maps to a known unit count; tool-group unit behavior is covered by the
# Textual scenario above and by the windowing suites, not re-measured here.

_REGISTRY_SIZES = (1_000, 10_000, 50_000)
_REGISTRY_TOP_SIZE = 50_000
_REGISTRY_RESIDENT_UNITS = 24
_REGISTRY_RESIDENT_ROWS = 6
_REGISTRY_CHECKPOINT_STRIDE = 50
_REGISTRY_GEOMETRY_WIDTH = TRANSCRIPT_WIDTH
_REGISTRY_VIEWPORT_HEIGHT = TRANSCRIPT_HEIGHT
_REGISTRY_TICKS = 16
_REGISTRY_ANCHOR_REPS = 8
# ACTIVE-TURN VARIANT (deep-review finding 1). The fixture used to run only
# with _active_turn_start = None, so the presenter's active-turn pin path
# (_protection_sets: the start_index bisect and the pinned-suffix walk) was
# never measured at registry scale. The deep review measured that path at
# 52.6x from 1k to 50k and 12.4 ms per tick at 50k -- inside all three frozen
# P3b2b bounds, so none of them could see the regression. The variant below
# sets a non-None start on this same fixture: a turn begins at the registry
# tail (its own newest units are the resident ones a real turn pins), so the
# pinned suffix is a constant 24 units at every size. A return to the
# registry-wide walk then fails the flat-scaling bound: a scratch probe of the
# pre-fix loop on this fixture's 50k registry measured 2.48 ms p50 against
# 0.0023 ms for the shipped suffix walk, i.e. ~50x from 1k to 50k.
# SCOPE LIMIT (deep-review round 5, disposition): the per-tick pin walk is
# O(turn units) -- the suffix bisect is O(log units) but the pin loop visits
# every unit of the turn -- and this variant fixes the turn at the resident
# unit count, so the frozen P3b2b ceilings are met at every registry size.
# A turn LONGER than _REGISTRY_RESIDENT_UNITS is therefore NOT observed by
# the frozen ceilings: those ticks stay proportional to the turn, not to
# registry size, which is the property the ceilings bound. Recorded only; no
# large-turn variant is measured here.
_REGISTRY_ACTIVE_TURN_UNITS = _REGISTRY_RESIDENT_UNITS
# Queries per timed sample. One indexed anchor lookup costs microseconds, below
# what the millisecond percentile rounding can resolve, so each sample batches
# them and the ratio compares like with like.
_REGISTRY_ANCHOR_QUERIES = 32
# Wall-clock guard only, not a per-measurement ceiling: the recorded baseline
# runs in roughly 11 s against the package's 60 s budget.
_REGISTRY_TIMEOUT = 180

# FROZEN CEILINGS (P0a). Non-regression upper bounds: exceeding one is a
# blocker, never a threshold change. PERF SCOPE: these apply in BOTH perf
# scopes -- the default 400-entry x 2-rep transcript smoke and
# CHARTREUX_PERF_TRANSCRIPT_FULL=1 -- because this fixture is scope-independent.
# Baseline: the committed perf-baselines/p0a-registry-baseline.jsonl run at
# 57217e5f (whole-file order, so the registry fixture runs after the ~380 s
# Textual scenario in the same process): admission p50 13.9/150.6/2098.2 ms,
# tick p50 0.79/8.88/80.65 ms. Run-to-run variance on this host: 50k admission
# 2.10-3.17 s, 50k tick p50 80.65-84.80 ms. An isolated -k run measures lower,
# so these cover both. Values are 4x (latency) or 6x (tick max, memory) that
# recorded baseline, so host load cannot mask a real regression.
# MEMORY CEILINGS ARE FIRST-OBSERVATION BASELINES: no earlier recorded values
# exist for them (tick alloc peak 0.013/0.082/0.387 MiB, admission alloc peak
# 3.06/30.83/157.36 MiB, admission RSS delta 0.0/16.5/122.3 MiB -- RSS deltas
# vary with heap warmth, hence the 1k floor at 25 MiB).
_REGISTRY_ADMIT_P50_CEILING_MS = {1_000: 62.0, 10_000: 640.0, 50_000: 12_700.0}
_REGISTRY_TICK_P50_CEILING_MS = {1_000: 3.2, 10_000: 56.0, 50_000: 340.0}
_REGISTRY_TICK_MAX_CEILING_MS = {1_000: 6.5, 10_000: 105.0, 50_000: 540.0}
_REGISTRY_TICK_ALLOC_CEILING_MIB = {1_000: 0.08, 10_000: 0.5, 50_000: 2.4}
_REGISTRY_ADMIT_ALLOC_CEILING_MIB = {1_000: 19.0, 10_000: 185.0, 50_000: 945.0}
_REGISTRY_ADMIT_RSS_CEILING_MIB = {1_000: 25.0, 10_000: 74.0, 50_000: 460.0}

# FROZEN IMPROVEMENT BOUNDS (P0a) -- acceptance values for LATER packages, not
# baseline properties. A miss by the named package is a blocker for that
# package. Recorded denominators (committed artifact, second run in brackets):
# tick p50 1k=0.79 ms / 50k=80.65 ms, 102x (107x), with 6 registry visits per
# unit per tick (6,000 / 300,000); tick allocation peak 50k=0.387 MiB; anchor
# query 50k=13.50 ms against 1k=0.101 ms, 134x (196x), because the oldest-first
# scan walked the whole registry. The 1k anchor sample is sub-0.1 ms, so its
# ratio is the noisy side of that bound; the 50k absolute value is stable.
# ANCHOR SAMPLES ARE NOW BATCHED: P3b1's ordered placement index answers the
# query in microseconds, so each sample times _REGISTRY_ANCHOR_QUERIES lookups
# and the ratio stays a like-for-like comparison of per-query cost.
_P3B2B_FLAT_SCALING_MAX_RATIO = 2.0
_P3B2B_UNCHANGED_TICK_MAX_MS = 5.0
_P3B2B_ALLOC_PLATEAU_MAX_MIB = 0.05
_P3B1_ANCHOR_SCALING_MAX_RATIO = 2.0
# The indexed query reads content order and live rows only, so it must not walk
# the registry at all; the replaced oldest-first scan visited every unit
# (~50k per query at the top size).
_P3B1_ANCHOR_REGISTRY_VISIT_CEILING = 0

# FROZEN IMPROVEMENT BOUND (P2b). P0a recorded admission-time retention cost
# WITH entry_snapshots live: traced admission alloc peak 3.056 / 30.8276 /
# 157.3558 MiB at 1k / 10k / 50k, from the committed
# perf-baselines/p0a-registry-baseline.meta.json run at 57217e5f. P2b replaced
# those per-entry deep copies with per-entry fingerprints. Measured on this
# tree (f82706a6): 1.4364 / 14.4111 / 76.9472 MiB, i.e. 53.0% / 53.3% / 51.1%
# below the baseline. The bound freezes 40%, ~11 points under the worst
# measured size, so a regression that brings any share of the copies back is
# a blocker. Full record: perf-baselines/p2b-snapshot-retention-bound.meta.json.
_P2B_BASELINE_ADMIT_ALLOC_PEAK_MIB = {1_000: 3.056, 10_000: 30.8276, 50_000: 157.3558}
_P2B_ADMIT_ALLOC_REDUCTION_MIN_PCT = 40.0

# Populated by test_transcript_registry_scale_baseline; the bound tests below
# read it. File order populates it before they could run.
_REGISTRY_BASELINE: dict[int, dict[str, float]] = {}


def _registry_entry(index: int) -> PublicHistoryEntry:
    if index % _REGISTRY_CHECKPOINT_STRIDE == _REGISTRY_CHECKPOINT_STRIDE - 1:
        return PublicCheckpointEntry(
            id=f"registry-checkpoint-{index:06d}",
            session_id="registry-perf-session",
            turn_id=None,
            created_at=index,
            updated_at=index,
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            kind="compaction",
            message="Synthetic registry compaction checkpoint",
        )
    return PublicMessageEntry(
        id=f"registry-message-{index:06d}",
        session_id="registry-perf-session",
        turn_id=f"registry-turn-{index // 6:06d}",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role="assistant" if index % 2 else "user",
        content=[TextContentBlock(text=f"Registry fixture body {index:06d}.")],
        source="harness",
    )


def _build_registry_entries(entry_count: int) -> list[PublicHistoryEntry]:
    return [_registry_entry(index) for index in range(entry_count)]


class _StubPlacementStream(Widget):
    """Placement parent that lets the fixture register roots without a mount.

    ``register_mounted`` validates that a unit's roots share one ``Widget``
    parent and are contiguous there; layout stays stubbed, so ``children`` and
    ``size`` stand in for measured geometry.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self._stub_children: list[Widget] = []
        self._stub_size = Size(width, _REGISTRY_VIEWPORT_HEIGHT)

    @property
    def children(self) -> Sequence[Widget]:
        return self._stub_children

    @property
    def size(self) -> Size:
        return self._stub_size

    def attach(self, placement: Widget) -> None:
        self._stub_children.append(placement)


class _PlacementStub(Widget):
    """Screen-geometry stand-in for one mounted placement root.

    The registry reads ``region`` off mounted roots and walks their children.
    A real Textual mount would measure layout instead of registry scale and
    cost seconds per size, so the fixture registers these with the real
    ``register_mounted`` seam and stubs the measured region.
    """

    def __init__(self, parent: _StubPlacementStream, top: int, height: int) -> None:
        super().__init__()
        self._stub_parent = parent
        self._stub_region = Region(0, top, parent.size.width, height)

    @property
    def parent(self) -> _StubPlacementStream | None:
        return self._stub_parent

    @property
    def region(self) -> Region:
        return self._stub_region

    @property
    def text_selection(self) -> None:
        # The selection gate reads this off mounted rows; an unmounted stub
        # has none (and no screen to resolve one against).
        return None


class _StubPlaceholder:
    """Stand-in for an evicted unit's PlacementPlaceholder with fixed geometry."""

    __slots__ = ("parent", "region")

    def __init__(self, top: int, height: int = 8) -> None:
        self.parent = None
        self.region = Region(0, top, _REGISTRY_GEOMETRY_WIDTH, height)

    def invalidate_geometry(self) -> None:
        return None


class _SpikeRoot:
    """Mounted-root stand-in for the mounted-spike phase (finding G).

    The spike mounts the whole registry and is evicted before any tick runs,
    so no scan ever reads these roots; they exist only to drive
    ``shadow.mounted`` through its peak and back down.
    """

    __slots__ = ("region",)

    def __init__(self) -> None:
        self.region = Region(0, -8, _REGISTRY_GEOMETRY_WIDTH, 8)


class _VisitCounter:
    """Counts registry unit visits made by the reconcile scans in one tick."""

    __slots__ = ("visits",)

    def __init__(self) -> None:
        self.visits = 0


class _CountingUnitIds(list[str]):
    """Registry id list that counts the ``unit_ids`` traversals scans perform.

    Slices are not counted (the checkpoint-neighbour slice adds at most three
    visits per checkpoint unit); the full traversals are the denominators.
    """

    def __init__(self, unit_ids: list[str], counter: _VisitCounter) -> None:
        super().__init__(unit_ids)
        self.counter = counter

    def __iter__(self) -> Iterator[str]:
        for unit_id in super().__iter__():
            self.counter.visits += 1
            yield unit_id


class _StubMessagesArea:
    """Minimal stand-in for the app's message area, which is reconcile's stream."""

    def __init__(self, width: int, height: int) -> None:
        self.size = SimpleNamespace(width=width, height=height)
        self.virtual_size = SimpleNamespace(width=width, height=height)
        self.region = SimpleNamespace(y=0, bottom=height, height=height)
        self.parent = None


class _StubChatScroll:
    """Minimal stand-in for the chat scroll container."""

    def __init__(self, height: int) -> None:
        self.region = SimpleNamespace(y=0, bottom=height, height=height)
        self.is_at_bottom = True


class _PresenterHostStub:
    """Exactly the host attributes ``request_transcript_reconcile`` reads."""

    def __init__(self, width: int, height: int) -> None:
        self._shutdown_started = False
        self._chat_widget = _StubChatScroll(height)
        self._messages_area = _StubMessagesArea(width, height)
        self.event_handler = None
        self._rewind_highlighted_widget = None
        self._queue_selected_widget = None
        self._agent_transcript_focus_target = None
        self._active_turn_start = None


class _RetainedWidgetsStub:
    """``event_handler`` stand-in whose retained set holds the given widgets."""

    def __init__(self, widgets: list[Widget]) -> None:
        self._widgets = widgets

    def retained_widgets(self) -> set[Widget]:
        return set(self._widgets)


def _place_resident_units(
    window: TranscriptWindow, stream: _StubPlacementStream
) -> None:
    """Give the tail units fixed placement geometry inside the viewport.

    The resident window sits at the END of the registry -- a long session
    scrolled to its newest messages -- so the anchor query resolves against
    resident rows only while resident rows stay far under the eviction high
    mark (24 * 6 rows against 1500), which keeps residency fixed across sizes.
    """
    resident_ids = window.unit_ids[-_REGISTRY_RESIDENT_UNITS:]
    for offset, unit_id in enumerate(resident_ids):
        stub = _PlacementStub(
            stream, offset * _REGISTRY_RESIDENT_ROWS, _REGISTRY_RESIDENT_ROWS
        )
        stream.attach(stub)
        window.register_mounted(unit_id, [stub])


def _evict_to_placeholders(window: TranscriptWindow, resident: int) -> int:
    """Give every non-resident unit a stub placeholder above the viewport.

    Simulates evicting all but the resident tail, so placeholders dominate the
    registry (deep-review rerun E1: the idle and active-turn variants assert
    zero placeholders and cannot see the restore path's per-tick cost). The
    stub regions keep the placement regions monotonic in content order -- the
    resident tail sits at the viewport top, the evicted slots above it.
    """
    total = len(window.unit_ids)
    for position, unit_id in enumerate(window.unit_ids):
        if position >= total - resident:
            continue
        unit = window.units[unit_id]
        unit.mounted_roots = []
        unit.placeholders = [cast("Any", _StubPlaceholder((position - total) * 8))]
        window._shadow_unit(unit)
    window._placements = list(window.unit_ids)
    return total - resident


def _frozen_registry_baseline() -> dict[int, dict[str, float]]:
    assert _REGISTRY_BASELINE, (
        "the gated bound tests read the recorded baseline; run "
        "test_transcript_registry_scale_baseline first (file order does this)"
    )
    return _REGISTRY_BASELINE


async def _sample_ticks(
    tick: Callable[[], Awaitable[None]], counter: _VisitCounter
) -> tuple[list[float], list[int]]:
    """Warm up, then time ``_REGISTRY_TICKS`` reconcile ticks.

    Returns one wall-clock millisecond sample per tick plus the registry
    visits the counting ``unit_ids`` observed during it; ``counter`` resets
    per sample.
    """
    await tick()  # warmup, excluded from the samples and the visit counts
    tick_ms: list[float] = []
    visits_per_tick: list[int] = []
    for _ in range(_REGISTRY_TICKS):
        counter.visits = 0
        started = perf_counter_ns()
        await tick()
        tick_ms.append((perf_counter_ns() - started) / 1_000_000)
        visits_per_tick.append(counter.visits)
    return tick_ms, visits_per_tick


async def _measure_engine_window(
    entries: list[PublicHistoryEntry], entry_count: int
) -> dict[str, object]:
    """Admit a bare engine window and time its registry-wide scans.

    Kept in its own frame so the registry is released when it returns; the
    presenter phase below builds its own window.
    """
    window = TranscriptWindow(
        tools_collapsed=True,
        expansion_state=ToolGroupExpansionState(),
        entry_expansion_state=EntryExpansionState(),
    )
    admission_ms: list[float] = []

    async def admit() -> None:
        started = perf_counter_ns()
        window.admit(entries, start_index=0, geometry_width=_REGISTRY_GEOMETRY_WIDTH)
        admission_ms.append((perf_counter_ns() - started) / 1_000_000)

    rss_before = rss_mib()
    admission_lag = await measure_loop_lag(admit())
    rss_after = rss_mib()
    rss_peak = rss_peak_mib()

    assert len(window.unit_ids) == entry_count
    assert len(window.units) == entry_count
    placements = _StubPlacementStream(_REGISTRY_GEOMETRY_WIDTH)
    _place_resident_units(window, placements)
    view = _StubMessagesArea(_REGISTRY_GEOMETRY_WIDTH, _REGISTRY_VIEWPORT_HEIGHT)

    def placeholders_scan() -> None:
        any(unit.placeholders for unit in window.units.values())

    def eviction_scan() -> None:
        window.eviction_candidates(view.region.y, view.region.height)

    scan_ms = {
        # The presenter's placeholder gate: a full miss visits every unit.
        "has_placeholders": percentiles(
            repeat_timed_ms(placeholders_scan, reps=_REGISTRY_TICKS)
        ),
        "eviction_candidates": percentiles(
            repeat_timed_ms(eviction_scan, reps=_REGISTRY_TICKS)
        ),
    }

    # The anchor query is batched per sample: a single indexed lookup costs
    # microseconds, which the millisecond percentile rounding cannot resolve.
    counter = _VisitCounter()
    window.unit_ids = _CountingUnitIds(window.unit_ids, counter)

    def anchor_scan() -> None:
        for _ in range(_REGISTRY_ANCHOR_QUERIES):
            reading = window.capture_anchor(cast(Widget, view), following=True)
            assert reading is not None

    counter.visits = 0
    scan_ms["capture_anchor"] = percentiles(
        repeat_timed_ms(anchor_scan, reps=_REGISTRY_ANCHOR_REPS)
    )
    anchor_visits = counter.visits

    # Residency must stay fixed: nothing evicted, nothing replaced.
    assert window.eviction_candidates(view.region.y, view.region.height) == []
    assert (
        sum(len(unit.mounted_roots) for unit in window.units.values())
        == _REGISTRY_RESIDENT_UNITS
    )
    return {
        "admission_ms": percentiles(admission_ms),
        "admission_loop_lag_ms": admission_lag,
        "rss_delta_mib": round(rss_after - rss_before, 3),
        "rss_after_mib": round(rss_after, 3),
        "rss_peak_mib": round(rss_peak, 3),
        "scan_ms": scan_ms,
        "anchor_registry_visits": anchor_visits,
    }


@pytest.mark.asyncio
@pytest.mark.perf
@pytest.mark.timeout(_REGISTRY_TIMEOUT)
@pytest.mark.parametrize("entry_count", _REGISTRY_SIZES)
async def test_transcript_registry_scale_baseline(entry_count: int) -> None:
    """Record registry-scale reconcile costs and assert the frozen ceilings."""
    entries = _build_registry_entries(entry_count)
    checkpoint_count = entry_count // _REGISTRY_CHECKPOINT_STRIDE

    engine = await _measure_engine_window(entries, entry_count)
    engine_scan_ms = cast(dict[str, dict[str, float]], engine["scan_ms"])
    admission_stats = cast(dict[str, float], engine["admission_ms"])
    rss_delta = cast(float, engine["rss_delta_mib"])

    # --- presenter window: the real per-tick gate -------------------------
    presenter = TranscriptPresenter(get_show_timestamps=lambda: False)
    presenter._bind_app(  # the presenter's documented late-bind seam
        cast(
            ChartreuxApp,
            _PresenterHostStub(_REGISTRY_GEOMETRY_WIDTH, _REGISTRY_VIEWPORT_HEIGHT),
        )
    )
    window = presenter._transcript

    # Allocation pass: tracing inflates latency, so this one is memory only.
    with alloc_peak() as admission_alloc:
        traced_started = perf_counter_ns()
        window.admit(entries, start_index=0, geometry_width=_REGISTRY_GEOMETRY_WIDTH)
        traced_admission_ms = (perf_counter_ns() - traced_started) / 1_000_000

    _place_resident_units(window, _StubPlacementStream(_REGISTRY_GEOMETRY_WIDTH))
    counter = _VisitCounter()
    window.unit_ids = _CountingUnitIds(window.unit_ids, counter)

    async def tick() -> None:
        presenter.request_transcript_reconcile()
        task = window._reconcile_task
        assert task is not None
        await task

    tick_ms, visits_per_tick = await _sample_ticks(tick, counter)
    assert len(set(visits_per_tick)) == 1, "registry visits drifted between ticks"
    # Post-P3b2b the gate answers from the reconcile indexes, so an unchanged
    # tick no longer traverses unit_ids at all; the count below records that.

    with alloc_peak() as tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()

    # --- ACTIVE-TURN VARIANT: the same registry, a live turn ---------------
    # The app sets this marker at TurnStarted; the fixture's host starts idle,
    # so the assignment below is the same transition on the same registry. The
    # pin loop then bisects the start_index-ordered unit_ids to the turn's
    # suffix instead of walking the registry, and it iterates a plain-list
    # slice, so _CountingUnitIds does not count those visits: tick time, not
    # the visit counter, is what bounds this path.
    active_turn_start = entry_count - _REGISTRY_ACTIVE_TURN_UNITS
    presenter._app._active_turn_start = active_turn_start
    active_tick_ms, active_visits_per_tick = await _sample_ticks(tick, counter)
    assert len(set(active_visits_per_tick)) == 1, (
        "active-turn registry visits drifted between ticks"
    )
    pinned_ids = {
        unit_id
        for unit_id in window.unit_ids
        if window.units[unit_id].start_index >= active_turn_start
    }
    # Pin coverage must equal the start_index range the bisect resolves: a
    # cheaper tick that pins less (or pins the whole registry) is a behavior
    # change this bound must not absorb.
    assert pinned_ids == set(window.unit_ids[-_REGISTRY_ACTIVE_TURN_UNITS:]), (
        "active-turn pin coverage diverged from the start_index range"
    )
    assert all(
        "active-turn" in window.units[unit_id].pin_reasons for unit_id in pinned_ids
    ), "the active turn did not pin every unit in its range"
    assert not any(
        "active-turn" in unit.pin_reasons
        for unit_id, unit in window.units.items()
        if unit_id not in pinned_ids
    ), "the active turn pinned units before its start"

    with alloc_peak() as active_tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()

    assert all(
        window.units[unit_id].mounted_roots
        for unit_id in window.unit_ids[-_REGISTRY_RESIDENT_UNITS:]
    )
    assert not any(unit.placeholders for unit in window.units.values())

    # --- PLACEHOLDERS-HEAVY VARIANT (deep-review rerun E1) -----------------
    # The variants above assert zero placeholders, so they cannot see the
    # restore path's per-tick cost. This one evicts every unit except the
    # resident tail, so placeholders dominate the registry; the frozen P3b2b
    # bounds below must hold on it exactly as on the idle/active-turn variants.
    presenter._app._active_turn_start = None
    for unit in window.units.values():
        unit.pin_reasons.clear()
    evicted_count = _evict_to_placeholders(window, _REGISTRY_RESIDENT_UNITS)
    assert evicted_count == entry_count - _REGISTRY_RESIDENT_UNITS
    assert window.has_placeholders
    placeholders_tick_ms, placeholders_visits = await _sample_ticks(tick, counter)
    assert len(set(placeholders_visits)) == 1, (
        "placeholders-variant registry visits drifted between ticks"
    )
    with alloc_peak() as placeholders_tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()
    # Residency must stay fixed through the variant: nothing was restored.
    assert all(
        window.units[unit_id].mounted_roots
        for unit_id in window.unit_ids[-_REGISTRY_RESIDENT_UNITS:]
    )
    assert len(window._shadow.placeholders) == evicted_count

    # --- COMBINED LIVE+PLACEHOLDERS VARIANT (deep-review round 3, finding 1) --
    # The placeholders variant above leaves every unit non-live, so its dirty
    # set empties and the empty-clear releases the hash table. This variant
    # keeps the resident tail LIVE while the registry-wide dirty spike
    # settles, so the surviving live members keep live_dirty nonempty; its
    # table must compact back to live membership, or every tick's
    # sorted(live_dirty) scan walks the spike-sized table. Measured pre-fix:
    # 0.05648/0.11711/0.34933 ms at 1k/10k/50k = 6.185x against the frozen
    # 2.0x flat-scaling bound, with a ~2 MB table behind 24 members.
    resident_ids = window.unit_ids[-_REGISTRY_RESIDENT_UNITS:]
    presenter._app.event_handler = cast(
        "Any",
        _RetainedWidgetsStub([
            window.units[unit_id].mounted_roots[0] for unit_id in resident_ids
        ]),
    )
    window._mark_live_dirty(window.units.keys())  # the placement-transition spike
    live_tick_ms, live_visits = await _sample_ticks(tick, counter)
    assert len(set(live_visits)) == 1, (
        "live+placeholders-variant registry visits drifted between ticks"
    )
    with alloc_peak() as live_tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()
    # Residency and liveness must stay fixed through the variant: nothing
    # restored, the resident tail live, everything else observed dead.
    assert all(window.units[unit_id].mounted_roots for unit_id in resident_ids)
    assert len(window._shadow.placeholders) == evicted_count
    assert window._shadow.live == set(resident_ids)
    assert set(window._shadow.live_dirty) == set(resident_ids)

    # --- MOUNTED-SPIKE VARIANT (deep-review round 4, finding G) ------------
    # The variants above mount only the resident tail before creating
    # placeholders, so shadow.mounted never spikes and its hash table stays
    # small. This variant mounts the WHOLE registry first (the spike), then
    # evicts back down to the resident tail with placeholders: mounted ends
    # at 24 members behind a peak-sized table, and the per-tick selection and
    # eviction scans iterate mounted, so pre-fix they walk the spike table
    # (measured tick p50 0.095/0.206/0.681 ms at 1k/10k/50k = 7.18x against
    # the frozen 2.0x bound; rehashing mounted alone drops the 50k tick to
    # 0.087 ms).
    for unit_id in window.unit_ids[:-_REGISTRY_RESIDENT_UNITS]:
        unit = window.units[unit_id]
        unit.placeholders = []
        unit.mounted_roots = [cast("Any", _SpikeRoot())]
        window._shadow_unit(unit)
    assert len(window._shadow.mounted) == entry_count
    assert _evict_to_placeholders(window, _REGISTRY_RESIDENT_UNITS) == evicted_count
    mounted_spike_tick_ms, mounted_spike_visits = await _sample_ticks(tick, counter)
    assert len(set(mounted_spike_visits)) == 1, (
        "mounted-spike-variant registry visits drifted between ticks"
    )
    with alloc_peak() as mounted_spike_tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()
    # Residency must stay fixed through the variant: nothing was restored,
    # the resident tail mounted, everything else a placeholder again.
    assert all(window.units[unit_id].mounted_roots for unit_id in resident_ids)
    assert len(window._shadow.placeholders) == evicted_count
    assert set(window._shadow.mounted) == set(resident_ids)
    resident_set = set(resident_ids)

    # --- LIVE-MEMBERSHIP SPIKE VARIANT (deep-review round 5, finding 1) ---
    # No variant above leaves ``live`` behind a peak-sized table: it only
    # grows while a unit stays retained, and the variants that settle the
    # dirty set re-observe the same resident tail. This one retains the WHOLE
    # registry for one observation pass (the live spike), then settles back
    # onto the resident tail. The presenter reads ``frozenset(shadow.live)``
    # on every protection tick, so pre-fix the tick walked the spike-sized
    # table: measured tick p50 0.100/0.127/0.215 ms at 1k/10k/50k = 2.15x
    # against the frozen 2.0x bound, at 2,097,368 B behind 24 members.
    window._mark_live_dirty(window.unit_ids)
    for unit_id in window.unit_ids:
        window._observe_live(unit_id, True)  # the live-membership spike
    assert len(window._shadow.live) == entry_count
    live_spike_table_bytes = sys.getsizeof(window._shadow.live)
    window._mark_live_dirty(window.unit_ids)
    for unit_id in window.unit_ids:
        window._observe_live(unit_id, unit_id in resident_set)
    assert window._shadow.live == resident_set
    live_settled_table_bytes = sys.getsizeof(window._shadow.live)
    live_spike_tick_ms, live_spike_visits = await _sample_ticks(tick, counter)
    assert len(set(live_spike_visits)) == 1, (
        "live-spike-variant registry visits drifted between ticks"
    )
    with alloc_peak() as live_spike_tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()
    # Membership must stay settled through the variant: the resident tail
    # live, everything else placeholder.
    assert window._shadow.live == resident_set
    assert len(window._shadow.placeholders) == evicted_count

    # --- CHECKPOINT-NEIGHBOUR SPIKE VARIANT (round 5, finding 2) -----------
    # ``neighbors`` only holds checkpoints with a mounted neighbour, and this
    # fixture's checkpoints sit on a stride, so no variant above can push the
    # set past entry_count // _REGISTRY_CHECKPOINT_STRIDE. This one makes
    # EVERY unit a checkpoint while the whole registry is mounted (the
    # checkpoint-dense spike), then settles back onto the resident tail. The
    # presenter reads ``frozenset(shadow.neighbors)`` on every protection
    # tick, so pre-fix the tick walked the spike-sized table: measured tick
    # p50 0.101/0.141/0.272 ms = 2.69x against the frozen 2.0x bound, at
    # 2,097,368 B behind 24 members. Pre-fix this variant also inherited the
    # previous variant's uncompacted live table (live never shrank either);
    # that is the same defect and this change releases both.
    # ``checkpoints`` goes through the same spike but is read per transition,
    # never per tick, so it is out of this variant's scope.
    saved_entries = {
        unit_id: window.units[unit_id].entries for unit_id in window.unit_ids
    }
    for unit_id in window.unit_ids[:-_REGISTRY_RESIDENT_UNITS]:
        unit = window.units[unit_id]
        unit.placeholders = []
        unit.mounted_roots = [cast("Any", _SpikeRoot())]
        window._shadow_unit(unit)
    spike_checkpoint = _registry_entry(_REGISTRY_CHECKPOINT_STRIDE - 1)
    for unit_id in window.unit_ids:
        unit = window.units[unit_id]
        unit.entries = [spike_checkpoint]
        window._shadow_entries(unit)
    # The triple re-derivation can discard a unit whose neighbour had not
    # been re-derived yet, so re-derive every unit's triple once more.
    for unit_id in window.unit_ids:
        window._shadow_neighbors(unit_id)
    assert len(window._shadow.neighbors) == entry_count
    neighbors_spike_table_bytes = sys.getsizeof(window._shadow.neighbors)
    assert _evict_to_placeholders(window, _REGISTRY_RESIDENT_UNITS) == evicted_count
    for unit_id in window.unit_ids:
        if unit_id in resident_set:
            continue
        window.units[unit_id].entries = saved_entries[unit_id]
        window._shadow_entries(window.units[unit_id])
    assert set(window._shadow.neighbors) == resident_set
    neighbors_settled_table_bytes = sys.getsizeof(window._shadow.neighbors)
    neighbors_spike_tick_ms, neighbors_spike_visits = await _sample_ticks(tick, counter)
    assert len(set(neighbors_spike_visits)) == 1, (
        "neighbors-spike-variant registry visits drifted between ticks"
    )
    with alloc_peak() as neighbors_spike_tick_alloc:
        for _ in range(_REGISTRY_TICKS):
            await tick()
    # Membership must stay settled: every resident unit is a checkpoint with
    # a mounted neighbour, and nothing outside the tail is either.
    assert set(window._shadow.neighbors) == resident_set
    assert len(window._shadow.placeholders) == evicted_count
    # The spike's checkpoint stubs are not part of the fixture's registry, so
    # put the resident tail's entries back now that the variant is measured.
    for unit_id in resident_ids:
        window.units[unit_id].entries = saved_entries[unit_id]
        window._shadow_entries(window.units[unit_id])
    del saved_entries
    # Set semantics survive the settle: re-derive neighbour membership from
    # the registry and require the index to agree.
    settled_shadow = window._shadow
    expected_neighbors = {
        unit_id
        for unit_id in window.unit_ids
        if any(
            isinstance(entry, PublicCheckpointEntry)
            for entry in window.units[unit_id].entries
        )
        and any(
            neighbour is not None and neighbour in settled_shadow.mounted
            for neighbour in (
                settled_shadow.before.get(unit_id),
                settled_shadow.after.get(unit_id),
            )
        )
    }
    assert set(settled_shadow.neighbors) == expected_neighbors, (
        "neighbor membership diverged from the checkpoint adjacency derivation"
    )

    tick_p50 = percentiles(tick_ms)["p50"]
    tick_max = percentiles(tick_ms)["max"]
    active_tick_p50 = percentiles(active_tick_ms)["p50"]
    active_tick_max = percentiles(active_tick_ms)["max"]
    placeholders_tick_p50 = percentiles(placeholders_tick_ms)["p50"]
    placeholders_tick_max = percentiles(placeholders_tick_ms)["max"]
    live_tick_p50 = percentiles(live_tick_ms)["p50"]
    live_tick_max = percentiles(live_tick_ms)["max"]
    mounted_spike_tick_p50 = percentiles(mounted_spike_tick_ms)["p50"]
    mounted_spike_tick_max = percentiles(mounted_spike_tick_ms)["max"]
    live_spike_tick_p50 = percentiles(live_spike_tick_ms)["p50"]
    live_spike_tick_max = percentiles(live_spike_tick_ms)["max"]
    neighbors_spike_tick_p50 = percentiles(neighbors_spike_tick_ms)["p50"]
    neighbors_spike_tick_max = percentiles(neighbors_spike_tick_ms)["max"]
    anchor_p50 = engine_scan_ms["capture_anchor"]["p50"]
    anchor_registry_visits = cast(int, engine["anchor_registry_visits"])
    admission_p50 = cast(dict[str, float], engine["admission_ms"])["p50"]

    record(
        f"transcript_registry.{entry_count}",
        {
            "package": "P0a",
            "entry_count": entry_count,
            "registry_units": len(window.units),
            "checkpoint_units": checkpoint_count,
            "resident_units": _REGISTRY_RESIDENT_UNITS,
            "resident_rows": _REGISTRY_RESIDENT_UNITS * _REGISTRY_RESIDENT_ROWS,
            "geometry": {
                "width": _REGISTRY_GEOMETRY_WIDTH,
                "viewport_height": _REGISTRY_VIEWPORT_HEIGHT,
            },
            "measured_ticks": _REGISTRY_TICKS,
            "anchor_query_reps": _REGISTRY_ANCHOR_REPS,
            "admission_ms": admission_stats,
            "admission_loop_lag_ms": engine["admission_loop_lag_ms"],
            "admission_rss_delta_mib": engine["rss_delta_mib"],
            "admission_rss_after_mib": engine["rss_after_mib"],
            "admission_rss_peak_mib": engine["rss_peak_mib"],
            "admission_traced": {
                "peak_mib": admission_alloc.peak_mib,
                "net_mib": admission_alloc.net_mib,
                "traced_ms_not_a_baseline": round(traced_admission_ms, 1),
            },
            "engine_scan_ms": engine_scan_ms,
            "anchor_queries_per_sample": _REGISTRY_ANCHOR_QUERIES,
            "anchor_registry_visits_per_sample": anchor_registry_visits,
            "presenter_tick_ms": percentiles(tick_ms),
            "registry_visits_per_tick": visits_per_tick[0],
            "presenter_tick_alloc_traced": {
                "peak_mib": tick_alloc.peak_mib,
                "net_mib": tick_alloc.net_mib,
                "ticks": _REGISTRY_TICKS,
            },
            "active_turn": {
                "turn_start_index": active_turn_start,
                "pinned_units": len(pinned_ids),
                "presenter_tick_ms": percentiles(active_tick_ms),
                "registry_visits_per_tick": active_visits_per_tick[0],
                "presenter_tick_alloc_traced": {
                    "peak_mib": active_tick_alloc.peak_mib,
                    "net_mib": active_tick_alloc.net_mib,
                    "ticks": _REGISTRY_TICKS,
                },
            },
            "placeholders_variant": {
                "evicted_units": evicted_count,
                "presenter_tick_ms": percentiles(placeholders_tick_ms),
                "registry_visits_per_tick": placeholders_visits[0],
                "presenter_tick_alloc_traced": {
                    "peak_mib": placeholders_tick_alloc.peak_mib,
                    "net_mib": placeholders_tick_alloc.net_mib,
                    "ticks": _REGISTRY_TICKS,
                },
            },
            "live_placeholders_variant": {
                "evicted_units": evicted_count,
                "live_units": _REGISTRY_RESIDENT_UNITS,
                "presenter_tick_ms": percentiles(live_tick_ms),
                "registry_visits_per_tick": live_visits[0],
                "presenter_tick_alloc_traced": {
                    "peak_mib": live_tick_alloc.peak_mib,
                    "net_mib": live_tick_alloc.net_mib,
                    "ticks": _REGISTRY_TICKS,
                },
            },
            "mounted_spike_variant": {
                "spike_units": entry_count,
                "evicted_units": evicted_count,
                "mounted_units": _REGISTRY_RESIDENT_UNITS,
                "presenter_tick_ms": percentiles(mounted_spike_tick_ms),
                "registry_visits_per_tick": mounted_spike_visits[0],
                "presenter_tick_alloc_traced": {
                    "peak_mib": mounted_spike_tick_alloc.peak_mib,
                    "net_mib": mounted_spike_tick_alloc.net_mib,
                    "ticks": _REGISTRY_TICKS,
                },
            },
            "live_spike_variant": {
                "spike_units": entry_count,
                "live_units": _REGISTRY_RESIDENT_UNITS,
                "evicted_units": evicted_count,
                "spike_table_bytes": live_spike_table_bytes,
                "settled_table_bytes": live_settled_table_bytes,
                "presenter_tick_ms": percentiles(live_spike_tick_ms),
                "registry_visits_per_tick": live_spike_visits[0],
                "presenter_tick_alloc_traced": {
                    "peak_mib": live_spike_tick_alloc.peak_mib,
                    "net_mib": live_spike_tick_alloc.net_mib,
                    "ticks": _REGISTRY_TICKS,
                },
            },
            "neighbors_spike_variant": {
                "spike_units": entry_count,
                "neighbor_units": _REGISTRY_RESIDENT_UNITS,
                "evicted_units": evicted_count,
                "spike_table_bytes": neighbors_spike_table_bytes,
                "settled_table_bytes": neighbors_settled_table_bytes,
                "presenter_tick_ms": percentiles(neighbors_spike_tick_ms),
                "registry_visits_per_tick": neighbors_spike_visits[0],
                "presenter_tick_alloc_traced": {
                    "peak_mib": neighbors_spike_tick_alloc.peak_mib,
                    "net_mib": neighbors_spike_tick_alloc.net_mib,
                    "ticks": _REGISTRY_TICKS,
                },
            },
            "measurement_scope": (
                "no Textual mount: engine scans on a bare TranscriptWindow with "
                "synthetic placement geometry; the tick drives the real presenter "
                "gate against a stub host, so DOM layout, restore and paint costs "
                "are out of scope"
            ),
            "memory_scope": (
                "RSS deltas are first-observation baselines: they vary with heap "
                "warmth and have no earlier comparison point. The traced admission "
                "peak is compared against the P0a snapshot-retention baseline by "
                "the P2b bound; both are valid on the baseline machine only"
            ),
            "perf_scope": (
                "applies in both perf scopes (default 400-entry x 2-rep smoke and "
                "CHARTREUX_PERF_TRANSCRIPT_FULL=1); this fixture is scope-independent"
            ),
            "frozen_ceilings": {
                "admit_p50_ms": _REGISTRY_ADMIT_P50_CEILING_MS[entry_count],
                "tick_p50_ms": _REGISTRY_TICK_P50_CEILING_MS[entry_count],
                "tick_max_ms": _REGISTRY_TICK_MAX_CEILING_MS[entry_count],
                "tick_alloc_mib": _REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count],
                "admit_alloc_mib": _REGISTRY_ADMIT_ALLOC_CEILING_MIB[entry_count],
                "admit_rss_mib": _REGISTRY_ADMIT_RSS_CEILING_MIB[entry_count],
            },
            "frozen_improvement_bounds": {
                "p3b2b_flat_scaling_max_ratio": _P3B2B_FLAT_SCALING_MAX_RATIO,
                "p3b2b_unchanged_tick_max_ms": _P3B2B_UNCHANGED_TICK_MAX_MS,
                "p3b2b_alloc_plateau_max_mib": _P3B2B_ALLOC_PLATEAU_MAX_MIB,
                "p3b1_anchor_scaling_max_ratio": _P3B1_ANCHOR_SCALING_MAX_RATIO,
                "p2b_admit_alloc_reduction_min_pct": (
                    _P2B_ADMIT_ALLOC_REDUCTION_MIN_PCT
                ),
            },
            "active_assertions": [
                "p2b:test_transcript_registry_scale_baseline",
                "p3b1:test_transcript_registry_anchor_scaling_bound",
                "p3b2b:test_transcript_registry_flat_scaling_bound",
                "p3b2b:test_transcript_registry_unchanged_tick_bound",
                "p3b2b:test_transcript_registry_allocation_plateau_bound",
            ],
            "machine": machine_context(),
        },
    )

    _REGISTRY_BASELINE[entry_count] = {
        "admission_p50_ms": admission_p50,
        "admission_alloc_peak_mib": admission_alloc.peak_mib,
        "admission_rss_delta_mib": rss_delta,
        "tick_p50_ms": tick_p50,
        "tick_max_ms": tick_max,
        "tick_alloc_peak_mib": tick_alloc.peak_mib,
        "active_turn_tick_p50_ms": active_tick_p50,
        "active_turn_tick_max_ms": active_tick_max,
        "active_turn_tick_alloc_peak_mib": active_tick_alloc.peak_mib,
        "active_turn_pinned_units": float(len(pinned_ids)),
        "placeholders_tick_p50_ms": placeholders_tick_p50,
        "placeholders_tick_max_ms": placeholders_tick_max,
        "placeholders_tick_alloc_peak_mib": placeholders_tick_alloc.peak_mib,
        "placeholders_units": float(evicted_count),
        "live_placeholders_tick_p50_ms": live_tick_p50,
        "live_placeholders_tick_max_ms": live_tick_max,
        "live_placeholders_tick_alloc_peak_mib": live_tick_alloc.peak_mib,
        "mounted_spike_tick_p50_ms": mounted_spike_tick_p50,
        "mounted_spike_tick_max_ms": mounted_spike_tick_max,
        "mounted_spike_tick_alloc_peak_mib": mounted_spike_tick_alloc.peak_mib,
        "live_spike_tick_p50_ms": live_spike_tick_p50,
        "live_spike_tick_max_ms": live_spike_tick_max,
        "live_spike_tick_alloc_peak_mib": live_spike_tick_alloc.peak_mib,
        "live_spike_table_bytes": float(live_spike_table_bytes),
        "live_spike_settled_table_bytes": float(live_settled_table_bytes),
        "neighbors_spike_tick_p50_ms": neighbors_spike_tick_p50,
        "neighbors_spike_tick_max_ms": neighbors_spike_tick_max,
        "neighbors_spike_tick_alloc_peak_mib": neighbors_spike_tick_alloc.peak_mib,
        "neighbors_spike_table_bytes": float(neighbors_spike_table_bytes),
        "neighbors_spike_settled_table_bytes": float(neighbors_settled_table_bytes),
        "anchor_p50_ms": anchor_p50,
        "anchor_registry_visits": float(anchor_registry_visits),
        "registry_units": float(len(window.units)),
        "registry_visits_per_tick": float(visits_per_tick[0]),
    }

    # Frozen ceilings: non-regression upper bounds, always asserted.
    assert admission_p50 <= _REGISTRY_ADMIT_P50_CEILING_MS[entry_count], (
        f"registry admission p50 {admission_p50:.1f} ms exceeds the frozen "
        f"ceiling {_REGISTRY_ADMIT_P50_CEILING_MS[entry_count]} ms"
    )
    assert tick_p50 <= _REGISTRY_TICK_P50_CEILING_MS[entry_count], (
        f"presenter tick p50 {tick_p50:.1f} ms exceeds the frozen ceiling "
        f"{_REGISTRY_TICK_P50_CEILING_MS[entry_count]} ms"
    )
    assert tick_max <= _REGISTRY_TICK_MAX_CEILING_MS[entry_count], (
        f"presenter tick max {tick_max:.1f} ms exceeds the frozen ceiling "
        f"{_REGISTRY_TICK_MAX_CEILING_MS[entry_count]} ms"
    )
    assert tick_alloc.peak_mib <= _REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count], (
        f"tick allocation peak {tick_alloc.peak_mib} MiB exceeds the frozen "
        f"ceiling {_REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]} MiB"
    )
    assert live_tick_p50 <= _REGISTRY_TICK_P50_CEILING_MS[entry_count], (
        f"live+placeholders presenter tick p50 {live_tick_p50:.1f} ms exceeds "
        f"the frozen ceiling {_REGISTRY_TICK_P50_CEILING_MS[entry_count]} ms"
    )
    assert live_tick_max <= _REGISTRY_TICK_MAX_CEILING_MS[entry_count], (
        f"live+placeholders presenter tick max {live_tick_max:.1f} ms exceeds "
        f"the frozen ceiling {_REGISTRY_TICK_MAX_CEILING_MS[entry_count]} ms"
    )
    assert live_tick_alloc.peak_mib <= _REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count], (
        f"live+placeholders tick allocation peak {live_tick_alloc.peak_mib} MiB "
        f"exceeds the frozen ceiling "
        f"{_REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]} MiB"
    )
    assert mounted_spike_tick_p50 <= _REGISTRY_TICK_P50_CEILING_MS[entry_count], (
        f"mounted-spike presenter tick p50 {mounted_spike_tick_p50:.1f} ms exceeds "
        f"the frozen ceiling {_REGISTRY_TICK_P50_CEILING_MS[entry_count]} ms"
    )
    assert mounted_spike_tick_max <= _REGISTRY_TICK_MAX_CEILING_MS[entry_count], (
        f"mounted-spike presenter tick max {mounted_spike_tick_max:.1f} ms exceeds "
        f"the frozen ceiling {_REGISTRY_TICK_MAX_CEILING_MS[entry_count]} ms"
    )
    assert (
        mounted_spike_tick_alloc.peak_mib
        <= _REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]
    ), (
        f"mounted-spike tick allocation peak {mounted_spike_tick_alloc.peak_mib} "
        f"MiB exceeds the frozen ceiling "
        f"{_REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]} MiB"
    )
    assert live_spike_tick_p50 <= _REGISTRY_TICK_P50_CEILING_MS[entry_count], (
        f"live-spike presenter tick p50 {live_spike_tick_p50:.1f} ms exceeds "
        f"the frozen ceiling {_REGISTRY_TICK_P50_CEILING_MS[entry_count]} ms"
    )
    assert live_spike_tick_max <= _REGISTRY_TICK_MAX_CEILING_MS[entry_count], (
        f"live-spike presenter tick max {live_spike_tick_max:.1f} ms exceeds "
        f"the frozen ceiling {_REGISTRY_TICK_MAX_CEILING_MS[entry_count]} ms"
    )
    assert (
        live_spike_tick_alloc.peak_mib <= _REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]
    ), (
        f"live-spike tick allocation peak {live_spike_tick_alloc.peak_mib} MiB "
        f"exceeds the frozen ceiling "
        f"{_REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]} MiB"
    )
    assert neighbors_spike_tick_p50 <= _REGISTRY_TICK_P50_CEILING_MS[entry_count], (
        f"neighbors-spike presenter tick p50 {neighbors_spike_tick_p50:.1f} ms "
        f"exceeds the frozen ceiling {_REGISTRY_TICK_P50_CEILING_MS[entry_count]} ms"
    )
    assert neighbors_spike_tick_max <= _REGISTRY_TICK_MAX_CEILING_MS[entry_count], (
        f"neighbors-spike presenter tick max {neighbors_spike_tick_max:.1f} ms "
        f"exceeds the frozen ceiling {_REGISTRY_TICK_MAX_CEILING_MS[entry_count]} ms"
    )
    assert (
        neighbors_spike_tick_alloc.peak_mib
        <= _REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]
    ), (
        f"neighbors-spike tick allocation peak "
        f"{neighbors_spike_tick_alloc.peak_mib} MiB exceeds the frozen ceiling "
        f"{_REGISTRY_TICK_ALLOC_CEILING_MIB[entry_count]} MiB"
    )
    assert admission_alloc.peak_mib <= _REGISTRY_ADMIT_ALLOC_CEILING_MIB[entry_count], (
        f"admission allocation peak {admission_alloc.peak_mib} MiB exceeds the "
        f"frozen ceiling {_REGISTRY_ADMIT_ALLOC_CEILING_MIB[entry_count]} MiB"
    )
    assert rss_delta <= _REGISTRY_ADMIT_RSS_CEILING_MIB[entry_count], (
        f"admission RSS delta {rss_delta} MiB exceeds the frozen ceiling "
        f"{_REGISTRY_ADMIT_RSS_CEILING_MIB[entry_count]} MiB"
    )

    # FROZEN IMPROVEMENT BOUND (P2b): admission keeps per-entry fingerprints
    # instead of deep copies, so the traced admission peak must sit well below
    # the P0a snapshot-retention baseline recorded with the copies live.
    p0a_peak = _P2B_BASELINE_ADMIT_ALLOC_PEAK_MIB[entry_count]
    reduction_pct = (1 - admission_alloc.peak_mib / p0a_peak) * 100
    assert reduction_pct >= _P2B_ADMIT_ALLOC_REDUCTION_MIN_PCT, (
        f"admission allocation peak {admission_alloc.peak_mib} MiB is only "
        f"{reduction_pct:.1f}% below the P0a snapshot-retention baseline "
        f"{p0a_peak} MiB; P2b must hold "
        f"{_P2B_ADMIT_ALLOC_REDUCTION_MIN_PCT}%"
    )


# TURNED ON by P3b2b's gate cutover (the reconcile indexes are now the live
# path): these three are that package's acceptance values. They are asserted
# on ALL SEVEN recorded variants -- the idle gate, the active-turn pin path,
# the placeholders-heavy restore path, the combined live+placeholders
# variant, the mounted-spike variant, and the live-membership and
# checkpoint-neighbour spike variants -- because an active turn, a
# placeholder-dominated registry, a dirty spike that settles onto live
# survivors, a residency spike that settles onto the resident tail, and a
# live/neighbour spike that settles behind a peak-sized hash table are the
# cases the deep-review rounds found unbounded.
_IDLE_VARIANT = "idle"
_ACTIVE_TURN_VARIANT = "active_turn"
_PLACEHOLDERS_VARIANT = "placeholders"
_LIVE_PLACEHOLDERS_VARIANT = "live_placeholders"
_MOUNTED_SPIKE_VARIANT = "mounted_spike"
_LIVE_SPIKE_VARIANT = "live_spike"
_NEIGHBORS_SPIKE_VARIANT = "neighbors_spike"
_REGISTRY_TICK_VARIANTS = (
    _IDLE_VARIANT,
    _ACTIVE_TURN_VARIANT,
    _PLACEHOLDERS_VARIANT,
    _LIVE_PLACEHOLDERS_VARIANT,
    _MOUNTED_SPIKE_VARIANT,
    _LIVE_SPIKE_VARIANT,
    _NEIGHBORS_SPIKE_VARIANT,
)


def _variant_tick_metrics(entry_count: int, variant: str) -> dict[str, float]:
    """Recorded tick metrics for one variant at one registry size.

    The idle variant's keys are the unprefixed P0a record; the active-turn
    variant's carry its prefix.
    """
    baseline = _frozen_registry_baseline()[entry_count]
    prefix = "" if variant == _IDLE_VARIANT else f"{variant}_"
    return {
        "tick_p50_ms": baseline[f"{prefix}tick_p50_ms"],
        "tick_alloc_peak_mib": baseline[f"{prefix}tick_alloc_peak_mib"],
    }


def test_transcript_registry_flat_scaling_bound() -> None:
    for variant in _REGISTRY_TICK_VARIANTS:
        root = _variant_tick_metrics(_REGISTRY_SIZES[0], variant)["tick_p50_ms"]
        top = _variant_tick_metrics(_REGISTRY_TOP_SIZE, variant)["tick_p50_ms"]
        ratio = top / root
        assert ratio <= _P3B2B_FLAT_SCALING_MAX_RATIO, (
            f"{variant} presenter tick p50 grew {ratio:.2f}x from "
            f"{_REGISTRY_SIZES[0]} to {_REGISTRY_TOP_SIZE} units at fixed "
            f"residency; P3b2b must hold {_P3B2B_FLAT_SCALING_MAX_RATIO}x"
        )


def test_transcript_registry_unchanged_tick_bound() -> None:
    for variant in _REGISTRY_TICK_VARIANTS:
        tick_p50 = _variant_tick_metrics(_REGISTRY_TOP_SIZE, variant)["tick_p50_ms"]
        assert tick_p50 <= _P3B2B_UNCHANGED_TICK_MAX_MS, (
            f"{variant} presenter tick p50 at {_REGISTRY_TOP_SIZE} units is "
            f"{tick_p50:.2f} ms; P3b2b must hold the unchanged-tick bound "
            f"{_P3B2B_UNCHANGED_TICK_MAX_MS} ms"
        )


def test_transcript_registry_allocation_plateau_bound() -> None:
    for variant in _REGISTRY_TICK_VARIANTS:
        peak = _variant_tick_metrics(_REGISTRY_TOP_SIZE, variant)["tick_alloc_peak_mib"]
        assert peak <= _P3B2B_ALLOC_PLATEAU_MAX_MIB, (
            f"{variant} traced tick allocation peak at {_REGISTRY_TOP_SIZE} "
            f"units is {peak:.4f} MiB; P3b2b must hold the allocation plateau "
            f"{_P3B2B_ALLOC_PLATEAU_MAX_MIB} MiB"
        )


# TURNED ON by P3b1: the ordered placement index replaced the oldest-first
# anchor scan this bound measures.
def test_transcript_registry_anchor_scaling_bound() -> None:
    baseline = _frozen_registry_baseline()
    root = baseline[_REGISTRY_SIZES[0]]["anchor_p50_ms"]
    top = baseline[_REGISTRY_TOP_SIZE]["anchor_p50_ms"]
    ratio = top / root
    assert ratio <= _P3B1_ANCHOR_SCALING_MAX_RATIO, (
        f"anchor query p50 grew {ratio:.2f}x from {_REGISTRY_SIZES[0]} to "
        f"{_REGISTRY_TOP_SIZE} units; P3b1 must hold "
        f"{_P3B1_ANCHOR_SCALING_MAX_RATIO}x"
    )
    visits = baseline[_REGISTRY_TOP_SIZE]["anchor_registry_visits"]
    assert visits <= _P3B1_ANCHOR_REGISTRY_VISIT_CEILING, (
        f"anchor query walks {visits:.0f} registry units per sample at "
        f"{_REGISTRY_TOP_SIZE}; the placement index must answer from content "
        f"order alone"
    )
