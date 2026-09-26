from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
import os
from time import perf_counter_ns
from typing import TypedDict
from weakref import WeakKeyDictionary

import pytest
from textual.geometry import Size
from textual.pilot import Pilot
from textual.screen import Screen
from textual.widget import Widget
from textual.widgets import Button

from chartreux.app_server.models import (
    CompletedEffectState,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicHistoryEntry,
    PublicMessageEntry,
)
from chartreux.app_server.session import AppServerSession
from chartreux.cli.textual_ui.app import _TRANSCRIPT_HIGH_MARK, ChartreuxApp, ChatScroll
from chartreux.cli.textual_ui.widgets.collapsible import CollapsibleSection
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.load_more import HistoryLoadMoreMessage
from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.widgets.tools import ToolGroup, ToolResultMessage
from chartreux.cli.textual_ui.windowing.history import _build_history_widgets_raw
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.perf._metrics import machine_context, percentiles, record
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
_SCENARIO_TIMEOUT = 1800 if _FULL_SCOPE else 420


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


def _load_more_label(app: ChartreuxApp) -> str | None:
    load_more_messages = list(app.query(HistoryLoadMoreMessage))
    if not load_more_messages:
        return None
    return str(load_more_messages[0].query_one(Button).label)


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
                (
                    str(message.query_one(Button).label),
                    message.query_one(Button).disabled,
                )
                for message in load_more_messages
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
    while _load_more_label(app) is not None:
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
    restored_anchor = next(
        (
            widget
            for widget, index in app._history_widget_indices.items()
            if index == entry_index and widget.is_mounted
        ),
        None,
    )
    assert restored_anchor is not None
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
        session: AppServerSession, session_id: str
    ) -> None:
        assert session_id == session.session_id

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
