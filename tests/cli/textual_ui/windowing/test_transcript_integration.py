from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Offset, Size
from textual.selection import Selection
from textual.widget import Widget

from chartreux.app_server.events import TurnCompleted
from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicHistoryEntry,
    PublicTurn,
    PublicTurnStatus,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.load_more import HistoryLoadMoreRequested
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupKey
from chartreux.cli.textual_ui.widgets.tools import ToolGroup
from chartreux.cli.textual_ui.windowing.state import SessionWindowing
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.cli.textual_ui.test_event_handler_error_muting import _call_event, _ok_result
from tests.cli.textual_ui.test_history_grouping import _message
from tests.cli.textual_ui.windowing.test_session_windowing import _checkpoint
from tests.conftest import build_test_chartreux_app
from tests.stubs.app_server import CoreEventProjection


class _App(App[None]):
    CSS = "#stream { width: 60; height: 10; layout: stream; overflow-y: auto; }"

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")


class _Measured(Widget):
    _to_write_buffer: str = ""

    def __init__(self, height: int) -> None:
        super().__init__()
        self.height = height

    def render(self) -> str:
        return "copyable transcript text"

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.height


@pytest.mark.asyncio
async def test_first_post_mount_layout_clamp_does_not_supersede_reading_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = TranscriptWindow()
    window.admit([_message(i) for i in range(9)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(8) for _ in range(9)]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        stream.scroll_to(y=48, animate=False, immediate=True)
        await pilot.pause()
        anchor_y = roots[6].region.y
        await window.evict_unit(window.unit_ids[0])
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(8)])
        assert await window.restore_unit(
            window.unit_ids[0], stream, WeakKeyDictionary(), follow_bottom=False
        )
        await pilot.pause()
        assert roots[6].region.y == anchor_y


@pytest.mark.asyncio
async def test_restoring_unit_anchor_uses_roots_before_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = TranscriptWindow()
    window.admit([_message(i) for i in range(7)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(8) for _ in range(7)]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        key = window.unit_ids[2]
        placeholder = (await window.evict_unit(key))[0]
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(8)])
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def initialize(_roots: object) -> None:
            entered.set()
            await finish.wait()

        task = asyncio.create_task(
            window.restore_unit(
                key,
                stream,
                WeakKeyDictionary(),
                follow_bottom=False,
                initialize=initialize,
            )
        )
        await entered.wait()
        unit = window.units[key]
        assert unit.restoring and unit.mounted_roots
        assert window._placement(unit) is unit.mounted_roots[0]
        assert placeholder._reserved
        finish.set()
        assert await task


def test_prepend_counts_only_unadmitted_entries() -> None:
    history: list[PublicHistoryEntry] = [_message(index) for index in range(40)]
    window = TranscriptWindow()
    window.admit(history[20:], start_index=20)
    pages = SessionWindowing(load_more_batch_size=10)
    pages.set_backfill(history[:20])
    batch = pages.next_load_more_batch()
    assert batch is not None
    window.admit(batch.entries, start_index=batch.start_index)
    assert window.admitted_start_index == 10
    assert pages.recompute_backfill(history, admitted_start_index=10)
    assert pages.remaining == 10
    assert len(window.unit_ids) == 30
    assert len(set(window.unit_ids)) == 30


@pytest.mark.asyncio
async def test_selected_and_live_rows_are_never_evicted() -> None:
    window = TranscriptWindow()
    window.admit([_message(i) for i in range(9)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        stream.scroll_to(y=48, animate=False, immediate=True)
        await pilot.pause()
        window.units[window.unit_ids[0]].pin_reasons.add("active-turn")
        excluded = window.eviction_candidates(
            stream.region.y,
            stream.region.height,
            low_mark=10,
            high_mark=20,
            selected=frozenset({window.unit_ids[1]}),
            live=frozenset({window.unit_ids[2]}),
        )
        assert window.unit_ids[0] not in excluded
        assert window.unit_ids[1] not in excluded
        assert window.unit_ids[2] not in excluded
        assert excluded
        window.units[window.unit_ids[0]].pin_reasons.clear()
        assert window.unit_ids[0] in window.eviction_candidates(
            stream.region.y, stream.region.height, low_mark=10, high_mark=20
        )


@pytest.mark.asyncio
async def test_reset_during_in_flight_restore_clears_old_session_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = TranscriptWindow()
    window.admit([_message(i) for i in range(3)], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        key = window.unit_ids[0]
        await window.evict_unit(key)
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(12)])
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def initialize(_roots: object) -> None:
            entered.set()
            await finish.wait()

        restore = asyncio.create_task(
            window.restore_unit(
                key,
                stream,
                WeakKeyDictionary(),
                follow_bottom=False,
                initialize=initialize,
            )
        )
        await entered.wait()
        window.reset()
        await stream.remove_children()
        finish.set()
        assert not await restore
        assert not window.units
        assert not stream.children


@pytest.mark.asyncio
async def test_under_budget_scroll_reuses_resident_geometry_without_scans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CountingUnits(dict[str, object]):
        scans = 0

        def values(self):  # type: ignore[override]
            self.scans += 1
            return super().values()

    window = TranscriptWindow()
    window.admit([_message(i) for i in range(4)], start_index=0)
    units = CountingUnits(window.units)
    window.units = units  # type: ignore[assignment]
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(8) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])

        candidates = Mock(wraps=window.eviction_candidates)
        monkeypatch.setattr(window, "eviction_candidates", candidates)
        target = SimpleNamespace(
            _shutdown_started=False,
            _transcript=window,
            _messages_area=stream,
            _chat_widget=SimpleNamespace(is_at_bottom=False, region=stream.region),
            _transcript_under_budget_key=None,
            _selected_transcript_units=Mock(return_value=frozenset()),
            event_handler=None,
            _rewind_highlighted_widget=None,
            _queue_selected_widget=None,
            _agent_transcript_focus_target=None,
            _history_widget_indices=WeakKeyDictionary(),
            _active_turn_start=None,
        )

        def tick() -> None:
            ChartreuxApp._request_transcript_reconcile(
                cast("ChartreuxApp", target), scroll_driven=True
            )

        tick()
        assert candidates.call_count == 1
        assert window._reconcile_task is None
        scans = units.scans
        for _ in range(5):
            tick()
        assert candidates.call_count == 1
        assert units.scans == scans
        assert window._reconcile_task is None

        window.geometry_version += 1
        tick()
        assert candidates.call_count == 2
        assert units.scans > scans
        assert window._reconcile_task is None

        window._reconcile_task = asyncio.create_task(asyncio.sleep(10))
        window._reconcile_active = True
        try:
            tick()
            assert candidates.call_count == 3
            assert window._reconcile_pending
        finally:
            window._reconcile_task.cancel()
            await asyncio.gather(window._reconcile_task, return_exceptions=True)
            window._reconcile_task = None
            window._reconcile_active = False

        await window.evict_unit(window.unit_ids[0])
        await pilot.pause()
        tick()
        assert candidates.call_count == 3  # Placeholder short-circuits candidates.
        assert window._reconcile_task is not None
        await cast("asyncio.Task[None]", window._reconcile_task)
        assert not window.units[window.unit_ids[0]].placeholders
        assert window.units[window.unit_ids[0]].mounted_roots


def test_admissions_do_not_rescan_unchanged_units() -> None:
    class CountingUnits(dict[str, object]):
        scans = 0

        def values(self):  # type: ignore[override]
            self.scans += 1
            return super().values()

    window = TranscriptWindow()
    window.admit(
        [_message(i) for i in range(40, 60)], start_index=40, geometry_width=60
    )
    units = CountingUnits(window.units)
    window.units = units  # type: ignore[assignment]
    for start in (30, 20, 10, 0):
        window.admit(
            [_message(i) for i in range(start, start + 10)],
            start_index=start,
            geometry_width=60,
        )
    assert units.scans == 0
    assert window.unit_ids == [f"message-{i}" for i in range(60)]
    window.admit([_message(60)], start_index=60, geometry_width=61)
    assert units.scans == 1  # Actual width changes still invalidate old geometry.


@pytest.mark.asyncio
async def test_settled_assistant_can_leave_live_residency() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30)], start_index=0)
        stream = app._messages_area
        assistant = AssistantMessage("settled")
        roots: list[Widget] = [assistant, *[_Measured(12) for _ in range(29)]]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        handler = app.event_handler
        assert handler is not None
        handler._turn_assistant_message = assistant
        handler.settle_turn()
        assert assistant not in handler.retained_widgets()
        stream.scroll_to(y=250, animate=False, immediate=True)
        await pilot.pause()
        assert window.unit_ids[0] in window.eviction_candidates(
            stream.region.y + 250, 10, low_mark=1, high_mark=20
        )


@pytest.mark.asyncio
async def test_load_more_prepend_preserves_app_reading_row() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30, 60)], start_index=30)
        stream = app._messages_area
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        app._chat_widget.scroll_to(y=150, animate=False, immediate=True)
        await pilot.pause()
        app._windowing.set_backfill([_message(i) for i in range(30)])
        await app._load_more.show(stream, remaining=30)
        await pilot.pause()
        anchor = roots[15]
        row = anchor.region.y
        await app.on_history_load_more_requested(HistoryLoadMoreRequested())
        await pilot.pause()
        assert anchor.region.y == row


@pytest.mark.asyncio
async def test_final_load_more_batch_hiding_button_preserves_reading_row() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30, 60)], start_index=30)
        stream = app._messages_area
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        app._chat_widget.scroll_to(y=150, animate=False, immediate=True)
        await pilot.pause()
        # A single-batch backfill: serving it exhausts Load More, so the handler
        # removes the button above the reading anchor while keeping the row.
        app._windowing.set_backfill([_message(i) for i in range(20, 30)])
        await app._load_more.show(stream, remaining=10)
        await pilot.pause()
        anchor = roots[15]
        row = anchor.region.y
        await app.on_history_load_more_requested(HistoryLoadMoreRequested())
        await pilot.pause()
        assert app._load_more.widget is None
        assert anchor.region.y == row


@pytest.mark.asyncio
async def test_failed_older_page_fetch_hiding_button_preserves_reading_row() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30, 60)], start_index=30)
        stream = app._messages_area
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        app._chat_widget.scroll_to(y=150, animate=False, immediate=True)
        await pilot.pause()
        await app._load_more.show(stream, remaining=10)
        await pilot.pause()
        # No backfill and no older-page cursor: the older-page fetch fails, so
        # the handler hides the button the user scrolled up to click while
        # keeping the reading row.
        anchor = roots[15]
        row = anchor.region.y
        await app.on_history_load_more_requested(HistoryLoadMoreRequested())
        await pilot.pause()
        assert app._load_more.widget is None
        assert anchor.region.y == row


@pytest.mark.asyncio
async def test_refresh_windowing_hiding_button_preserves_reading_row() -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30, 60)], start_index=30)
        stream = app._messages_area
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        app._chat_widget.scroll_to(y=150, animate=False, immediate=True)
        await pilot.pause()
        await app._load_more.show(stream, remaining=10)
        await pilot.pause()
        # The live transcript's backfill and older-page cursor are both gone,
        # so the refresh hides the button while keeping the reading row.
        anchor = roots[15]
        row = anchor.region.y
        await app._refresh_windowing_from_history()
        await pilot.pause()
        assert app._load_more.widget is None
        assert anchor.region.y == row


@pytest.mark.asyncio
async def test_production_watcher_does_not_treat_restore_clamp_as_user_scroll(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30)], start_index=0)
        stream = app._messages_area
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        app._chat_widget.scroll_to(y=250, animate=False, immediate=True)
        await pilot.pause()
        key = window.unit_ids[0]
        await window.evict_unit(key)
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(12)])
        anchor = roots[22]
        screen_row = anchor.region.y
        revision = window._scroll_revision
        assert await window.restore_unit(
            key, stream, app._history_widget_indices, follow_bottom=False
        )
        await pilot.pause()
        assert anchor.region.y == screen_row
        assert window._scroll_revision == revision


@pytest.mark.asyncio
async def test_animated_user_scroll_supersedes_active_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(30)], start_index=0)
        stream = app._messages_area
        roots = [_Measured(12) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        chat = app._chat_widget
        chat.scroll_to(y=250, animate=False, immediate=True)
        await pilot.pause()
        key = window.unit_ids[0]
        await window.evict_unit(key)
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(12)])
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def initialize(_roots: object) -> None:
            entered.set()
            await finish.wait()

        restore = asyncio.create_task(
            window.restore_unit(
                key,
                stream,
                app._history_widget_indices,
                follow_bottom=False,
                initialize=initialize,
            )
        )
        await entered.wait()
        assert chat._transcript_layout_change
        revision = window._scroll_revision
        user_target = max(0, chat.scroll_y - 20)
        assert user_target != chat.scroll_y
        chat.scroll_to(
            y=user_target, animate=True, duration=0.1, easing="linear", immediate=True
        )
        assert app.animator.is_being_animated(chat, "scroll_y")
        for _ in range(10):
            if window._scroll_revision > revision:
                break
            await pilot.pause(0.005)
        assert window._scroll_revision > revision
        assert app.animator.is_being_animated(chat, "scroll_y")
        await pilot.pause(0.15)
        user_scroll_position = chat.scroll_offset.y
        finish.set()
        assert await restore
        assert chat.scroll_target_y == pytest.approx(user_scroll_position)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True])
async def test_aborted_submit_clears_active_turn_pins_before_reconcile(
    monkeypatch: pytest.MonkeyPatch, raises: bool
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(0)], start_index=0)
        unit = window.units[window.unit_ids[0]]
        app._active_turn_start = 0

        async def abort(_message: str, *, optimistic_start: bool) -> bool:
            assert optimistic_start
            app._request_transcript_reconcile()
            assert "active-turn" in unit.pin_reasons
            if raises:
                raise RuntimeError("prompt preparation aborted")
            return False

        monkeypatch.setattr(app, "_enqueue_prompt_with_resources", abort)
        if raises:
            with pytest.raises(RuntimeError, match="prompt preparation aborted"):
                await app._handle_user_message("abort")
        else:
            await app._handle_user_message("abort")
        await pilot.pause()
        app._request_transcript_reconcile()
        assert app._active_turn_start is None
        assert all(
            "active-turn" not in item.pin_reasons for item in window.units.values()
        )


@pytest.mark.asyncio
async def test_admission_failure_settles_turn_and_listener_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    release_events = asyncio.Event()
    listener_processed_events = asyncio.Event()
    settled = 0

    async def events():
        await release_events.wait()
        for index in range(2):
            yield TurnCompleted(
                PublicTurn(
                    id=f"turn-{index}",
                    session_id="test-session",
                    status=PublicTurnStatus.COMPLETED,
                    started_at=1,
                )
            )
        listener_processed_events.set()

    async with app.run_test() as pilot:
        handler = app.event_handler
        assert handler is not None
        original_settle = handler.settle_turn

        def fail_admission() -> None:
            raise ValueError("invalid admitted batch")

        def settle() -> None:
            nonlocal settled
            settled += 1
            original_settle()

        monkeypatch.setattr(app, "_admit_live_history", fail_admission)
        monkeypatch.setattr(handler, "settle_turn", settle)
        monkeypatch.setattr(app.app_server, "events", events)
        app._active_turn_start = 0
        app._transcript.admit([_message(0)], start_index=0)
        unit = app._transcript.units[app._transcript.unit_ids[0]]
        unit.pin_reasons.add("active-turn")
        listener = asyncio.create_task(app._listen_app_server_events())
        release_events.set()
        await asyncio.wait_for(listener_processed_events.wait(), timeout=2)
        await listener
        await pilot.pause()
        assert settled == 2
        assert app._active_turn_start is None
        assert "active-turn" not in unit.pin_reasons


@pytest.mark.asyncio
async def test_terminal_tool_group_admission_defensively_extends_finalized_group() -> (
    None
):
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        handler = app.event_handler
        assert handler is not None
        projection = CoreEventProjection()
        await projection.dispatch(_call_event("tool-a"), handler.handle_event)
        assert handler.current_tool_group is not None
        assert not app._transcript.unit_ids
        await projection.dispatch(_ok_result("tool-a"), handler.handle_event)
        await pilot.pause()
        assert handler.current_tool_group is None
        history = app.app_server._state.projection.state.history
        assert history is not None
        history.extend(projection.history)
        app._admit_live_history()
        assert app._transcript.unit_ids
        assert app._transcript.units[app._transcript.unit_ids[0]].mounted_roots
        await projection.dispatch(_call_event("tool-b"), handler.handle_event)
        await projection.dispatch(_ok_result("tool-b"), handler.handle_event)
        history.extend(projection.history[1:])
        # Explicitly exercise the admission helper's defensive pre-settlement path.
        app._admit_live_history()
        assert len(app._transcript.unit_ids) == 1
        assert app._transcript.units[app._transcript.unit_ids[0]].member_entry_ids == [
            "tool-a",
            "tool-b",
        ]


@pytest.mark.asyncio
async def test_session_replacement_stops_plan_watcher_and_drops_detached_widgets(
    tmp_path: Path,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test():
        handler = app.event_handler
        assert handler is not None
        await handler._handle_start_plan_review(tmp_path / "plan.md")
        plan = handler.plan_file_message
        assert plan is not None and plan.parent is app._messages_area
        stop = Mock(wraps=plan.stop_watching)
        plan.stop_watching = stop
        handler.current_tool_group = ToolGroup(key=ToolGroupKey("current"))
        handler._finalized_tool_group = ToolGroup(key=ToolGroupKey("finalized"))
        await app._rebuild_transcript_from_current_session()
        assert handler.plan_file_message is None
        assert plan.parent is None
        stop.assert_called_once()
        assert handler.current_tool_group is None
        assert handler._finalized_tool_group is None


@pytest.mark.asyncio
async def test_app_reconcile_protects_selection_stream_active_turn_and_compaction() -> (
    None
):
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        entries: list[PublicHistoryEntry] = [_message(i) for i in range(35)]
        entries[2] = cast(
            "PublicCheckpointEntry",
            _checkpoint(2).model_copy(update={"kind": "compaction"}),
        )
        window.admit(entries, start_index=0)
        stream = app._messages_area
        roots = [_Measured(60) for _ in window.unit_ids]
        roots[1]._to_write_buffer = "pending markdown"
        await stream.mount_all(roots)
        app.screen.selections[roots[0]] = Selection.from_offsets(
            Offset(0, 0), Offset(1, 0)
        )
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        app._active_turn_start = 33
        app._chat_widget.scroll_to(y=1100, animate=False, immediate=True)
        await pilot.pause()
        app._request_transcript_reconcile()
        task = window._reconcile_task
        assert task is not None
        await task
        for index in (0, 1, 2, 33, 34):
            assert window.units[window.unit_ids[index]].mounted_roots
        assert app.screen.get_selected_text()
        assert roots[1]._to_write_buffer == "pending markdown"
        assert window.units[window.unit_ids[33]].pin_reasons == {"active-turn"}
        assert any(unit.placeholders for unit in window.units.values())


@pytest.mark.asyncio
async def test_app_resume_scroll_remount_preserves_group_content_and_expansion() -> (
    None
):
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        projection = CoreEventProjection()
        projection.project(_call_event("restore-tool"))
        projection.project(_ok_result("restore-tool"))
        history: list[PublicHistoryEntry] = [
            *projection.history,
            *[_message(i) for i in range(18)],
        ]
        app._tool_group_expansion_state.set_collapsed(
            ToolGroupKey("restore-tool"), False
        )
        session_history = app.app_server._state.projection.state.history
        assert session_history is not None
        session_history.extend(history)
        await app._resume_history_from_messages()
        await pilot.pause()
        window = app._transcript
        key = window.unit_ids[0]
        original = window.units[key].mounted_roots[0]
        assert isinstance(original, ToolGroup)
        assert not original.is_collapsed
        app._chat_widget.scroll_to(
            y=app._chat_widget.max_scroll_y, animate=False, immediate=True
        )
        await pilot.pause()
        await window.evict_unit(key)
        app._chat_widget.scroll_to(y=0, animate=False, immediate=True)
        await pilot.pause()
        if window._reconcile_task is not None:
            await window._reconcile_task
        assert not window.units[key].placeholders
        restored = window.units[key].mounted_roots[0]
        assert isinstance(restored, ToolGroup)
        assert restored is not original
        assert not restored.is_collapsed
        assert len(restored.content_container.children) == 2


@pytest.mark.asyncio
async def test_app_session_replacement_cancels_pending_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test() as pilot:
        window = app._transcript
        window.admit([_message(i) for i in range(5)], start_index=0)
        stream = app._messages_area
        roots = [_Measured(20) for _ in window.unit_ids]
        await stream.mount_all(roots)
        await pilot.pause()
        for key, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(key, [root])
        key = window.unit_ids[0]
        await window.evict_unit(key)
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(20)])
        entered = asyncio.Event()
        finish = asyncio.Event()

        async def initialize(_roots: object) -> None:
            entered.set()
            await finish.wait()

        pending = window.request_reconcile(
            stream,
            app._history_widget_indices,
            follow_bottom=False,
            initialize=initialize,
        )
        await entered.wait()
        await app._rebuild_transcript_from_current_session()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert not window.units
        assert not stream.children
