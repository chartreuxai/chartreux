from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any
from weakref import WeakKeyDictionary

from textual.widget import Widget

from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicHistoryEntry,
    PublicMessageEntry,
)
from chartreux.cli.textual_ui.widgets.collapsible import CollapsibleSection
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.load_more import HistoryLoadMoreRequested
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ReasoningMessage,
    StreamingMessageBase,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    entry_keeps_tool_group,
)
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing import (
    HISTORY_RESUME_TAIL_MESSAGES,
    LOAD_MORE_BATCH_SIZE,
    HistoryLoadMoreManager,
    SessionWindowing,
    create_resume_plan,
    shift_history_widget_indices,
    should_resume_history,
    sync_backfill_state,
)
from chartreux.cli.textual_ui.windowing.history import build_history_widgets
from chartreux.cli.textual_ui.windowing.transcript import (
    TranscriptAnchor,
    TranscriptWindow,
)

if TYPE_CHECKING:
    from chartreux.cli.textual_ui.app import ChartreuxApp, ChatScroll

# Reconcile high-water mark behind the under-budget scroll cache. app.py
# re-exports this name, so the module-global seam (read by
# tests/perf/test_transcript_windowing.py from app) is preserved.
_TRANSCRIPT_HIGH_MARK = 1500


class _SessionTranscriptWindow(TranscriptWindow):
    """Apply current interface preferences even to rematerialized history units."""

    # Late-bound host reference, assigned by the owner after construction and
    # never captured through the constructor.
    _app: ChartreuxApp

    def __init__(self, get_show_timestamps: Callable[[], bool], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._get_show_timestamps = get_show_timestamps

    def build_unit(
        self, unit_id: str, history_widget_indices: WeakKeyDictionary[Widget, int]
    ) -> list[Widget]:
        unit = self.units[unit_id]
        return build_history_widgets(
            unit.entries,
            start_index=unit.start_index,
            history_widget_indices=history_widget_indices,
            tools_collapsed=self.tools_collapsed,
            expansion_state=self.expansion_state,
            entry_expansion_state=self.entry_expansion_state,
            show_message_timestamps=self._get_show_timestamps(),
        )


class TranscriptPresenter:
    """Owns the transcript presentation state for one app.

    Plain object (no mixins): the app binds itself late via ``_bind_app``
    (never constructor-captured), and collaborators are invoked as
    ``self._app._method(...)`` at call time so instance patches keep
    intercepting. The app exposes a forwarding property (plus a setter
    exactly where app code assigns) for every field, so direct-attribute
    consumers see no change.
    """

    # Late-bound host reference, assigned via _bind_app after construction.
    _app: ChartreuxApp

    def __init__(self, *, get_show_timestamps: Callable[[], bool]) -> None:
        self._tools_collapsed = True
        self._tool_group_expansion_state = ToolGroupExpansionState()
        self._preview_tool_group_expansion_state = ToolGroupExpansionState()
        self._entry_expansion_state = EntryExpansionState()
        self._preview_entry_expansion_state = EntryExpansionState()
        self._windowing = SessionWindowing(load_more_batch_size=LOAD_MORE_BATCH_SIZE)
        self._transcript = _SessionTranscriptWindow(
            get_show_timestamps,
            expansion_state=self._tool_group_expansion_state,
            entry_expansion_state=self._entry_expansion_state,
        )
        self._load_more = HistoryLoadMoreManager()
        self._history_widget_indices: WeakKeyDictionary[Widget, int] = (
            WeakKeyDictionary()
        )
        self._transcript_under_budget_key: tuple[int, int, int] | None = None
        self._transcript_following = True
        self._resize_anchor_task: asyncio.Task[None] | None = None
        self._last_transcript_anchor: TranscriptAnchor | None = None

    def _bind_app(self, app: ChartreuxApp) -> None:
        """Bind the host app after construction and propagate to the window."""
        self._app = app
        self._transcript._app = app

    def mounted_timing_widgets(self, entry_id: str) -> list[Widget]:
        self._app._admit_live_history()
        return self._transcript.mounted_entry_widgets(entry_id)

    def admit_live_history(self) -> None:
        open_group = (
            self._app.event_handler.current_tool_group
            if self._app.event_handler
            else None
        )
        history = self._app.app_server.history
        start = self._transcript.admitted_end_index
        while start < len(history) and self._app.event_handler is not None:
            entry = history[start]
            if not isinstance(entry, PublicMessageEntry) or entry.role != "assistant":
                break
            root = self._app.event_handler.assistant_entry_widgets.get(entry.id)
            if root is None or not self._transcript.admit_reused_root(
                entry, root, index=start
            ):
                break
            start += 1
            self._app._request_transcript_reconcile()
        if start >= len(history):
            return
        batch = history[start:]
        if open_group is not None:
            # Admit completed roots preceding the still-growing group, but leave
            # the group's membership open until its finalization.
            end = len(batch)
            while end and entry_keeps_tool_group(batch[end - 1]):
                end -= 1
            batch = batch[:end]
            if not batch:
                return
        if self._transcript.unit_ids and all(
            entry_keeps_tool_group(entry) for entry in batch
        ):
            # Defensive for callers that admit while a finalized group is still
            # held; normal turn finalization has no partial group to extend here.
            last = self._transcript.units[self._transcript.unit_ids[-1]]
            if (
                last.group_key is not None
                and last.mounted_roots
                and self._app.event_handler is not None
                and self._app.event_handler._finalized_tool_group
                is last.mounted_roots[0]
            ):
                self._transcript.admit(
                    batch,
                    start_index=start,
                    geometry_width=self._app._messages_area.size.width,
                    extend_group=True,
                )
                self._app._request_transcript_reconcile()
                return
        preview = TranscriptWindow()
        preview.admit(batch, start_index=start)
        mounted = {
            root
            for unit in self._transcript.units.values()
            for root in unit.mounted_roots
        }
        candidates = [
            child
            for child in self._app._messages_area.children
            if child not in mounted
            and child is not open_group
            and type(child)
            in {
                UserMessage,
                AssistantMessage,
                ReasoningMessage,
                ToolCallMessage,
                ToolResultMessage,
                ToolGroup,
                CompactMessage,
            }
        ]
        counts = [
            2
            if unit.group_key is None
            and len(unit.entries) == 1
            and isinstance(unit.entries[0], PublicEffectEntry)
            else 1
            for unit in (preview.units[key] for key in preview.unit_ids)
        ]
        if sum(counts) != len(candidates):
            return
        self._transcript.admit(
            batch, start_index=start, geometry_width=self._app._messages_area.size.width
        )
        offset = 0
        for key, count in zip(preview.unit_ids, counts, strict=True):
            roots = candidates[offset : offset + count]
            self._transcript.register_mounted(key, roots)
            offset += count
        self._app._request_transcript_reconcile()

    def on_chat_scroll_activity(self, chat: ChatScroll, new_value: float) -> None:
        """Host reaction to ChatScroll scroll activity (injected into the widget)."""
        engine = self._transcript
        engine_scroll = getattr(chat, "_transcript_engine_scroll", False)
        layout_clamp = (
            getattr(chat, "_transcript_layout_change", False)
            and new_value != chat.scroll_target_y
            and not self._app.animator.is_being_animated(chat, "scroll_y")
        )
        if not (engine_scroll or layout_clamp):
            engine._scroll_revision += 1
            if (
                not self._app._picker.previewing
                and not self._app._picker_origin_restoring
            ):
                self._transcript_following = chat.scroll_y >= chat.max_scroll_y
                chat.call_after_refresh(self._app._remember_transcript_anchor)
        self._app._request_transcript_reconcile(scroll_driven=True)

    def request_transcript_reconcile(self, *, scroll_driven: bool = False) -> None:
        if self._app._shutdown_started or not self._transcript.unit_ids:
            return
        chat = self._app._chat_widget
        transcript = self._transcript
        messages = self._app._messages_area
        idle = transcript._reconcile_task is None or transcript._reconcile_task.done()
        budget_key = (
            transcript.geometry_version,
            messages.virtual_size.height,
            messages.size.width,
        )
        if scroll_driven and idle and budget_key == self._transcript_under_budget_key:
            # Eviction/restoration bumps geometry_version; no resident scan is
            # needed on scroll ticks while the measured placement is unchanged.
            return
        if not scroll_driven:
            self._transcript_under_budget_key = None
        has_placeholders = any(unit.placeholders for unit in transcript.units.values())
        needs_protection = has_placeholders or bool(
            transcript.eviction_candidates(chat.region.y, chat.region.height)
        )
        if scroll_driven and idle and not needs_protection:
            # Only cache the genuine under-budget case: a temporarily protected
            # over-budget unit must be reconsidered on the next scroll.
            budgeted_rows = sum(
                unit.mounted_roots[-1].region.bottom - unit.mounted_roots[0].region.y
                for unit in transcript.units.values()
                if unit.mounted_roots
                and not unit.pin_reasons
                and unit.mounted_roots[-1].region.bottom
                - unit.mounted_roots[0].region.y
                <= _TRANSCRIPT_HIGH_MARK
            )
            if budgeted_rows <= _TRANSCRIPT_HIGH_MARK:
                self._transcript_under_budget_key = budget_key
            return
        selected = (
            self._selected_transcript_units() if needs_protection else frozenset()
        )
        retained = (
            self._app.event_handler.retained_widgets()
            if self._app.event_handler
            else set()
        )
        for widget in (
            self._app._rewind_highlighted_widget,
            self._app._queue_selected_widget,
            self._app._agent_transcript_focus_target,
        ):
            if widget is not None:
                retained.add(widget)
        live: set[str] = set()
        neighbors: set[str] = set()
        for key in self._transcript.unit_ids:
            unit = self._transcript.units[key]
            if (
                self._app._active_turn_start is not None
                and unit.start_index >= self._app._active_turn_start
            ):
                unit.pin_reasons.add("active-turn")
            if needs_protection and any(
                node in retained
                or getattr(node, "_is_spinning", False)
                or getattr(node, "_stream_write_timer", None) is not None
                or getattr(node, "_stream_message_buffer", None)
                or getattr(node, "_write_timer", None) is not None
                or getattr(node, "_to_write_buffer", None)
                for root in unit.mounted_roots
                for node in (root, *root.walk_children())
            ):
                live.add(key)
        for index, key in enumerate(self._transcript.unit_ids):
            unit = self._transcript.units[key]
            if not any(
                isinstance(entry, PublicCheckpointEntry) for entry in unit.entries
            ):
                continue
            adjacent = self._transcript.unit_ids[max(index - 1, 0) : index + 2]
            if any(
                self._transcript.units[item].mounted_roots
                for item in adjacent
                if item != key
            ):
                neighbors.add(key)
        self._transcript.request_reconcile(
            self._app._messages_area,
            self._history_widget_indices,
            follow_bottom=chat.is_at_bottom,
            selected=selected,
            live=frozenset(live),
            compaction_neighbors=frozenset(neighbors),
        )

    def request_scroll_driven_reconcile(self) -> None:
        """Kick a scroll-driven reconcile tick (injected into ChatScroll)."""
        self._app._request_transcript_reconcile(scroll_driven=True)

    def _selected_transcript_units(self) -> frozenset[str]:
        selected: set[str] = set()
        for key, unit in self._transcript.units.items():
            for root in unit.mounted_roots:
                if any(
                    getattr(node, "text_selection", None)
                    for node in (root, *root.walk_children())
                ):
                    selected.add(key)
        return frozenset(selected)

    def remember_transcript_anchor(self) -> None:
        if not self._app._picker.previewing and (
            self._resize_anchor_task is None or self._resize_anchor_task.done()
        ):
            self._last_transcript_anchor = self._transcript.capture_anchor(
                self._app._messages_area, following=self._transcript_following
            )

    def reset_ui_state(self) -> None:
        self._last_transcript_anchor = None
        self._windowing.reset()
        self._transcript.reset()
        self._app._active_turn_start = None
        self._history_widget_indices = WeakKeyDictionary()
        if self._app.event_handler is not None:
            self._app.event_handler._finalize_tool_group()
            self._app.event_handler._handle_stop_plan_review()
            self._app.event_handler.cancel_retry_presentation()

    async def rebuild_transcript_from_current_session(self) -> None:
        # batch_update coalesces the teardown and re-mount into one refresh, so the
        # message area doesn't flash empty mid-rebuild.
        async with self._app._custom_tools_deprecation_message_lock:
            self._app._reset_ui_state()
            with self._app.batch_update():
                await self._load_more.hide()
                await self._app._messages_area.remove_children()
                await self._app._resume_history_from_messages()
            self._app._custom_tools_deprecation_message = None
        await self._app._show_custom_tools_deprecation_warning()

    async def restore_picker_history_range(self) -> None:
        """Re-admit the origin's inspected pages before restoring its viewport."""
        if self._app._picker_origin_range is None or not self._transcript.unit_ids:
            return
        start, _ = self._app._picker_origin_range
        current = self._transcript.admitted_start_index
        if start >= current:
            return
        history = self._app.app_server.history
        # The timeline may have changed during preview; admit only available rows.
        prefix = history[max(0, start) : min(current, len(history))]
        if prefix:
            await self._app._mount_history_batch(
                prefix, self._app._messages_area, start_index=max(0, start), before=0
            )
            self._windowing.recompute_backfill(
                history, admitted_start_index=self._transcript.admitted_start_index
            )
            await self._load_more.set_visible(
                self._app._messages_area,
                visible=self.has_older_history,
                remaining=self.history_backfill_remaining,
            )

    async def resume_history_from_messages(self) -> None:
        messages_area = self._app._messages_area
        if self._transcript.admitted_end_index or not should_resume_history(
            list(messages_area.children)
        ):
            return

        if (
            plan := create_resume_plan(
                self._app.app_server.history, HISTORY_RESUME_TAIL_MESSAGES
            )
        ) is None:
            return
        await self._app._mount_history_batch(
            plan.tail_entries, messages_area, start_index=plan.tail_start_index
        )
        self._app.call_after_refresh(self._app._chat_widget.anchor)
        self._windowing.set_backfill(plan.backfill_entries)
        await self._load_more.set_visible(
            messages_area,
            visible=self.has_older_history,
            remaining=self.history_backfill_remaining,
        )

    async def mount_history_batch(
        self,
        batch: list[PublicHistoryEntry],
        messages_area: Widget,
        *,
        start_index: int,
        before: Widget | int | None = None,
        after: Widget | None = None,
    ) -> None:
        reading_anchor = (
            self._transcript._reading_anchor(messages_area)
            if (before is not None or after is not None)
            and not self._app._chat_widget.is_at_bottom
            else None
        )
        self._transcript.tools_collapsed = self._tools_collapsed
        self._transcript.expansion_state = (
            self._preview_tool_group_expansion_state
            if self._app._picker.previewing
            else self._tool_group_expansion_state
        )
        self._transcript.entry_expansion_state = (
            self._preview_entry_expansion_state
            if self._app._picker.previewing
            else self._entry_expansion_state
        )
        previous = set(self._transcript.unit_ids)
        self._transcript.admit(
            batch, start_index=start_index, geometry_width=messages_area.size.width
        )
        new_ids = [key for key in self._transcript.unit_ids if key not in previous]
        widgets_by_unit = [
            (key, self._transcript.build_unit(key, self._history_widget_indices))
            for key in new_ids
        ]
        widgets = [widget for _, roots in widgets_by_unit for widget in roots]

        with self._app.batch_update():
            if not widgets:
                return
            if before is not None:
                await messages_area.mount_all(widgets, before=before)
            elif after is not None:
                await messages_area.mount_all(widgets, after=after)
            else:
                await messages_area.mount_all(widgets)

        for widget in widgets:
            if isinstance(widget, StreamingMessageBase):
                await widget.write_initial_content()
        for key, roots in widgets_by_unit:
            self._transcript.register_mounted(key, roots)
        if reading_anchor is not None:
            await self._app._restore_reading_anchor(reading_anchor, messages_area)
        self._app._request_transcript_reconcile()

    async def restore_reading_anchor(
        self, reading_anchor: tuple[str, int, int], messages_area: Widget
    ) -> None:
        """Scroll the viewport so a recorded reading anchor stays on its row."""
        anchor_id, row_offset, screen_row = reading_anchor
        unit = self._transcript.units.get(anchor_id)
        placement = self._transcript._placement(unit) if unit else None
        if placement is None:
            return
        for _ in range(3):
            await self._transcript._layout_pass(messages_area)
            delta = placement.region.y + row_offset - screen_row
            if delta:
                self._transcript._set_scroll_flag(
                    self._app._chat_widget, "_transcript_engine_scroll", True
                )
                try:
                    self._app._chat_widget.scroll_to(
                        y=self._app._chat_widget.scroll_offset.y + delta,
                        animate=False,
                        immediate=True,
                    )
                finally:
                    self._transcript._set_scroll_flag(
                        self._app._chat_widget, "_transcript_engine_scroll", False
                    )

    async def on_load_more_requested(self, _: HistoryLoadMoreRequested) -> None:
        self._load_more.set_enabled(False)
        try:
            if not self._windowing.has_backfill:
                if not await self.load_older_history_page():
                    messages_area = self._app._messages_area
                    # A failed/empty older-page fetch hides the button the user
                    # scrolled up to click; keep the reading anchor's row.
                    hiding_load_more = self._load_more.widget is not None
                    reading_anchor = None
                    if hiding_load_more and not self._app._chat_widget.is_at_bottom:
                        await self._transcript._layout_pass(messages_area)
                        reading_anchor = self._transcript._reading_anchor(messages_area)
                    await self._load_more.hide()
                    if reading_anchor is not None:
                        await self._app._restore_reading_anchor(
                            reading_anchor, messages_area
                        )
                    return
            if (batch := self._windowing.next_load_more_batch()) is None:
                await self._load_more.hide()
                return
            messages_area = self._app._messages_area
            if self._load_more.widget:
                before: Widget | int | None = None
                after: Widget | None = self._load_more.widget
            else:
                before = 0
                after = None
            await self._app._mount_history_batch(
                batch.entries,
                messages_area,
                start_index=batch.start_index,
                before=before,
                after=after,
            )
            # Removing an exhausted Load More button shrinks the content above the
            # reading anchor; record the anchor first and restore its row after
            # the removal so the view does not jump.
            hiding_load_more = (
                not self.has_older_history and self._load_more.widget is not None
            )
            reading_anchor = None
            if hiding_load_more and not self._app._chat_widget.is_at_bottom:
                # The prepend compensation's final scroll must be laid out before
                # placement regions describe the settled reading anchor.
                await self._transcript._layout_pass(messages_area)
                reading_anchor = self._transcript._reading_anchor(messages_area)
            await self._load_more.set_visible(
                messages_area,
                visible=self.has_older_history,
                remaining=self.history_backfill_remaining,
            )
            if reading_anchor is not None:
                await self._app._restore_reading_anchor(reading_anchor, messages_area)
        finally:
            self._load_more.set_enabled(True)

    @property
    def has_older_history(self) -> bool:
        if self._windowing.has_backfill:
            return True
        # During picker preview the live session's server-side cursor is irrelevant:
        # the preview history was fetched in full via session/history/get, so only
        # the local windowing backfill determines whether Load More is available.
        if self._app._picker.previewing:
            return False
        return self._app.app_server.resources.sessions.history_before_cursor is not None

    @property
    def history_backfill_remaining(self) -> int | None:
        if self._app._picker.previewing:
            return self._windowing.remaining or None
        if self._app.app_server.resources.sessions.history_before_cursor is not None:
            return None
        return self._windowing.remaining or None

    async def load_older_history_page(self) -> bool:
        before = self._app.app_server.resources.sessions.history_before_cursor
        if before is None:
            return False
        page = await self._app.app_server.resources.sessions.load_before(
            before, LOAD_MORE_BATCH_SIZE
        )
        if not page.data:
            return False
        shift_history_widget_indices(self._history_widget_indices, len(page.data))
        self._windowing.set_backfill(page.data)
        return True

    async def refresh_windowing_from_history(self) -> None:
        if self._load_more.widget is None:
            return
        messages_area = self._app._messages_area
        has_backfill = sync_backfill_state(
            history=self._app.app_server.history,
            transcript=self._transcript,
            windowing=self._windowing,
        )
        visible = (
            has_backfill
            or self._app.app_server.resources.sessions.history_before_cursor is not None
        )
        # Hiding a live transcript's Load More button shrinks the content above
        # the reading anchor; keep its row. Fresh mounts and rebuilds never get
        # here because they have no button to hide.
        reading_anchor = None
        if not visible and not self._app._chat_widget.is_at_bottom:
            await self._transcript._layout_pass(messages_area)
            reading_anchor = self._transcript._reading_anchor(messages_area)
        await self._load_more.set_visible(
            messages_area, visible=visible, remaining=self.history_backfill_remaining
        )
        if reading_anchor is not None:
            await self._app._restore_reading_anchor(reading_anchor, messages_area)

    def handle_resize(self) -> None:
        if (
            self._transcript.unit_ids
            and not self._app._picker.previewing
            and (self._resize_anchor_task is None or self._resize_anchor_task.done())
        ):
            anchor = self._last_transcript_anchor or self._transcript.capture_anchor(
                self._app._messages_area, following=self._transcript_following
            )
            if anchor is not None:
                anchor = replace(
                    anchor, scroll_revision=self._transcript._scroll_revision
                )

                async def restore_after_reflow() -> None:
                    # App resize precedes the screen's resize message. Wait for
                    # the chat viewport to adopt the new width before restoring
                    # a wrapped-message anchor against its reflowed geometry.
                    for _ in range(100):
                        if (
                            not self._app.is_running
                            or not self._app._chat_widget.display
                            or self._app._chat_widget.region.width
                            == self._app.size.width
                        ):
                            break
                        await asyncio.sleep(0.01)
                    # Restore against the available geometry if reflow never settles.
                    self._app._messages_area.refresh(layout=True)
                    await self._transcript._layout_pass(self._app._messages_area)
                    await self._transcript.restore_anchor(
                        self._app._messages_area, anchor
                    )
                    self._app._request_transcript_reconcile()

                self._resize_anchor_task = asyncio.create_task(restore_after_reflow())
                self._resize_anchor_task.add_done_callback(
                    lambda _: self._app.call_after_refresh(
                        self._app._remember_transcript_anchor
                    )
                )
        self._app._request_transcript_reconcile()

    async def toggle_tool(self) -> None:
        groups = list(self._app.query(ToolGroup))
        sections = list(self._app.query(CollapsibleSection))
        reasoning = list(self._app.query(ReasoningMessage))
        group_state = (
            self._preview_tool_group_expansion_state
            if self._app._picker.previewing
            else self._tool_group_expansion_state
        )
        entry_state = (
            self._preview_entry_expansion_state
            if self._app._picker.previewing
            else self._entry_expansion_state
        )
        expanded = [not group_state.is_collapsed(key) for key in group_state.keys]
        expanded.extend(
            not group.is_collapsed for group in groups if group._key is None
        )
        self._tools_collapsed = all(expanded) if expanded else not self._tools_collapsed
        group_state.set_all_collapsed(self._tools_collapsed)
        entry_state.set_all_collapsed(self._tools_collapsed)
        for group in groups:
            group.set_collapsed(self._tools_collapsed)
        for section in sections:
            section.set_collapsed(self._tools_collapsed)
        for node in reasoning:
            await node.set_collapsed(self._tools_collapsed)
        self._app._request_transcript_reconcile()
