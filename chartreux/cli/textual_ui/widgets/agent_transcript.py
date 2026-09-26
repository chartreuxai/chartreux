from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import time
from typing import ClassVar
from weakref import WeakKeyDictionary

from textual import __version__ as _TEXTUAL_VERSION, events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static
from textual.worker import Worker, WorkerCancelled, WorkerFailed

from chartreux.app_server.models import (
    ContentBlock,
    EffectResultDisplay,
    FailedEffectState,
    ImageAttachment,
    ImageContentBlock,
    InlineImageSource,
    PublicEffectEntry,
    PublicError,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicReasoningEntry,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptSource,
    AgentTranscriptState,
    AgentTranscriptToolStatus,
)
from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    effect_state_is_terminal,
)
from chartreux.cli.textual_ui.widgets.tools import ToolGroup
from chartreux.cli.textual_ui.windowing.history import build_history_widgets
from chartreux.observability.logging import get_effective_log_level, logger

DEFAULT_TRANSCRIPT_PAGE_SIZE = 50
MAX_MOUNTED_TRANSCRIPT_ENTRIES = 100
_SAVED_SNAPSHOT_NOTICE = "Saved transcript — may lag the running agent."
_TRANSCRIPT_SESSION_ID = "agent-transcript"
_TEXTUAL_REPARENT_VERSION = "8.2.8"
_MOUNT_YIELD_BUDGET_SECONDS = 0.03
_DISPOSAL_BATCH_SIZE = 8
_DISPOSAL_YIELD_BUDGET_SECONDS = 0.03
_HEARTBEAT_DRIFT_WARNING_SECONDS = 0.25


def _detach_for_reparenting(parent: Widget, child: Widget) -> None:
    if _TEXTUAL_VERSION != _TEXTUAL_REPARENT_VERSION:
        raise RuntimeError(
            "Textual reparenting internals must be reviewed for this version"
        )
    # Textual 8.2.8 has no public child-reparent API: Widget.remove() prunes the
    # node from the app. Recheck for a public move API and these internals whenever
    # the exact Textual pin in pyproject.toml changes.
    parent._nodes._remove(child)
    child._detach()


@dataclass(slots=True)
class _MountedUnit:
    """A top-level history widget, or a grouped run of projected entries."""

    widgets: list[Widget]
    entry_ids: list[str]
    entries: dict[str, PublicHistoryEntry]
    entry_widgets: dict[str, list[Widget]]
    group: ToolGroup | None = None


class AgentTranscriptViewer(Vertical):
    """A read-only, paginated view of an agent transcript."""

    can_focus = True

    DEFAULT_CSS = """
    AgentTranscriptViewer {
        width: 100%;
        height: 1fr;
        border: solid $foreground-muted;
        background: $surface;
    }

    #agent-transcript-header {
        height: auto;
        padding: 0 1;
        background: $boost;
    }

    #agent-transcript-notice {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }

    #agent-transcript-content-scroll {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    #agent-transcript-content {
        width: 100%;
        height: auto;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("pageup", "older_page", "Older"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "close", "Close", priority=True),
    ]

    class Closed(Message):
        """Posted when the viewer should be removed by its owner."""

        def __init__(self, viewer: AgentTranscriptViewer) -> None:
            self.viewer = viewer
            super().__init__()

    def __init__(
        self,
        source: AgentTranscriptSource,
        agent_id: str,
        *,
        profile: str = "unknown",
        live: bool = False,
        page_size: int = DEFAULT_TRANSCRIPT_PAGE_SIZE,
    ) -> None:
        super().__init__(id="agent-transcript-viewer")
        self._source = source
        self._agent_id = agent_id
        self._profile = profile
        self._live = live
        self._live_timer: Timer | None = None
        self._on_newest_page = True
        self._page_size = page_size
        self._content: Vertical | None = None
        self._status_widget: Static | None = None
        self._content_scroll: VerticalScroll | None = None
        self._cursor: str | None = None
        self._has_more = False
        self._known_entries: dict[str, AgentTranscriptEntry] = {}
        self._units: list[_MountedUnit] = []
        self._entry_units: dict[str, _MountedUnit] = {}
        self._request_epoch = 0
        self._reading_page = False
        self._operation_active = False
        self._operation_worker: Worker[None] | None = None
        self._pending_request: tuple[str | None, bool] | None = None
        self._viewer_closed = False
        self._disposal_complete = False
        self._heartbeat_timer: Timer | None = None
        self._last_heartbeat = 0.0
        self._max_heartbeat_drift = 0.0
        self._expansion_state = ToolGroupExpansionState()

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def compose(self) -> ComposeResult:
        yield Static(
            f"Subagent: {self._agent_id} · {self._profile}  [PageUp: older · r: refresh · Esc: close]",
            id="agent-transcript-header",
        )
        yield Static(self._notice_text(), id="agent-transcript-notice")
        with VerticalScroll(id="agent-transcript-content-scroll") as content_scroll:
            self._content_scroll = content_scroll
            with Vertical(id="agent-transcript-content") as content:
                self._content = content
                self._status_widget = Static(self._loading_text())
                yield self._status_widget

    def on_mount(self) -> None:
        self.focus()
        self._last_heartbeat = time.monotonic()
        # Diagnostic timers are created only at mount; /log-level changes take
        # effect on the next mount (the handler still filters existing records).
        if get_effective_log_level() == "DEBUG":
            self._heartbeat_timer = self.set_interval(0.5, self._record_heartbeat)
        self._start_refresh()
        if self._live:
            self._live_timer = self.set_interval(1.0, self._append_live_output)

    def _record_heartbeat(self) -> None:
        now = time.monotonic()
        drift = now - self._last_heartbeat - 0.5
        self._last_heartbeat = now
        self._max_heartbeat_drift = max(self._max_heartbeat_drift, drift)
        if drift > _HEARTBEAT_DRIFT_WARNING_SECONDS:
            logger.debug("Agent transcript heartbeat drift: %.3fs", drift)

    def on_unmount(self) -> None:
        self._stop_requests()
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.stop()
            self._heartbeat_timer = None
            logger.debug(
                "Agent transcript max heartbeat drift: %.3fs", self._max_heartbeat_drift
            )
        if self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None

    def set_live(self, live: bool) -> None:
        """Update live state without replacing the converged transcript view."""
        if self._viewer_closed:
            return
        was_live = self._live
        self._live = live
        if self.is_mounted and live and self._live_timer is None:
            self._live_timer = self.set_interval(1.0, self._append_live_output)
        elif not live and self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None
            if was_live:
                self._submit_request(before=None, replace=False)
        self._update_notice()

    def _loading_text(self) -> str:
        return "Loading live output…" if self._live else "Loading saved transcript…"

    def _notice_text(self) -> str:
        return (
            "Live output — refreshes about every second."
            if self._live
            else _SAVED_SNAPSHOT_NOTICE
        )

    def _update_notice(self, extra: str = "") -> None:
        notice = self.query_one("#agent-transcript-notice", Static)
        notice.update(f"{self._notice_text()}{extra}")

    def _append_live_output(self) -> None:
        if self._viewer_closed:
            return
        if self._live and self._on_newest_page:
            self._submit_request(before=None, replace=False)
        elif self._live and not self._on_newest_page:
            self._update_notice(" New output available.")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_close()

    def action_older_page(self) -> None:
        if self._reading_page or not self._has_more or self._cursor is None:
            return
        self._on_newest_page = False
        self._submit_request(before=self._cursor, replace=False)

    def action_refresh(self) -> None:
        self._start_refresh()

    def _stop_requests(self) -> None:
        if not self._viewer_closed:
            self._viewer_closed = True
            self._request_epoch += 1
        self._pending_request = None
        if self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None

    def action_close(self) -> None:
        if self._viewer_closed:
            return
        self._stop_requests()
        self.post_message(self.Closed(self))

    async def dispose(self) -> None:
        """Quiesce the viewer, then drain merged groups before removing the shell."""
        self._stop_requests()
        worker = self._operation_worker
        if worker is not None:
            worker.cancel()
            try:
                await worker.wait()
            except (WorkerCancelled, WorkerFailed):
                # The worker is finished even when it failed or was cancelled.
                pass
            self._operation_worker = None
        if self._disposal_complete or not self.is_mounted:
            return

        batch_start = time.perf_counter()
        batch_count = 0
        for unit in self._units:
            if unit.group is None:
                continue
            for child in self._unit_entry_widgets(unit):
                if child.is_attached:
                    await child.remove()
                    batch_count += 1
                    if (
                        batch_count >= _DISPOSAL_BATCH_SIZE
                        or time.perf_counter() - batch_start
                        >= _DISPOSAL_YIELD_BUDGET_SECONDS
                    ):
                        await asyncio.sleep(0)
                        batch_start = time.perf_counter()
                        batch_count = 0

        # Empty merged groups now have cheap shells. Also remove ungrouped units
        # incrementally so they cannot accumulate in a single removal call.
        content = self._content
        if content is not None:
            for widget in tuple(content.children):
                if widget.is_attached:
                    await widget.remove()
                    batch_count += 1
                    if (
                        batch_count >= _DISPOSAL_BATCH_SIZE
                        or time.perf_counter() - batch_start
                        >= _DISPOSAL_YIELD_BUDGET_SECONDS
                    ):
                        await asyncio.sleep(0)
                        batch_start = time.perf_counter()
                        batch_count = 0
        self._units.clear()
        self._entry_units.clear()
        self._known_entries.clear()
        self._disposal_complete = True

    def _start_refresh(self) -> None:
        if self._viewer_closed:
            return
        self._on_newest_page = True
        self._cursor = None
        self._has_more = False
        self._submit_request(before=None, replace=True)

    def _submit_request(self, *, before: str | None, replace: bool) -> None:
        if self._viewer_closed:
            return
        if self._operation_active:
            self._pending_request = (before, replace)
            return
        self._operation_active = True
        self._reading_page = True
        self._operation_worker = self.run_worker(self._operate(before, replace))

    async def _operate(self, before: str | None, replace: bool) -> None:
        try:
            while True:
                await self._execute_request(before, replace)
                pending = self._pending_request
                self._pending_request = None
                if pending is None or self._viewer_closed:
                    return
                before, replace = pending
        finally:
            self._operation_active = False
            self._reading_page = False
            pending = self._pending_request
            if pending is not None and not self._viewer_closed:
                self._pending_request = None
                self._submit_request(before=pending[0], replace=pending[1])

    async def _execute_request(self, before: str | None, replace: bool) -> None:
        self._request_epoch += 1
        epoch = self._request_epoch
        debug = get_effective_log_level() == "DEBUG"
        fetch_start = time.perf_counter() if debug else 0.0
        try:
            response = await self._source.read_agent_transcript(
                self._agent_id, before=before, limit=self._page_size
            )
        except Exception:
            if self._is_current(epoch):
                await self._reset_rendered_content()
                if not self._is_current(epoch):
                    return
                self._known_entries.clear()
                await self._show_status("Unable to load saved transcript.")
            return
        if debug:
            logger.debug(
                "Agent transcript fetch: %.3fs, %d entries",
                time.perf_counter() - fetch_start,
                len(response.entries or []),
            )
        if not self._is_current(epoch):
            return
        apply_start = time.perf_counter() if debug else 0.0
        await self._apply_response(
            response, epoch=epoch, replace=replace, before=before
        )
        if debug:
            logger.debug(
                "Agent transcript apply: %.3fs, %d entries",
                time.perf_counter() - apply_start,
                len(response.entries or []),
            )

    def _is_current(self, epoch: int) -> bool:
        return (
            not self._viewer_closed and self.is_mounted and epoch == self._request_epoch
        )

    async def _apply_response(  # noqa: PLR0911, PLR0912
        self,
        response: AgentTranscriptGetResponse,
        *,
        epoch: int,
        replace: bool,
        before: str | None,
    ) -> None:
        if response.state is AgentTranscriptState.NO_SAVED_TRANSCRIPT:
            await self._reset_rendered_content()
            if not self._is_current(epoch):
                return
            self._known_entries.clear()
            await self._show_status("No saved transcript available.")
            return
        if response.state is AgentTranscriptState.CHANGED:
            await self._reset_rendered_content()
            if not self._is_current(epoch):
                return
            self._known_entries.clear()
            await self._show_status("Transcript changed; refresh")
            return
        if response.state is AgentTranscriptState.EXCEEDS_VIEWER_LIMIT:
            await self._reset_rendered_content()
            if not self._is_current(epoch):
                return
            self._known_entries.clear()
            await self._show_status("Transcript exceeds viewer limit")
            return
        if response.state is not AgentTranscriptState.AVAILABLE:
            await self._reset_rendered_content()
            if not self._is_current(epoch):
                return
            self._known_entries.clear()
            await self._show_status("Unable to load saved transcript.")
            return

        staged: dict[str, AgentTranscriptEntry] = (
            {} if replace else dict(self._known_entries)
        )
        if replace:
            await self._reset_rendered_content()
            if not self._is_current(epoch):
                return

        entries = response.entries or []
        changed: list[AgentTranscriptEntry] = []
        new_entries: list[AgentTranscriptEntry] = []
        timeline_entries: list[AgentTranscriptEntry] = []
        for entry in entries:
            previous = staged.get(entry.entry_id)
            if previous is None:
                staged[entry.entry_id] = entry
                if entry.kind is AgentTranscriptEntryKind.ASSISTANT_TEXT:
                    timeline_entries.append(entry)
                else:
                    new_entries.append(entry)
            elif previous.digest != entry.digest:
                staged[entry.entry_id] = entry
                if (
                    previous.kind is AgentTranscriptEntryKind.ASSISTANT_TEXT
                    and not previous.display_text
                    and entry.display_text
                ):
                    timeline_entries.append(entry)
                else:
                    changed.append(entry)

        for entry in changed:
            await self._replace_changed_entry(entry, staged)
            if not self._is_current(epoch):
                return

        await self._mount_response_batch(
            [*new_entries, *timeline_entries],
            staged=staged,
            epoch=epoch,
            before=before,
            has_entries=bool(entries),
            replace=replace,
            timeline_insert=bool(timeline_entries),
        )
        if not self._is_current(epoch):
            return
        self._known_entries = staged
        self._cursor = response.oldest_cursor
        self._has_more = response.has_more or False

    async def _mount_response_batch(
        self,
        entries: Sequence[AgentTranscriptEntry],
        *,
        staged: dict[str, AgentTranscriptEntry],
        epoch: int,
        before: str | None,
        has_entries: bool,
        replace: bool,
        timeline_insert: bool,
    ) -> None:
        batch_entries = sorted(entries, key=lambda entry: entry.created_at)
        if batch_entries:
            await self._mount_entries(
                batch_entries,
                staged=staged,
                epoch=epoch,
                prepend=before is not None,
                timeline_ordered=timeline_insert and before is None,
            )
            if not self._is_current(epoch):
                return
        elif not self._units and (has_entries or replace):
            await self._show_status("Saved transcript is empty.")
            if not self._is_current(epoch):
                return
        await self._enforce_mount_cap(
            staged=staged,
            preserve_entry_ids={entry.entry_id for entry in batch_entries},
            evict_newest=before is not None,
        )
        if not self._is_current(epoch):
            return
        self._scroll_to_newest_if_needed()

    async def _reset_rendered_content(self) -> None:
        content = self._content
        if content is not None:
            debug = get_effective_log_level() == "DEBUG"
            count = len(content.children) if debug else 0
            start = time.perf_counter() if debug else 0.0
            try:
                await content.remove_children()
            finally:
                if debug:
                    logger.debug(
                        "Agent transcript reset remove_children: %.3fs, %d children agent=%s",
                        time.perf_counter() - start,
                        count,
                        self._agent_id,
                    )
        self._status_widget = None
        self._units.clear()
        self._entry_units.clear()

    async def _show_status(self, message: str) -> None:
        content = self._content
        if content is None:
            return
        debug = get_effective_log_level() == "DEBUG"
        count = len(content.children) if debug else 0
        start = time.perf_counter() if debug else 0.0
        try:
            await content.remove_children()
        finally:
            if debug:
                logger.debug(
                    "Agent transcript status remove_children: %.3fs, %d children agent=%s",
                    time.perf_counter() - start,
                    count,
                    self._agent_id,
                )
        status = Static(message)
        self._status_widget = status
        await content.mount(status)
        self._scroll_to_newest_if_needed()

    async def _clear_status(self) -> None:
        status = self._status_widget
        if status is None:
            return
        self._status_widget = None
        if status.parent is self._content:
            await status.remove()

    def _history_entry(self, entry: AgentTranscriptEntry) -> PublicHistoryEntry:
        common = {
            "id": entry.entry_id,
            "session_id": _TRANSCRIPT_SESSION_ID,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "generation_status": entry.generation_status,
        }
        match entry.kind:
            case AgentTranscriptEntryKind.USER_TEXT:
                content: list[ContentBlock] = [
                    TextContentBlock(text=_entry_display_text(entry))
                ]
                names = list(entry.attachment_names)
                if entry.attachment_count > len(names):
                    remaining = entry.attachment_count - len(names)
                    names.append(
                        f"{remaining} more image{'s' if remaining != 1 else ''}"
                    )
                content.extend(
                    ImageContentBlock(
                        attachment=ImageAttachment(
                            source=InlineImageSource(data=""),
                            alias=name,
                            mime_type="image/placeholder",
                        )
                    )
                    for name in names
                )
                return PublicMessageEntry(role="user", content=content, **common)
            case AgentTranscriptEntryKind.ASSISTANT_TEXT:
                content: list[ContentBlock] = [
                    TextContentBlock(text=_entry_display_text(entry))
                ]
                return PublicMessageEntry(role="assistant", content=content, **common)
            case AgentTranscriptEntryKind.REASONING:
                return PublicReasoningEntry(text=_entry_display_text(entry), **common)
            case (
                AgentTranscriptEntryKind.TOOL_CALL
                | AgentTranscriptEntryKind.TOOL_RESULT
            ):
                if entry.detail is None or entry.state is None:
                    raise ValueError(
                        "Projected tool entry is missing its effect payload"
                    )
                detail = entry.detail
                state = entry.state
                if entry.truncated:
                    detail_display = detail.display.model_copy(
                        update={
                            "message": _entry_display_text(
                                entry, detail.display.message or detail.display.summary
                            )
                        }
                    )
                    detail = detail.model_copy(update={"display": detail_display})
                    state_display = getattr(state, "display", None)
                    if state_display is not None:
                        state = state.model_copy(
                            update={
                                "display": state_display.model_copy(
                                    update={
                                        "message": _entry_display_text(
                                            entry, state_display.message
                                        )
                                    }
                                )
                            }
                        )
                if (
                    not self._live
                    and entry.kind is AgentTranscriptEntryKind.TOOL_CALL
                    and entry.status is AgentTranscriptToolStatus.PENDING
                ):
                    interrupted_message = _entry_display_text(
                        entry, "Tool call was interrupted before a result was saved."
                    )
                    detail = detail.model_copy(
                        update={
                            "display": detail.display.model_copy(
                                update={
                                    "settled_verb": "Interrupted",
                                    "settled_message": interrupted_message,
                                }
                            )
                        }
                    )
                    state = FailedEffectState(
                        error=PublicError(message=interrupted_message),
                        output_text="",
                        duration_ms=0,
                        display=EffectResultDisplay(
                            success=False,
                            verb="Interrupted",
                            message=interrupted_message,
                        ),
                    )
                return PublicEffectEntry(
                    title=entry.title or entry.tool_name or "Tool",
                    detail=detail,
                    state=state,
                    related_entry_id=entry.tool_call_id,
                    **common,
                )

    def _build_history_batch(
        self, entries: Sequence[AgentTranscriptEntry]
    ) -> tuple[
        list[AgentTranscriptEntry],
        list[Widget],
        dict[int, tuple[AgentTranscriptEntry, PublicHistoryEntry]],
        WeakKeyDictionary[Widget, int],
    ]:
        paired_call_ids = {
            entry.tool_call_id
            for entry in entries
            if entry.kind is AgentTranscriptEntryKind.TOOL_CALL
            and entry.tool_call_id is not None
        }
        rendered_entries = [
            entry
            for entry in entries
            if not (
                entry.kind is AgentTranscriptEntryKind.TOOL_RESULT
                and entry.tool_call_id in paired_call_ids
            )
        ]
        if not rendered_entries:
            return [], [], {}, WeakKeyDictionary()
        history = [self._history_entry(entry) for entry in rendered_entries]
        start_index = rendered_entries[0].created_at
        history_by_index = {
            start_index + offset: (entry, projected)
            for offset, (entry, projected) in enumerate(
                zip(rendered_entries, history, strict=True)
            )
        }
        widget_indices: WeakKeyDictionary[Widget, int] = WeakKeyDictionary()
        tools_collapsed = bool(getattr(self.app, "_tools_collapsed", True))
        self._expansion_state.default_collapsed = tools_collapsed
        widgets = build_history_widgets(
            history,
            start_index=start_index,
            history_widget_indices=widget_indices,
            tools_collapsed=tools_collapsed,
            expansion_state=self._expansion_state,
        )
        return rendered_entries, widgets, history_by_index, widget_indices

    def _build_units(
        self, entries: Sequence[AgentTranscriptEntry]
    ) -> list[_MountedUnit]:
        rendered_entries, widgets, history_by_index, widget_indices = (
            self._build_history_batch(entries)
        )
        units: list[_MountedUnit] = []
        for widget in widgets:
            if isinstance(widget, ToolGroup):
                entry_widgets: dict[str, list[Widget]] = {}
                projected_entries: dict[str, PublicHistoryEntry] = {}
                for child in widget.content_container.children:
                    index = widget_indices.get(child)
                    indexed_entry = (
                        history_by_index.get(index) if index is not None else None
                    )
                    if indexed_entry is None:
                        continue
                    projected, history_entry = indexed_entry
                    entry_widgets.setdefault(projected.entry_id, []).append(child)
                    projected_entries[projected.entry_id] = history_entry
                entry_ids = [
                    entry.entry_id
                    for entry in rendered_entries
                    if entry.entry_id in entry_widgets
                ]
                if entry_ids:
                    units.append(
                        _MountedUnit(
                            [widget],
                            entry_ids,
                            projected_entries,
                            entry_widgets,
                            widget,
                        )
                    )
                continue

            index = widget_indices.get(widget)
            indexed_entry = history_by_index.get(index) if index is not None else None
            if indexed_entry is None:
                continue
            projected, history_entry = indexed_entry
            if (
                units
                and units[-1].group is None
                and units[-1].entry_ids == [projected.entry_id]
            ):
                unit = units[-1]
                unit.widgets.append(widget)
                unit.entry_widgets[projected.entry_id].append(widget)
            else:
                units.append(
                    _MountedUnit(
                        [widget],
                        [projected.entry_id],
                        {projected.entry_id: history_entry},
                        {projected.entry_id: [widget]},
                    )
                )
        return units

    async def _mount_entries(  # noqa: PLR0911, PLR0912, PLR0915
        self,
        entries: Sequence[AgentTranscriptEntry],
        *,
        staged: dict[str, AgentTranscriptEntry],
        epoch: int,
        prepend: bool,
        timeline_ordered: bool = False,
    ) -> None:
        await self._clear_status()
        if not self._is_current(epoch):
            return
        incoming = self._build_units([
            entry
            for entry in entries
            if not self._has_mounted_counterpart(entry, staged)
        ])
        if not incoming:
            return

        content = self._content
        if content is None:
            return
        budget_start = time.perf_counter()
        if timeline_ordered:
            for unit in incoming:
                unit_start = self._unit_timeline_start(unit)
                position = next(
                    (
                        index
                        for index, mounted in enumerate(self._units)
                        if self._unit_timeline_start(mounted) > unit_start
                    ),
                    len(self._units),
                )
                before = (
                    self._units[position].widgets[0]
                    if position < len(self._units)
                    else None
                )
                await content.mount(*unit.widgets, before=before)
                self._units.insert(position, unit)
                await self._initialize_streaming_widgets(self._unit_entry_widgets(unit))
                if time.perf_counter() - budget_start >= _MOUNT_YIELD_BUDGET_SECONDS:
                    await asyncio.sleep(0)
                    if not self._is_current(epoch):
                        return
                    budget_start = time.perf_counter()
        else:
            inserted_units = incoming
            if prepend and self._units:
                older = incoming[-1]
                newer = self._units[0]
                if (
                    older.group is not None
                    and newer.group is not None
                    and self._units_are_adjacent(older, newer, staged)
                ):
                    await self._merge_groups(newer, older, prepend=True)
                    if not self._is_current(epoch):
                        return
                    await self._initialize_streaming_widgets(
                        self._unit_entry_widgets(older)
                    )
                    if not self._is_current(epoch):
                        return
                    inserted_units = incoming[:-1]
            elif self._units:
                older = self._units[-1]
                newer = incoming[0]
                if (
                    older.group is not None
                    and newer.group is not None
                    and self._units_are_adjacent(older, newer, staged)
                ):
                    await self._merge_groups(older, newer, prepend=False)
                    if not self._is_current(epoch):
                        return
                    await self._initialize_streaming_widgets(
                        self._unit_entry_widgets(newer)
                    )
                    if not self._is_current(epoch):
                        return
                    inserted_units = incoming[1:]

            anchor = self._units[0].widgets[0] if prepend and self._units else None
            mounted = 0
            for unit in inserted_units:
                await content.mount(*unit.widgets, before=anchor)
                if prepend:
                    self._units.insert(mounted, unit)
                    mounted += 1
                else:
                    self._units.append(unit)
                await self._initialize_streaming_widgets(self._unit_entry_widgets(unit))
                if time.perf_counter() - budget_start >= _MOUNT_YIELD_BUDGET_SECONDS:
                    await asyncio.sleep(0)
                    if not self._is_current(epoch):
                        return
                    budget_start = time.perf_counter()
        self._reindex_mounted_entries()

    @staticmethod
    def _unit_timeline_start(unit: _MountedUnit) -> int:
        return min(entry.created_at for entry in unit.entries.values())

    def _has_mounted_counterpart(
        self, entry: AgentTranscriptEntry, staged: dict[str, AgentTranscriptEntry]
    ) -> bool:
        if entry.tool_call_id is None or entry.kind not in {
            AgentTranscriptEntryKind.TOOL_CALL,
            AgentTranscriptEntryKind.TOOL_RESULT,
        }:
            return False
        counterpart_kind = (
            AgentTranscriptEntryKind.TOOL_RESULT
            if entry.kind is AgentTranscriptEntryKind.TOOL_CALL
            else AgentTranscriptEntryKind.TOOL_CALL
        )
        return any(
            (mounted := staged.get(entry_id)) is not None
            and mounted.kind is counterpart_kind
            and mounted.tool_call_id == entry.tool_call_id
            for unit in self._units
            for entry_id in unit.entry_ids
        )

    def _units_are_adjacent(
        self,
        older: _MountedUnit,
        newer: _MountedUnit,
        staged: dict[str, AgentTranscriptEntry],
    ) -> bool:
        if set(older.entry_ids).intersection(newer.entry_ids):
            return False
        older_entries = [
            staged[entry_id] for entry_id in older.entry_ids if entry_id in staged
        ]
        newer_entries = [
            staged[entry_id] for entry_id in newer.entry_ids if entry_id in staged
        ]
        if not older_entries or not newer_entries:
            return False
        older_call_ids = {
            entry.tool_call_id
            for entry in older_entries
            if entry.tool_call_id is not None
        }
        newer_call_ids = {
            entry.tool_call_id
            for entry in newer_entries
            if entry.tool_call_id is not None
        }
        older_entries.extend(
            entry for entry in staged.values() if entry.tool_call_id in older_call_ids
        )
        newer_entries.extend(
            entry for entry in staged.values() if entry.tool_call_id in newer_call_ids
        )
        return max(entry.created_at for entry in older_entries) + 1 == min(
            entry.created_at for entry in newer_entries
        )

    async def _merge_groups(
        self, target: _MountedUnit, incoming: _MountedUnit, *, prepend: bool
    ) -> None:
        target_group = target.group
        incoming_group = incoming.group
        if target_group is None or incoming_group is None:
            return
        incoming_children = [
            child
            for entry_id in incoming.entry_ids
            for child in incoming.entry_widgets[entry_id]
        ]
        incoming_container = incoming_group.content_container
        for child in incoming_children:
            _detach_for_reparenting(incoming_container, child)

        if incoming_children:
            if prepend:
                first_existing = next(
                    child
                    for entry_id in target.entry_ids
                    for child in target.entry_widgets[entry_id]
                )
                await target_group.content_container.mount(
                    *incoming_children, before=first_existing
                )
            else:
                await target_group.content_container.mount(*incoming_children)

        target.entries = (
            {**incoming.entries, **target.entries}
            if prepend
            else {**target.entries, **incoming.entries}
        )
        target.entry_ids = (
            [*incoming.entry_ids, *target.entry_ids]
            if prepend
            else [*target.entry_ids, *incoming.entry_ids]
        )
        target.entry_widgets = (
            {**incoming.entry_widgets, **target.entry_widgets}
            if prepend
            else {**target.entry_widgets, **incoming.entry_widgets}
        )
        for entry_id in incoming.entry_ids:
            history_entry = incoming.entries[entry_id]
            if isinstance(history_entry, PublicEffectEntry):
                target_group.add_call_kind(history_entry.detail.kind)
                target_group.record_effect(
                    history_entry.created_at, history_entry.state
                )
            elif isinstance(history_entry, PublicReasoningEntry):
                target_group.mark_reasoning()
        target_group.resume()
        if not any(
            isinstance(entry, PublicEffectEntry)
            and not effect_state_is_terminal(entry.state)
            for entry in incoming.entries.values()
        ):
            target_group.finalize()

    async def _replace_changed_entry(
        self, entry: AgentTranscriptEntry, staged: dict[str, AgentTranscriptEntry]
    ) -> None:
        unit = self._entry_units.get(entry.entry_id)
        if unit is None:
            return
        projected = self._history_entry(entry)
        if unit.group is not None:
            await self._replace_group_entry(unit, entry.entry_id, projected, staged)
            return

        replacement_entries = [
            staged[item_id] for item_id in unit.entry_ids if item_id in staged
        ]
        replacement_units = self._build_units(replacement_entries)
        await self._replace_unit(unit, replacement_units)

    async def _replace_group_entry(
        self,
        unit: _MountedUnit,
        entry_id: str,
        history_entry: PublicHistoryEntry,
        staged: dict[str, AgentTranscriptEntry],
    ) -> None:
        group = unit.group
        content = self._content
        if group is None or content is None:
            return
        replacement = self._build_units([staged[entry_id]])
        if not replacement or replacement[0].group is None:
            return
        fresh = replacement[0]
        fresh_group = fresh.group
        if fresh_group is None:
            return
        new_children = fresh.entry_widgets.get(entry_id, [])
        old_children = unit.entry_widgets.get(entry_id, [])
        if not new_children or not old_children:
            return
        fresh_container = fresh_group.content_container
        for child in new_children:
            _detach_for_reparenting(fresh_container, child)
        await group.content_container.mount(*new_children, before=old_children[0])
        await self._initialize_streaming_widgets(new_children)
        for child in old_children:
            await child.remove()
        unit.entry_widgets[entry_id] = new_children
        unit.entries[entry_id] = history_entry
        if isinstance(history_entry, PublicEffectEntry):
            group.add_call_kind(history_entry.detail.kind)
            group.record_effect(history_entry.created_at, history_entry.state)
        elif isinstance(history_entry, PublicReasoningEntry):
            group.mark_reasoning()
        group.resume()
        if not any(
            isinstance(item, PublicEffectEntry)
            and not effect_state_is_terminal(item.state)
            for item in unit.entries.values()
        ):
            group.finalize()

    async def _replace_unit(
        self, unit: _MountedUnit, replacement: list[_MountedUnit]
    ) -> None:
        content = self._content
        if content is None:
            return
        try:
            position = self._units.index(unit)
        except ValueError:
            return
        old_widgets = unit.widgets
        new_widgets = [
            widget for new_unit in replacement for widget in new_unit.widgets
        ]
        if old_widgets and new_widgets:
            await content.mount(*new_widgets, before=old_widgets[0])
            await self._initialize_streaming_widgets(new_widgets)
        for widget in old_widgets:
            await widget.remove()
        self._units[position : position + 1] = replacement
        self._reindex_mounted_entries()

    async def _enforce_mount_cap(
        self,
        *,
        staged: dict[str, AgentTranscriptEntry],
        preserve_entry_ids: set[str] | None = None,
        evict_newest: bool = False,
    ) -> None:
        preserved_ids = preserve_entry_ids or set()
        mounted_count = sum(len(unit.entry_ids) for unit in self._units)
        while mounted_count > MAX_MOUNTED_TRANSCRIPT_ENTRIES and self._units:
            excess = mounted_count - MAX_MOUNTED_TRANSCRIPT_ENTRIES
            unit_indices = (
                range(len(self._units) - 1, -1, -1)
                if evict_newest
                else range(len(self._units))
            )
            eviction: tuple[int, list[str]] | None = None
            for preserve in (True, False):
                for unit_index in unit_indices:
                    unit = self._units[unit_index]
                    entry_ids = (
                        reversed(unit.entry_ids) if evict_newest else unit.entry_ids
                    )
                    removable = [
                        entry_id
                        for entry_id in entry_ids
                        if not preserve or entry_id not in preserved_ids
                    ]
                    if removable:
                        eviction = (unit_index, removable[:excess])
                        break
                if eviction is not None:
                    break
            if eviction is None:
                break

            unit_index, removed_ids = eviction
            unit = self._units[unit_index]
            if len(unit.entry_ids) > len(removed_ids):
                for entry_id in removed_ids:
                    history_entry = unit.entries.pop(entry_id, None)
                    if unit.group is not None and isinstance(
                        history_entry, PublicEffectEntry
                    ):
                        unit.group.forget_effect(history_entry.created_at)
                    for widget in unit.entry_widgets.pop(entry_id, []):
                        await widget.remove()
                    self._entry_units.pop(entry_id, None)
                removed_id_set = set(removed_ids)
                unit.entry_ids = [
                    entry_id
                    for entry_id in unit.entry_ids
                    if entry_id not in removed_id_set
                ]
                if unit.group is None:
                    unit.widgets = [
                        widget
                        for entry_id in unit.entry_ids
                        for widget in unit.entry_widgets[entry_id]
                    ]
                mounted_count -= len(removed_ids)
                continue

            self._units.pop(unit_index)
            for entry_id in unit.entry_ids:
                self._entry_units.pop(entry_id, None)
            for widget in unit.widgets:
                await widget.remove()
            mounted_count -= len(unit.entry_ids)
        self._reindex_mounted_entries()
        self._prune_known_entries(staged)

    def _prune_known_entries(self, staged: dict[str, AgentTranscriptEntry]) -> None:
        retained_ids = {entry_id for unit in self._units for entry_id in unit.entry_ids}
        mounted_by_call: dict[str, list[AgentTranscriptEntry]] = {}
        for entry_id in retained_ids:
            entry = staged.get(entry_id)
            if entry is not None and entry.tool_call_id is not None:
                mounted_by_call.setdefault(entry.tool_call_id, []).append(entry)

        for call_id, mounted_entries in mounted_by_call.items():
            mounted_kinds = {entry.kind for entry in mounted_entries}
            if {
                AgentTranscriptEntryKind.TOOL_CALL,
                AgentTranscriptEntryKind.TOOL_RESULT,
            } <= mounted_kinds:
                continue
            counterpart_kind = (
                AgentTranscriptEntryKind.TOOL_RESULT
                if AgentTranscriptEntryKind.TOOL_CALL in mounted_kinds
                else AgentTranscriptEntryKind.TOOL_CALL
            )
            candidates = [
                entry
                for entry in staged.values()
                if entry.entry_id not in retained_ids
                and entry.tool_call_id == call_id
                and entry.kind is counterpart_kind
            ]
            if candidates:
                mounted_time = min(entry.created_at for entry in mounted_entries)
                counterpart = min(
                    candidates, key=lambda entry: abs(entry.created_at - mounted_time)
                )
                retained_ids.add(counterpart.entry_id)

        for entry_id in tuple(staged):
            if entry_id not in retained_ids:
                del staged[entry_id]

    def _reindex_mounted_entries(self) -> None:
        self._entry_units = {
            entry_id: unit for unit in self._units for entry_id in unit.entry_ids
        }

    @staticmethod
    def _unit_entry_widgets(unit: _MountedUnit) -> list[Widget]:
        return [
            widget
            for entry_id in unit.entry_ids
            for widget in unit.entry_widgets[entry_id]
        ]

    async def _initialize_streaming_widgets(self, widgets: Sequence[Widget]) -> None:
        for widget in widgets:
            if isinstance(widget, StreamingMessageBase):
                await widget.write_initial_content()

    def _scroll_to_newest_if_needed(self) -> None:
        if self._on_newest_page and self._content_scroll is not None:
            self.call_after_refresh(
                self._content_scroll.scroll_end,
                animate=False,
                immediate=True,
                x_axis=False,
            )


def _entry_display_text(entry: AgentTranscriptEntry, text: str | None = None) -> str:
    display_text = entry.display_text if text is None else text
    return f"{display_text} [truncated]" if entry.truncated else display_text


__all__ = ["AgentTranscriptViewer"]
