from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from weakref import WeakKeyDictionary

from textual.widget import Widget

from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicNoticeEntry,
    PublicReasoningEntry,
    RootRecoveryNoticeDetail,
    RouteTableAppliedNoticeDetail,
    ScheduledLoopFiredNoticeDetail,
)
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.message_header import MinuteKey
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ReasoningMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.minute_group_header import MinuteGroupHeader
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    effect_state_is_failure,
    entry_keeps_tool_group,
    tool_group_key,
)
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)


def history_entry_renders_widget(entry: PublicHistoryEntry) -> bool:
    match entry:
        case PublicMessageEntry(role="user"):
            return True
        case PublicMessageEntry(role="assistant"):
            return bool(entry.text)
        case PublicReasoningEntry() | PublicEffectEntry():
            return True
        case PublicCheckpointEntry(kind="compaction"):
            return True
        case _:
            return False


def build_history_widgets(
    batch: Sequence[PublicHistoryEntry],
    *,
    start_index: int,
    history_widget_indices: WeakKeyDictionary[Widget, int],
    tools_collapsed: bool,
    expansion_state: ToolGroupExpansionState | None = None,
    entry_expansion_state: EntryExpansionState | None = None,
    show_message_timestamps: bool = True,
) -> list[Widget]:
    """Build restored timeline widgets with the live tool-grouping policy.

    ``start_index`` is deliberately used for status ordering, while group keys
    are anchored to the first reasoning/effect entry that creates the live group.
    The key remains valid when older pages are prepended and indices are shifted.
    Each batch is built independently, so a run that crosses a pagination boundary
    remains two consistently-rendered groups until pagination can merge widgets.
    """
    from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow

    window = TranscriptWindow(
        tools_collapsed=tools_collapsed,
        expansion_state=expansion_state,
        entry_expansion_state=entry_expansion_state,
    )
    # The batch build stamps the minutes it swallows, so it needs the same gate
    # the caller applies to the widgets.
    window._get_show_timestamps = lambda: show_message_timestamps
    window.admit(batch, start_index=start_index)
    widgets = window.flat_widgets(history_widget_indices)
    for widget in widgets:
        if isinstance(
            widget,
            UserMessage
            | AssistantMessage
            | ToolGroup
            | ToolCallMessage
            | ToolResultMessage,
        ):
            widget.set_show_message_timestamps(show_message_timestamps)
    return widgets


def _build_history_widgets_raw(
    batch: Sequence[PublicHistoryEntry],
    *,
    start_index: int,
    history_widget_indices: WeakKeyDictionary[Widget, int],
    tools_collapsed: bool,
    expansion_state: ToolGroupExpansionState | None = None,
    entry_expansion_state: EntryExpansionState | None = None,
    show_message_timestamps: bool = True,
) -> list[Widget]:
    """Build one batch's widgets, stamping the minutes inside a fold.

    A batch's first record is stamped at the transcript-spanning scope -- the
    window mounts it above the unit, and a batch boundary has no earlier record
    to compare against -- so this build stamps only the later records: the ones
    whose minute the walk has not rendered yet.
    """
    widgets: list[Widget] = []
    current_group: ToolGroup | None = None
    current_group_errors: list[tuple[int, ToolResultMessage]] = []
    markers = minute_marker_minutes(batch)

    def finalize_current_group() -> None:
        """Apply the same individual-error policy as live rendering."""
        nonlocal current_group, current_group_errors
        if current_group is None:
            return
        for timeline_index, result in current_group_errors:
            result.set_error_escalation(
                escalate=not current_group.failure_is_muted(timeline_index)
            )
        current_group.finalize()
        widgets.append(current_group)
        current_group = None
        current_group_errors = []

    for position, (history_index, entry) in enumerate(
        zip(range(start_index, start_index + len(batch)), batch, strict=True)
    ):
        marker = _opened_marker(markers, position, show_message_timestamps)
        if entry_keeps_tool_group(entry):
            created_group = False
            if current_group is None:
                # Hook notices preserve a group but do not create one live.
                if not isinstance(entry, PublicReasoningEntry | PublicEffectEntry):
                    continue
                current_group = ToolGroup(
                    key=tool_group_key(entry), expansion_state=expansion_state
                )
                if expansion_state is None:
                    current_group.set_collapsed(tools_collapsed)
                created_group = True
            _place_marker(
                marker, group=current_group, opens_fold=created_group, outside=widgets
            )
            entry_widgets = _entry_widgets(
                entry, history_index, tools_collapsed, entry_expansion_state
            )
            for widget in entry_widgets:
                current_group.add_content_child(widget)
                history_widget_indices[widget] = history_index
            if isinstance(entry, PublicEffectEntry):
                current_group.add_call_kind(entry.detail.kind)
                current_group.record_effect(history_index, entry.state)
                if effect_state_is_failure(entry.state):
                    result = next(
                        (
                            widget
                            for widget in entry_widgets
                            if isinstance(widget, ToolResultMessage)
                        ),
                        None,
                    )
                    if result is not None:
                        current_group_errors.append((history_index, result))
            elif isinstance(entry, PublicReasoningEntry):
                current_group.mark_reasoning()
            continue

        if current_group is not None:
            finalize_current_group()
        _place_marker(marker, group=None, opens_fold=True, outside=widgets)
        entry_widgets = _entry_widgets(
            entry, history_index, tools_collapsed, entry_expansion_state
        )
        if isinstance(entry, PublicEffectEntry) and effect_state_is_failure(
            entry.state
        ):
            result = next(
                (
                    widget
                    for widget in entry_widgets
                    if isinstance(widget, ToolResultMessage)
                ),
                None,
            )
            if result is not None:
                # Standalone effects have no grouped follow-up that can recover
                # their failure, so their restored result is terminal.
                result.set_error_escalation(escalate=True)
        widgets.extend(entry_widgets)
        for widget in entry_widgets:
            history_widget_indices[widget] = history_index

    if current_group is not None:
        finalize_current_group()

    return widgets


def _opened_marker(
    markers: Sequence[MinuteKey | None], position: int, show: bool
) -> MinuteGroupHeader | None:
    """The stamp a non-leading record opens; a batch's first record owns none."""
    if position == 0 or not show:
        return None
    minute = markers[position]
    return None if minute is None else MinuteGroupHeader(minute)


def _place_marker(
    marker: MinuteGroupHeader | None,
    *,
    group: ToolGroup | None,
    opens_fold: bool,
    outside: list[Widget],
) -> None:
    """Put a stamp above the record it opens: above a fold, or inside one."""
    if marker is None:
        return
    if group is None or opens_fold:
        outside.append(marker)
    else:
        group.add_content_child(marker)


def _entry_widgets(
    entry: PublicHistoryEntry,
    history_index: int,
    tools_collapsed: bool,
    entry_expansion_state: EntryExpansionState | None,
) -> list[Widget]:
    match entry:
        case PublicMessageEntry(role="user"):
            return [
                UserMessage(
                    entry.text,
                    history_entry_id=entry.id,
                    images=entry.images or None,
                    posted_at=entry.posted_at,
                )
            ]
        case PublicMessageEntry(role="assistant"):
            return (
                [
                    AssistantMessage(
                        entry.text,
                        posted_at=entry.posted_at,
                        turn_duration_ms=entry.turn_duration_ms,
                    )
                ]
                if entry.text
                else []
            )
        case PublicReasoningEntry():
            return [
                ReasoningMessage(
                    entry.text,
                    collapsed=tools_collapsed,
                    entry_id=entry.id,
                    expansion_state=entry_expansion_state,
                    completed=(
                        entry.generation_status is PublicEntryGenerationStatus.COMPLETED
                    ),
                )
            ]
        case PublicEffectEntry():
            call = ToolCallMessage(entry)
            return [
                call,
                ToolResultMessage(entry, call, expansion_state=entry_expansion_state),
            ]
        case PublicCheckpointEntry(kind="compaction"):
            message = CompactMessage()
            message.set_complete()
            return [message]
        case _:
            return []


def notice_renders_record(entry: PublicNoticeEntry) -> bool:
    """Whether a notice is a transcript record with a time of its own.

    The design stamps the notices that render as records; the remaining kinds
    (waiting-for-input, session-title updates, hook notices folded into a tool
    group) are status or inline content and carry no stamp.
    """
    return isinstance(
        entry.detail,
        ScheduledLoopFiredNoticeDetail
        | RootRecoveryNoticeDetail
        | RouteTableAppliedNoticeDetail,
    )


def renders_minute_record(entry: PublicHistoryEntry) -> bool:
    """Whether the transcript shows *entry* as a record of its own."""
    if isinstance(entry, PublicNoticeEntry):
        return notice_renders_record(entry)
    return history_entry_renders_widget(entry)


def entry_minute(entry: PublicHistoryEntry) -> MinuteKey | None:
    """The local wall-clock minute *entry* stamps, or None without a real clock.

    Clock sources are chosen by transport. A user/assistant message carries its
    canonical posting time on both transports. A record keys on the persisted
    per-event clock (``event_at``) whenever it has one: that is the only real
    clock a restored record offers, because restore reuses ``created_at`` for
    its synthetic timestamp+index ordinal. A record without one keys on
    ``created_at`` only when the live projector stamped it -- the live projector
    stamps its turn id on every entry it creates and restored projections clear
    it, so a turn id marks a live emission whose ``created_at`` is the real
    clock. A record written before v0.7.4 therefore stamps no minute, and
    history is never backfilled.
    """
    match entry:
        case PublicMessageEntry():
            posted_at = entry.posted_at
            if posted_at is None:
                return None
            local = posted_at.astimezone()
        case PublicReasoningEntry() | PublicEffectEntry():
            event_at = entry.event_at
            if event_at is None:
                if entry.turn_id is None:
                    return None
                event_at = entry.created_at
            local = datetime.fromtimestamp(event_at // 1000).astimezone()
        case PublicCheckpointEntry():
            if entry.turn_id is None:
                return None
            local = datetime.fromtimestamp(entry.created_at // 1000).astimezone()
        case PublicNoticeEntry():
            if not notice_renders_record(entry) or entry.turn_id is None:
                return None
            local = datetime.fromtimestamp(entry.created_at // 1000).astimezone()
        case _:
            return None
    return (local.year, local.month, local.day, local.hour, local.minute)


def advance_rendered_minute(
    rendered: MinuteKey | None, entry: PublicHistoryEntry
) -> tuple[MinuteKey | None, MinuteKey | None]:
    """Step the rendered-minute marker over one visible record.

    Returns the marker the walk carries after this record and the stamp the
    record itself opens. A message header renders the minute instead of a group
    header, so it moves the marker and opens nothing; a record whose minute the
    marker already carries opens nothing; a record without a real clock leaves
    the marker where it was.
    """
    minute = entry_minute(entry)
    if minute is None:
        return rendered, None
    if isinstance(entry, PublicMessageEntry):
        return minute, None
    if minute == rendered:
        return rendered, None
    return minute, minute


def minute_marker_minutes(
    entries: Sequence[PublicHistoryEntry],
) -> list[MinuteKey | None]:
    """The minute bucket each visible record opens, in content order.

    Records bucket by wall-clock minute and a bucket stamps only its first
    record. A record that renders a header of its own (a user/assistant message)
    advances the rendered-minute marker instead of opening a group header, so
    the records continuing in that minute spawn none; a record without a real
    clock leaves the marker untouched.
    """
    rendered: MinuteKey | None = None
    markers: list[MinuteKey | None] = []
    for entry in entries:
        rendered, marker = advance_rendered_minute(rendered, entry)
        markers.append(marker)
    return markers


def split_history_tail(
    history: list[PublicHistoryEntry], tail_size: int
) -> tuple[list[PublicHistoryEntry], list[PublicHistoryEntry], int]:
    tail = history[-tail_size:]
    backfill = history[:-tail_size]
    return tail, backfill, len(history) - len(tail)


def visible_history_indices(
    children: list[Widget], history_widget_indices: WeakKeyDictionary[Widget, int]
) -> list[int]:
    indices: list[int] = []
    for child in children:
        if isinstance(child, ToolGroup):
            indices.extend(
                index
                for group_child in child.content_container.children
                if (index := history_widget_indices.get(group_child)) is not None
            )
        elif (index := history_widget_indices.get(child)) is not None:
            indices.append(index)
    return indices


def visible_history_widgets_count(children: list[Widget]) -> int:
    history_widget_types = (
        UserMessage,
        AssistantMessage,
        CompactMessage,
        ReasoningMessage,
        ToolCallMessage,
        ToolResultMessage,
        ToolGroup,
    )
    return sum(isinstance(child, history_widget_types) for child in children)


def shift_history_widget_indices(
    history_widget_indices: WeakKeyDictionary[Widget, int], offset: int
) -> None:
    for widget, index in list(history_widget_indices.items()):
        history_widget_indices[widget] = index + offset
