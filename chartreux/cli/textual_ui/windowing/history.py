from __future__ import annotations

from collections.abc import Sequence
from weakref import WeakKeyDictionary

from textual.widget import Widget

from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicReasoningEntry,
)
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ReasoningMessage,
    UserMessage,
)
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
    window.admit(batch, start_index=start_index)
    return window.flat_widgets(history_widget_indices)


def _build_history_widgets_raw(
    batch: Sequence[PublicHistoryEntry],
    *,
    start_index: int,
    history_widget_indices: WeakKeyDictionary[Widget, int],
    tools_collapsed: bool,
    expansion_state: ToolGroupExpansionState | None = None,
    entry_expansion_state: EntryExpansionState | None = None,
) -> list[Widget]:
    widgets: list[Widget] = []
    current_group: ToolGroup | None = None
    current_group_errors: list[tuple[int, ToolResultMessage]] = []

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

    for history_index, entry in zip(
        range(start_index, start_index + len(batch)), batch, strict=True
    ):
        if entry_keeps_tool_group(entry):
            if current_group is None:
                # Hook notices preserve a group but do not create one live.
                if not isinstance(entry, PublicReasoningEntry | PublicEffectEntry):
                    continue
                current_group = ToolGroup(
                    key=tool_group_key(entry), expansion_state=expansion_state
                )
                if expansion_state is None:
                    current_group.set_collapsed(tools_collapsed)
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
                    entry.text, history_entry_id=entry.id, images=entry.images or None
                )
            ]
        case PublicMessageEntry(role="assistant"):
            return [AssistantMessage(entry.text)] if entry.text else []
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
