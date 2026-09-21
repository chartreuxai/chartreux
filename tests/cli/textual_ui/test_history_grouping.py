from __future__ import annotations

from weakref import WeakKeyDictionary

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    FailedEffectState,
    FileEditEffectDetail,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    TextContentBlock,
)
from chartreux.cli.textual_ui.widgets.status_message import IndicatorState
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing.history import build_history_widgets


def _effect(index: int, *, success: bool = True) -> PublicEffectEntry:
    state = CompletedEffectState(
        display=EffectResultDisplay(success=success, message="done")
    )
    if not success:
        state = FailedEffectState(
            error={"message": "failed"},  # type: ignore[arg-type]
            display=EffectResultDisplay(success=False, message="failed"),
        )
    return PublicEffectEntry(
        id=f"effect-{index}",
        session_id="session-1",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="tool",
        detail=GenericEffectDetail(
            tool_name="tool",
            display=EffectCallDisplay(summary="tool", status_text="run"),
        ),
        state=state,
    )


def _file_edit_effect(index: int, *, success: bool = True) -> PublicEffectEntry:
    effect = _effect(index, success=success)
    effect.detail = FileEditEffectDetail(
        tool_name="edit", display=EffectCallDisplay(summary="edit", status_text="edit")
    )
    return effect


def _message(index: int) -> PublicMessageEntry:
    return PublicMessageEntry(
        id=f"message-{index}",
        session_id="session-1",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role="assistant",
        content=[TextContentBlock(text="separator")],
        source="harness",
    )


def _build(entries: list[object], *, start_index: int = 0, state=None):
    return build_history_widgets(
        entries,  # type: ignore[arg-type]
        start_index=start_index,
        history_widget_indices=WeakKeyDictionary(),
        tools_collapsed=True,
        expansion_state=state,
    )


def test_restored_history_uses_live_group_boundaries_and_timeline_status() -> None:
    widgets = _build([_effect(0, success=False), _effect(1), _message(2), _effect(3)])

    groups = [widget for widget in widgets if isinstance(widget, ToolGroup)]
    assert len(groups) == 2
    assert [len(group.content_container.children) for group in groups] == [4, 2]
    # The last timeline call wins, matching live TimelineStatus behavior.
    assert groups[0].header._last_state is IndicatorState.SUCCESS


def test_group_key_uses_oldest_entry_not_page_offset() -> None:
    group = _build([_effect(10), _effect(11)], start_index=100)[0]

    assert isinstance(group, ToolGroup)
    assert group._key is not None
    assert group._key.first_entry_id == "effect-10"


def test_restored_manual_shell_breaks_tool_groups() -> None:
    manual = _effect(1)
    manual.detail.tool_name = "shell"

    widgets = _build([_effect(0), manual, _effect(2)])

    assert [type(widget) for widget in widgets] == [
        ToolGroup,
        ToolCallMessage,
        ToolResultMessage,
        ToolGroup,
    ]

    state = ToolGroupExpansionState()
    first = _build([_effect(1), _effect(2)], state=state)[0]
    assert isinstance(first, ToolGroup)
    first.set_collapsed(False)

    rebuilt = _build([_effect(1), _effect(2)], start_index=20, state=state)[0]
    assert isinstance(rebuilt, ToolGroup)
    assert not rebuilt.is_collapsed


def test_restored_file_edits_break_groups_and_escalate_failures() -> None:
    widgets = _build([_effect(0), _file_edit_effect(1, success=False), _effect(2)])

    assert [type(widget) for widget in widgets] == [
        ToolGroup,
        ToolCallMessage,
        ToolResultMessage,
        ToolGroup,
    ]
    standalone_result = widgets[2]
    assert isinstance(standalone_result, ToolResultMessage)
    assert standalone_result._should_escalate is True
