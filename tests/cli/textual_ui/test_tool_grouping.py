from __future__ import annotations

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    EffectState,
    FailedEffectState,
    FileEditEffectDetail,
    FileWriteEffectDetail,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    SkippedEffectState,
    TextContentBlock,
)
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    GroupIndicator,
    TimelineStatus,
    ToolGroupExpansionState,
    entry_keeps_tool_group,
    is_manual_shell_entry,
    starts_tool_group,
    tool_group_key,
)


def _effect(index: int, state: EffectState | None = None) -> PublicEffectEntry:
    return PublicEffectEntry(
        id=f"effect-{index}",
        session_id="session-1",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="tool",
        detail=GenericEffectDetail(
            tool_name="tool",
            display=EffectCallDisplay(summary="tool", status_text="Running tool"),
        ),
        state=state
        or CompletedEffectState(
            display=EffectResultDisplay(success=True, message="completed")
        ),
    )


def _message(index: int, role: str) -> PublicMessageEntry:
    return PublicMessageEntry(
        id=f"message-{index}",
        session_id="session-1",
        created_at=index,
        updated_at=index,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        role=role,  # type: ignore[arg-type]
        content=[TextContentBlock(text="content")],
        source="harness",
    )


def test_consecutive_tool_calls_share_a_group_until_message_content() -> None:
    first = _effect(1)
    second = _effect(2)
    user = _message(3, "user")
    third = _effect(4)
    assistant = _message(5, "assistant")

    assert starts_tool_group(first, None)
    assert entry_keeps_tool_group(second)
    assert not starts_tool_group(second, first)
    assert not entry_keeps_tool_group(user)
    assert starts_tool_group(third, user)
    assert not entry_keeps_tool_group(assistant)


def test_group_key_is_stable_when_backfill_includes_the_same_oldest_entry() -> None:
    oldest = _effect(10)
    later = _effect(11)

    key_from_tail = tool_group_key(oldest)
    key_after_backfill = tool_group_key(oldest)

    assert key_from_tail == key_after_backfill
    assert key_from_tail.first_entry_id == oldest.id
    assert not starts_tool_group(later, oldest)


def test_timeline_status_uses_the_last_call_not_the_worst_result() -> None:
    status = TimelineStatus()
    status.record_effect(
        10,
        FailedEffectState(
            error={"message": "failed"},  # type: ignore[arg-type]
            display=EffectResultDisplay(success=False, message="failed"),
        ),
    )
    status.record_effect(
        11,
        CompletedEffectState(
            display=EffectResultDisplay(success=True, message="completed")
        ),
    )

    assert status.indicator is GroupIndicator.SUCCESS


def test_expansion_state_is_owned_by_a_group_key() -> None:
    key = tool_group_key(_effect(1))
    state = ToolGroupExpansionState(default_collapsed=True)

    assert state.is_collapsed(key)
    state.set_collapsed(key, False)
    assert not state.is_collapsed(key)
    assert ToolGroupExpansionState(default_collapsed=False).is_collapsed(key) is False


def test_manual_shell_exclusion_is_the_default_shared_policy() -> None:
    effect = _effect(1)
    effect.detail.tool_name = "shell"

    assert is_manual_shell_entry(effect)
    assert not entry_keeps_tool_group(effect)


def test_file_edit_and_write_exclusion_is_the_default_shared_policy() -> None:
    for detail in (
        FileEditEffectDetail(
            tool_name="edit",
            display=EffectCallDisplay(summary="edit", status_text="edit"),
        ),
        FileWriteEffectDetail(
            tool_name="write_file",
            display=EffectCallDisplay(summary="write", status_text="write"),
        ),
    ):
        effect = _effect(1)
        effect.detail = detail

        assert not entry_keeps_tool_group(effect)


def test_skipped_effect_maps_to_a_muted_indicator() -> None:
    status = TimelineStatus()
    status.record_effect(
        1,
        SkippedEffectState(
            reason="not needed",
            display=EffectResultDisplay(success=True, message="skipped"),
        ),
    )

    assert status.indicator is GroupIndicator.MUTED
