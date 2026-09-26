from __future__ import annotations

from bisect import bisect_right
from unittest.mock import patch
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget

from chartreux.app_server.events import HistoryEntryAdded
from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    FailedEffectState,
    FileEditEffectDetail,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicError,
    PublicReasoningEntry,
)
from chartreux.cli.textual_ui.widgets.collapsible import CollapsibleSection
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.messages import ReasoningMessage
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.widgets.tools import ToolGroup, ToolResultMessage
from chartreux.cli.textual_ui.windowing.history import build_history_widgets
from tests.conftest import build_test_chartreux_app, build_test_vibe_config


class _HistoryApp(App[None]):
    CSS_PATH = "../../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield Vertical(id="history")


def _reasoning(entry_id: str = "reasoning") -> PublicReasoningEntry:
    return PublicReasoningEntry(
        id=entry_id,
        session_id="session",
        turn_id="turn",
        created_at=1,
        updated_at=2,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        text="Reasoning body",
    )


def _effect(entry_id: str = "effect", *, standalone: bool = False) -> PublicEffectEntry:
    detail = (
        FileEditEffectDetail(
            tool_name="edit",
            display=EffectCallDisplay(summary="Edit file", status_text="Edit"),
        )
        if standalone
        else GenericEffectDetail(
            tool_name="read",
            display=EffectCallDisplay(summary="Read file", status_text="Read"),
        )
    )
    return PublicEffectEntry(
        id=entry_id,
        session_id="session",
        turn_id="turn",
        created_at=2,
        updated_at=3,
        generation_status=PublicEntryGenerationStatus.COMPLETED,
        title="tool",
        detail=detail,
        state=(
            FailedEffectState(
                error=PublicError(message="Failure details"),
                display=EffectResultDisplay(success=False, message="Failed"),
            )
            if standalone
            else CompletedEffectState(
                output_text="Result body",
                display=EffectResultDisplay(success=True, message="Result"),
            )
        ),
    )


def _build(
    entries: list[PublicReasoningEntry | PublicEffectEntry],
    entry_state: EntryExpansionState,
    group_state: ToolGroupExpansionState,
) -> list[Widget]:
    return build_history_widgets(
        entries,
        start_index=0,
        history_widget_indices=WeakKeyDictionary(),
        tools_collapsed=True,
        expansion_state=group_state,
        entry_expansion_state=entry_state,
    )


@pytest.mark.asyncio
async def test_individual_choices_survive_destroy_and_rebuild() -> None:
    state = EntryExpansionState()
    groups = ToolGroupExpansionState()
    entries = [_reasoning(), _effect(), _effect("standalone", standalone=True)]
    app = _HistoryApp()
    async with app.run_test() as pilot:
        history = app.query_one("#history", Vertical)
        await history.mount_all(_build(entries, state, groups))
        await pilot.pause()
        group = app.query_one(ToolGroup)
        group.set_collapsed(False)
        await app.query_one(ReasoningMessage).set_collapsed(False)
        result = list(app.query(ToolResultMessage))[-1]
        section = result.query_one(CollapsibleSection)
        section.set_collapsed(False)
        await pilot.pause()
        assert not state.is_collapsed("reasoning")
        assert not state.is_collapsed("standalone")
        revision = state.revision
        assert state.changed_ids_since(0) == {"reasoning", "standalone"}

        await history.remove_children()
        assert state.revision == revision
        await history.mount_all(_build(entries, state, groups))
        await pilot.pause()
        assert not app.query_one(ReasoningMessage).collapsed
        assert (
            not list(app.query(ToolResultMessage))[-1]
            .query_one(CollapsibleSection)
            .is_collapsed
        )
        assert app.query_one(ToolGroup).is_collapsed is False

        await history.remove_children()
        await history.mount_all(_build(entries, state, groups))
        await pilot.pause()
        assert (
            not list(app.query(ToolResultMessage))[-1]
            .query_one(CollapsibleSection)
            .is_collapsed
        )


@pytest.mark.asyncio
async def test_group_and_child_fold_independently() -> None:
    state = EntryExpansionState()
    groups = ToolGroupExpansionState()
    app = _HistoryApp()
    async with app.run_test() as pilot:
        history = app.query_one("#history", Vertical)
        await history.mount_all(_build([_reasoning(), _effect()], state, groups))
        await pilot.pause()
        group = app.query_one(ToolGroup)
        group.set_collapsed(False)
        await app.query_one(ReasoningMessage).set_collapsed(False)
        group.set_collapsed(True)
        assert not state.is_collapsed("reasoning")
        assert state.is_collapsed("effect")
        assert group.is_collapsed
        await history.remove_children()
        await history.mount_all(_build([_reasoning(), _effect()], state, groups))
        assert app.query_one(ToolGroup).is_collapsed
        assert not app.query_one(ReasoningMessage).collapsed


@pytest.mark.asyncio
async def test_live_entry_ids_are_shared_with_restored_widgets() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    entries = [_reasoning(), _effect("standalone", standalone=True)]
    async with app.run_test() as pilot:
        handler = app.event_handler
        assert handler is not None
        for entry in entries:
            await handler.handle_event(HistoryEntryAdded(entry))
        await pilot.pause()
        assert app._entry_expansion_state.entry_ids == {"reasoning", "standalone"}
        await app._messages_area.query_one(ReasoningMessage).set_collapsed(False)
        app._messages_area.query_one(ToolResultMessage).query_one(
            CollapsibleSection
        ).set_collapsed(False)
        await app._messages_area.remove_children()
        await app._messages_area.mount_all(
            _build(entries, app._entry_expansion_state, app._tool_group_expansion_state)
        )
        await pilot.pause()
        assert not app._messages_area.query_one(ReasoningMessage).collapsed
        assert not app._messages_area.query_one(CollapsibleSection).is_collapsed


@pytest.mark.asyncio
async def test_ctrl_o_updates_unmounted_entries_and_overrides() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    state = app._entry_expansion_state
    state.register("offscreen")
    state.set_collapsed("offscreen", False)
    state.register("other")
    async with app.run_test():
        await app.action_toggle_tool()
        assert not state.default_collapsed
        assert not state.is_collapsed("other")
        await app.action_toggle_tool()
        assert state.default_collapsed
        assert state.is_collapsed("offscreen")
        assert state.is_collapsed("other")
        assert "offscreen" in state.changed_ids_since(0)


def test_session_replacement_resets_overrides_and_preserves_revision() -> None:
    state = EntryExpansionState()
    state.register("same-id")
    state.set_collapsed("same-id", False)
    previous_revision = state.revision
    state.reset()
    assert state.revision > previous_revision
    assert not state.entry_ids
    assert not state.changed_ids_since(previous_revision)
    assert state.register("same-id")


def test_changed_ids_since_indexes_only_the_recent_delta() -> None:
    state = EntryExpansionState()
    for index in range(10_000):
        state.set_collapsed(f"old-{index}", False)
    checkpoint = state.revision
    state.set_collapsed("recent", False)
    state.set_collapsed("old-0", True)
    with patch(
        "chartreux.cli.textual_ui.widgets.entry_expansion.bisect_right",
        wraps=bisect_right,
    ) as search:
        assert state.changed_ids_since(checkpoint) == {"recent", "old-0"}
    search.assert_called_once_with(state._change_revisions, checkpoint)
    assert state._changed_ids[-2:] == ["recent", "old-0"]


@pytest.mark.asyncio
async def test_preview_entry_choices_do_not_leak_into_active_session() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    active = app._entry_expansion_state
    active.register("reasoning")
    active.set_collapsed("reasoning", False)
    checkpoint = active.revision
    async with app.run_test() as pilot:
        app._picker.preview_session_id = "preview-session"
        await app._apply_picker_preview("preview-session", [_reasoning()])
        await pilot.pause()
        preview_node = app._messages_area.query_one(ReasoningMessage)
        assert preview_node.collapsed
        await preview_node.set_collapsed(False)
        await app.action_toggle_tool()
        assert active.entry_ids == {"reasoning"}
        assert active.revision == checkpoint
        assert not active.is_collapsed("reasoning")
        await app._exit_picker_to_input()
        await pilot.pause()
        assert app._tools_collapsed is active.default_collapsed
        assert app._tools_collapsed is app._tool_group_expansion_state.default_collapsed
        assert active.entry_ids == {"reasoning"}
        assert not active.is_collapsed("reasoning")

        await app._messages_area.remove_children()
        await app._mount_history_batch(
            [_reasoning("rebuilt-reasoning")], app._messages_area, start_index=0
        )
        await pilot.pause()
        rebuilt_node = app._messages_area.query_one(ReasoningMessage)
        assert rebuilt_node.collapsed is app._tools_collapsed
        assert rebuilt_node.collapsed is active.default_collapsed


@pytest.mark.asyncio
async def test_app_session_replacement_resets_expansion() -> None:
    app = build_test_chartreux_app(config=build_test_vibe_config())
    async with app.run_test():
        app._entry_expansion_state.set_collapsed("old-id", False)
        await app._reset_presentation_after_resume()
        assert app._entry_expansion_state.is_collapsed("old-id")
        assert not app._entry_expansion_state.entry_ids
