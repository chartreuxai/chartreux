from __future__ import annotations

import gc
from weakref import WeakKeyDictionary, ref

import pytest
from textual.widget import Widget

from chartreux.app_server.models import PublicNoticeEntry, TextContentBlock
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing import transcript as transcript_module
from chartreux.cli.textual_ui.windowing.history import (
    _build_history_widgets_raw,
    build_history_widgets,
)
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.cli.textual_ui.test_history_grouping import (
    _effect,
    _file_edit_effect,
    _message,
)


def test_composite_membership_order_and_page_boundaries() -> None:
    window = TranscriptWindow()
    window.admit([_effect(2), _effect(3), _file_edit_effect(4)], start_index=2)
    window.admit([_effect(0), _effect(1)], start_index=0)
    assert window.unit_ids == ["effect-0", "effect-2", "effect-4"]
    assert [window.units[key].member_entry_ids for key in window.unit_ids] == [
        ["effect-0", "effect-1"],
        ["effect-2", "effect-3"],
        ["effect-4"],
    ]
    assert [window.units[key].start_index for key in window.unit_ids] == [0, 2, 4]
    widgets = window.flat_widgets(WeakKeyDictionary())
    assert [type(widget) for widget in widgets] == [
        ToolGroup,
        ToolGroup,
        ToolCallMessage,
        ToolResultMessage,
    ]


def test_rebuild_group_failure_muting_expansion_and_fresh_widgets() -> None:
    group_state = ToolGroupExpansionState()
    entry_state = EntryExpansionState()
    window = TranscriptWindow(
        expansion_state=group_state, entry_expansion_state=entry_state
    )
    window.admit([_effect(0, success=False), _effect(1)], start_index=0)
    first = window.build_unit("effect-0", WeakKeyDictionary())[0]
    assert isinstance(first, ToolGroup)
    results = [
        widget
        for widget in first.content_container.children
        if isinstance(widget, ToolResultMessage)
    ]
    assert results[0]._should_escalate is False
    assert first._key is not None
    first.set_collapsed(False)
    entry_state.set_collapsed("effect-0", False)
    second = window.build_unit("effect-0", WeakKeyDictionary())[0]
    assert isinstance(second, ToolGroup)
    assert second is not first
    assert not second.is_collapsed
    assert all(
        left is not right
        for left, right in zip(
            first.content_container.children,
            second.content_container.children,
            strict=True,
        )
    )


def test_idempotence_prepend_and_non_rendering_entries() -> None:
    window = TranscriptWindow()
    empty = _message(1)
    empty.content = [TextContentBlock(text="")]
    system = _message(2)
    system.role = "system"  # type: ignore[assignment]
    batch = [_file_edit_effect(3), empty, system]
    window.admit(batch, start_index=10)
    original = window.units["effect-3"]
    revision = window.geometry_version
    window.admit(batch, start_index=10)
    assert window.geometry_version == revision
    assert window.unit_ids == ["effect-3"]
    assert window.units["effect-3"] is original
    window.admit([_message(0)], start_index=9)
    assert window.unit_ids == ["message-0", "effect-3"]
    assert window.units["effect-3"].start_index == 10


def test_updated_entry_invalidates_geometry_without_changing_identity() -> None:
    window = TranscriptWindow()
    entry = _file_edit_effect(0)
    window.admit([entry], start_index=0)
    unit = window.units[entry.id]
    unit.content_height = 10
    updated = entry.model_copy(update={"title": "updated"})
    window.admit([updated], start_index=0)
    assert window.units[entry.id] is unit
    assert unit.entries == [updated]
    assert unit.content_height is None
    assert unit.geometry_version == window.geometry_version


def test_replacing_entry_cannot_adopt_stale_object_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = TranscriptWindow()
    original = _file_edit_effect(0)
    old_address = id(original)
    original_ref = ref(original)
    window.admit([original], start_index=0)
    replacement = _file_edit_effect(0)
    window.admit([replacement], start_index=0)
    del original
    gc.collect()
    assert original_ref() is None
    candidate = _file_edit_effect(1)
    # Force the allocator-reuse case even on interpreters that do not recycle it.
    monkeypatch.setattr(
        transcript_module, "id", lambda _entry: old_address, raising=False
    )
    window.admit([candidate], start_index=1)
    assert window.unit_ids == ["effect-0", "effect-1"]
    assert window.units["effect-0"].entries == [replacement]
    assert window.units["effect-1"].entries == [candidate]


def test_prepend_shifts_existing_indices_without_changing_unit_ids() -> None:
    window = TranscriptWindow()
    window.admit([_effect(2), _effect(3), _message(4)], start_index=0)
    ids = list(window.unit_ids)
    units = [window.units[key] for key in ids]
    window.admit([_message(0), _message(1)], start_index=0)
    window.admit([_effect(2), _effect(3), _message(4)], start_index=2)
    assert window.unit_ids == ["message-0", "message-1", *ids]
    assert [window.units[key] for key in ids] == units
    assert [window.units[key].start_index for key in ids] == [2, 4]
    indices = WeakKeyDictionary()
    widgets = window.flat_widgets(indices)
    assert isinstance(widgets[2], ToolGroup)
    assert [indices[child] for child in widgets[2].content_container.children] == [
        2,
        2,
        3,
        3,
    ]
    assert indices[widgets[3]] == 4


def test_admission_and_width_changes_invalidate_geometry() -> None:
    window = TranscriptWindow()
    assert window.geometry_version == 0
    window.admit([_effect(0)], start_index=0, geometry_width=80)
    unit = window.units["effect-0"]
    assert unit.geometry_width == 80
    assert unit.geometry_version == window.geometry_version > 0
    unit.content_height = 12
    version = window.geometry_version
    window.admit([_effect(0)], start_index=0, geometry_width=100)
    assert unit.geometry_width == 100
    assert unit.geometry_version == window.geometry_version > version
    assert unit.content_height is None
    version = window.geometry_version
    window.admit([_effect(1)], start_index=1, geometry_width=100)
    assert window.geometry_version > version
    assert window.units["effect-1"].geometry_width == 100


def test_mutating_retained_entry_invalidates_geometry() -> None:
    window = TranscriptWindow()
    entry = _file_edit_effect(0)
    window.admit([entry], start_index=0)
    unit = window.units[entry.id]
    unit.content_height = 12
    version = window.geometry_version
    entry.title = "changed in place"
    window.admit([entry], start_index=0)
    assert unit.entries[0] is entry
    assert unit.content_height is None
    assert unit.geometry_version == window.geometry_version > version


def test_identity_adoption_and_reset() -> None:
    window = TranscriptWindow()
    entry = _file_edit_effect(0)
    entry.id = ""
    window.admit([entry], start_index=0)
    fallback = window.unit_ids[0]
    assert fallback.startswith("transient-")
    entry.id = "public-0"
    window.admit([entry], start_index=0)
    assert window.unit_ids == ["public-0"]
    assert window.units["public-0"].member_entry_ids == ["public-0"]
    window.reset()
    assert not window.unit_ids
    assert not window.units


def test_group_and_entry_revision_invalidate_cached_height() -> None:
    groups = ToolGroupExpansionState()
    entries = EntryExpansionState()
    window = TranscriptWindow(expansion_state=groups, entry_expansion_state=entries)
    window.admit([_effect(0), _effect(1)], start_index=0)
    unit = window.units["effect-0"]
    assert unit.group_key is not None
    unit.content_height = 10
    groups.set_collapsed(unit.group_key, False)
    assert groups.changed_keys_since(0) == {unit.group_key}
    assert window.sync_expansion() == {unit.id}
    assert unit.content_height is None
    previous = window.geometry_version
    unit.content_height = 10
    entries.set_collapsed("effect-1", False)
    assert window.sync_expansion() == {unit.id}
    assert window.geometry_version > previous
    assert unit.content_height is None
    previous = groups.revision
    groups.set_collapsed(unit.group_key, False)
    assert groups.revision == previous
    groups.reset()
    assert groups.revision > previous
    window.reset()
    assert not window.units


def test_bulk_group_fold_invalidates_unbuilt_group() -> None:
    groups = ToolGroupExpansionState()
    window = TranscriptWindow(expansion_state=groups)
    window.admit([_effect(0)], start_index=0)
    unit = window.units["effect-0"]
    unit.content_height = 12
    groups.set_all_collapsed(False)
    assert window.sync_expansion() == {unit.id}
    assert unit.content_height is None
    rebuilt = window.build_unit(unit.id, WeakKeyDictionary())[0]
    assert isinstance(rebuilt, ToolGroup)
    assert not rebuilt.is_collapsed


def test_group_reset_with_unchanged_default_invalidates_cached_height() -> None:
    groups = ToolGroupExpansionState()
    window = TranscriptWindow(expansion_state=groups)
    window.admit([_effect(0)], start_index=0)
    unit = window.units["effect-0"]
    assert unit.group_key is not None
    groups.set_collapsed(unit.group_key, False)
    assert window.sync_expansion() == {unit.id}
    unit.content_height = 12
    groups.reset()
    assert window.sync_expansion() == {unit.id}
    assert unit.content_height is None
    rebuilt = window.build_unit(unit.id, WeakKeyDictionary())[0]
    assert isinstance(rebuilt, ToolGroup)
    assert rebuilt.is_collapsed


def _widget_signature(
    widget: Widget, indices: WeakKeyDictionary[Widget, int]
) -> tuple[object, ...]:
    if isinstance(widget, ToolGroup):
        return (
            type(widget),
            widget._key,
            widget.is_collapsed,
            widget.header._last_state,
            tuple(
                _widget_signature(child, indices)
                for child in widget.content_container.children
            ),
        )
    if isinstance(widget, ToolCallMessage):
        return (
            type(widget),
            indices[widget],
            widget.tool_call_id,
            widget.get_content(),
            widget.get_content_suffix(),
            widget._state,
        )
    if isinstance(widget, ToolResultMessage):
        return (
            type(widget),
            indices[widget],
            widget._entry.id,
            widget._get_result_parts(),
            widget._should_escalate,
        )
    if isinstance(widget, AssistantMessage | UserMessage):
        return (type(widget), indices[widget], widget.get_content())
    return (type(widget), indices[widget])


def test_flat_adapter_matches_original_builder_and_indices() -> None:
    batch = [_effect(0, success=False), _effect(1), _message(2), _file_edit_effect(3)]
    actual_indices = WeakKeyDictionary()
    expected_indices = WeakKeyDictionary()
    actual = build_history_widgets(
        batch,
        start_index=20,
        history_widget_indices=actual_indices,
        tools_collapsed=True,
    )
    expected = _build_history_widgets_raw(
        batch,
        start_index=20,
        history_widget_indices=expected_indices,
        tools_collapsed=True,
    )
    assert [_widget_signature(widget, actual_indices) for widget in actual] == [
        _widget_signature(widget, expected_indices) for widget in expected
    ]
    assert isinstance(actual[0], ToolGroup)
    assert len(actual[0].content_container.children) == 4
    grouped_result = actual[0].content_container.children[1]
    assert isinstance(grouped_result, ToolResultMessage)
    assert grouped_result._should_escalate is False
    assert isinstance(actual[-1], ToolResultMessage)
    assert actual[-1]._should_escalate is False


def test_blank_id_adapter_matches_raw_builder() -> None:
    effect = _effect(0)
    effect.id = ""
    user = _message(1)
    user.id = ""
    user.role = "user"  # type: ignore[assignment]
    batch = [effect, user]
    actual_indices = WeakKeyDictionary()
    expected_indices = WeakKeyDictionary()
    actual = build_history_widgets(
        batch,
        start_index=20,
        history_widget_indices=actual_indices,
        tools_collapsed=True,
    )
    expected = _build_history_widgets_raw(
        batch,
        start_index=20,
        history_widget_indices=expected_indices,
        tools_collapsed=True,
    )
    assert [_widget_signature(widget, actual_indices) for widget in actual] == [
        _widget_signature(widget, expected_indices) for widget in expected
    ]
    assert isinstance(actual[0], ToolGroup)
    assert actual[0]._key is not None
    assert actual[0]._key.first_entry_id == ""
    assert isinstance(actual[1], UserMessage)
    assert actual[1].history_entry_id == ""


def test_hook_notice_and_other_non_rendering_entries() -> None:
    window = TranscriptWindow()
    notice = PublicNoticeEntry.model_construct(id="notice", detail=None)
    window.admit([notice], start_index=0)
    assert window.unit_ids == []
    assert not window.flat_widgets(WeakKeyDictionary())
