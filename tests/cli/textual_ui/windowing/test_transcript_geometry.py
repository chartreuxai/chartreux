from __future__ import annotations

import asyncio
import gc
from weakref import WeakKeyDictionary, ref

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.widget import Widget

from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    ToolGroupExpansionState,
    ToolGroupKey,
)
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing.placeholder import PlacementPlaceholder
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.cli.textual_ui.test_history_grouping import (
    _effect,
    _file_edit_effect,
    _message,
)


class _TranscriptApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #stream { height: 10; width: 60; layout: stream; overflow-y: auto; }
    #stream > .first { margin: 1 0 3 0; padding: 2 1 1 1; }
    #stream > .middle { margin: 4 0 1 0; padding: 1 0 2 0; }
    #stream > .last { margin: 2 0 0 0; }
    #stream > .edge0 { margin: 1 0 6 0; }
    #stream > .edge1 { margin: 2 0 5 0; padding: 1 0 1 0; }
    #stream > .edge2 { margin: 7 0 3 0; }
    #stream > .edge3 { margin: 1 0 2 0; }
    #stream > .horizontal { margin: 0 4 0 3; padding: 1 1 1 1; }
    """

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")


class _Measured(Widget):
    def __init__(
        self,
        height: int,
        *,
        margin: tuple[int, int, int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        super().__init__()
        self.height = height
        self.styles.margin = margin
        self.styles.padding = padding

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.height


def _window(count: int) -> TranscriptWindow:
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(count)], start_index=0)
    return window


@pytest.mark.asyncio
async def test_placement_round_trip_preserves_virtual_size_and_scroll_offset() -> None:
    window = _window(3)
    app = _TranscriptApp()
    roots = [
        _Measured(15, margin=(1, 0, 3, 0), padding=(2, 1, 1, 1)),
        _Measured(11, margin=(4, 0, 1, 0), padding=(1, 0, 2, 0)),
        _Measured(19, margin=(2, 0, 0, 0)),
    ]
    for root, css_class in zip(roots, ("first", "middle", "last"), strict=True):
        root.add_class(css_class)
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        stream.scroll_to(y=12, animate=False)
        await pilot.pause()
        before = (stream.virtual_size, stream.scroll_offset)
        placeholder = (await window.evict_unit(window.unit_ids[1]))[0]
        await pilot.pause()
        assert (stream.virtual_size, stream.scroll_offset) == before
        assert placeholder.content_height == roots[1].height
        assert (
            placeholder.styles._base_styles.margin
            == roots[1].styles._base_styles.margin
        )
        restored = _Measured(11, margin=(4, 0, 1, 0), padding=(1, 0, 2, 0))
        restored.add_class("middle")
        with app.batch_update():
            await stream.mount(restored, before=placeholder)
            await placeholder.remove()
        await pilot.pause()
        assert (stream.virtual_size, stream.scroll_offset) == before


@pytest.mark.asyncio
async def test_adjacent_margin_collapse_and_multi_placement_unit() -> None:
    window = _window(3)
    app = _TranscriptApp()
    roots = [
        _Measured(4, margin=(1, 0, 6, 0)),
        _Measured(3, margin=(2, 0, 5, 0), padding=(1, 0, 1, 0)),
        _Measured(2, margin=(7, 0, 3, 0)),
        _Measured(5, margin=(1, 0, 2, 0)),
    ]
    for root, css_class in zip(
        roots, ("edge0", "edge1", "edge2", "edge3"), strict=True
    ):
        root.add_class(css_class)
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], roots[:1])
        window.register_mounted(window.unit_ids[1], roots[1:3])
        window.register_mounted(window.unit_ids[2], roots[3:])
        original = (stream.virtual_size, [root.region.y for root in roots])
        placeholders = await window.evict_unit(window.unit_ids[1])
        await pilot.pause()
        assert len(placeholders) == 2
        assert (
            stream.virtual_size,
            [root.region.y for root in (roots[0], *placeholders, roots[3])],
        ) == original
        assert all(isinstance(node, PlacementPlaceholder) for node in placeholders)


@pytest.mark.asyncio
async def test_standalone_call_and_result_evict_jointly() -> None:
    window = TranscriptWindow()
    window.admit([_file_edit_effect(0)], start_index=0)
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = window.build_unit(window.unit_ids[0], WeakKeyDictionary())
        assert [type(root) for root in roots] == [ToolCallMessage, ToolResultMessage]
        await stream.mount_all(roots)
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], roots)
        placeholders = await window.evict_unit(window.unit_ids[0])
        await pilot.pause()
        assert list(stream.children) == placeholders
        assert len(placeholders) == 2
        assert not window.units[window.unit_ids[0]].mounted_roots


@pytest.mark.asyncio
async def test_budget_hysteresis_and_exclusions() -> None:
    window = _window(8)
    app = _TranscriptApp()
    roots = [
        _Measured(height, margin=(0, 0, 0, 0)) for height in (8, 8, 30, 8, 8, 8, 8, 8)
    ]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        assert window.eviction_candidates(50, 5, low_mark=100, high_mark=120) == []
        window.units[window.unit_ids[0]].pin_reasons.add("manual")
        excluded = frozenset({window.unit_ids[1]})
        candidates = window.eviction_candidates(
            50,
            5,
            low_mark=8,
            high_mark=15,
            selected=excluded,
            live=frozenset({window.unit_ids[3]}),
            compaction_neighbors=frozenset({window.unit_ids[4]}),
        )
        assert not set(candidates) & {window.unit_ids[index] for index in range(5)}
        assert candidates
        assert set(candidates) <= set(window.unit_ids[5:])
        assert window.unit_ids[2] not in candidates
        assert window.eviction_candidates(50, 5, low_mark=8, high_mark=15)
        for unit_id in candidates:
            await window.evict_unit(unit_id)
        await pilot.pause()
        assert not window.eviction_candidates(
            50,
            5,
            low_mark=8,
            high_mark=15,
            selected=excluded,
            live=frozenset({window.unit_ids[3]}),
            compaction_neighbors=frozenset({window.unit_ids[4]}),
        )
        assert all(
            window.units[unit_id].mounted_roots for unit_id in window.unit_ids[:5]
        )


@pytest.mark.asyncio
async def test_cancelled_eviction_rolls_back_and_blocks_reentry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = _window(1)
    unit_id = window.unit_ids[0]
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        root = _Measured(12, margin=(0, 0, 0, 0))
        await stream.mount(root)
        await pilot.pause()
        window.register_mounted(unit_id, [root])
        original_mount_all = stream.mount_all
        entered = asyncio.Event()
        release = asyncio.Event()

        async def pause_after_mount(
            widgets: list[PlacementPlaceholder], *, before: Widget
        ) -> None:
            await original_mount_all(widgets, before=before)
            entered.set()
            await release.wait()

        monkeypatch.setattr(stream, "mount_all", pause_after_mount)
        task = asyncio.create_task(window.evict_unit(unit_id))
        await entered.wait()
        assert window.eviction_candidates(0, 10, low_mark=1, high_mark=2) == []
        with pytest.raises(ValueError, match="not evictable"):
            await window.evict_unit(unit_id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await pilot.pause()
        assert list(stream.children) == [root]
        assert window.units[unit_id].mounted_roots == [root]
        assert not window.units[unit_id].placeholders
        monkeypatch.setattr(stream, "mount_all", original_mount_all)
        assert len(await window.evict_unit(unit_id)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_twice", [False, True])
async def test_cancelled_eviction_restores_removed_root(
    monkeypatch: pytest.MonkeyPatch, cancel_twice: bool
) -> None:
    window = _window(1)
    unit_id = window.unit_ids[0]
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        root = _Measured(12, margin=(0, 0, 0, 0))
        await stream.mount(root)
        await pilot.pause()
        window.register_mounted(unit_id, [root])
        original_remove = root.remove
        original_mount_all = stream.mount_all
        removed = asyncio.Event()
        rollback_entered = asyncio.Event()
        resume_rollback = asyncio.Event()

        async def pause_after_remove() -> None:
            await original_remove()
            removed.set()
            await asyncio.Event().wait()

        async def pause_after_remount(
            widgets: list[Widget], *, before: Widget | None = None
        ) -> None:
            await original_mount_all(widgets, before=before)
            if root in widgets:
                rollback_entered.set()
                await resume_rollback.wait()

        monkeypatch.setattr(root, "remove", pause_after_remove)
        monkeypatch.setattr(stream, "mount_all", pause_after_remount)
        task = asyncio.create_task(window.evict_unit(unit_id))
        await removed.wait()
        assert root.parent is not stream
        task.cancel()
        await rollback_entered.wait()
        assert window.eviction_candidates(0, 10, low_mark=1, high_mark=2) == []
        if cancel_twice:
            task.cancel()
        resume_rollback.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await pilot.pause()
        assert list(stream.children) == [root]
        assert window.units[unit_id].mounted_roots == [root]
        assert not window.units[unit_id].placeholders
        monkeypatch.setattr(stream, "mount_all", original_mount_all)
        monkeypatch.setattr(root, "remove", original_remove)
        assert len(await window.evict_unit(unit_id)) == 1


@pytest.mark.asyncio
async def test_grouped_members_evict_together() -> None:
    window = TranscriptWindow()
    window.admit([_effect(0), _effect(1), _effect(2)], start_index=0)
    assert len(window.unit_ids) == 1
    unit = window.units[window.unit_ids[0]]
    assert unit.member_entry_ids == [f"effect-{i}" for i in range(3)]
    roots = window.build_unit(unit.id, WeakKeyDictionary())
    assert len(roots) == 1 and isinstance(roots[0], ToolGroup)
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        window.register_mounted(unit.id, roots)
        placeholders = await window.evict_unit(unit.id)
        await pilot.pause()
        assert list(stream.children) == placeholders
        assert unit.placeholders == placeholders
        assert not unit.mounted_roots


@pytest.mark.asyncio
async def test_width_change_invalidates_evicted_placement() -> None:
    window = _window(1)
    unit = window.units[window.unit_ids[0]]
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        original = _Measured(12, margin=(0, 0, 0, 0))
        await stream.mount(original)
        await pilot.pause()
        window.register_mounted(unit.id, [original])
        placeholder = (await window.evict_unit(unit.id))[0]
        await pilot.pause()
        old_height = placeholder.content_height
        old_width = placeholder.geometry_width
        stream.styles.width = 40
        await pilot.pause()
        assert placeholder.content_height is None
        assert (
            placeholder.get_content_height(Size(40, 10), Size(40, 10), old_width - 1)
            == 0
        )
        window.admit([_message(0)], start_index=0, geometry_width=40)
        assert unit.content_height is None
        assert placeholder.content_height is None
        assert old_height == 12
        restored = _Measured(18, margin=(0, 0, 0, 0))
        with app.batch_update():
            await stream.mount(restored, before=placeholder)
            await placeholder.remove()
        await pilot.pause()
        window.register_restored(unit.id, [restored])
        assert unit.content_height == 18
        assert unit.geometry_width == stream.size.width


@pytest.mark.asyncio
async def test_horizontal_margins_use_stream_width_baseline() -> None:
    window = _window(1)
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        original = _Measured(12, margin=(0, 4, 0, 3), padding=(1, 1, 1, 1))
        original.add_class("horizontal")
        await stream.mount(original)
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], [original])
        before = stream.virtual_size
        original_width = original.region.width
        original_content_width = original.content_region.width
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        await pilot.pause()
        assert placeholder.geometry_width == (
            original_width
            + original.styles._base_styles.margin.totals[0]
            - original.styles._base_styles.gutter.totals[0]
        )
        assert placeholder.geometry_width != original_content_width
        assert placeholder.content_height == 12
        assert stream.virtual_size == before
        assert (
            placeholder.get_content_height(
                stream.size, app.size, placeholder.geometry_width
            )
            == 12
        )
        assert placeholder.content_height == 12
        stream.styles.width = 40
        await pilot.pause()
        assert placeholder.content_height is None
        assert placeholder.get_content_height(stream.size, app.size, 39) == 0


@pytest.mark.asyncio
async def test_prefix_rebase_keeps_evicted_height() -> None:
    window = _window(1)
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        root = _Measured(12, margin=(0, 0, 0, 0))
        await stream.mount(root)
        await pilot.pause()
        unit = window.units[window.unit_ids[0]]
        window.register_mounted(unit.id, [root])
        placeholder = (await window.evict_unit(unit.id))[0]
        await pilot.pause()
        before = stream.virtual_size
        version = unit.geometry_version
        window.admit([_message(9)], start_index=0)
        await pilot.pause()
        assert unit.start_index == 1
        assert unit.geometry_version > version
        assert unit.content_height == 12
        assert placeholder.content_height == 12
        assert (
            placeholder.get_content_height(
                stream.size, app.size, placeholder.geometry_width
            )
            == 12
        )
        assert stream.virtual_size == before


@pytest.mark.asyncio
async def test_fold_invalidates_evicted_placeholder() -> None:
    state = ToolGroupExpansionState()
    window = TranscriptWindow(expansion_state=state)
    window.admit([_effect(0)], start_index=0)
    app = _TranscriptApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        unit = window.units[window.unit_ids[0]]
        roots = window.build_unit(unit.id, WeakKeyDictionary())
        await stream.mount_all(roots)
        await pilot.pause()
        window.register_mounted(unit.id, roots)
        placeholder = (await window.evict_unit(unit.id))[0]
        assert placeholder.content_height is not None
        assert unit.content_height is not None
        assert unit.group_key is not None
        state.set_collapsed(unit.group_key, False)
        assert window.sync_expansion() == {unit.id}
        assert unit.content_height is None
        assert placeholder.content_height is None


@pytest.mark.asyncio
async def test_scrolled_stream_uses_screen_space_viewport() -> None:
    window = _window(8)
    app = _TranscriptApp()
    roots = [_Measured(8, margin=(0, 0, 0, 0)) for _ in window.unit_ids]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        stream.scroll_to(y=24, animate=False)
        await pilot.pause()
        assert stream.scroll_offset.y == 24
        top = stream.region.y
        height = stream.region.height
        visible = {
            unit_id
            for unit_id, root in zip(window.unit_ids, roots, strict=True)
            if root.region.y < top + height and root.region.bottom > top
        }
        assert visible
        candidates = window.eviction_candidates(top, height, low_mark=8, high_mark=16)
        assert candidates
        assert not visible.intersection(candidates)
        assert candidates[0] == window.unit_ids[-1]  # Furthest in screen space.
        assert all(
            roots[window.unit_ids.index(unit_id)].region.y >= top + 2 * height
            or roots[window.unit_ids.index(unit_id)].region.bottom <= top - height
            for unit_id in candidates
        )


@pytest.mark.asyncio
async def test_budget_boundaries_and_exemptions() -> None:
    window = _window(3)
    app = _TranscriptApp()
    roots = [_Measured(500, margin=(0, 0, 0, 0)) for _ in window.unit_ids]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        # A single evictable placement lies outside the overscan area.
        candidates = lambda: window.eviction_candidates(0, 10)
        for heights, expected in [
            ((499, 500, 0), []),
            ((500, 500, 0), []),
            ((500, 500, 500), []),
            ((500, 500, 501), [window.unit_ids[2]]),
        ]:
            for root, height in zip(roots, heights, strict=True):
                root.height = height
                root.refresh(layout=True)
            await pilot.pause()
            assert candidates() == expected
        for root, height in zip(roots, (500, 500, 501), strict=True):
            root.height = height
            root.refresh(layout=True)
        await pilot.pause()
        window.units[window.unit_ids[0]].pin_reasons.add("manual")
        assert not candidates()
        window.units[window.unit_ids[0]].pin_reasons.clear()
        roots[0].height = 1501
        roots[0].refresh(layout=True)
        await pilot.pause()
        assert not candidates()  # An oversized placement does not trigger eviction.
        roots[0].height = 500
        roots[0].refresh(layout=True)
        await pilot.pause()
        assert window.eviction_candidates(
            0,
            10,
            selected=frozenset({window.unit_ids[0]}),
            live=frozenset({window.unit_ids[1]}),
        ) == [window.unit_ids[2]]
        assert not window.eviction_candidates(
            0,
            10,
            selected=frozenset({window.unit_ids[0]}),
            live=frozenset({window.unit_ids[1]}),
            compaction_neighbors=frozenset({window.unit_ids[2]}),
        )
        await window.evict_unit(window.unit_ids[2])
        await pilot.pause()
        assert window.units[window.unit_ids[2]].placeholders
        assert not candidates()  # Placeholders do not consume mounted-row budget.


def test_entry_reset_generation_invalidates_cached_units() -> None:
    state = EntryExpansionState()
    window = TranscriptWindow(entry_expansion_state=state)
    window.admit([_file_edit_effect(0)], start_index=0)
    unit = window.units[window.unit_ids[0]]
    state.set_collapsed("effect-0", False)
    assert window.sync_expansion() == {unit.id}
    unit.content_height = 12
    state.reset()
    assert window.sync_expansion() == {unit.id}
    assert unit.content_height is None
    assert state.reset_generation == 1


def test_interleaved_admission_rejected_without_mutating_registry() -> None:
    window = _window(3)
    version = window.geometry_version
    with pytest.raises(ValueError, match="interleaved"):
        window.admit([_message(0), _message(8), _message(2)], start_index=0)
    assert window.geometry_version == version
    assert window.unit_ids == ["message-0", "message-1", "message-2"]
    with pytest.raises(ValueError, match="interleaved"):
        window.admit([_message(9)], start_index=1)
    with pytest.raises(ValueError, match="interleaved"):
        window.admit([_message(2), _message(3)], start_index=1)
    assert window.unit_ids == ["message-0", "message-1", "message-2"]
    assert window._admitted_indices == {f"message-{i}": i for i in range(3)}


def test_all_new_overlap_and_append_holes_rejected() -> None:
    window = _window(1)
    version = window.geometry_version
    with pytest.raises(ValueError, match="interleaved"):
        window.admit([_message(9), _message(8)], start_index=-1)
    with pytest.raises(ValueError, match="interleaved"):
        window.admit([_message(9)], start_index=2)
    assert window._admitted_indices == {"message-0": 0}
    assert window.geometry_version == version
    window.admit([_message(9)], start_index=-1)
    window.admit([_message(8)], start_index=1)
    assert window._admitted_indices == {"message-9": -1, "message-0": 0, "message-8": 1}


def test_equal_content_replacement_reuses_geometry() -> None:
    window = _window(1)
    unit = window.units[window.unit_ids[0]]
    unit.content_height = 10
    version = window.geometry_version
    replacement = _message(0)
    window.admit([replacement], start_index=0)
    assert unit.entries[0] is replacement
    assert unit.content_height == 10
    assert window.geometry_version == version


def test_blank_group_uses_widget_visible_key_and_tracks_fold() -> None:
    state = ToolGroupExpansionState()
    window = TranscriptWindow(expansion_state=state)
    effect = _effect(0)
    effect.id = ""
    window.admit([effect], start_index=0)
    unit = window.units[window.unit_ids[0]]
    widget = window.build_unit(unit.id, WeakKeyDictionary())[0]
    assert isinstance(widget, ToolGroup)
    assert widget._key == unit.group_key == ToolGroupKey("")
    assert widget._key is not None
    unit.content_height = 10
    state.set_collapsed(widget._key, False)
    assert window.sync_expansion() == {unit.id}
    assert unit.content_height is None


def test_unmatched_expansion_changes_do_not_bump_geometry() -> None:
    entries = EntryExpansionState()
    groups = ToolGroupExpansionState()
    window = TranscriptWindow(entry_expansion_state=entries, expansion_state=groups)
    window.admit([_message(0)], start_index=0)
    version = window.geometry_version
    entries.set_collapsed("not-admitted", False)
    groups.set_collapsed(ToolGroupKey("not-admitted"), False)
    assert window.sync_expansion() == set()
    assert window.geometry_version == version


def test_dead_object_references_are_pruned() -> None:
    window = _window(1)
    transient = _message(4)
    window.admit([transient], start_index=1)
    key = id(transient)
    weak = ref(transient)
    assert key in window._object_ids
    replacement = _message(4)
    window.admit([replacement], start_index=1)
    del transient
    gc.collect()
    assert weak() is None
    assert key not in window._object_ids
