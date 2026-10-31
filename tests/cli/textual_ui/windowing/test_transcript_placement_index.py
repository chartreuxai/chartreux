from __future__ import annotations

from bisect import bisect_left
from collections.abc import Iterator

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.widget import Widget

from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.cli.textual_ui.test_history_grouping import _message


class _IndexApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #stream { height: 12; width: 60; layout: stream; overflow-y: auto; }
    """

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")


class _Row(Widget):
    def __init__(self, height: int) -> None:
        super().__init__()
        self.row_height = height

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.row_height


class _WindowedApp(App[None]):
    """Stream below a header, so the viewport top is not screen row 0.

    ``_restore_candidates`` edges its window at ``top - height``; with the
    stream at row 0 that edge is negative and a zero-region placement reads as
    *below* it, so the probe could not reach the ordering break a zero-region
    tail causes.
    """

    CSS = """
    Screen { layout: vertical; }
    #header { height: 20; }
    #stream { height: 12; width: 60; layout: stream; overflow-y: auto; }
    """

    def compose(self) -> ComposeResult:
        yield Widget(id="header")
        yield VerticalScroll(id="stream")


def _ordered_lower_bound(window: TranscriptWindow, edge: int) -> int:
    """First index position whose placement ends below ``edge``, in content order."""
    index = window._placements
    for position, unit_id in enumerate(index):
        placement = window._placement(window.units[unit_id])
        assert placement is not None
        if placement.region.bottom > edge:
            return position
    return len(index)


def _walked_anchor(window: TranscriptWindow, top: int) -> tuple[str, int, int] | None:
    """Content-ordered walk over live placements, the index's oracle."""
    for unit_id in window.unit_ids:
        placement = window._placement(window.units[unit_id])
        if placement is not None and placement.region.bottom > top:
            offset = max(0, top - placement.region.y)
            return unit_id, offset, placement.region.y + offset
    return None


def _placed_units(window: TranscriptWindow) -> list[str]:
    return [
        unit_id
        for unit_id in window.unit_ids
        if window._placement(window.units[unit_id]) is not None
    ]


def _content_anchor(window: TranscriptWindow, top: int) -> tuple[str, int, int] | None:
    """Content-ordered (start_index) walk over live placements, the index oracle."""
    for unit_id in sorted(
        window.unit_ids, key=lambda key: window.units[key].start_index
    ):
        placement = window._placement(window.units[unit_id])
        if placement is not None and placement.region.bottom > top:
            offset = max(0, top - placement.region.y)
            return unit_id, offset, placement.region.y + offset
    return None


class _CountingIds(list[str]):
    """Registry id list that counts the traversals an anchor query performs."""

    def __init__(self, unit_ids: list[str]) -> None:
        super().__init__(unit_ids)
        self.visits = 0

    def __iter__(self) -> Iterator[str]:
        for unit_id in super().__iter__():
            self.visits += 1
            yield unit_id


@pytest.mark.asyncio
async def test_anchor_query_reads_live_rows_and_survives_scrolling() -> None:
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(5)], start_index=0)
    app = _IndexApp()
    rows = [_Row(4) for _ in range(5)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()
        index = window._placements
        assert list(index) == window.unit_ids

        stream.scroll_to(y=6, animate=False)
        await pilot.pause()
        top = stream.region.y
        scrolled = window.units[window.unit_ids[1]].mounted_roots[0]
        assert scrolled.region.y < top
        expected = _walked_anchor(window, top)

        window.unit_ids = counted = _CountingIds(window.unit_ids)
        counted.visits = 0
        # Same index, same answer, no registry walk: rows are read live from
        # the placements, so the anchor follows the scroll offset.
        assert window._placements is index
        assert window._reading_anchor(stream) == (
            window.unit_ids[1],
            top - scrolled.region.y,
            top,
        )
        assert window._reading_anchor(stream) == expected
        assert counted.visits == 0, "the anchor query walked the registry"

        stream.scroll_to(y=0, animate=False)
        await pilot.pause()
        top = stream.region.y
        counted.visits = 0
        assert window._placements is index
        assert window._reading_anchor(stream) == (window.unit_ids[0], 0, top)
        assert counted.visits == 0, "the anchor query walked the registry"


@pytest.mark.asyncio
async def test_anchor_query_matches_ordered_walk_across_scroll_eviction_and_removal() -> (
    None
):
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(6)], start_index=0)
    app = _IndexApp()
    rows = [_Row(4) for _ in range(6)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()

        for y in (0, 3, 8):
            stream.scroll_to(y=y, animate=False)
            await pilot.pause()
            assert window._reading_anchor(stream) == _walked_anchor(
                window, stream.region.y
            )

        evicted = window.unit_ids[1]
        await window.evict_unit(evicted)
        await pilot.pause()
        # The placeholder keeps the slot, so the index keeps the unit.
        assert evicted in window._placements
        assert window._placements == _placed_units(window)
        assert window._reading_anchor(stream) == _walked_anchor(window, stream.region.y)

        removed = window.unit_ids[0]
        window.remove_unit(removed)
        await pilot.pause()
        assert removed not in window._placements
        assert window._placements == _placed_units(window)
        assert window._reading_anchor(stream) == _walked_anchor(window, stream.region.y)


@pytest.mark.asyncio
async def test_prepended_placements_join_the_index_in_content_order() -> None:
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(3, 6)], start_index=3)
    app = _IndexApp()
    rows = [_Row(4) for _ in range(3)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()

        window.admit([_message(index) for index in range(3)], start_index=0)
        prefix_rows = [_Row(4) for _ in range(3)]
        await stream.mount_all(prefix_rows, before=rows[0])
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids[:3], prefix_rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()

        assert window.unit_ids[:3] == [f"message-{index}" for index in range(3)]
        assert window._placements == window.unit_ids
        stream.scroll_to(y=4, animate=False)
        await pilot.pause()
        assert window._reading_anchor(stream) == _walked_anchor(window, stream.region.y)


@pytest.mark.asyncio
async def test_placement_leaving_the_dom_falls_back_to_the_ordered_walk() -> None:
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(4)], start_index=0)
    app = _IndexApp()
    rows = [_Row(4) for _ in range(4)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()
        stale = window.unit_ids[0]
        window.units[stale].mounted_roots = []
        await pilot.pause()

        assert stale in window._placements
        assert window._reading_anchor(stream) == _walked_anchor(window, stream.region.y)


@pytest.mark.asyncio
async def test_replaced_member_start_index_move_reslots_the_placement_index() -> None:
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(4)], start_index=0)
    app = _IndexApp()
    rows = [_Row(4) for _ in range(4)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()
        assert window._placements == window.unit_ids

        # A re-admitted member can move its unit's start_index; the mounted
        # unit must re-slot in the content-ordered placement index, and the
        # reflowed row follows its new content position.
        unit = window.units["message-1"]
        window._replace_member(unit, "message-1", _message(1), index=10)
        assert unit.start_index == 10
        assert window._placements == [
            "message-0",
            "message-2",
            "message-3",
            "message-1",
        ]
        starts = [window.units[unit_id].start_index for unit_id in window._placements]
        assert starts == sorted(starts)
        # unit_ids re-slots too: the presenter's active-turn bisect resolves
        # against its start_index order, so a moved member must not leave it
        # unsorted.
        assert window.unit_ids == ["message-0", "message-2", "message-3", "message-1"]
        unit_starts = [window.units[unit_id].start_index for unit_id in window.unit_ids]
        assert unit_starts == sorted(unit_starts)
        first = bisect_left(
            window.unit_ids, 5, key=lambda key: window.units[key].start_index
        )
        assert window.unit_ids[first:] == ["message-1"]
        stream.move_child(rows[1], after=rows[3])
        await pilot.pause()

        for y in (0, 5, 9, 13):
            stream.scroll_to(y=y, animate=False)
            await pilot.pause()
            assert window._reading_anchor(stream) == _content_anchor(
                window, stream.region.y
            )


@pytest.mark.asyncio
async def test_anchor_ignores_a_placement_awaiting_its_layout() -> None:
    """An unmeasured placeholder must not steer the anchor off the viewport top.

    ``evict_unit`` mounts the placeholder before the layout that measures it,
    so for one frame the evicted slot reads as row 0 -- above every
    scrolled-up neighbour -- and the placement index stops being ordered by
    screen row. Bisecting that index resolves to a placement far below the
    viewport, and capturing it as the reading anchor scrolls the reader there
    once the unit remounts.
    """
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(64)], start_index=0)
    app = _IndexApp()
    rows = [_Row(8) for _ in range(64)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()
        stream.scroll_to(y=0, animate=False)
        await pilot.pause()

        placeholder = (await window.evict_unit(window.unit_ids[32]))[0]
        assert placeholder.region.width == 0, (
            "the placeholder was laid out before the query: this scenario "
            "must exercise the unmeasured placement"
        )
        top = stream.region.y
        assert window._reading_anchor(stream) == _walked_anchor(window, top)
        anchor = window.capture_anchor(stream, following=False)
        assert anchor is not None
        assert (anchor.entry_id, anchor.history_index, anchor.row_offset) == (
            "message-0",
            0,
            0,
        )


@pytest.mark.asyncio
async def test_identity_adoption_during_eviction_marks_the_public_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An adoption inside an awaited eviction must name the unit's current id.

    ``evict_unit`` adds its mark once the awaited DOM operations finish, and
    ``admit`` may adopt the entry's public id while one of them is in flight
    (admission does not take the reconcile lock). A mark captured under the
    entry-time id then names no unit, the pending check discards it, and the
    anchor bisects the unmeasured placeholder it was meant to guard.
    """
    window = TranscriptWindow()
    entries = [_message(index) for index in range(64)]
    pending = entries[32]
    pending.id = ""
    window.admit(entries, start_index=0)
    transient_id = window.unit_ids[32]
    assert transient_id.startswith("transient-")
    rows = [_Row(8) for _ in range(64)]
    app = _IndexApp()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()
        stream.scroll_to(y=0, animate=False)
        await pilot.pause()

        root = rows[32]
        remove = root.remove

        async def adopt_then_remove() -> None:
            # The public id arrives while the eviction is mid-flight.
            pending.id = "message-32"
            window.admit([pending], start_index=32)
            assert window.unit_ids[32] == "message-32"
            assert not window._unmeasured_placements
            await remove()

        monkeypatch.setattr(root, "remove", adopt_then_remove)
        placeholder = (await window.evict_unit(transient_id))[0]
        assert placeholder.region.width == 0, (
            "the placeholder was laid out before the query: this scenario "
            "must exercise the unmeasured placement"
        )
        assert window._unmeasured_placements == {"message-32"}, (
            "the eviction marked the entry-time id instead of the adopted one"
        )
        assert window._unmeasured_placement_pending()
        top = stream.region.y
        assert window._reading_anchor(stream) == _walked_anchor(window, top)
        anchor = window.capture_anchor(stream, following=False)
        assert anchor is not None
        assert (anchor.entry_id, anchor.history_index, anchor.row_offset) == (
            "message-0",
            0,
            0,
        )
        assert "message-32" in window._restore_candidates(top, stream.region.height), (
            "the restore scan lost the unit whose mark did not follow the rename"
        )


@pytest.mark.asyncio
async def test_zero_region_tail_block_keeps_the_anchor_and_the_restore_window() -> None:
    """A zero-region block at the index tail must not steer either bisect.

    A placement mounted ahead of its layout reads as row 0, i.e. *above* a
    scrolled-up viewport. With such a block at the tail the placement index
    stops being ordered by screen row, and the bisects drag far past the
    boundary they never visit: the anchor resolves to nothing although a row
    spans the viewport top, and the restore window collapses to nothing while
    a placeholder owns a slot inside it. Both consumers answer from the
    content-ordered walk instead.
    """
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(64)], start_index=0)
    app = _WindowedApp()
    rows = [_Row(8) for _ in range(64)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        stream.scroll_to(y=256, animate=False, immediate=True)
        await pilot.pause()

        # The unit spanning the viewport top is evicted and measured, so its
        # placeholder is the restore scan's target and every mark has drained.
        in_window = window.unit_ids[32]
        await window.evict_unit(in_window)
        await pilot.pause()
        assert not window._unmeasured_placement_pending()

        # A block at least half the index long joins the tail unmeasured: the
        # frame between a mount-ahead and the layout that measures it.
        window.admit([_message(index) for index in range(64, 128)], start_index=64)
        tail_rows = [_Row(8) for _ in range(64)]
        for row in tail_rows:
            row.styles.display = "none"
        await stream.mount_all(tail_rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids[64:], tail_rows, strict=True):
            window.register_mounted(unit_id, [row])
        assert all(row.region.width == 0 for row in tail_rows), (
            "the tail block was measured: this scenario must exercise the "
            "unmeasured placement"
        )
        assert not window._unmeasured_placement_pending(), (
            "an edge admission must not be the guarded case this test measures"
        )

        top, height = stream.region.y, stream.region.height
        assert top - height >= 0, (
            "the viewport must sit below its own height, or an unmeasured row "
            "reads as above the window edge instead of inside it"
        )
        assert window._reading_anchor(stream) == ("message-32", 0, top)
        assert window._reading_anchor(stream) == window._walk_anchor(top)
        anchor = window.capture_anchor(stream, following=False)
        assert anchor is not None
        assert (anchor.entry_id, anchor.history_index, anchor.row_offset) == (
            "message-32",
            32,
            0,
        )
        assert in_window in window._restore_candidates(top, height), (
            "the restore scan dropped the placeholder whose slot is on screen"
        )


@pytest.mark.asyncio
async def test_zero_region_prepended_block_keeps_the_windowed_lower_bound() -> None:
    """A zero-region block at the head must not steer the window bisect.

    ``_window_low`` bisects the first placement whose bottom passes
    ``top - height``. A prepended block held unmeasured reads as row 0, which
    sits *below* a negative edge: the non-monotone prefix then drags the
    search into the measured middle, past the placeholders the window has to
    restore. The answer must stay the content-ordered one.
    """
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(32, 96)], start_index=32)
    app = _IndexApp()
    rows = [_Row(8) for _ in range(64)]
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(rows)
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids, rows, strict=True):
            window.register_mounted(unit_id, [row])
        await pilot.pause()

        window.admit([_message(index) for index in range(32)], start_index=0)
        prefix_rows = [_Row(8) for _ in range(32)]
        for row in prefix_rows:
            row.styles.display = "none"
        await stream.mount_all(prefix_rows, before=rows[0])
        await pilot.pause()
        for unit_id, row in zip(window.unit_ids[:32], prefix_rows, strict=True):
            window.register_mounted(unit_id, [row])
        assert all(row.region.width == 0 for row in prefix_rows), (
            "the prepended block was measured: this scenario must exercise the "
            "unmeasured placement"
        )

        stream.scroll_to(y=200, animate=False, immediate=True)
        await pilot.pause()
        top, height = stream.region.y, stream.region.height
        edge = top - height
        assert edge < 0
        assert window._placements == window.unit_ids
        assert not window._unmeasured_placement_pending(), (
            "an edge admission must not be the guarded case this test measures"
        )
        assert window._window_low(window._placements, edge) == _ordered_lower_bound(
            window, edge
        )
        assert window._window_low(window._placements, edge) == 0, (
            "the zero-region head is the first row below the negated edge"
        )
        assert window._reading_anchor(stream) == window._walk_anchor(top)
