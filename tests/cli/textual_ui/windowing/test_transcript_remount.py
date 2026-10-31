from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.geometry import Size
from textual.widget import Widget

from chartreux.app_server.models import TextContentBlock
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.messages import StreamingMessageBase
from chartreux.cli.textual_ui.widgets.virtual_output import VirtualOutputText
from chartreux.cli.textual_ui.windowing.placeholder import PlacementPlaceholder
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from tests.cli.textual_ui.test_history_grouping import _message


class _App(App[None]):
    CSS = "#stream { width: 60; height: 10; layout: stream; overflow-y: auto; }"

    def compose(self) -> ComposeResult:
        yield VerticalScroll(id="stream")


class _Measured(Widget):
    def __init__(self, height: int) -> None:
        super().__init__()
        self.height = height

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        return self.height


async def _fixture(count: int = 8) -> tuple[TranscriptWindow, list[_Measured]]:
    window = TranscriptWindow()
    window.admit([_message(index) for index in range(count)], start_index=0)
    roots = [_Measured(8) for _ in range(count)]
    return window, roots


@pytest.mark.asyncio
async def test_delayed_initialization_and_exact_reading_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture()
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        stream.scroll_to(y=24, animate=False)
        await pilot.pause()
        anchor = roots[3].region.y
        for index, height in [(0, 13), (7, 15)]:
            unit_id = window.unit_ids[index]
            placeholder = (await window.evict_unit(unit_id))[0]
            await pilot.pause()
            started = asyncio.Event()
            release = asyncio.Event()
            restored = _Measured(0)
            monkeypatch.setattr(
                window, "build_unit", lambda *_, widget=restored: [widget]
            )

            async def initialize(
                _widgets: object,
                start: asyncio.Event = started,
                done: asyncio.Event = release,
                widget: _Measured = restored,
                final_height: int = height,
            ) -> None:
                start.set()
                await done.wait()
                widget.height = final_height
                widget.refresh(layout=True)

            task = asyncio.create_task(
                window.restore_unit(
                    unit_id,
                    stream,
                    WeakKeyDictionary(),
                    initialize=initialize,
                    follow_bottom=False,
                )
            )
            await started.wait()
            await pilot.pause()
            assert placeholder.parent is stream
            assert placeholder._reserved
            assert stream.scroll_offset.y > 0
            release.set()
            assert await task
            await pilot.pause()
            assert placeholder.parent is None
            assert roots[3].region.y == anchor
            assert window.units[unit_id].content_height == height


@pytest.mark.asyncio
async def test_markdown_init_must_complete_before_reservation_is_released(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(6)
    entry = _message(0)
    entry.content = [
        TextContentBlock(text="\n".join(f"paragraph {i}" for i in range(30)))
    ]
    window.admit([entry], start_index=0)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        stream.scroll_to(y=16, animate=False)
        await pilot.pause()
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        await pilot.pause()
        started = asyncio.Event()
        release = asyncio.Event()
        original_write = StreamingMessageBase.write_initial_content

        async def delayed_write(widget: StreamingMessageBase) -> None:
            started.set()
            await release.wait()
            await original_write(widget)

        monkeypatch.setattr(
            StreamingMessageBase, "write_initial_content", delayed_write
        )
        task = asyncio.create_task(
            window.restore_unit(
                window.unit_ids[0], stream, WeakKeyDictionary(), follow_bottom=False
            )
        )
        await started.wait()
        await pilot.pause()
        assert placeholder.parent is stream and placeholder._reserved
        assert stream.scroll_offset.y > 0
        release.set()
        assert await task
        mounted = window.units[window.unit_ids[0]].mounted_roots[0]
        assert isinstance(mounted, StreamingMessageBase)
        assert mounted._content_initialized
        assert placeholder.parent is None


@pytest.mark.asyncio
async def test_failed_remount_retry_and_supersession(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(2)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        unit_id = window.unit_ids[0]
        placeholder = (await window.evict_unit(unit_id))[0]
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(12)])

        async def fail(_widgets: object) -> None:
            raise RuntimeError("initialization failed")

        with pytest.raises(RuntimeError, match="initialization failed"):
            await window.restore_unit(
                unit_id,
                stream,
                WeakKeyDictionary(),
                follow_bottom=False,
                initialize=fail,
            )
        assert placeholder.parent is stream
        assert not window.units[unit_id].restoring
        assert not placeholder._reserved
        assert await window.restore_unit(
            unit_id, stream, WeakKeyDictionary(), follow_bottom=False
        )
        assert not window.units[unit_id].placeholders


@pytest.mark.asyncio
async def test_coalesced_reconcile_and_rapid_reversal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(4)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        await window.evict_unit(window.unit_ids[0])
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(8)])
        indices: WeakKeyDictionary[Widget, int] = WeakKeyDictionary()
        original_reconcile = window._reconcile_once
        passes = 0

        async def count_passes(
            parent: Widget,
            mapping: WeakKeyDictionary[Widget, int],
            _initialize: Callable[[Sequence[Widget]], Awaitable[None]] | None,
            follow_bottom: bool,
            low_mark: int,
            high_mark: int,
            selected: frozenset[str],
            live: frozenset[str],
            compaction_neighbors: frozenset[str],
        ) -> None:
            nonlocal passes
            passes += 1
            await original_reconcile(
                parent,
                mapping,
                _initialize,
                follow_bottom,
                low_mark,
                high_mark,
                selected,
                live,
                compaction_neighbors,
            )

        monkeypatch.setattr(window, "_reconcile_once", count_passes)
        before_swap = stream.virtual_size.height
        entered = asyncio.Event()
        release = asyncio.Event()

        async def initialize(_widgets: object) -> None:
            await pilot.pause()
            assert stream.virtual_size.height == before_swap
            entered.set()
            await release.wait()

        first = window.request_reconcile(
            stream, indices, initialize=initialize, follow_bottom=False
        )
        await entered.wait()
        assert window._reconcile_active
        assert window.units[window.unit_ids[0]].placeholders[0]._reserved
        stream.scroll_to(y=20, animate=False, immediate=True)
        second = window.request_reconcile(stream, indices, follow_bottom=False)
        stream.scroll_to(y=0, animate=False, immediate=True)
        third = window.request_reconcile(stream, indices, follow_bottom=False)
        stream.scroll_to(y=20, animate=False, immediate=True)
        last = window.request_reconcile(stream, indices, follow_bottom=False)
        assert len({first, second, third, last}) == 4
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        release.set()
        await asyncio.gather(first, third, last)
        await pilot.pause()
        assert passes == 2
        assert not window.units[window.unit_ids[0]].placeholders
        assert len(stream.children) == 4
        assert stream.scroll_offset.y == 20
        assert not window._reconcile_active


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["reset", "remove", "supersede"])
async def test_stale_completion_does_not_register_roots(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    window, roots = await _fixture(2)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        unit_id = window.unit_ids[0]
        placeholder = (await window.evict_unit(unit_id))[0]
        unit = window.units[unit_id]
        restored = _Measured(11)
        monkeypatch.setattr(window, "build_unit", lambda *_: [restored])
        entered = asyncio.Event()
        resume = asyncio.Event()

        async def initialize(_widgets: object) -> None:
            entered.set()
            await resume.wait()

        task = asyncio.create_task(
            window.restore_unit(
                unit_id,
                stream,
                WeakKeyDictionary(),
                follow_bottom=False,
                initialize=initialize,
            )
        )
        await entered.wait()
        if change == "reset":
            window.reset()
        elif change == "remove":
            window.remove_unit(unit_id)
        else:
            updated = _message(0)
            updated.content = [
                updated.content[0].model_copy(update={"text": "changed"})
            ]
            window.admit([updated], start_index=0)
        resume.set()
        assert not await task
        assert restored.parent is None
        assert placeholder.parent is (stream if change == "supersede" else None)
        assert not unit.restoring
        assert not placeholder._reserved
        if change == "supersede":
            assert unit.placeholders == [placeholder]
            assert await window.restore_unit(
                unit_id, stream, WeakKeyDictionary(), follow_bottom=False
            )


@pytest.mark.asyncio
async def test_follow_bottom_skips_reading_compensation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(5)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        await pilot.pause()
        stream.scroll_end(animate=False)
        await pilot.pause()
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(15)])
        monkeypatch.setattr(
            window,
            "_reading_anchor",
            lambda *_: pytest.fail("follow mode took an anchor"),
        )
        assert await window.restore_unit(
            window.unit_ids[0], stream, WeakKeyDictionary(), follow_bottom=True
        )
        await pilot.pause()
        assert placeholder.parent is None
        assert stream.is_vertical_scroll_end


@pytest.mark.asyncio
async def test_scroll_during_initialization_supersedes_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(6)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        await pilot.pause()
        stream.scroll_to(y=16, animate=False, immediate=True)
        restored = _Measured(13)
        monkeypatch.setattr(window, "build_unit", lambda *_: [restored])
        entered = asyncio.Event()
        release = asyncio.Event()

        async def initialize(_widgets: object) -> None:
            entered.set()
            await release.wait()

        task = asyncio.create_task(
            window.restore_unit(
                window.unit_ids[0],
                stream,
                WeakKeyDictionary(),
                initialize=initialize,
                follow_bottom=False,
            )
        )
        await entered.wait()
        stream.scroll_to(y=24, animate=False, immediate=True)
        release.set()
        assert await task
        assert placeholder.parent is None
        assert stream.scroll_offset.y == 24


@pytest.mark.asyncio
async def test_reading_mode_at_bottom_still_captures_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(5)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        await window.evict_unit(window.unit_ids[0])
        stream.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert stream.is_vertical_scroll_end
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(15)])
        original_anchor = window._reading_anchor
        anchors: list[tuple[str, int, int] | None] = []

        def record_anchor(parent: Widget) -> tuple[str, int, int] | None:
            anchor = original_anchor(parent)
            anchors.append(anchor)
            return anchor

        monkeypatch.setattr(window, "_reading_anchor", record_anchor)
        assert await window.restore_unit(
            window.unit_ids[0], stream, WeakKeyDictionary(), follow_bottom=False
        )
        assert anchors and anchors[0] is not None


@pytest.mark.asyncio
async def test_resize_and_offscreen_expansion_remeasure_on_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = EntryExpansionState()
    window = TranscriptWindow(entry_expansion_state=state)
    window.admit([_message(0), _message(1)], start_index=0, geometry_width=60)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        roots = [_Measured(10), _Measured(10)]
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        stream.styles.width = 40
        await pilot.pause()
        window.admit([_message(1)], start_index=1, geometry_width=stream.size.width)
        state.set_collapsed("message-0", False)
        assert window.sync_expansion() == {"message-0"}
        assert window.units["message-0"].content_height is None
        assert placeholder.content_height is None
        monkeypatch.setattr(
            window,
            "build_unit",
            lambda *_: [_Measured(20 if stream.size.width == 40 else 10)],
        )
        assert await window.restore_unit(
            "message-0", stream, WeakKeyDictionary(), follow_bottom=False
        )
        assert window.units["message-0"].content_height == 20


@pytest.mark.asyncio
async def test_rollback_failure_chains_original_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(1)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        window.register_mounted(window.unit_ids[0], roots)

        async def fail_remove() -> None:
            raise RuntimeError("original")

        async def fail_rollback(_placeholder: PlacementPlaceholder) -> None:
            raise RuntimeError("rollback")

        monkeypatch.setattr(roots[0], "remove", fail_remove)
        monkeypatch.setattr(PlacementPlaceholder, "remove", fail_rollback)
        with pytest.raises(RuntimeError, match="rollback") as exc:
            await window.evict_unit(window.unit_ids[0])
        assert isinstance(exc.value.__cause__, RuntimeError)
        assert str(exc.value.__cause__) == "original"


@pytest.mark.asyncio
@pytest.mark.parametrize("changing_passes", [5, 20])
async def test_reservation_settles_or_reaches_bound(
    monkeypatch: pytest.MonkeyPatch, changing_passes: int
) -> None:
    window, roots = await _fixture(2)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        restored = _Measured(8)
        monkeypatch.setattr(window, "build_unit", lambda *_: [restored])
        original_pass = window._layout_pass
        passes = 0

        async def advance(parent: Widget) -> None:
            nonlocal passes
            passes += 1
            if passes <= changing_passes and placeholder.parent is stream:
                assert placeholder._reserved
                restored.height += 1
                restored.refresh(layout=True)
            await original_pass(parent)

        monkeypatch.setattr(window, "_layout_pass", advance)
        assert await window.restore_unit(
            window.unit_ids[0], stream, WeakKeyDictionary(), follow_bottom=False
        )
        assert passes >= min(changing_passes + 1, 12)
        assert placeholder.parent is None
        assert not placeholder._reserved


@pytest.mark.asyncio
async def test_width_invalidation_preserves_virtual_output_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(2)
    window.admit([_message(1)], start_index=1, geometry_width=60)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        body = VirtualOutputText("\n".join(str(i) for i in range(6)))
        await roots[0].mount(body)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        placeholder = (await window.evict_unit(window.unit_ids[0]))[0]
        assert placeholder.exempt_body_height == 6
        window._sync_width(40)
        assert placeholder.content_height is None
        assert placeholder.get_content_height(Size(40, 10), Size(40, 10), 40) == 6
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(14)])
        assert await window.restore_unit(
            window.unit_ids[0], stream, WeakKeyDictionary(), follow_bottom=False
        )
        assert window.units[window.unit_ids[0]].content_height == 14


@pytest.mark.asyncio
async def test_initializer_can_restore_another_unit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, roots = await _fixture(2)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        for unit_id in window.unit_ids:
            await window.evict_unit(unit_id)
        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(8)])

        async def initialize(_widgets: object) -> None:
            assert await window.restore_unit(
                window.unit_ids[1], stream, WeakKeyDictionary(), follow_bottom=False
            )

        assert await window.restore_unit(
            window.unit_ids[0],
            stream,
            WeakKeyDictionary(),
            follow_bottom=False,
            initialize=initialize,
        )
        assert all(not unit.placeholders for unit in window.units.values())


@pytest.mark.asyncio
async def test_restore_scan_sees_an_eviction_before_its_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An eviction is a restore candidate before its placeholder is measured.

    ``evict_unit`` mounts the placeholder and returns; the layout that measures
    it is a later frame. A scroll arriving in between reaches the reconcile
    with a zero-width placeholder, which reads as row 0 above every
    scrolled-up neighbour and breaks the placement index's screen ordering --
    a bisected window then walks past the unit that was just evicted (the
    scroll round-trip never remounted it).
    """
    window, roots = await _fixture()
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        stream.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        unit_id = window.unit_ids[0]
        placeholder = (await window.evict_unit(unit_id))[0]
        assert placeholder.region.width == 0, (
            "the placeholder was laid out before the scroll: this scenario "
            "must exercise the unmeasured placement"
        )

        stream.scroll_to(y=0, animate=False, force=True, immediate=True)
        top, height = stream.region.y, stream.region.height
        assert unit_id in window._restore_candidates(top, height), (
            "the restore scan dropped the unit evicted this frame"
        )

        monkeypatch.setattr(window, "build_unit", lambda *_: [_Measured(8)])
        await window._reconcile_once(
            stream,
            WeakKeyDictionary(),
            None,
            False,
            1000,
            1500,
            frozenset(),
            frozenset(),
            frozenset(),
        )
        assert window.units[unit_id].mounted_roots
        assert not window.units[unit_id].placeholders


@pytest.mark.asyncio
async def test_unmeasured_placement_settles_after_the_layout() -> None:
    """The unmeasured-placement guard clears once the rows land.

    The full placeholder scan it selects is a fallback for the eviction frame
    only: a member that stays marked keeps the per-tick restore scan
    proportional to the registry's placeholder count.
    """
    window, roots = await _fixture(4)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        unit_id = window.unit_ids[0]
        placeholder = (await window.evict_unit(unit_id))[0]
        assert window._unmeasured_placements == {unit_id}
        assert window._unmeasured_placement_pending()
        await pilot.pause()
        assert placeholder.region.width > 0
        assert not window._unmeasured_placement_pending()
        assert window._unmeasured_placements == set()


@pytest.mark.asyncio
async def test_identity_adoption_carries_the_unmeasured_mark() -> None:
    """Adopting a public id moves the unmeasured mark with the unit.

    ``adopt_identity`` carries every shadow facet across the rename; a mark
    left on the transient id names a unit that no longer exists, so the
    pending check discards it and the restore scan windows past the still
    unmeasured placeholder -- the just-evicted unit is dropped again.
    """
    window = TranscriptWindow()
    entries = [_message(index) for index in range(64)]
    pending = entries[32]
    pending.id = ""
    window.admit(entries, start_index=0)
    transient_id = window.unit_ids[32]
    assert transient_id.startswith("transient-")
    roots = [_Measured(8) for _ in range(64)]
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        await pilot.pause()
        stream.scroll_to(y=0, animate=False)
        await pilot.pause()

        placeholder = (await window.evict_unit(transient_id))[0]
        assert placeholder.region.width == 0, (
            "the placeholder was laid out before the adoption: this scenario "
            "must exercise the unmeasured placement"
        )
        pending.id = "message-32"
        window.admit([pending], start_index=32)
        assert window.unit_ids[32] == "message-32"
        assert window._unmeasured_placements == {"message-32"}
        assert window._unmeasured_placement_pending()
        top, height = stream.region.y, stream.region.height
        assert "message-32" in window._restore_candidates(top, height), (
            "the restore scan lost the unit whose mark did not follow the rename"
        )


@pytest.mark.asyncio
async def test_removed_and_reset_units_forget_the_unmeasured_mark() -> None:
    """A mark must not outlive the unit it names.

    ``reset`` clears the registry, the placement index and the evicting set,
    and ``remove_unit`` drops a unit outright; a mark left behind keeps a
    stale id in the guard until the next restore scan drains it.
    """
    window, roots = await _fixture(4)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])

        removed = window.unit_ids[0]
        await window.evict_unit(removed)
        assert window._unmeasured_placements == {removed}
        window.remove_unit(removed)
        assert removed not in window._unmeasured_placements

        await window.evict_unit(window.unit_ids[0])
        assert window._unmeasured_placements
        window.reset()
        assert window._unmeasured_placements == set()
        assert not window._unmeasured_placement_pending()


def test_new_id_cannot_claim_known_index_and_width_invalidates_off_batch() -> None:
    window = TranscriptWindow()
    window.admit([_message(i) for i in range(4)], start_index=0, geometry_width=60)
    with pytest.raises(ValueError, match="interleaved"):
        window.admit([_message(1), _message(8), _message(3)], start_index=1)
    assert window._admitted_indices == {f"message-{i}": i for i in range(4)}
    off_batch = window.units[window.unit_ids[0]]
    off_batch.content_height = 8
    window.admit([_message(3)], start_index=3, geometry_width=40)
    assert off_batch.content_height is None
    assert off_batch.geometry_width == 40


@pytest.mark.asyncio
async def test_restore_marks_the_root_before_its_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restored root has no rows until the layout pass that measures it.

    ``restore_unit`` mounts the unit's real roots while its placeholders still
    hold the slot, so for that frame the unit's placement reads as row 0 --
    above every scrolled-up neighbour -- and the placement index stops being
    ordered by screen row, exactly as for a freshly mounted placeholder. An
    anchor captured in that window (load-more prepend, resize, picker open)
    bisects an out-of-order index and resolves far below the viewport.
    """
    window, roots = await _fixture(64)
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        await pilot.pause()
        for unit_id in window.unit_ids:
            await window.evict_unit(unit_id)
        await pilot.pause()
        assert not window._unmeasured_placement_pending()

        unit_id = window.unit_ids[32]
        restored = _Measured(8)
        # A root the stream never arranges stands in for the frame between the
        # mount and its measuring layout: display:none keeps its region empty.
        restored.styles.display = "none"
        monkeypatch.setattr(window, "build_unit", lambda *_: [restored])

        # The measuring layout is the frame after the mount; hold it back so
        # the probe observes the in-flight restore rather than its result.
        async def held_layout(_parent: object) -> None:
            return None

        monkeypatch.setattr(window, "_layout_pass", held_layout)
        observed: list[tuple[bool, object]] = []

        async def initialize(_widgets: object) -> None:
            assert window.units[unit_id].restoring
            assert restored.region.width == 0, (
                "the restored root was measured before the probe: this "
                "scenario must exercise the unmeasured placement"
            )
            observed.append((
                window._unmeasured_placement_pending(),
                window._reading_anchor(stream),
            ))

        assert await window.restore_unit(
            unit_id,
            stream,
            WeakKeyDictionary(),
            follow_bottom=False,
            initialize=initialize,
        )
        assert observed
        pending, anchor = observed[0]
        assert pending, "the in-flight restore left its unmeasured root unmarked"
        assert anchor == window._walk_anchor(stream.region.y)
        assert window._unmeasured_placement_pending()
        captured = window.capture_anchor(stream, following=False)
        assert captured is not None
        assert (captured.entry_id, captured.history_index, captured.row_offset) == (
            "message-0",
            0,
            0,
        )


@pytest.mark.asyncio
async def test_identity_adoption_during_restore_keeps_both_representatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rename mid-restore must not detach the unit's roots and placeholders.

    ``restore_unit`` mounts the real roots while the placeholders still hold
    the unit's slot, and admission adopts the entry's public id during any of
    the awaits in between (admission does not take the reconcile lock). The
    completion and rollback checks addressed the unit by its entry-time id, so
    the rename read as "the unit left the registry": the rollback removed both
    representatives from the DOM, the release and the mark were skipped, and
    the renamed unit kept a detached placement mid-index -- its slot never
    measured again and its placeholder unreachable for a retry.
    """
    window = TranscriptWindow()
    entries = [_message(index) for index in range(64)]
    pending = entries[32]
    pending.id = ""
    window.admit(entries, start_index=0)
    transient_id = window.unit_ids[32]
    assert transient_id.startswith("transient-")
    roots = [_Measured(8) for _ in range(64)]
    app = _App()
    async with app.run_test() as pilot:
        stream = app.query_one("#stream", VerticalScroll)
        await stream.mount_all(roots)
        await pilot.pause()
        for unit_id, root in zip(window.unit_ids, roots, strict=True):
            window.register_mounted(unit_id, [root])
        await pilot.pause()
        await window.evict_unit(transient_id)
        await pilot.pause()
        placeholder = window.units[transient_id].placeholders[0]
        assert not window._unmeasured_placement_pending()
        assert placeholder.parent is stream

        restored = _Measured(8)
        monkeypatch.setattr(window, "build_unit", lambda *_: [restored])

        async def adopt(_widgets: object) -> None:
            # The public id arrives between the mount and the completion: the
            # window where the restore holds no reconcile lock.
            assert window.units[transient_id].restoring
            pending.id = "message-32"
            window.admit([pending], start_index=32)
            assert window.unit_ids[32] == "message-32"
            assert "message-32" in window._unmeasured_placements, (
                "the restore's mark did not follow the adopted id"
            )

        # The rename re-fingerprints the entry, so the attempt aborts: the
        # roots it mounted were built for the pre-rename id.
        assert not await window.restore_unit(
            transient_id,
            stream,
            WeakKeyDictionary(),
            follow_bottom=False,
            initialize=adopt,
        )
        unit = window.units["message-32"]
        assert not unit.mounted_roots, "the aborted restore kept stale roots"
        assert restored.parent is None
        assert unit.placeholders == [placeholder]
        assert placeholder.parent is stream, (
            "the rollback detached the placeholder the unit still owns"
        )
        assert window._placement(unit) is placeholder
        for unit_id in window._placements:
            placement = window._placement(window.units[unit_id])
            assert placement is not None and placement.parent is stream, (
                f"{unit_id} kept a detached representative inside the index"
            )
        await pilot.pause()
        assert not window._unmeasured_placement_pending()
        assert window._unmeasured_placements == set()

        # The slot was never lost, so the retry restores the renamed unit.
        assert await window.restore_unit(
            "message-32", stream, WeakKeyDictionary(), follow_bottom=False
        ), "the adopted unit could not be restored again"
        assert not unit.placeholders
        assert unit.mounted_roots and unit.mounted_roots[0].parent is stream
        await pilot.pause()
        assert not window._unmeasured_placement_pending()
        assert window._unmeasured_placements == set()
        assert window._reading_anchor(stream) == window._walk_anchor(stream.region.y)
        # The restored slot is registered, so a second retry has nothing to do.
        assert not await window.restore_unit(
            "message-32", stream, WeakKeyDictionary(), follow_bottom=False
        )
