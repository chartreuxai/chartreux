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
