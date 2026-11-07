"""PKG-03 — resize settle: screen repaint, strip-cache clear, re-arm.

The v0.7.3 resize path has three coupled defects (see the accepted design
``rendering-repaint-fix-design.md``):

1. nothing repaints the cells a resize vacates — ``Screen``, ``#chat`` and
   ``#messages`` are all ``background: transparent`` (app.tcss:11-31,
   112-119), so a vacated cell keeps whatever the terminal held before it
   (the stale right strip and bottom band in the evidence screenshots);
2. a resize arriving while the single-flight reflow task runs is dropped,
   leaving a half-applied reflow until the next resize;
3. the width-keyed stream caches (``_SimpleListCanvas`` strips and
   ``SimpleListStreamEngine.layout_cache``) are only rebuilt on a new write,
   so a pure resize serves strips wrapped at the old width.

These tests assert the fix at cell granularity: every screen cell is claimed
and painted with the app conversation background after a resize settles, in
the plain path and in each deterministic storm case (empty transcript, picker
preview, and a resize interlaced with a streaming append).
"""

from __future__ import annotations

import asyncio

import pytest
from textual import events
from textual.geometry import Region, Size

from chartreux.app_server.models import TextContentBlock
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    SimpleListStreamEngine,
)
from tests.cli.textual_ui.test_history_grouping import _message
from tests.conftest import build_test_chartreux_app

SIMPLE_CHUNKS = ["Plan under construction:\n\n"] + [
    f"- Item {index:03d}: a streamed bullet whose text is long enough to wrap "
    "and force list-aware markdown layout on each flush\n"
    for index in range(8)
]


def _record_screen_repaints(monkeypatch: pytest.MonkeyPatch, app) -> list[object]:
    """Record every full-screen repaint the app requests on the resize path."""
    repaints: list[object] = []
    original = app.screen.refresh

    def record(*args, **kwargs):
        if not args and not kwargs.get("layout"):
            repaints.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(app.screen, "refresh", record)
    return repaints


def _assert_screen_repaint_requested(repaints: list[object]) -> None:
    assert repaints, (
        "the resize did not end with a screen-level repaint of the app "
        "conversation background"
    )


def _cell_backgrounds(app) -> list[list[object]]:
    """The composited background of every screen cell after the last frame."""
    rows: list[list[object]] = []
    for strip in app.screen._compositor.render_strips():
        row: list[object] = []
        for segment in strip:
            row.extend([segment.style.bgcolor] * segment.cell_length)
        rows.append(row)
    return rows


def _app_background(app):
    """The app conversation background, as a Rich color (what a cell carries)."""
    return app.styles.background.rich_color


def _scrollbar_regions(app) -> list[Region]:
    """Widgets that legitimately paint their own background over the screen."""
    chat = app._chat_widget
    return [chat.vertical_scrollbar.region] if chat.vertical_scrollbar.display else []


def _assert_every_cell_painted(app) -> None:
    """Cell coverage: no gaps, and every screen cell carries a background."""
    rows = _cell_backgrounds(app)
    size = app.size
    assert len(rows) == size.height
    for y, row in enumerate(rows):
        assert len(row) == size.width, (
            f"row {y} claims {len(row)} of {size.width} cells"
        )
        for x, background in enumerate(row):
            assert background is not None, f"cell ({x},{y}) is not painted"


def _assert_conversation_painted_with_app_background(app) -> None:
    """Every conversation cell is painted with the app conversation background."""
    rows = _cell_backgrounds(app)
    chat = app._chat_widget
    expected = _app_background(app)
    excluded = _scrollbar_regions(app)

    def owned(x: int, y: int) -> bool:
        return any(region.contains(x, y) for region in excluded)

    region = chat.region
    assert region.height > 0
    for y in range(region.y, region.bottom):
        for x in range(region.x, region.right):
            if owned(x, y):
                continue
            assert rows[y][x] == expected, (
                f"conversation cell ({x},{y}) painted {rows[y][x]}, expected {expected}"
            )


async def _settle(app, pilot) -> None:
    """Let a pending resize reflow settle, then let the repaint render."""
    task = app._resize_anchor_task
    if task is not None:
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
    await _render_pending_frames(pilot)


async def _settle_chain(app, pilot, limit: int = 8) -> None:
    """Await the settle task and any successor it arms at the reflow cap."""
    seen: list[asyncio.Task[None]] = []
    for _ in range(limit):
        task = app._resize_anchor_task
        if task is None or task.done():
            return
        if task in seen:
            raise AssertionError("the resize settle task was re-awaited")
        seen.append(task)
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
        await _render_pending_frames(pilot)
    raise AssertionError("the resize settle chain never quiesced")


async def _render_pending_frames(pilot) -> None:
    for _ in range(4):
        await pilot.pause(0.05)


def _rendered_lines(app, message: AssistantMessage) -> list[str]:
    """The message's rendered rows via the compositor (widget region cropped)."""
    strips = app.screen._compositor.render_strips(app.screen.size)
    region = message.region
    return [
        strips[y].text[region.x : region.right].rstrip()
        for y in range(region.y, min(region.bottom, app.screen.size.height))
    ]


def _completed_item_rows(engine: SimpleListStreamEngine) -> list[list[str]]:
    """The rendered rows of each completed item (the engine's strip cache)."""
    return [[strip.text for strip in entry.strips] for entry in engine._completed]


def _flat_rows(rows: list[list[str]]) -> list[str]:
    return [line for item in rows for line in item]


async def _await_engine_rendered(pilot, engine: SimpleListStreamEngine) -> None:
    """Wait for the fast path to engage and render its completed items."""
    for _ in range(10):
        if engine.mode == "fast" and all(entry.strips for entry in engine._completed):
            return
        await pilot.pause(0.05)
    raise AssertionError("the list engine never rendered its completed items")


def _fill_history(app, count: int = 12) -> None:
    history = app.app_server._state.projection.state.history
    assert history is not None
    history.extend(
        _message(i).model_copy(
            update={"content": [TextContentBlock(text=f"entry-{i} " * 90)]}
        )
        for i in range(count)
    )


@pytest.mark.asyncio
async def test_resize_settle_repaints_every_cell_with_app_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        _fill_history(app)
        await app._resume_history_from_messages()
        await pilot.pause()

        repaints = _record_screen_repaints(monkeypatch, app)
        await pilot.resize_terminal(64, 24)
        task = app._resize_anchor_task
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
        await _render_pending_frames(pilot)

        _assert_screen_repaint_requested(repaints)
        _assert_every_cell_painted(app)
        _assert_conversation_painted_with_app_background(app)


@pytest.mark.asyncio
async def test_rapid_double_resize_is_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        _fill_history(app)
        await app._resume_history_from_messages()
        await pilot.pause()

        transcript = app._transcript
        restores: list[object] = []
        original = transcript.restore_anchor

        async def counting_restore(messages_area, anchor):
            restores.append(anchor)
            return await original(messages_area, anchor)

        monkeypatch.setattr(transcript, "restore_anchor", counting_restore)

        # A reflow pass is in flight when the next resize arrives (the pilot's
        # own resize action without its pause, which would let the pass finish).
        app._transcript_presenter.handle_resize()
        in_flight = app._resize_anchor_task
        assert in_flight is not None and not in_flight.done()
        restores.clear()

        resized = Size(48, 20)
        driver = app._driver
        assert driver is not None
        driver._size = resized
        app.post_message(events.Resize(resized, resized))
        await _settle(app, pilot)

        assert app.size == resized
        assert len(restores) >= 2, (
            "the resize that arrived during the reflow was dropped: "
            f"{len(restores)} reflow pass(es) ran"
        )
        _assert_every_cell_painted(app)
        _assert_conversation_painted_with_app_background(app)


@pytest.mark.asyncio
async def test_pure_resize_rewraps_streamed_list_strips() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 30)) as pilot:
        await app._session_ready.wait()
        message = AssistantMessage("")
        await app._messages_area.mount(message)
        await pilot.pause()
        for chunk in SIMPLE_CHUNKS:
            await message.append_content(chunk)
            await message._flush_write_buffer()
            await pilot.pause()

        engine = message._list_engine
        assert engine is not None
        assert isinstance(engine, SimpleListStreamEngine)
        await _await_engine_rendered(pilot, engine)
        assert engine._cache_key is not None
        old_width = engine._cache_key[0]
        old_rows = _completed_item_rows(engine)
        parsed_before = list(engine.parse_bytes_log)

        await pilot.resize_terminal(50, 30)
        await _settle(app, pilot)

        # A pure resize parses no new bytes: the re-wrap comes from the cached
        # parsed content.
        assert engine.parse_bytes_log == parsed_before
        resized_rows = _completed_item_rows(engine)
        assert _flat_rows(resized_rows) != _flat_rows(old_rows), (
            f"the completed strips were still wrapped for the old width: {old_rows}"
        )
        # The strips are wrapped at the new width, so every completed item
        # renders exactly like a fresh render of the same document at that
        # width.
        reference = AssistantMessage("".join(SIMPLE_CHUNKS))
        await app._messages_area.mount(reference)
        await pilot.pause()
        await reference.write_initial_content()
        reference_engine = reference._list_engine
        assert reference_engine is not None
        await _await_engine_rendered(pilot, reference_engine)
        reference_rows = _completed_item_rows(reference_engine)
        shared = min(len(resized_rows), len(reference_rows))
        assert shared >= 5
        assert resized_rows[:shared] == reference_rows[:shared], (
            "the resize re-wrap does not match a fresh render at the new width:"
            f" {resized_rows[:shared]!r} != {reference_rows[:shared]!r}"
        )
        assert engine._cache_key is not None
        assert engine._cache_key[0] < old_width


@pytest.mark.asyncio
async def test_empty_transcript_resize_paints_app_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(90, 28)) as pilot:
        await app._session_ready.wait()
        await pilot.pause()
        assert not app._transcript.unit_ids
        assert app._resize_anchor_task is None

        await pilot.resize_terminal(60, 24)
        await _settle(app, pilot)

        # The repaint is not conditional on the reflow having run.
        repaints = _record_screen_repaints(monkeypatch, app)
        app._transcript_presenter.handle_resize()
        await _render_pending_frames(pilot)

        _assert_screen_repaint_requested(repaints)
        _assert_every_cell_painted(app)
        _assert_conversation_painted_with_app_background(app)


@pytest.mark.asyncio
async def test_picker_preview_resize_paints_app_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(90, 28)) as pilot:
        await app._session_ready.wait()
        _fill_history(app, count=4)
        await app._resume_history_from_messages()
        await pilot.pause()
        app._picker.previewing = True
        try:
            await pilot.resize_terminal(60, 24)
            await _settle(app, pilot)

            # A picker preview skips the anchor reflow; the repaint still runs.
            repaints = _record_screen_repaints(monkeypatch, app)
            app._transcript_presenter.handle_resize()
            assert app._resize_anchor_task is None
            await _render_pending_frames(pilot)

            _assert_screen_repaint_requested(repaints)
            _assert_every_cell_painted(app)
            _assert_conversation_painted_with_app_background(app)
        finally:
            app._picker.previewing = False


@pytest.mark.asyncio
async def test_resize_interlaced_with_streaming_append_paints_app_background(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(92, 30)) as pilot:
        await app._session_ready.wait()
        _fill_history(app, count=3)
        await app._resume_history_from_messages()
        await pilot.pause()

        message = AssistantMessage("")
        await app._messages_area.mount(message)
        await pilot.pause()
        await message.append_content(SIMPLE_CHUNKS[0])
        await message._flush_write_buffer()
        await pilot.pause()

        repaints = _record_screen_repaints(monkeypatch, app)
        await pilot.resize_terminal(58, 24)
        task = app._resize_anchor_task
        assert task is not None
        await asyncio.wait_for(asyncio.shield(task), timeout=5)
        await _render_pending_frames(pilot)
        await message.append_content(SIMPLE_CHUNKS[1])
        await message._flush_write_buffer()
        await _render_pending_frames(pilot)

        _assert_screen_repaint_requested(repaints)
        _assert_every_cell_painted(app)
        _assert_conversation_painted_with_app_background(app)
        engine = message._list_engine
        assert engine is not None
        new_width = message._get_markdown().content_region.width
        assert engine._cache_key is not None
        assert engine._cache_key[0] == new_width


def _drifting_passes(
    monkeypatch: pytest.MonkeyPatch,
    app,
    pilot,
    storm: list[Size],
    pass_sizes: list[Size],
    settle_tasks: list[asyncio.Task[None]],
) -> None:
    """Make every settle pass end on an injected resize from ``storm``.

    Ordered injected events only: a pass records its geometry, restores, then
    lands the next resize and awaits its screen update, so the passing loop
    observes drift on returning without any sleep or background timing task.
    """
    transcript = app._transcript
    pending = iter(storm)
    original = transcript.restore_anchor

    async def drifting_restore(messages_area, anchor):
        task = app._resize_anchor_task
        assert task is not None and not task.done(), (
            "the settle passes overlapped: a second settle task was armed "
            "while the first was still running"
        )
        pass_sizes.append(app.size)
        settle_tasks.append(task)
        result = await original(messages_area, anchor)
        if (size := next(pending, None)) is not None:
            await pilot.resize_terminal(size.width, size.height)
            assert app.size == size
        return result

    monkeypatch.setattr(transcript, "restore_anchor", drifting_restore)


@pytest.mark.asyncio
async def test_resize_drifting_through_the_reflow_cap_gets_a_successor_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A storm that ends on a drifting pass is re-armed once, not abandoned.

    Deterministic by construction: every pass of the settle task injects the
    next ordered resize and awaits its screen update, so each of the four
    bounded passes ends on changed geometry and the fourth (the cap) leaves
    the transcript reflowed for geometry the terminal has already left. Only
    a successor pass restores the anchor against the geometry the storm
    settled on; the storm list is exhausted before that pass, so the chain
    terminates without sleeps or background timing.
    """
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        _fill_history(app)
        await app._resume_history_from_messages()
        await pilot.pause()

        storm = [Size(92, 28), Size(84, 26), Size(76, 25), Size(68, 24)]
        pass_sizes: list[Size] = []
        settle_tasks: list[asyncio.Task[None]] = []
        _drifting_passes(monkeypatch, app, pilot, storm, pass_sizes, settle_tasks)

        await pilot.resize_terminal(97, 29)
        await _settle_chain(app, pilot)

        # Four bounded passes, each ending on drift, then the successor pass
        # at the geometry the storm settled on.
        assert pass_sizes == [Size(97, 29), *storm], (
            "the drift the last bounded pass observed was abandoned instead of "
            f"re-armed: {pass_sizes}"
        )
        assert app.size == storm[-1]
        assert pass_sizes[-1] == app.size
        # Single-flight survived the re-arm: one task per bounded chain.
        assert len(settle_tasks) == 5
        assert len({id(task) for task in settle_tasks}) == 2
        assert app._resize_anchor_task is not None
        assert app._resize_anchor_task.done()
        # The final pass reflowed at the settled width, so the settled
        # geometry is coherent and every cell is claimed and painted.
        assert app._chat_widget.region.width == app.size.width
        _assert_every_cell_painted(app)
        _assert_conversation_painted_with_app_background(app)


@pytest.mark.asyncio
async def test_resize_drifting_past_the_second_cap_keeps_repairing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drift that outlives the successor chain's own cap is still repaired.

    The settle slot is single-flight and each chain is bounded to four passes,
    but the repair is not: a chain that exhausts its passes on drift hands the
    slot to the next one, so a storm longer than two chains still ends with the
    anchor restored against the geometry the terminal settled on. The storm
    list is exhausted after eight passes, so the third chain's pass runs on
    stable geometry and the repair terminates.
    """
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        _fill_history(app)
        await app._resume_history_from_messages()
        await pilot.pause()

        storm = [
            Size(92, 28),
            Size(84, 26),
            Size(76, 25),
            Size(68, 24),
            Size(64, 22),
            Size(60, 21),
            Size(54, 20),
            Size(46, 19),
        ]
        pass_sizes: list[Size] = []
        settle_tasks: list[asyncio.Task[None]] = []
        _drifting_passes(monkeypatch, app, pilot, storm, pass_sizes, settle_tasks)

        await pilot.resize_terminal(97, 29)
        await _settle_chain(app, pilot)

        # Eight drifting passes across two bounded chains, then a third chain's
        # pass at the geometry the storm settled on.
        assert pass_sizes == [Size(97, 29), *storm], (
            "drift that outlived the second bounded chain was abandoned instead "
            f"of repaired: {pass_sizes}"
        )
        assert app.size == storm[-1]
        assert pass_sizes[-1] == app.size
        # Single-flight survived both re-arms: one live task per bounded chain.
        assert len(settle_tasks) == 9
        assert len({id(task) for task in settle_tasks}) == 3
        assert app._resize_anchor_task is not None
        assert app._resize_anchor_task.done()
        # The trailing pass reflowed at the settled width, so the final
        # geometry is coherent and every cell is claimed and painted.
        assert app._chat_widget.region.width == app.size.width
        _assert_every_cell_painted(app)
        _assert_conversation_painted_with_app_background(app)


@pytest.mark.asyncio
async def test_shutdown_does_not_rearm_the_reflow_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A drifting storm during teardown must not spawn a successor pass."""
    app = build_test_chartreux_app()
    async with app.run_test(size=(100, 30)) as pilot:
        await app._session_ready.wait()
        _fill_history(app)
        await app._resume_history_from_messages()
        await pilot.pause()

        storm = [Size(92, 28), Size(84, 26), Size(76, 25), Size(68, 24)]
        pass_sizes: list[Size] = []
        settle_tasks: list[asyncio.Task[None]] = []
        _drifting_passes(monkeypatch, app, pilot, storm, pass_sizes, settle_tasks)

        app._shutdown_started = True
        try:
            await pilot.resize_terminal(97, 29)
            await _settle_chain(app, pilot)
        finally:
            app._shutdown_started = False

        assert pass_sizes == [Size(97, 29), *storm[:-1]]
        assert len({id(task) for task in settle_tasks}) == 1
        assert app._resize_anchor_task is not None
        assert app._resize_anchor_task.done()
