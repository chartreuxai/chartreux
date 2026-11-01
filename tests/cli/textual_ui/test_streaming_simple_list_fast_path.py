"""P5b — simple-list streaming fast path: scoping, equivalence, and cache keys.

SimpleListStreamEngine (chartreux/cli/textual_ui/widgets/messages.py) renders
parser-confirmed simple bullet lists incrementally: completed items are mounted
once and never reparsed while the fast path holds, and only the active (last)
item is reparsed per frame. Non-simple input (loose lists, code fences, nested
containers, late reference definitions) takes the whole-document path; each of
those fallback shapes has an equivalence fixture below asserting rendered-text
equality against a non-streamed reference at mid-stream prefixes and at
completion. The reference is a second AssistantMessage so both render at the
identical effective width.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.pilot import Pilot
from textual.widget import Widget
from textual.widgets import Static
from textual.widgets._markdown import MarkdownParagraph

from chartreux.cli.textual_ui.app import ChatScroll
from chartreux.cli.textual_ui.widgets.messages import (
    STREAM_WRITE_FRAME_CORRECTION_GAIN,
    STREAM_WRITE_FRAME_DEADBAND_SECONDS,
    STREAM_WRITE_FRAME_GAP_RESET_SECONDS,
    STREAM_WRITE_FRAME_SECONDS,
    STREAM_WRITE_FRAME_SECONDS_MAX,
    STREAM_WRITE_FRAME_SECONDS_MIN,
    AssistantMessage,
    SimpleListStreamEngine,
)

CSS_PATH = Path(__file__).parents[3] / "chartreux/cli/textual_ui/app.tcss"

SIMPLE_CHUNKS = ["Plan under construction:\n\n"] + [
    f"- Item {index:03d}: a streamed bullet whose text is long enough to wrap "
    "and force list-aware markdown layout on each flush\n"
    for index in range(12)
]

# One fixture per fallback shape: each must take the whole-document path.
FALLBACK_FIXTURES: dict[str, list[str]] = {
    "loose_list": ["- alpha\n", "\n", "- beta\n", "- gamma\n"],
    "code_fence": [
        "- alpha\n",
        "- beta\n",
        "\n",
        "```python\n",
        "print('hi')\n",
        "```\n",
        "- gamma\n",
    ],
    "nested_container": ["- alpha\n", "  - nested bullet\n", "- beta\n"],
    "late_reference_definition": [
        "- see [the docs][ref] for details\n",
        "\n",
        "[ref]: https://example.com\n",
    ],
}


class _ListApp(App[None]):
    CSS_PATH = CSS_PATH

    def compose(self) -> ComposeResult:
        yield Vertical(id="history")


class _ChatApp(App[None]):
    CSS_PATH = CSS_PATH

    def compose(self) -> ComposeResult:
        with ChatScroll(id="chat"):
            yield Static("\n".join(f"filler line {i}" for i in range(100)))


def _rendered_lines(app: App[Any], widget: Widget) -> list[str]:
    """The widget's rendered text via the compositor (widget-region cropped)."""
    strips = app.screen._compositor.render_strips(app.screen.size)
    region = widget.region
    return [
        strips[y].text[region.x : region.right].rstrip()
        for y in range(region.y, min(region.bottom, app.screen.size.height))
    ]


async def _mount_pair(
    app: _ListApp, pilot: Pilot[Any]
) -> tuple[AssistantMessage, AssistantMessage]:
    """Mount a non-streamed reference message first, then the streamed one."""
    history = app.query_one("#history", Vertical)
    ref_msg = AssistantMessage("")
    await history.mount(ref_msg)
    msg = AssistantMessage("")
    await history.mount(msg)
    await pilot.pause()
    return msg, ref_msg


def _markdown_of(msg: AssistantMessage):
    markdown = msg._markdown
    assert markdown is not None
    return markdown


async def _stream_chunks(
    pilot: Pilot[Any],
    msg: AssistantMessage,
    ref_msg: AssistantMessage,
    chunks: list[str],
) -> list[bool]:
    """Stream chunk by chunk, flushing each frame; compare against the
    non-streamed reference at every mid-stream prefix and at completion.
    """
    app = pilot.app
    ref_markdown = _markdown_of(ref_msg)
    markdown = _markdown_of(msg)
    results: list[bool] = []
    for index, chunk in enumerate(chunks):
        await msg.append_content(chunk)
        await msg._flush_write_buffer()
        await pilot.pause(0.02)
        prefix = "".join(chunks[: index + 1])
        await ref_markdown.update(prefix)
        await pilot.pause(0.02)
        results.append(
            _rendered_lines(app, markdown) == _rendered_lines(app, ref_markdown)
        )
    await msg.stop_stream()
    await pilot.pause(0.05)
    await ref_markdown.update("".join(chunks))
    await pilot.pause(0.05)
    results.append(_rendered_lines(app, markdown) == _rendered_lines(app, ref_markdown))
    return results


def _engine(msg: AssistantMessage) -> SimpleListStreamEngine:
    engine = msg._list_engine
    assert engine is not None
    return engine


@pytest.mark.asyncio
async def test_simple_list_fast_path_matches_reference_at_every_prefix() -> None:
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, ref_msg = await _mount_pair(app, pilot)
        results = await _stream_chunks(pilot, msg, ref_msg, SIMPLE_CHUNKS)

        assert all(results), "rendered text diverged from the non-streamed reference"
        engine = _engine(msg)
        assert engine.mode == "fast"
        assert engine.fallback_count == 0
        assert engine._item_count == len(SIMPLE_CHUNKS) - 1
        assert msg.get_content() == "".join(SIMPLE_CHUNKS)
        assert _markdown_of(msg)._markdown == "".join(SIMPLE_CHUNKS)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", sorted(FALLBACK_FIXTURES))
async def test_fallback_shapes_take_whole_document_path_and_match_reference(
    shape: str,
) -> None:
    chunks = FALLBACK_FIXTURES[shape]
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, ref_msg = await _mount_pair(app, pilot)
        results = await _stream_chunks(pilot, msg, ref_msg, chunks)

        assert all(results), f"{shape}: rendered text diverged from the reference"
        engine = _engine(msg)
        assert engine.mode == "fallback", f"{shape}: expected the whole-document path"
        assert engine.fallback_count >= 1
        assert _markdown_of(msg)._markdown == "".join(chunks)


@pytest.mark.asyncio
async def test_reparse_scoped_to_active_item_while_fast_path_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from textual.widgets import Markdown

    append_calls = 0
    original_append = Markdown.append

    def counting_append(self: Markdown, markdown: str) -> object:
        nonlocal append_calls
        append_calls += 1
        return original_append(self, markdown)

    monkeypatch.setattr(Markdown, "append", counting_append)

    updates: dict[int, int] = {}
    original_update = MarkdownParagraph._update_from_block

    async def counting_update(self: MarkdownParagraph, block: object) -> None:
        updates[id(self)] = updates.get(id(self), 0) + 1
        await original_update(self, block)  # type: ignore[arg-type]

    monkeypatch.setattr(MarkdownParagraph, "_update_from_block", counting_update)

    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, _ref_msg = await _mount_pair(app, pilot)
        append_calls_at_engagement: int | None = None
        before_last: dict[int, int] = {}
        for chunk in SIMPLE_CHUNKS:
            before_last = dict(updates)
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
            engine = msg._list_engine
            if (
                engine is not None
                and engine.mode == "fast"
                and append_calls_at_engagement is None
            ):
                append_calls_at_engagement = append_calls
        await msg.stop_stream()
        await pilot.pause(0.05)

        engine = _engine(msg)
        assert engine.mode == "fast"
        # Once engaged, the fast path never reparses through Markdown.append
        # (the single pre-engagement intro chunk may use the plain stream).
        assert append_calls_at_engagement is not None
        assert append_calls == append_calls_at_engagement <= 1
        # Per-flush parse volume stays at item size, not document size: a
        # flush parses at most the finalized item, the new item, and the
        # active item again — three item-sized ranges, never the document.
        per_flush = engine.parse_bytes_log
        max_item_bytes = max(len(chunk.encode("utf-8")) for chunk in SIMPLE_CHUNKS[1:])
        assert per_flush, "the fast path recorded no per-flush parse volume"
        assert max(per_flush[1:]) <= 3 * max_item_bytes + 8
        assert sum(per_flush) <= 4 * len("".join(SIMPLE_CHUNKS).encode("utf-8"))
        # In the final flush only the finalizing and active items' paragraphs
        # were re-rendered — never an already-completed item.
        active_para = cast(MarkdownParagraph | None, engine._active_para)
        assert active_para is not None
        touched = [
            pid for pid, count in updates.items() if count > before_last.get(pid, 0)
        ]
        assert id(active_para) in touched
        assert len(touched) <= 2


@pytest.mark.asyncio
async def test_completed_item_identities_stable_across_invalidation() -> None:
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, _ref_msg = await _mount_pair(app, pilot)
        for chunk in SIMPLE_CHUNKS[:6]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        engine = _engine(msg)
        completed = list(engine._completed)
        assert len(completed) == 4
        assert engine._item_count == 5
        # Stream the rest: the active item is invalidated on every flush.
        for chunk in SIMPLE_CHUNKS[6:]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        await msg.stop_stream()
        await pilot.pause(0.05)
        final_completed = engine._completed
        # Completed cache entries keep their identity across every later
        # flush — they are appended, never replaced or re-created — and each
        # still carries the strips it was rendered to.
        assert final_completed[: len(completed)] == completed
        assert len(final_completed) == len(SIMPLE_CHUNKS) - 2
        assert all(entry.strips for entry in final_completed)


@pytest.mark.asyncio
async def test_layout_cache_re_renders_completed_items_on_width_change() -> None:
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, ref_msg = await _mount_pair(app, pilot)
        for chunk in SIMPLE_CHUNKS[:6]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        engine = _engine(msg)
        key_before = engine.current_layout_key()
        invalidations_before = engine.cache_invalidations
        await pilot.resize_terminal(44, 120)
        await pilot.pause(0.05)
        for chunk in SIMPLE_CHUNKS[6:]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        await msg.stop_stream()
        await pilot.pause(0.05)

        key_after = engine.current_layout_key()
        assert key_after[0] != key_before[0], "effective width must key the cache"
        assert key_after[0] == _markdown_of(msg).content_region.width
        assert engine.cache_invalidations > invalidations_before
        assert key_after in engine.layout_cache
        cached = engine.layout_cache[key_after]
        final_completed = engine._completed
        # The cache holds the completed items as of the key change, and later
        # flushes only append after them (identities stay stable).
        assert final_completed[: len(cached)] == cached
        assert len(cached) < len(final_completed)
        # Completed items re-rendered correctly at the new width.
        await _markdown_of(ref_msg).update("".join(SIMPLE_CHUNKS))
        await pilot.pause(0.05)
        assert _rendered_lines(app, _markdown_of(msg)) == _rendered_lines(
            app, _markdown_of(ref_msg)
        )


@pytest.mark.asyncio
async def test_layout_cache_re_renders_completed_items_on_theme_change() -> None:
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, ref_msg = await _mount_pair(app, pilot)
        for chunk in SIMPLE_CHUNKS[:6]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        engine = _engine(msg)
        key_before = engine.current_layout_key()
        invalidations_before = engine.cache_invalidations
        app.theme = "textual-light"
        await pilot.pause(0.05)
        for chunk in SIMPLE_CHUNKS[6:]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        await msg.stop_stream()
        await pilot.pause(0.05)

        key_after = engine.current_layout_key()
        assert key_after[1] != key_before[1], (
            "theme/style generation must key the cache"
        )
        assert key_after[0] == key_before[0]
        assert engine.cache_invalidations > invalidations_before
        await _markdown_of(ref_msg).update("".join(SIMPLE_CHUNKS))
        await pilot.pause(0.05)
        assert _rendered_lines(app, _markdown_of(msg)) == _rendered_lines(
            app, _markdown_of(ref_msg)
        )


@pytest.mark.asyncio
async def test_bottom_follow_preserved_only_when_already_at_bottom() -> None:
    app = _ChatApp()
    async with app.run_test(size=(60, 24)) as pilot:
        chat = app.query_one("#chat", ChatScroll)
        msg = AssistantMessage("")
        await chat.mount(msg)
        await pilot.pause()
        chat.scroll_end(animate=False)
        chat.anchor()
        await pilot.pause(0.05)
        assert chat.is_at_bottom

        for chunk in SIMPLE_CHUNKS[:4]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        markdown = _markdown_of(msg)
        assert markdown._markdown == "".join(SIMPLE_CHUNKS[:4])
        assert chat.is_at_bottom

        # Scroll away (as a user would): the anchor is released.
        chat.release_anchor()
        chat.scroll_to(y=0, animate=False)
        await pilot.pause(0.05)
        assert not chat.is_at_bottom
        for chunk in SIMPLE_CHUNKS[4:8]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        # Chunks buffer without rendering and without forcing a scroll back.
        assert markdown._markdown == "".join(SIMPLE_CHUNKS[:4])
        assert msg._to_write_buffer == "".join(SIMPLE_CHUNKS[4:8])
        assert not chat.is_at_bottom

        chat.scroll_end(animate=False)
        chat.anchor()
        await pilot.pause(0.05)
        assert chat.is_at_bottom
        for chunk in SIMPLE_CHUNKS[8:]:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        await msg.stop_stream()
        await pilot.pause(0.05)
        assert markdown._markdown == "".join(SIMPLE_CHUNKS)
        assert msg.get_content() == "".join(SIMPLE_CHUNKS)


@pytest.mark.asyncio
async def test_non_list_content_stays_on_the_plain_stream_path() -> None:
    chunks = ["Just prose, ", "no list markers ", "at all.\n", "Second paragraph.\n"]
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, ref_msg = await _mount_pair(app, pilot)
        results: list[bool] = []
        for index, chunk in enumerate(chunks):
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
            assert msg._stream is not None  # the plain stream carries prose
            prefix = "".join(chunks[: index + 1])
            await _markdown_of(ref_msg).update(prefix)
            await pilot.pause(0.02)
            results.append(
                _rendered_lines(app, _markdown_of(msg))
                == _rendered_lines(app, _markdown_of(ref_msg))
            )
        await msg.stop_stream()
        await pilot.pause(0.05)

        assert all(results)
        engine = msg._list_engine
        assert engine is None or engine.mode == "undecided"
        assert _markdown_of(msg)._markdown == "".join(chunks)


def test_adaptive_cadence_constants_unchanged() -> None:
    """P5b lands no cadence change: the existing target constants stay as-is."""
    assert STREAM_WRITE_FRAME_SECONDS == 0.05
    assert STREAM_WRITE_FRAME_SECONDS_MIN == 0.025
    assert STREAM_WRITE_FRAME_SECONDS_MAX == 0.15
    assert STREAM_WRITE_FRAME_CORRECTION_GAIN == 0.5
    assert STREAM_WRITE_FRAME_DEADBAND_SECONDS == 0.005
    assert STREAM_WRITE_FRAME_GAP_RESET_SECONDS == 0.25


@pytest.mark.asyncio
async def test_fast_path_keeps_frame_delay_within_the_cadence_clamp() -> None:
    app = _ListApp()
    async with app.run_test(size=(60, 120)) as pilot:
        msg, _ref_msg = await _mount_pair(app, pilot)
        for chunk in SIMPLE_CHUNKS:
            await msg.append_content(chunk)
            await msg._flush_write_buffer()
            await pilot.pause(0.02)
        await msg.stop_stream()
        await pilot.pause(0.05)
        assert STREAM_WRITE_FRAME_SECONDS_MIN <= msg._frame_delay
        assert msg._frame_delay <= STREAM_WRITE_FRAME_SECONDS_MAX
