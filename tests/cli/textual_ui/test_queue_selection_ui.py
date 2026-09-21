from __future__ import annotations

import asyncio
import time
from weakref import WeakKeyDictionary

import pytest

from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class _BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([[mock_llm_chunk(content="done")]] * 8)
        self.started = [asyncio.Event() for _ in range(8)]
        self.release = [asyncio.Event() for _ in range(8)]
        self.calls = 0

    async def complete(self, **kwargs):
        index = self.calls
        self.calls += 1
        self.started[index].set()
        await self.release[index].wait()
        return await super().complete(**kwargs)


_BACKENDS: WeakKeyDictionary[ChartreuxApp, _BlockingBackend] = WeakKeyDictionary()


@pytest.fixture
def chartreux_app() -> ChartreuxApp:
    backend = _BlockingBackend()
    app = build_test_chartreux_app(agent_loop=build_test_agent_loop(backend=backend))
    _BACKENDS[app] = backend
    return app


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await pilot.pause(0.05)
        if predicate():
            return True
    return False


async def _enqueue_prompt(pilot, app: ChartreuxApp, text: str) -> None:
    chat_input = app.query_one(ChatInputContainer)
    chat_input.value = text
    await pilot.press("enter")
    await pilot.pause(0.05)


async def _start_bash_and_wait_busy(pilot, app: ChartreuxApp) -> None:
    chat_input = app.query_one(ChatInputContainer)
    chat_input.value = "keep the turn active"
    await pilot.press("enter")
    assert await _wait_until(pilot, _BACKENDS[app].started[0].is_set, timeout=2.0)


@pytest.mark.asyncio
async def test_up_enters_selection_mode_when_queue_nonempty(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "first queued")
        await _enqueue_prompt(pilot, chartreux_app, "second queued")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)

        assert body._queue_cursor >= 0
        assert body.input_widget is not None
        assert body.input_widget._queue_selection_active
        assert body.input_widget.read_only


@pytest.mark.asyncio
async def test_up_down_navigates_queued_items(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "first")
        await _enqueue_prompt(pilot, chartreux_app, "second")
        await _enqueue_prompt(pilot, chartreux_app, "third")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 0

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 1

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 2

        await pilot.press("down")
        await pilot.pause(0.1)
        assert body._queue_cursor == 1

        await pilot.press("down")
        await pilot.pause(0.1)
        assert body._queue_cursor == 0


@pytest.mark.asyncio
async def test_down_past_newest_exits_selection_mode(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "only item")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor >= 0

        await pilot.press("down")
        await pilot.pause(0.1)
        assert body._queue_cursor < 0
        assert body.input_widget is not None
        assert not body.input_widget._queue_selection_active
        assert not body.input_widget.read_only


@pytest.mark.asyncio
async def test_escape_exits_selection_mode(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "queued item")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor >= 0

        await pilot.press("escape")
        await pilot.pause(0.1)
        assert body._queue_cursor < 0
        assert body.input_widget is not None
        assert not body.input_widget._queue_selection_active


@pytest.mark.asyncio
async def test_enter_from_selection_enters_edit_mode(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "edit me")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)

        await pilot.press("enter")
        await pilot.pause(0.1)

        assert body._queue_in_edit_mode
        assert body.input_widget is not None
        assert body.input_widget._queue_edit_active
        assert not body.input_widget.read_only
        assert "edit me" in body.input_widget.text


@pytest.mark.asyncio
async def test_escape_from_edit_returns_to_selection(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "edit me")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert body._queue_in_edit_mode

        await pilot.press("escape")
        await pilot.pause(0.1)

        assert not body._queue_in_edit_mode
        assert body._queue_cursor >= 0
        assert body.input_widget is not None
        assert body.input_widget._queue_selection_active
        assert body.input_widget.read_only


@pytest.mark.asyncio
async def test_enter_in_edit_mode_updates_queued_item(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "original text")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert body.input_widget is not None
        body.input_widget.clear_text()
        body.input_widget.load_text("edited text")
        await pilot.pause(0.05)

        await pilot.press("enter")
        await pilot.pause(0.1)

        assert not body._queue_in_edit_mode
        assert body._queue_cursor >= 0

        items = chartreux_app._queue.queue_item_texts()
        assert any("edited text" == content for _, content in items)


@pytest.mark.asyncio
async def test_edit_keeps_target_when_an_earlier_item_leaves_queue(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "oldest")
        await _enqueue_prompt(pilot, chartreux_app, "target")
        await _enqueue_prompt(pilot, chartreux_app, "newest")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert body._queue_in_edit_mode
        assert body.input_widget is not None
        assert body.input_widget.text == "target"

        # Simulate FIFO drain removing an older item while this one is edited.
        assert await chartreux_app._queue.pop_at(0)
        assert [content for _, content in chartreux_app._queue.queue_item_texts()] == [
            "target",
            "newest",
        ]

        body.input_widget.clear_text()
        body.input_widget.load_text("target edited")
        await pilot.press("enter")
        await pilot.pause(0.2)

        assert [content for _, content in chartreux_app._queue.queue_item_texts()] == [
            "target edited",
            "newest",
        ]


@pytest.mark.asyncio
async def test_queue_index_of_widget_resolves_by_identity_not_highlight(
    chartreux_app: ChartreuxApp,
) -> None:
    # The edit-submit handler must re-resolve the edited item by the widget it
    # captured at submit time, not the live highlight: after submit the body is
    # back in selection mode, so Up/Down can move _queue_selected_widget away
    # from the edited item while _prepare_prompt_or_abort is awaiting. Verifies
    # _queue_index_of_widget tracks the passed widget, ignoring the highlight.
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "oldest")
        await _enqueue_prompt(pilot, chartreux_app, "newest")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 0  # highlighting newest
        newest_widget = chartreux_app._queue_selected_widget
        assert newest_widget is not None
        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 1  # highlighting oldest
        oldest_widget = chartreux_app._queue_selected_widget
        assert oldest_widget is not None
        assert newest_widget is not oldest_widget

        # Resolving the newest widget ignores that the highlight points at oldest.
        assert chartreux_app._queue_index_of_widget(newest_widget) is not None
        assert chartreux_app._queue_index_of_widget(oldest_widget) is not None
        assert chartreux_app._queue_index_of_widget(newest_widget) != (
            chartreux_app._queue_index_of_widget(oldest_widget)
        )
        assert chartreux_app._queue_index_of_widget(None) is None
        assert chartreux_app._queue_index_of_widget(object()) is None  # unknown widget


@pytest.mark.asyncio
async def test_backspace_deletes_selected_item(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "first")
        await _enqueue_prompt(pilot, chartreux_app, "second")
        await _enqueue_prompt(pilot, chartreux_app, "third")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        assert len(chartreux_app._queue) == 3

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 0

        await pilot.press("backspace")
        await pilot.pause(0.1)

        assert len(chartreux_app._queue) == 2
        remaining = [content for _, content in chartreux_app._queue.queue_item_texts()]
        assert "third" not in remaining

        assert body._queue_cursor == 0


@pytest.mark.asyncio
async def test_delete_deletes_selected_item(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "keep me")
        await _enqueue_prompt(pilot, chartreux_app, "delete me")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 0

        await pilot.press("delete")
        await pilot.pause(0.1)

        assert len(chartreux_app._queue) == 1
        remaining = [content for _, content in chartreux_app._queue.queue_item_texts()]
        assert "delete me" not in remaining
        assert "keep me" in remaining


@pytest.mark.asyncio
async def test_delete_all_items_exits_selection_mode(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "only item")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor >= 0

        await pilot.press("backspace")
        await pilot.pause(0.1)

        assert body._queue_cursor < 0
        assert len(chartreux_app._queue) == 0


@pytest.mark.asyncio
async def test_selection_does_not_activate_when_queue_empty(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)

        assert body._queue_cursor < 0
        assert body.input_widget is not None
        assert not body.input_widget._queue_selection_active


@pytest.mark.asyncio
async def test_selection_does_not_activate_when_idle(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _enqueue_prompt(pilot, chartreux_app, "some prompt")
        await pilot.pause(0.2)

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)

        assert body._queue_cursor < 0


@pytest.mark.asyncio
async def test_queue_selected_css_class_toggled(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "first")
        await _enqueue_prompt(pilot, chartreux_app, "second")

        await pilot.press("up")
        await pilot.pause(0.1)

        selected = [
            w for w in chartreux_app._queue.widgets if w.has_class("queue-selected")
        ]
        assert len(selected) == 1

        await pilot.press("up")
        await pilot.pause(0.1)

        selected = [
            w for w in chartreux_app._queue.widgets if w.has_class("queue-selected")
        ]
        assert len(selected) == 1

        await pilot.press("escape")
        await pilot.pause(0.1)

        selected = [
            w for w in chartreux_app._queue.widgets if w.has_class("queue-selected")
        ]
        assert len(selected) == 0


@pytest.mark.asyncio
async def test_edit_submit_returns_to_selection_not_stuck(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "first")
        await _enqueue_prompt(pilot, chartreux_app, "second")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert body.input_widget is not None
        body.input_widget.clear_text()
        body.input_widget.load_text("edited")
        await pilot.pause(0.05)

        await pilot.press("enter")
        await pilot.pause(0.1)

        assert not body._queue_in_edit_mode
        assert body._queue_cursor >= 0
        assert body.input_widget is not None
        assert body.input_widget._queue_selection_active

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 1


@pytest.mark.asyncio
async def test_input_locked_during_selection_mode(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "queued")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)

        assert body.input_widget is not None
        assert body.input_widget.read_only
        assert not body.input_widget.show_cursor

        original_text = body.input_widget.text
        for char in "hello world":
            await pilot.press(char)
        await pilot.pause(0.1)

        assert body.input_widget.text == original_text


@pytest.mark.asyncio
async def test_input_unlocked_in_edit_mode(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "edit me")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert body.input_widget is not None
        assert not body.input_widget.read_only
        assert body.input_widget.show_cursor


@pytest.mark.asyncio
async def test_input_unlocked_after_exit(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "queued")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body.input_widget is not None
        assert body.input_widget.read_only

        await pilot.press("escape")
        await pilot.pause(0.1)
        assert not body.input_widget.read_only
        assert body.input_widget.show_cursor


@pytest.mark.asyncio
async def test_delete_keeps_later_indices_correct(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "a")
        await _enqueue_prompt(pilot, chartreux_app, "b")
        await _enqueue_prompt(pilot, chartreux_app, "c")

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor == 0  # newest = "c"

        await pilot.press("backspace")
        await pilot.pause(0.1)
        assert [c for _, c in chartreux_app._queue.queue_item_texts()] == ["a", "b"]
        assert body._queue_cursor == 0  # now points at "b"

        await pilot.press("backspace")
        await pilot.pause(0.1)
        assert [c for _, c in chartreux_app._queue.queue_item_texts()] == ["a"]
        assert body._queue_cursor == 0


@pytest.mark.asyncio
async def test_edit_mode_hint_shown_on_enter(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "edit me")

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)

        assert chartreux_app._inline_notice.content == "Enter to save · Esc to discard"


@pytest.mark.asyncio
async def test_edit_mode_hint_persists_until_exit(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "edit me")

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert chartreux_app._inline_notice.display

        # The hint must persist for as long as edit mode is active — it must
        # not self-hide after a timeout while the user is still editing.
        await pilot.pause(4.0)
        assert chartreux_app._inline_notice.display
        assert chartreux_app._inline_notice.content == "Enter to save · Esc to discard"

        # Leaving edit mode (Escape) clears the hint.
        await pilot.press("escape")
        await pilot.pause(0.15)
        assert not chartreux_app._inline_notice.display


@pytest.mark.asyncio
async def test_selection_exits_when_queue_drained_empty(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "queued")
        assert len(chartreux_app._queue) == 1

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        assert body._queue_cursor >= 0

        backend = _BACKENDS[chartreux_app]
        backend.release[0].set()
        assert await _wait_until(pilot, backend.started[1].is_set, timeout=2.0)
        # The drain reaches the client queue a few event-loop hops after the next
        # turn starts, so wait for it rather than asserting point-in-time.
        assert await _wait_until(pilot, lambda: len(chartreux_app._queue) == 0)

        # Next navigation re-syncs against the empty queue and exits selection.
        await pilot.press("up")
        await pilot.pause(0.2)
        assert body._queue_cursor < 0


@pytest.mark.asyncio
async def test_consumed_prompt_edit_copy_on_write_requeues_as_prompt(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        await _start_bash_and_wait_busy(pilot, chartreux_app)
        await _enqueue_prompt(pilot, chartreux_app, "queued")
        assert len(chartreux_app._queue) == 1

        body = chartreux_app.query_one(ChatInputContainer)._body
        assert body is not None

        await pilot.press("up")
        await pilot.pause(0.1)
        await pilot.press("enter")
        await pilot.pause(0.1)
        assert body._queue_in_edit_mode

        backend = _BACKENDS[chartreux_app]
        backend.release[0].set()
        assert await _wait_until(pilot, backend.started[1].is_set, timeout=2.0)
        assert await _wait_until(pilot, lambda: len(chartreux_app._queue) == 0)

        assert body.input_widget is not None
        body.input_widget.clear_text()
        body.input_widget.load_text("edited prompt")
        await pilot.pause(0.05)

        # First Enter: item was consumed -> copy-on-write notice, stays in edit.
        await pilot.press("enter")
        await pilot.pause(0.15)
        assert body._queue_edit_consumed
        assert chartreux_app._inline_notice.display

        # Second Enter: re-submit the edited text as a fresh item.
        await pilot.press("enter")
        await pilot.pause(0.15)

        assert [content for _, content in chartreux_app._queue.queue_item_texts()] == [
            "edited prompt"
        ]
        assert not chartreux_app._inline_notice.display
        backend.release[1].set()
