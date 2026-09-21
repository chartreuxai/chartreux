from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual.widget import Widget

from chartreux.cli.commands import CommandRegistry
from chartreux.cli.history_manager import HistoryManager
from chartreux.cli.input_modes import InputMode
from chartreux.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class ChatInputBody(Widget):
    class Submitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class QueueEditSubmitted(Message):
        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class QueueEditConsumed(Message):
        """Posted when the user submits an edit but the item was already consumed."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    class QueueRemoveRequested(Message):
        pass

    class QueueSelectionScroll(Message):
        def __init__(self, queue_index: int) -> None:
            self.queue_index = queue_index
            super().__init__()

    class QueueModeExited(Message):
        """Posted when queue selection/edit mode is fully exited."""

    class CompletionResetRequested(Message):
        pass

    class InlineNoticeRequested(Message):
        def __init__(self, message: str, *, timeout: float | None = 4.0) -> None:
            self.message = message
            self.timeout = timeout
            super().__init__()

    class InlineNoticeCleared(Message):
        pass

    def __init__(
        self,
        command_registry: CommandRegistry,
        history_file: Path | None = None,
        queue_edit_active_getter: Callable[[], bool] | None = None,
        queue_items_getter: Callable[[], list[tuple[int, str]]] | None = None,
        queue_selected_index_getter: Callable[[], int | None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.input_widget: ChatTextArea | None = None
        self.prompt_widget: NoMarkupStatic | None = None
        self._command_registry = command_registry
        self._queue_edit_active_getter = queue_edit_active_getter
        self._queue_items_getter = queue_items_getter
        self._queue_selected_index_getter = queue_selected_index_getter
        # (queue_index, content), newest-first (reversed from queue order).
        self._queue_items: list[tuple[int, str]] = []
        self._queue_cursor: int = -1
        self._queue_original_text: str = ""
        self._queue_in_edit_mode: bool = False
        self._queue_edit_consumed: bool = False

        if history_file:
            self.history = HistoryManager(history_file)
        else:
            self.history = None

    def compose(self) -> ComposeResult:
        with Horizontal():
            self.prompt_widget = NoMarkupStatic(">", id="prompt")
            yield self.prompt_widget

            self.input_widget = ChatTextArea(
                id="input", command_registry=self._command_registry
            )
            yield self.input_widget

    def on_mount(self) -> None:
        if self.input_widget:
            self.input_widget.focus()

    def _parse_mode_and_text(self, text: str) -> tuple[InputMode, str]:
        if text.startswith("!"):
            return "!", text[1:]
        elif text.startswith("/"):
            return "/", text[1:]
        else:
            return ">", text

    def _update_prompt(self) -> None:
        if not self.input_widget or not self.prompt_widget:
            return

        self.prompt_widget.update(self.input_widget.input_mode)

    def on_chat_text_area_mode_changed(self, event: ChatTextArea.ModeChanged) -> None:
        if self.prompt_widget:
            self.prompt_widget.update(event.mode)

    def _load_history_entry(self, text: str, cursor_col: int | None = None) -> None:
        if not self.input_widget:
            return

        mode, display_text = self._parse_mode_and_text(text)

        self.input_widget._navigating_history = True
        self.input_widget.set_mode(mode)
        self.input_widget.load_text(display_text)

        first_line = display_text.split("\n")[0]
        col = cursor_col if cursor_col is not None else len(first_line)
        cursor_pos = (0, col)

        self.input_widget.move_cursor(cursor_pos)
        self.input_widget._cursor_pos_after_load = cursor_pos
        self.input_widget._cursor_moved_since_load = False

        self._update_prompt()
        self._notify_completion_reset()

    def on_chat_text_area_history_previous(
        self, _event: ChatTextArea.HistoryPrevious
    ) -> None:
        if not self.input_widget:
            return

        self.input_widget._navigating_history = False

        if self._queue_cursor >= 0:
            return

        if self._try_enter_queue_selection():
            return

        if not self.history:
            return

        if not self.history.is_navigating():
            self.input_widget._original_text = self.input_widget.text

        previous = self.history.get_previous(self.input_widget._original_text)

        if previous is not None:
            self._load_history_entry(previous)

    def on_chat_text_area_history_next(self, _event: ChatTextArea.HistoryNext) -> None:
        if not self.input_widget:
            return

        self.input_widget._navigating_history = False

        if self._queue_cursor >= 0:
            return

        if not self.history:
            return

        next_entry = self.history.get_next()
        if next_entry is not None:
            self._load_history_entry(next_entry)

    def on_chat_text_area_history_reset(
        self, _event: ChatTextArea.HistoryReset
    ) -> None:
        if self.history:
            self.history.reset_navigation()
        if self.input_widget:
            self.input_widget._original_text = ""
            self.input_widget._cursor_pos_after_load = None
            self.input_widget._cursor_moved_since_load = False

    # -- queue selection + edit state machine -----------------------------

    def _is_queue_edit_available(self) -> bool:
        return (
            self._queue_edit_active_getter is not None
            and self._queue_edit_active_getter()
        )

    def _lock_input_for_selection(self) -> None:
        if not self.input_widget:
            return
        self.input_widget.read_only = True
        self.input_widget.cursor_blink = False
        self.input_widget.show_cursor = False

    def _unlock_input_for_edit(self) -> None:
        if not self.input_widget:
            return
        self.input_widget.read_only = False
        self.input_widget.cursor_blink = True
        self.input_widget.show_cursor = True

    def _resnapshot_queue_items(self) -> list[tuple[int, str]]:
        """Re-read the live queue (newest-first) so cached indices stay valid
        after the queue mutates (delete, or server promotion consuming an item).
        """
        if self._queue_items_getter is None:
            return []
        return list(reversed(self._queue_items_getter()))

    @property
    def in_queue_mode(self) -> bool:
        """Whether queue selection or edit mode is currently active."""
        return self._queue_cursor >= 0

    def _resync_selection(self) -> bool:
        """Reconcile the cached cursor with the live queue after promotion.

        The app server promotes the oldest item first, so a highlighted item
        that is still queued shifts to a lower absolute index. We track it by
        widget identity rather than by cached index, which would misread that
        shift as consumption.

        Returns False (and exits selection mode) if the queue is now empty.
        """
        if self._queue_cursor < 0:
            return True
        live_index = (
            self._queue_selected_index_getter()
            if self._queue_selected_index_getter is not None
            else None
        )
        self._queue_items = self._resnapshot_queue_items()
        if not self._queue_items:
            self._exit_queue_mode()
            return False
        if live_index is None:
            # Highlighted item was consumed: stay on the same relative position
            # (newest-first), clamped — that now points to the next item. Re-post
            # the scroll so the app re-syncs ``_queue_selected_widget`` to the
            # new item; without it a following Enter/Backspace resolves to the
            # removed widget (edit → stray copy-on-write, delete → no-op).
            self._queue_cursor = min(self._queue_cursor, len(self._queue_items) - 1)
            self._post_scroll()
            return True
        for pos, (qi, _) in enumerate(self._queue_items):
            if qi == live_index:
                self._queue_cursor = pos
                return True
        self._queue_cursor = min(self._queue_cursor, len(self._queue_items) - 1)
        return True

    def _drop_selected_from_cache(self) -> None:
        """Synchronously remove the highlighted item from the local cache and
        re-number the newer items whose absolute queue index shifted down.

        The app removes the item asynchronously (posted message), so a plain
        re-snapshot here would still see it; mutating the cache keeps indices
        consistent until the app catches up.
        """
        c = self._queue_cursor
        if c < 0 or c >= len(self._queue_items):
            return
        del self._queue_items[c]
        # Newer items (positions 0..c-1 in the newest-first list) had a higher
        # absolute index than the removed one and shift down by 1.
        for pos in range(c):
            qi, content = self._queue_items[pos]
            self._queue_items[pos] = (qi - 1, content)

    def _try_enter_queue_selection(self) -> bool:
        if not self.input_widget or not self._is_queue_edit_available():
            return False
        if self._queue_items_getter is None:
            return False

        self._queue_items = self._resnapshot_queue_items()
        if not self._queue_items:
            return False

        self._queue_cursor = 0
        self._queue_original_text = self.input_widget.text
        self._queue_in_edit_mode = False
        self._queue_edit_consumed = False
        self.input_widget._queue_selection_active = True
        self._lock_input_for_selection()
        self._post_scroll()
        self.post_message(
            self.InlineNoticeRequested(
                "Up/Down: select  ·  Enter: edit  ·  Backspace/Delete: remove  ·  Esc: exit",
                timeout=3.0,
            )
        )
        return True

    def on_chat_text_area_queue_selection_previous(
        self, _event: ChatTextArea.QueueSelectionPrevious
    ) -> None:
        if self._queue_cursor < 0 or self._queue_in_edit_mode:
            return
        if not self._resync_selection():
            return
        if self._queue_cursor < len(self._queue_items) - 1:
            self._queue_cursor += 1
            self._post_scroll()

    def on_chat_text_area_queue_selection_next(
        self, _event: ChatTextArea.QueueSelectionNext
    ) -> None:
        if self._queue_cursor < 0 or self._queue_in_edit_mode:
            return
        if not self._resync_selection():
            return
        if self._queue_cursor > 0:
            self._queue_cursor -= 1
            self._post_scroll()
        else:
            self._exit_queue_mode()

    def on_chat_text_area_queue_selection_enter(
        self, _event: ChatTextArea.QueueSelectionEnter
    ) -> None:
        if self._queue_cursor < 0 or self._queue_in_edit_mode or not self.input_widget:
            return
        if not self._resync_selection():
            return
        _, content = self._queue_items[self._queue_cursor]
        self._queue_in_edit_mode = True
        self.input_widget._queue_selection_active = False
        self.input_widget._queue_edit_active = True
        self._unlock_input_for_edit()
        self._load_history_entry(content)
        self.post_message(
            self.InlineNoticeRequested("Enter to save · Esc to discard", timeout=None)
        )

    def on_chat_text_area_queue_selection_remove(
        self, _event: ChatTextArea.QueueSelectionRemove
    ) -> None:
        if self._queue_cursor < 0 or self._queue_in_edit_mode:
            return
        if not self._resync_selection():
            return
        self.post_message(self.QueueRemoveRequested())

        # Synchronously drop the item from the local cache and re-number the
        # newer items whose absolute index shifted down. A live re-snapshot
        # here would still see the item (the app removes it asynchronously),
        # and a bare ``del`` would leave the shifted indices stale (F1).
        self._drop_selected_from_cache()
        if not self._queue_items:
            self._exit_queue_mode()
            return

        self._queue_cursor = max(0, self._queue_cursor - 1)
        self._post_scroll()

    def on_chat_text_area_queue_selection_exit(
        self, _event: ChatTextArea.QueueSelectionExit
    ) -> None:
        if self._queue_cursor < 0:
            return
        self._exit_queue_mode()

    def on_chat_text_area_queue_edit_cancelled(
        self, _event: ChatTextArea.QueueEditCancelled
    ) -> None:
        if self._queue_cursor < 0 or not self._queue_in_edit_mode:
            return
        self._queue_in_edit_mode = False
        self._queue_edit_consumed = False
        if self.input_widget:
            self.input_widget._queue_edit_active = False
            self.input_widget._queue_selection_active = True
            self.input_widget.clear_text()
            self._update_prompt()
        self._lock_input_for_selection()
        self._post_scroll()
        self.post_message(self.InlineNoticeCleared())

    def _exit_queue_mode(self) -> None:
        was_in_edit = self._queue_in_edit_mode
        self._queue_cursor = -1
        self._queue_items = []
        self._queue_in_edit_mode = False
        self._queue_edit_consumed = False
        if self.input_widget:
            self.input_widget._queue_selection_active = False
            self.input_widget._queue_edit_active = False
            self.input_widget.load_text(self._queue_original_text)
            self._update_prompt()
        if was_in_edit:
            if self.input_widget:
                self.input_widget.clear_text()
                self._update_prompt()
            self.post_message(self.InlineNoticeCleared())
        self._queue_original_text = ""
        if self.input_widget:
            self._unlock_input_for_edit()
        self.post_message(self.QueueModeExited())

    def _post_scroll(self) -> None:
        if self._queue_cursor < 0 or not self._queue_items:
            return
        queue_index = self._queue_items[self._queue_cursor][0]
        self.post_message(self.QueueSelectionScroll(queue_index))

    def _end_edit_mode_back_to_selection(
        self, *, scroll_to_selection: bool = True
    ) -> None:
        self._queue_in_edit_mode = False
        self._queue_edit_consumed = False
        if self.input_widget:
            self.input_widget._queue_edit_active = False
            self.input_widget._queue_selection_active = True
            self.input_widget.clear_text()
            self._update_prompt()
        self._notify_completion_reset()
        self._lock_input_for_selection()
        if scroll_to_selection:
            self._post_scroll()
        self.post_message(self.InlineNoticeCleared())

    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        event.stop()

        if not self.input_widget:
            return

        value = event.value.strip()
        if value and self._queue_in_edit_mode:
            if self._queue_edit_consumed:
                self.post_message(self.InlineNoticeCleared())
                self._queue_items = self._resnapshot_queue_items()
                if not self._queue_items:
                    self._exit_queue_mode()
                    self.input_widget.clear_text()
                    self._update_prompt()
                    self._notify_completion_reset()
                    self.post_message(self.QueueEditConsumed(value))
                    return
                if self._queue_cursor >= len(self._queue_items):
                    self._queue_cursor = len(self._queue_items) - 1
                self._end_edit_mode_back_to_selection()
                self.post_message(self.QueueEditConsumed(value))
                return
            # The app server may have promoted the item while the user was
            # typing. Detect that by widget identity because older prompts
            # starting can shift every later queue index. Keep the edit and let
            # the user re-submit it as new or discard it.
            if (
                self._queue_selected_index_getter is not None
                and self._queue_selected_index_getter() is None
            ):
                self._queue_edit_consumed = True
                self.post_message(
                    self.InlineNoticeRequested(
                        "This message was already processed — "
                        "press Enter to submit as new, or Escape to discard.",
                        timeout=8.0,
                    )
                )
                return
            # Keep the app's selected widget unchanged until it handles the edit.
            # An older item may have left the FIFO while this item was being
            # edited, making the cached numeric index stale. Re-scrolling with
            # that index would select a newer item and apply the edit there.
            self._end_edit_mode_back_to_selection(scroll_to_selection=False)
            self.post_message(self.QueueEditSubmitted(value))
            return

        if value:
            if self.history:
                self.history.add(value)
                self.history.reset_navigation()
                self.run_worker(
                    partial(self.history.persist, value),
                    group="history_persist",
                    thread=True,
                )

            self.input_widget.clear_text()
            self._update_prompt()

            self._notify_completion_reset()

        self.post_message(self.Submitted(value))

    @property
    def value(self) -> str:
        if not self.input_widget:
            return ""
        return self.input_widget.get_full_text()

    @value.setter
    def value(self, text: str) -> None:
        if self.input_widget:
            mode, display_text = self._parse_mode_and_text(text)
            self.input_widget.set_mode(mode)
            self.input_widget.load_text(display_text)
            self._update_prompt()

    def focus_input(self) -> None:
        if self.input_widget:
            self.input_widget.focus()

    def _notify_completion_reset(self) -> None:
        self.post_message(self.CompletionResetRequested())

    def replace_input(self, text: str, cursor_offset: int | None = None) -> None:
        if not self.input_widget:
            return

        self.input_widget.load_text(text)
        self.input_widget.reset_history_state()
        self._update_prompt()

        if cursor_offset is not None:
            self.input_widget.set_cursor_offset(max(0, min(cursor_offset, len(text))))
