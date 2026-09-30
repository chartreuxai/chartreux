from __future__ import annotations

import textwrap
from typing import Any

from rich.cells import cell_len
from textual import events
from textual.containers import Horizontal, VerticalScroll
from textual.message import Message
from textual.widgets import Static

from chartreux.cli.autocompletion.base import CompletionEntry

COMPLETION_POPUP_MAX_ROWS = 8
COMPLETION_POPUP_MAX_WIDTH = 92
COMPLETION_POPUP_PADDING_X = 1
SELECTED_CLASS = "completion-selected"
NO_DESCRIPTION_CLASS = "completion-no-description"


class _CompletionItem(Static):
    pass


class _CompletionRow(Horizontal):
    pass


class CompletionPopup(VerticalScroll):
    class SuggestionClicked(Message):
        def __init__(self, index: int) -> None:
            super().__init__()
            self.index = index

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="completion-popup", **kwargs)
        self.styles.display = "none"
        self.styles.max_height = COMPLETION_POPUP_MAX_ROWS + 2
        self.styles.padding = (0, COMPLETION_POPUP_PADDING_X)
        self.can_focus = False
        self._suggestions: list[CompletionEntry] = []
        self._desired_width = 0
        self._command_width = 0

    def update_suggestions(
        self, suggestions: list[CompletionEntry], selected: int
    ) -> None:
        if not suggestions:
            self.hide()
            return

        if suggestions != self._suggestions:
            rows = self._rebuild(suggestions)
        else:
            rows = list(self.query(_CompletionRow))
        self._select(rows, selected)
        self._update_max_height(selected)
        self.styles.display = "block"

    def on_resize(self) -> None:
        self.refresh_viewport_layout()

    def on_click(self, event: events.Click) -> None:
        if event.widget is None:
            return
        rows = list(self.query(_CompletionRow))
        for index, row in enumerate(rows):
            if row in event.widget.ancestors_with_self:
                event.stop()
                self.post_message(self.SuggestionClicked(index))
                return

    def refresh_viewport_layout(self) -> None:
        if self._suggestions:
            self.styles.width = min(
                COMPLETION_POPUP_MAX_WIDTH,
                max(1, self.app.size.width - 2),
                self._desired_width,
            )
            self._update_max_height()

    def _rebuild(self, suggestions: list[CompletionEntry]) -> list[_CompletionRow]:
        self.remove_children()
        self._suggestions = suggestions
        has_descriptions = any(entry.description for entry in suggestions)
        self.set_class(not has_descriptions, NO_DESCRIPTION_CLASS)
        command_width = max(
            cell_len(self._display_label(entry.label)) for entry in suggestions
        )
        self._command_width = command_width
        description_width = max(
            (cell_len(entry.description) for entry in suggestions), default=0
        )
        desired_width = (
            command_width + (2 + description_width if has_descriptions else 0) + 4
        )
        self._desired_width = desired_width
        available = max(1, self.app.size.width - 2)
        self.styles.width = min(COMPLETION_POPUP_MAX_WIDTH, available, desired_width)
        rows: list[_CompletionRow] = []
        for entry in suggestions:
            command = _CompletionItem(
                self._display_label(entry.label), classes="completion-command"
            )
            if has_descriptions:
                command.styles.width = command_width
            description_cell = _CompletionItem(
                entry.description, classes="completion-description"
            )
            rows.append(_CompletionRow(command, description_cell))
        self.mount_all(rows)
        return rows

    def _select(self, rows: list[_CompletionRow], selected: int) -> None:
        for idx, row in enumerate(rows):
            row.set_class(idx == selected, SELECTED_CLASS)
        if 0 <= selected < len(rows):
            rows[selected].scroll_visible(animate=False)

    def _update_max_height(self, selected: int | None = None) -> None:
        if selected is None:
            selected = next(
                (
                    idx
                    for idx, row in enumerate(self.query(_CompletionRow))
                    if row.has_class(SELECTED_CLASS)
                ),
                -1,
            )
        extra_rows = 0
        if 0 <= selected < len(self._suggestions) and self._command_width:
            popup_width = min(
                COMPLETION_POPUP_MAX_WIDTH,
                max(1, self.app.size.width - 2),
                self._desired_width,
            )
            inner_width = max(1, popup_width - 4)
            command_column = min(self._command_width, max(1, inner_width * 30 // 100))
            description_width = max(1, inner_width - command_column - 2)
            wrapped_lines = len(
                textwrap.wrap(
                    self._suggestions[selected].description,
                    width=description_width,
                    break_long_words=True,
                    break_on_hyphens=False,
                )
            )
            extra_rows = max(0, wrapped_lines - 1)
        available_height = max(1, self.app.size.height - max(0, self.region.y))
        self.styles.max_height = min(
            COMPLETION_POPUP_MAX_ROWS + extra_rows + 2, available_height
        )

    def hide(self) -> None:
        self.remove_children()
        self._suggestions = []
        self.styles.display = "none"

    @property
    def content_text(self) -> str:
        return "\n".join(str(child.render()) for child in self.query(_CompletionItem))

    @staticmethod
    def _display_label(label: str) -> str:
        if label.startswith("@"):
            return label[1:]
        return label
