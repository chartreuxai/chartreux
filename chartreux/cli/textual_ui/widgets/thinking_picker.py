from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.config import ThinkingLevel
from chartreux.ui.shortcut_hints import rich_theme_style, shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def _build_option_text(level: str, is_current: bool, muted_style: str = "") -> Text:
    text = Text(no_wrap=True)
    text.append("  ")
    text.append(level.capitalize())
    if is_current:
        text.append("  This session", style=muted_style)
    return text


class ThinkingPickerApp(Container):
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False)
    ]

    class ThinkingSelected(Message):
        level: ThinkingLevel

        def __init__(self, level: ThinkingLevel) -> None:
            self.level = level
            super().__init__()

    class Cancelled(Message):
        pass

    def __init__(
        self,
        thinking_levels: Sequence[ThinkingLevel],
        current_thinking: ThinkingLevel,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="thinkingpicker-app", **kwargs)
        self._thinking_levels = thinking_levels
        self._current_thinking = current_thinking
        self._selection_pending = False

    def compose(self) -> ComposeResult:
        options = [
            Option(
                _build_option_text(
                    level,
                    level == self._current_thinking,
                    rich_theme_style(self.app.theme_variables["text-muted"]),
                ),
                id=level,
            )
            for level in self._thinking_levels
        ]
        with Vertical(id="thinkingpicker-content"):
            yield NoMarkupStatic(
                "Thinking level for this session", classes="thinkingpicker-title"
            )
            yield NavigableOptionList(*options, id="thinkingpicker-options")
            error = NoMarkupStatic(
                "", id="thinkingpicker-error", classes="thinkingpicker-help"
            )
            error.display = False
            yield error
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('↑↓/jk')} Navigate  {shortcut('Enter')} Apply  "
                    f"{shortcut('Esc')} Cancel"
                ),
                classes="thinkingpicker-help",
            )

    def on_mount(self) -> None:
        option_list = self.query_one(OptionList)
        for i, level in enumerate(self._thinking_levels):
            if level == self._current_thinking:
                option_list.highlighted = i
                break
        option_list.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option.id and not self._selection_pending:
            self._selection_pending = True
            self.post_message(
                self.ThinkingSelected(cast(ThinkingLevel, event.option.id))
            )

    def action_cancel(self) -> None:
        self.post_message(self.Cancelled())

    def show_error(self, message: str) -> None:
        self._selection_pending = False
        error = self.query_one("#thinkingpicker-error", NoMarkupStatic)
        error.update(message)
        error.display = True
        self.query_one(OptionList).focus()

    def clear_error(self) -> None:
        self.query_one("#thinkingpicker-error", NoMarkupStatic).display = False
