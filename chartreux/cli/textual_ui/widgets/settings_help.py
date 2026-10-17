"""Clickable shortcut hint widgets for the settings screen."""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from rich.cells import cell_len
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.visual import VisualType

from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class SettingsHelpHost(Protocol):
    """Screen surface the hint widgets act on; the settings screen implements it."""

    _pointer_command_started: bool


class SettingsHints(NoMarkupStatic):
    """Clickable shortcut text does not add keyboard focus stops."""

    targets: list[tuple[int, int, str]]

    def __init__(
        self, *args: Any, host: SettingsHelpHost | None = None, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.targets = []
        self._host = host

    def pointer_action_at(self, x: int, y: int) -> str | None:
        return next(
            (action for start, end, action in self.targets if start <= x < end), None
        )

    async def on_click(self, event: events.Click) -> None:
        action = self.pointer_action_at(event.x, event.y)
        if action and event.chain == 1:
            event.stop()
            if self._host is not None:
                self._host._pointer_command_started = True
            await self.screen.run_action(action)


class SettingsHelp(SettingsHints):
    """The detail region opens help; named mutation shortcuts stay explicit."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "scroll_up", show=False),
        Binding("down,j", "scroll_down", show=False),
        Binding("pageup", "page_up", show=False),
        Binding("pagedown", "page_down", show=False),
        Binding("home", "scroll_home", show=False),
        Binding("end", "scroll_end", show=False),
    ]

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic(self.content, id="settings-help-text")

    def update(self, content: VisualType = "", *, layout: bool = True) -> None:
        super().update(content, layout=layout)
        if self.is_mounted:
            self.query_one("#settings-help-text", NoMarkupStatic).update(
                content, layout=layout
            )

    async def on_click(self, event: events.Click) -> None:
        event.prevent_default()
        # Child text clicks bubble with child-local coordinates.
        action = self.pointer_action_at(
            event.screen_x - self.region.x, event.screen_y - self.region.y
        )
        if action and event.chain == 1:
            event.stop()
            if self._host is not None:
                self._host._pointer_command_started = True
            await self.screen.run_action(action)

    def pointer_action_at(self, x: int, y: int) -> str | None:
        child = self.query_one("#settings-help-text", NoMarkupStatic)
        line = "".join(
            segment.text for segment in child.render_line(y + int(self.scroll_y))
        )
        for key, action in (("Ctrl+R", "remove_override"), ("Ctrl+D", "delete_item")):
            index = line.find(key)
            if index >= 0 and cell_len(line[:index]) <= x < cell_len(
                line[:index]
            ) + len(key):
                return action
        return "help"
