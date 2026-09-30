from __future__ import annotations

import os
from typing import Any, ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button
from textual.widgets.option_list import Option

from chartreux.config_values import DEFAULT_LOG_LEVEL
from chartreux.observability.logging import LOG_LEVELS, LogLevelChain
from chartreux.ui.shortcut_hints import rich_theme_style, shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

# Ordered for display; derived from the canonical set so they always agree.
ORDERED_LEVELS: list[str] = [
    lvl
    for lvl in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if lvl in LOG_LEVELS
]

_BADGE_SESSION = "session"
_BADGE_CONFIG = "config"


def _build_row(
    level: str,
    *,
    is_highlighted: bool,
    focused_badge: str,
    effective_level: str,
    session_level: str | None,
    config_level: str | None,
    muted_style: str = "",
    set_style: str = "bold",
    unset_style: str = "dim",
) -> Text:
    text = Text(no_wrap=True)

    session_here = session_level == level
    config_here = config_level == level

    text.append("  ")

    text.append(f"{level:<10}")
    if level == effective_level:
        text.append("Active ", style=muted_style)

    text.append("  ")
    for badge in (_BADGE_SESSION, _BADGE_CONFIG):
        is_set = (badge == _BADGE_SESSION and session_here) or (
            badge == _BADGE_CONFIG and config_here
        )
        _append_badge(
            text,
            badge,
            focused=is_highlighted and badge == focused_badge,
            is_set=is_set,
            set_style=set_style,
            unset_style=unset_style,
        )
        text.append("  ")

    return text


def _append_badge(
    text: Text,
    badge: str,
    *,
    focused: bool,
    is_set: bool,
    set_style: str,
    unset_style: str,
) -> None:
    # State and badge positions do not change as the cursor moves.
    state = "●" if is_set else "○"
    text.append(f"{state} {badge:<7}", style=set_style if is_set else unset_style)
    if focused:
        text.stylize("underline bold", len(text.plain) - 9, len(text.plain))


class LogLevelPickerApp(Container):
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+s", "apply", "Apply changes", show=False),
    ]

    class Applied(Message):
        def __init__(
            self,
            session_level: str | None,
            config_level: str | None,
            *,
            config_cleared: bool,
        ) -> None:
            self.session_level = session_level
            self.config_level = config_level
            # True when the user explicitly removed a previously set config value.
            self.config_cleared = config_cleared
            super().__init__()

    def __init__(self, chain: LogLevelChain, **kwargs: Any) -> None:
        super().__init__(id="loglevelpicker-app", **kwargs)
        self._chain = chain
        self._session_level: str | None = chain.session
        self._config_level: str | None = chain.config
        self._highlighted_level: str = chain.session or chain.effective
        self._focused_badge: str = _BADGE_SESSION
        self._confirming_discard = False

    def compose(self) -> ComposeResult:
        options = [Option(self._row_text(level), id=level) for level in ORDERED_LEVELS]
        with Vertical(id="loglevelpicker-content"):
            yield NoMarkupStatic("Log Level", classes="loglevelpicker-title")
            yield NoMarkupStatic(
                self._subtitle_text(),
                id="loglevelpicker-subtitle",
                classes="loglevelpicker-subtitle",
            )
            yield NavigableOptionList(*options, id="loglevelpicker-options")
            with Vertical(id="loglevelpicker-discard") as confirmation:
                confirmation.display = False
                yield NoMarkupStatic(
                    "Discard edits to session and config.toml log levels? Unsaved changes will be lost. "
                    "Cancel preserves both saved levels and your current draft."
                )
                with Horizontal():
                    yield Button("Cancel", id="loglevelpicker-keep")
                    yield Button("Discard edits", id="loglevelpicker-confirm-discard")
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('↑↓/jk')} Navigate  "
                    f"{shortcut('Tab')} Switch badge  "
                    f"{shortcut('Space')} Toggle  "
                    f"{shortcut('Enter')} Accept  {shortcut('Esc')} Cancel"
                ),
                id="loglevelpicker-help",
                classes="loglevelpicker-help",
            )

    def on_mount(self) -> None:
        option_list = self.query_one(NavigableOptionList)
        for i, level in enumerate(ORDERED_LEVELS):
            if level == self._highlighted_level:
                option_list.highlighted = i
                break
        option_list.focus()

    def on_option_list_option_highlighted(
        self, event: NavigableOptionList.OptionHighlighted
    ) -> None:
        if event.option.id and event.option.id in ORDERED_LEVELS:
            self._highlighted_level = event.option.id
            self._redraw()

    def on_key(self, event: events.Key) -> None:
        if self._confirming_discard:
            return
        if event.key == "shift+tab":
            event.stop()
            event.prevent_default()
            self._focused_badge = _BADGE_SESSION
            self._redraw()
        elif event.key == "tab":
            event.stop()
            event.prevent_default()
            self._focused_badge = _BADGE_CONFIG
            self._redraw()
        elif event.key == "space":
            event.stop()
            event.prevent_default()
            self._toggle_badge()
        elif event.key == "enter":
            event.stop()
            event.prevent_default()
            self.action_apply()

    def _toggle_badge(self) -> None:
        level = self._highlighted_level
        if self._focused_badge == _BADGE_SESSION:
            self._session_level = None if self._session_level == level else level
        else:
            self._config_level = None if self._config_level == level else level
        self._redraw()

    def _subtitle_text(self) -> str:
        effective = self._effective_level()
        if self._session_level:
            source = f"session override: {self._session_level}"
        elif self._chain.env:
            source = f"env LOG_LEVEL: {self._chain.env}"
        elif self._config_level:
            source = f"config.toml: {self._config_level}"
        else:
            source = "default"
        return f"Effective: {effective}  ({source})"

    def _redraw(self) -> None:
        self.query_one("#loglevelpicker-subtitle", NoMarkupStatic).update(
            self._subtitle_text()
        )
        option_list = self.query_one(NavigableOptionList)
        for level in ORDERED_LEVELS:
            option_list.replace_option_prompt(level, self._row_text(level))

    def _effective_level(self) -> str:
        # Recompute the priority chain against the current draft state so the
        # Active label and subtitle update live as the user toggles badges.
        env: str | None = None
        if os.environ.get("DEBUG_MODE") == "true":
            env = "DEBUG"
        else:
            raw = os.environ.get("LOG_LEVEL", "").upper()
            if raw in LOG_LEVELS:
                env = raw
        return self._session_level or env or self._config_level or DEFAULT_LOG_LEVEL

    def _row_text(self, level: str) -> Text:
        return _build_row(
            level,
            is_highlighted=level == self._highlighted_level,
            focused_badge=self._focused_badge,
            effective_level=self._effective_level(),
            session_level=self._session_level,
            config_level=self._config_level,
            muted_style=rich_theme_style(self.app.theme_variables["text-muted"])
            if self.is_attached
            else "",
            set_style=rich_theme_style(self.app.theme_variables["success"])
            if self.is_attached
            else "bold",
            unset_style=rich_theme_style(self.app.theme_variables["text-muted"])
            if self.is_attached
            else "dim",
        )

    def action_cancel(self) -> None:
        if self._confirming_discard:
            self._show_discard_confirmation(False)
        elif (self._session_level, self._config_level) != (
            self._chain.session,
            self._chain.config,
        ):
            self._show_discard_confirmation(True)
        else:
            self.post_message(self.Cancelled())

    def _show_discard_confirmation(self, show: bool) -> None:
        self._confirming_discard = show
        self.query_one("#loglevelpicker-discard", Vertical).display = show
        self.query_one(NavigableOptionList).display = not show
        self.query_one("#loglevelpicker-help", NoMarkupStatic).update(
            shortcut_hint(
                f"{shortcut('Tab')} Choose  {shortcut('Enter')} Select  "
                f"{shortcut('Esc')} Back"
                if show
                else f"{shortcut('↑↓/jk')} Navigate  {shortcut('Tab')} Switch badge  "
                f"{shortcut('Space')} Toggle  {shortcut('Enter')} Accept  "
                f"{shortcut('Esc')} Cancel"
            )
        )
        if show:
            self.query_one("#loglevelpicker-keep", Button).focus()
        else:
            self.query_one(NavigableOptionList).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "loglevelpicker-keep":
            self._show_discard_confirmation(False)
        elif event.button.id == "loglevelpicker-confirm-discard":
            self.post_message(self.Cancelled())

    class Cancelled(Message):
        pass

    def action_apply(self) -> None:
        if self._confirming_discard:
            return
        self.post_message(
            self.Applied(
                session_level=self._session_level,
                config_level=self._config_level,
                config_cleared=self._chain.config is not None
                and self._config_level is None,
            )
        )
