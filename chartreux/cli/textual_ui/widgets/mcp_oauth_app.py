from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import ClassVar, Protocol
import webbrowser

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Vertical
from textual.events import DescendantBlur
from textual.message import Message
from textual.widgets import OptionList
from textual.widgets.option_list import Option
from textual.worker import Worker

from chartreux.app_server.protocol import MCPAuthUrlParams
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.clipboard import copy_text_to_clipboard
from chartreux.ui.shortcut_hints import rich_theme_style, shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

_HELP_RETRY = (
    f"{shortcut('R')} Retry  {shortcut('Backspace')} Close  {shortcut('Esc')} Close"
)
_HELP_RUNNING = f"Login in progress  {shortcut('Esc')} Close"
_STATUS_FEEDBACK_RUNNING = "Login is in progress. Press Esc to close."
_STATUS_FEEDBACK_FAILED = (
    "This status row cannot be activated. Use R to retry or Esc to close."
)
_OPTION_PADDING = "  "


class _OAuthOptionId(StrEnum):
    OPEN = auto()
    COPY = auto()
    SHOW = auto()


@dataclass(frozen=True)
class _LoginResult:
    authenticated: bool
    error: str | None = None


class MCPLoginClient(Protocol):
    def login(self, name: str) -> AsyncIterator[MCPAuthUrlParams]: ...


class MCPOAuthApp(Container):
    can_focus_children = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Close", show=False),
        Binding("backspace", "close", "Close", show=False),
        Binding("r", "refresh", "Retry", show=False),
    ]

    class MCPOAuthClosed(Message):
        def __init__(self, *, refreshed: bool = False, server_name: str = "") -> None:
            super().__init__()
            self.refreshed = refreshed
            self.server_name = server_name

    def __init__(self, server_name: str, mcp: MCPLoginClient) -> None:
        super().__init__(id="mcpoauth-app")
        self._server_name = server_name
        self._mcp = mcp
        self._auth_url: str | None = None
        self._auth_url_visible = False
        self._status_message: str | None = None
        self._logging_in = False

    def compose(self) -> ComposeResult:
        with Vertical(id="mcpoauth-content"):
            yield NoMarkupStatic("", id="mcpoauth-title", classes="settings-title")
            yield OptionList(id="mcpoauth-options")
            yield NoMarkupStatic("", id="mcpoauth-detail")
            yield NoMarkupStatic("", id="mcpoauth-help", classes="settings-help")

    def on_mount(self) -> None:
        self.query_one("#mcpoauth-title", NoMarkupStatic).update(
            f"MCP Server: {self._server_name}"
        )
        self._start_login()
        self.query_one(OptionList).focus()

    def on_descendant_blur(self, _event: DescendantBlur) -> None:
        self.query_one(OptionList).focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = event.option.id or ""
        if option_id.startswith("status:"):
            self._set_help_text(
                _STATUS_FEEDBACK_RUNNING
                if self._logging_in
                else _STATUS_FEEDBACK_FAILED
            )
            return
        if option_id == _OAuthOptionId.OPEN:
            self._open_browser()
        elif option_id == _OAuthOptionId.COPY:
            self._copy_url()
        elif option_id == _OAuthOptionId.SHOW:
            self._toggle_url()

    def action_close(self) -> None:
        self.post_message(self.MCPOAuthClosed())

    async def action_refresh(self) -> None:
        if self._logging_in:
            return
        self._start_login()

    def _theme_style(self, role: str) -> str:
        return (
            rich_theme_style(self.app.theme_variables[role]) if self.is_attached else ""
        )

    def _start_login(self) -> None:
        self._auth_url = None
        self._auth_url_visible = False
        self._logging_in = True
        self._status_message = "Running: Preparing authentication"
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_option(
            Option(
                Text(
                    f"{chrome_glyph('running')} Running: Starting OAuth login",
                    style=self._theme_style("primary"),
                ),
                id="status:running",
            )
        )
        self.query_one("#mcpoauth-detail", NoMarkupStatic).update("")
        self._set_help_text(_HELP_RUNNING)
        self.run_worker(self._run_login(), exclusive=True, group="mcp_oauth_login")

    async def _run_login(self) -> _LoginResult:
        try:
            async for event in self._mcp.login(self._server_name):
                self._on_auth_url_available(event.url)
        except Exception as exc:
            return _LoginResult(authenticated=False, error=str(exc))
        return _LoginResult(authenticated=True)

    def on_option_list_option_highlighted(
        self, _event: OptionList.OptionHighlighted
    ) -> None:
        self._set_help_text(_HELP_RUNNING if self._logging_in else _HELP_RETRY)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.worker.group != "mcp_oauth_login" or not event.worker.is_finished:
            return
        self._logging_in = False
        result = event.worker.result
        if not isinstance(result, _LoginResult):
            self._on_login_failed("Authentication failed. Press R to retry.")
            return
        if result.authenticated:
            self.post_message(
                self.MCPOAuthClosed(refreshed=True, server_name=self._server_name)
            )
            return
        self._on_login_failed(
            result.error or "Authentication failed. Press R to retry."
        )

    def _on_auth_url_available(self, url: str) -> None:
        self._auth_url = url
        self._status_message = "Running: Waiting for browser sign-in"
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_option(
            Option(
                Text(
                    f"{chrome_glyph('running')} Running: Waiting for browser sign-in",
                    no_wrap=True,
                    style=self._theme_style("primary"),
                ),
                id="status:waiting",
            )
        )
        option_list.add_option(
            Option(
                Text(
                    f"{_OPTION_PADDING}Press enter to open auth in your browser",
                    no_wrap=True,
                ),
                id=_OAuthOptionId.OPEN,
            )
        )
        option_list.add_option(
            Option(
                Text(f"{_OPTION_PADDING}Copy URL to clipboard", no_wrap=True),
                id=_OAuthOptionId.COPY,
            )
        )
        option_list.add_option(
            Option(
                Text(f"{_OPTION_PADDING}Manually show the URL", no_wrap=True),
                id=_OAuthOptionId.SHOW,
            )
        )
        option_list.highlighted = option_list.get_option_index(_OAuthOptionId.OPEN)
        self._update_detail_text()
        self._set_help_text(_HELP_RUNNING)

    def _on_login_failed(self, message: str) -> None:
        self._status_message = f"Failed: {message}"
        option_list = self.query_one(OptionList)
        option_list.clear_options()
        option_list.add_option(
            Option(
                Text(
                    f"{chrome_glyph('error')} Failed: {message}",
                    style=self._theme_style("error"),
                ),
                id="status:failed",
            )
        )
        self.query_one("#mcpoauth-detail", NoMarkupStatic).update("")
        self._set_help_text(_HELP_RETRY)

    def _open_browser(self) -> None:
        if self._auth_url is None:
            return
        try:
            opened = webbrowser.open(self._auth_url)
        except Exception:
            opened = False
        self._status_message = (
            "Opened in browser"
            if opened
            else "Browser could not be opened; copy or show the URL below"
        )
        self._update_detail_text()
        self._set_help_text(_HELP_RUNNING if self._logging_in else _HELP_RETRY)

    def _copy_url(self) -> None:
        if self._auth_url is None:
            return
        copy_text_to_clipboard(
            self.app, self._auth_url, success_message="Auth URL copied to clipboard"
        )

    def _toggle_url(self) -> None:
        if self._auth_url is None:
            return
        self._auth_url_visible = not self._auth_url_visible
        self._update_detail_text()

    def _update_detail_text(self) -> None:
        detail = self.query_one("#mcpoauth-detail", NoMarkupStatic)
        text = Text()
        status_message = self._status_message
        if status_message == "Opened in browser":
            text.append(
                f"{chrome_glyph('success')} Opened in browser\n",
                style=self._theme_style("success"),
            )
        elif status_message is not None and status_message.startswith(
            "Browser could not"
        ):
            text.append(
                f"{chrome_glyph('error')} {status_message}\n",
                style=self._theme_style("error"),
            )
        if self._auth_url_visible and self._auth_url:
            text.append(f"{self._auth_url}\n\n")
        text.append("Once authenticated in your browser, return to Chartreux")
        detail.update(text)

    def _set_help_text(self, text: str) -> None:
        self.query_one("#mcpoauth-help", NoMarkupStatic).update(shortcut_hint(text))
