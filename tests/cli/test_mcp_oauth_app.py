from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import OptionList

from chartreux.app_server.protocol import MCPAuthUrlParams
from chartreux.cli.textual_ui.widgets.mcp_oauth_app import (
    MCPOAuthApp,
    _LoginResult,
    _OAuthOptionId,
)


class FakeMCPResource:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.login_calls: list[str] = []

    async def login(self, alias: str) -> AsyncGenerator[MCPAuthUrlParams, None]:
        self.login_calls.append(alias)
        yield MCPAuthUrlParams(name=alias, url="https://auth.example.com/oauth")
        if self.error:
            raise self.error


def _make_app(resource: FakeMCPResource | None = None) -> MCPOAuthApp:
    return MCPOAuthApp(
        server_name="oauth", mcp=cast(Any, resource or FakeMCPResource())
    )


def _wire_query(app: MCPOAuthApp) -> tuple[MagicMock, MagicMock, MagicMock]:
    option_list = MagicMock()
    option_list.get_option_index.return_value = 2
    detail = MagicMock()
    help_widget = MagicMock()

    def query(sel: object, *a: object, **kw: object) -> MagicMock:
        if sel is OptionList:
            return option_list
        s = str(sel)
        if "detail" in s:
            return detail
        if "help" in s:
            return help_widget
        return MagicMock()

    app.query_one = cast(Any, query)
    return option_list, detail, help_widget


class TestMCPOAuthApp:
    def test_widget_id(self) -> None:
        app = _make_app()
        assert app.id == "mcpoauth-app"

    def test_action_close_posts_cancelled_message(self) -> None:
        app = _make_app()
        app.post_message = MagicMock()

        app.action_close()

        msg = app.post_message.call_args.args[0]
        assert isinstance(msg, MCPOAuthApp.MCPOAuthClosed)
        assert msg.refreshed is False
        assert msg.server_name == ""

    def test_auth_url_available_shows_menu(self) -> None:
        app = _make_app()
        option_list, _detail, _help = _wire_query(app)

        app._on_auth_url_available("https://auth.example.com/oauth")

        assert app._auth_url == "https://auth.example.com/oauth"
        assert option_list.clear_options.called
        assert option_list.add_option.call_count == 4
        option_ids = [
            call.args[0].id
            for call in option_list.add_option.call_args_list
            if hasattr(call.args[0], "id") and call.args[0].id
        ]
        assert _OAuthOptionId.OPEN in option_ids
        assert _OAuthOptionId.COPY in option_ids
        assert _OAuthOptionId.SHOW in option_ids

        status_option = next(
            option
            for option in option_list.add_option.call_args_list
            if getattr(option.args[0], "id", None) == "status:waiting"
        )
        assert status_option.args[0].disabled is False

    def test_status_row_enter_shows_footer_feedback(self) -> None:
        app = _make_app()
        _option_list, _detail, help_widget = _wire_query(app)
        app._logging_in = True
        event = MagicMock()
        event.option.id = "status:waiting"

        app.on_option_list_option_selected(event)

        assert "in progress" in str(help_widget.update.call_args.args[0]).lower()
        assert "retry" not in str(help_widget.update.call_args.args[0]).lower()

    def test_failed_status_row_enter_shows_retry_feedback(self) -> None:
        app = _make_app()
        _option_list, _detail, help_widget = _wire_query(app)
        app._logging_in = False
        event = MagicMock()
        event.option.id = "status:failed"

        app.on_option_list_option_selected(event)

        feedback = str(help_widget.update.call_args.args[0]).lower()
        assert "status row cannot be activated" in feedback
        assert "retry" in feedback

    def test_running_help_omits_retry_and_failed_help_restores_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        app = _make_app()
        _option_list, _detail, help_widget = _wire_query(app)

        def close_worker(coroutine: Any, **_kwargs: Any) -> None:
            coroutine.close()

        monkeypatch.setattr(app, "run_worker", close_worker)

        app._start_login()
        running_help = str(help_widget.update.call_args.args[0]).lower()
        assert "in progress" in running_help
        assert "retry" not in running_help

        app._on_login_failed("bad auth")
        failed_help = str(help_widget.update.call_args.args[0]).lower()
        assert "retry" in failed_help

    @pytest.mark.asyncio
    async def test_run_login_streams_auth_url(self) -> None:
        resource = FakeMCPResource()
        app = _make_app(resource)
        _wire_query(app)

        result = await app._run_login()

        assert result == _LoginResult(authenticated=True)
        assert resource.login_calls == ["oauth"]
        assert app._auth_url == "https://auth.example.com/oauth"

    @pytest.mark.asyncio
    async def test_run_login_returns_error(self) -> None:
        resource = FakeMCPResource(error=ValueError("bad auth"))
        app = _make_app(resource)
        _wire_query(app)

        result = await app._run_login()

        assert result == _LoginResult(authenticated=False, error="bad auth")

    def test_worker_success_posts_closed(self) -> None:
        app = _make_app()
        app.post_message = MagicMock()
        worker = MagicMock()
        worker.group = "mcp_oauth_login"
        worker.is_finished = True
        worker.result = _LoginResult(authenticated=True)
        event = MagicMock()
        event.worker = worker

        app.on_worker_state_changed(event)

        msg = app.post_message.call_args.args[0]
        assert isinstance(msg, MCPOAuthApp.MCPOAuthClosed)
        assert msg.refreshed is True
        assert msg.server_name == "oauth"

    def test_worker_failure_shows_retry_message(self) -> None:
        app = _make_app()
        _wire_query(app)
        app.post_message = MagicMock()
        worker = MagicMock()
        worker.group = "mcp_oauth_login"
        worker.is_finished = True
        worker.result = _LoginResult(authenticated=False, error="bad auth")
        event = MagicMock()
        event.worker = worker

        app.on_worker_state_changed(event)

        app.post_message.assert_not_called()
        assert app._status_message == "Failed: bad auth"

    def test_open_browser_calls_webbrowser(self) -> None:
        app = _make_app()
        app._auth_url = "https://auth.example.com/oauth"
        app.query_one = MagicMock()

        with patch("chartreux.cli.textual_ui.widgets.mcp_oauth_app.webbrowser") as wb:
            app._open_browser()
            wb.open.assert_called_once_with("https://auth.example.com/oauth")

        assert app._status_message == "Opened in browser"
        detail = app.query_one.return_value
        assert any(
            "✓ Opened in browser" in call.args[0].plain
            for call in detail.update.call_args_list
        )

    @pytest.mark.parametrize("failure", [False, OSError("no browser")])
    def test_open_browser_failure_offers_manual_url(self, failure: object) -> None:
        app = _make_app()
        app._auth_url = "https://auth.example.com/oauth"
        app._logging_in = True
        _options, detail, _help = _wire_query(app)

        with patch(
            "chartreux.cli.textual_ui.widgets.mcp_oauth_app.webbrowser.open"
        ) as open_browser:
            if isinstance(failure, Exception):
                open_browser.side_effect = failure
            else:
                open_browser.return_value = failure
            app._open_browser()

        assert app._status_message is not None
        assert app._status_message.startswith("Browser could not be opened")
        assert any(
            "copy or show the URL" in call.args[0].plain
            for call in detail.update.call_args_list
        )
        assert not any(
            "Opened in browser" in call.args[0].plain
            for call in detail.update.call_args_list
        )

    def test_copy_url_calls_clipboard(self) -> None:
        app = cast(Any, _make_app())
        app._auth_url = "https://auth.example.com/oauth"

        with (
            patch.object(
                type(app), "app", new_callable=lambda: property(lambda s: MagicMock())
            ),
            patch(
                "chartreux.cli.textual_ui.widgets.mcp_oauth_app.copy_text_to_clipboard"
            ) as copy_fn,
        ):
            app._copy_url()

        copy_fn.assert_called_once()
        assert copy_fn.call_args.args[1] == "https://auth.example.com/oauth"
