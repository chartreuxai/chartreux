"""Bottom-panel surface controller for the TUI app (W4b C2, stage 7)."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import TYPE_CHECKING, Any, cast

from textual.widget import Widget
from textual.widgets import OptionList

from chartreux.app_server.config import THINKING_LEVELS
from chartreux.app_server.models import UserQuestionRequest
from chartreux.cli.textual_ui.widgets.chat_input import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserCommandMessage
from chartreux.observability.logging import get_log_level_chain

if TYPE_CHECKING:
    from collections.abc import Callable

    from chartreux.cli.commands import Command
    from chartreux.cli.textual_ui.app import ChartreuxApp

# Double-escape timing window for the interrupt ladder. app.py re-exports
# this name, so the module-global seam (read from app) is preserved.
DOUBLE_ESC_DELAY = 0.2


class BottomApp(StrEnum):
    """Bottom panel app types.

    Convention: Each value must match the widget class name with "App" suffix removed.
    E.g., QuestionApp -> Question.
    This allows dynamic lookup via: BottomApp[type(widget).__name__.removesuffix("App")]
    """

    Input = auto()
    LogLevelPicker = auto()
    MCP = auto()
    MCPOAuth = auto()
    ModelPicker = auto()
    ProxySetup = auto()
    Question = auto()
    ThemePicker = auto()
    ThinkingPicker = auto()
    Rewind = auto()
    SessionPicker = auto()


class BottomPanelController:
    """Owns the bottom-panel surface arbitration for one app.

    Plain object (no mixins): the app binds itself late via ``_bind_app``
    (never constructor-captured), and collaborators are invoked as
    ``self._app._method(...)`` at call time so instance patches keep
    intercepting. The app exposes thin same-name delegates (plus a
    forwarding property for moved state) so direct callers see no change.
    """

    # Late-bound host reference, assigned via _bind_app after construction.
    _app: ChartreuxApp

    def __init__(self, clock: Callable[[], float]) -> None:
        # Injected monotonic clock (wired to the app.py wrapper seam); the
        # double-escape timing reads it at call time.
        self._clock = clock
        self.current_app: BottomApp = BottomApp.Input
        self.settings_return_bottom_app: BottomApp | None = None
        self.settings_return_providers = False
        self.settings_return_web_search = False
        self.settings_return_state: tuple[str, str | None, int, str | None] | None = (
            None
        )
        self.last_escape_time: float | None = None

    def _bind_app(self, app: ChartreuxApp) -> None:
        """Bind the host app after construction."""
        self._app = app

    def secondary_surface_active(self) -> bool:
        return (
            self._app._agent_transcript_viewer is not None
            or self._app._agent_selection_target is not None
            or self.current_app not in {BottomApp.Input, BottomApp.Question}
            or any(
                screen.id
                in {
                    "settings-screen",
                    "usage-screen",
                    "status-line-settings-screen",
                    "shell-permissions-screen",
                    "websearch-screen",
                }
                or screen.__class__.__name__ == "ProviderWorkbenchScreen"
                for screen in self._app.screen_stack
            )
        )

    def indicate_pending_action(self) -> None:
        pending = bool(
            self._app._pending_callbacks or self._app._pending_local_question
        )
        for indication in (
            "#settings-pending-action",
            "#status-line-settings-pending-action",
            "#shell-permissions-pending-action",
            "#wb-pending-action",
            "#agent-transcript-pending-action",
        ):
            for screen in self._app.screen_stack:
                for widget in screen.query(indication):
                    widget.display = pending
        # The transcript viewer is mounted directly on the app, not on a screen.
        for widget in self._app.query("#agent-transcript-pending-action"):
            widget.display = pending
        for widget in self._app.query("#bottom-pending-action"):
            widget.display = pending and self.current_app not in {
                BottomApp.Input,
                BottomApp.Question,
            }

    async def switch_from_input(self, widget: Widget, scroll: bool = False) -> None:
        bottom_container = self._app.query_one("#bottom-app-container")
        chat = self._app._chat_widget
        should_scroll = scroll and chat.is_at_bottom
        stale_bottom_apps = self._mounted_non_input_bottom_apps()

        with self._app.batch_update():
            if self._app._chat_input_container:
                self._app._chat_input_container.display = False
                self._app._chat_input_container.disabled = True

            for stale in stale_bottom_apps:
                await stale.remove()

            self.current_app = BottomApp[type(widget).__name__.removesuffix("App")]
            await bottom_container.mount(widget)

        if self._app._loading_widget is not None:
            self._app._loading_widget.set_hint_suppressed(
                self.current_app == BottomApp.Question
            )

        self._app._indicate_pending_action()
        self._app.call_after_refresh(widget.focus)
        if should_scroll:
            self._app.call_after_refresh(chat.anchor)

    def _mounted_non_input_bottom_apps(self) -> list[Widget]:
        mounted: list[Widget] = []
        for app in BottomApp:
            if app == BottomApp.Input:
                continue
            try:
                mounted.append(self._app.query_one(f"#{app.value}-app"))
            except Exception:
                pass
        return mounted

    async def replace_bottom_app(self, widget: Widget, scroll: bool = False) -> None:
        bottom_container = self._app.query_one("#bottom-app-container")
        chat = self._app._chat_widget
        should_anchor = chat.is_at_bottom
        old_widgets: list[Widget] = []
        for app in BottomApp:
            if app == BottomApp.Input:
                continue
            try:
                old_widgets.append(self._app.query_one(f"#{app.value}-app"))
            except Exception:
                pass

        with self._app.batch_update():
            if self._app._chat_input_container:
                self._app._chat_input_container.display = False
                self._app._chat_input_container.disabled = True

            self.current_app = BottomApp[type(widget).__name__.removesuffix("App")]
            await bottom_container.mount(widget)
            for old_widget in old_widgets:
                await old_widget.remove()

        if self._app._loading_widget is not None:
            self._app._loading_widget.set_hint_suppressed(
                self.current_app == BottomApp.Question
            )

        self._app.call_after_refresh(widget.focus)
        if should_anchor or scroll:
            self._app.call_after_refresh(chat.anchor)

    def focus_current_bottom_app(self) -> None:
        # Late-bound factory lookup: the lazy _get_*_class factories stay in
        # app.py (module-global seam); importing them at call time keeps this
        # module import-cycle-free and preserves patch interception.
        from chartreux.cli.textual_ui.app import (
            _get_log_level_picker_app_class,
            _get_mcp_app_class,
            _get_mcp_oauth_app_class,
            _get_model_picker_app_class,
            _get_proxy_setup_app_class,
            _get_question_app_class,
            _get_rewind_app_class,
            _get_session_picker_app_class,
            _get_theme_picker_app_class,
            _get_thinking_picker_app_class,
        )

        focus_widget_by_app: dict[BottomApp, type[Widget]] = {
            BottomApp.LogLevelPicker: _get_log_level_picker_app_class(),
            BottomApp.ModelPicker: _get_model_picker_app_class(),
            BottomApp.ThemePicker: _get_theme_picker_app_class(),
            BottomApp.ThinkingPicker: _get_thinking_picker_app_class(),
            BottomApp.ProxySetup: _get_proxy_setup_app_class(),
            BottomApp.Question: _get_question_app_class(),
            BottomApp.SessionPicker: _get_session_picker_app_class(),
            BottomApp.MCP: _get_mcp_app_class(),
            BottomApp.MCPOAuth: _get_mcp_oauth_app_class(),
            BottomApp.Rewind: _get_rewind_app_class(),
        }
        try:
            if self.current_app == BottomApp.Input:
                self._app.query_one(ChatInputContainer).focus_input()
                return
            self._app.query_one(focus_widget_by_app[self.current_app]).focus()
        except Exception:
            pass

    async def switch_to_model_picker_app(self) -> None:
        if self.current_app == BottomApp.ModelPicker:
            return

        self._app._pending_picker_reload = None

        from chartreux.cli.textual_ui.app import _get_model_picker_app_class
        from chartreux.cli.textual_ui.widgets.model_picker import ModelOption

        models = [
            ModelOption(alias=model.alias, display_name=model.display_name)
            for model in self._app.config.models
        ]
        await self._app._switch_from_input(
            _get_model_picker_app_class()(
                models=models,
                current_model=self._app.config.active_model.alias,
                is_pinned=self._app.config.active_model_pinned,
                default_display_name=self._app.config.default_model_display_name,
            )
        )

    async def switch_to_thinking_picker_app(self) -> None:
        if self.current_app == BottomApp.ThinkingPicker:
            return

        self._app._pending_picker_reload = None

        from chartreux.cli.textual_ui.app import _get_thinking_picker_app_class

        await self._app._switch_from_input(
            _get_thinking_picker_app_class()(
                thinking_levels=THINKING_LEVELS,
                current_thinking=self._app.config.active_model.thinking,
            )
        )

    async def switch_to_log_level_picker_app(self) -> None:
        if self.current_app == BottomApp.LogLevelPicker:
            return

        from chartreux.cli.textual_ui.app import _get_log_level_picker_app_class

        await self._app._switch_from_input(
            _get_log_level_picker_app_class()(chain=get_log_level_chain())
        )

    async def switch_to_theme_picker_app(self) -> None:
        if self.current_app == BottomApp.ThemePicker:
            return

        from chartreux.cli.textual_ui.app import _get_theme_picker_app_class
        from chartreux.ui.widgets.theme_picker import sorted_theme_names

        picker = _get_theme_picker_app_class()(
            theme_names=sorted_theme_names(), current_theme=self._app.config.theme
        )
        await self._app._switch_from_input(picker)
        if self._app._pending_theme_selection in sorted_theme_names():
            picker.query_one(OptionList).highlighted = sorted_theme_names().index(
                self._app._pending_theme_selection
            )

    async def switch_to_proxy_setup_app(self) -> None:
        if self.current_app == BottomApp.ProxySetup:
            return

        from chartreux.cli.textual_ui.app import _get_proxy_setup_app_class

        try:
            settings = await self._app.app_server.resources.config.read_proxy()
        except Exception as exc:
            await self._app._mount_and_scroll(
                ErrorMessage(f"Failed to read proxy settings: {exc}")
            )
            return
        await self._app._mount_and_scroll(UserCommandMessage("Proxy setup opened..."))
        self._app._proxy_save_pending = False
        await self._app._switch_from_input(_get_proxy_setup_app_class()(settings))

    async def switch_to_question_app(self, args: UserQuestionRequest) -> None:
        from chartreux.cli.textual_ui.app import _get_question_app_class

        await self._app._switch_from_input(
            _get_question_app_class()(args=args), scroll=True
        )

    async def switch_to_input_app(self, *, return_to_settings: bool = True) -> None:
        previous = self.current_app
        if previous == BottomApp.MCP:
            self._app._show_mcp_discovery_failures()
        if return_to_settings and previous == self.settings_return_bottom_app:
            self.settings_return_bottom_app = None
            reopen_settings = True
        else:
            reopen_settings = False
        if self._app._chat_input_container:
            self._app._chat_input_container.disabled = False
            self._app._chat_input_container.display = True
            self.current_app = BottomApp.Input
        if self._app._loading_widget is not None:
            self._app._loading_widget.set_hint_suppressed(False)

        for app in BottomApp:
            if app != BottomApp.Input:
                try:
                    await self._app.query_one(f"#{app.value}-app").remove()
                except Exception:
                    pass

        if self._app._chat_input_container:
            self._app.call_after_refresh(self._app._chat_input_container.focus_input)
            if self._app._chat_widget.is_at_bottom:
                self._app.call_after_refresh(self._app._chat_widget.anchor)
        if reopen_settings:
            await self._app._show_settings()
        else:
            await self._app._present_pending_callback()

    async def record_settings_return(
        self, name: str, resolved: Command, args: str
    ) -> None:
        """Route a settings-issued command back to its bottom-panel target."""
        bottom_targets = {
            "theme": BottomApp.ThemePicker,
            "log-level": BottomApp.LogLevelPicker,
            "mcp": BottomApp.MCP,
            "proxy-setup": BottomApp.ProxySetup,
        }
        self.settings_return_bottom_app = bottom_targets.get(name)
        self.settings_return_providers = name == "providers"
        self.settings_return_web_search = name == "web-search"
        await self._app._invoke_resolved_command(name, resolved, args, name)
        if name in bottom_targets and self.current_app == BottomApp.Input:
            self.settings_return_bottom_app = None
            await self._app._show_settings()
        elif name == "providers" and (
            self._app._provider_management_worker is None
            or self._app._provider_management_worker.is_finished
        ):
            self.settings_return_providers = False
            await self._app._show_settings()
        elif name == "web-search" and (
            self._app._web_search_worker is None
            or self._app._web_search_worker.is_finished
        ):
            self.settings_return_web_search = False
            await self._app._show_settings()

    def handle_question_app_escape(self) -> None:
        from chartreux.cli.textual_ui.app import _get_question_app_class

        try:
            question_app = self._app.query_one(_get_question_app_class())
            if not question_app.is_within_grace_period():
                question_app.action_cancel()
        except Exception:
            pass
        self.last_escape_time = None

    def handle_log_level_picker_app_escape(self) -> None:
        from chartreux.cli.textual_ui.app import _get_log_level_picker_app_class

        try:
            log_level_picker = self._app.query_one(_get_log_level_picker_app_class())
            log_level_picker.action_cancel()
        except Exception:
            pass
        self.last_escape_time = None

    def handle_model_picker_app_escape(self) -> None:
        from chartreux.cli.textual_ui.app import _get_model_picker_app_class

        try:
            model_picker = self._app.query_one(_get_model_picker_app_class())
            model_picker.post_message(_get_model_picker_app_class().Cancelled())
        except Exception:
            pass
        self.last_escape_time = None

    def handle_theme_picker_app_escape(self) -> None:
        from chartreux.cli.textual_ui.app import _get_theme_picker_app_class

        try:
            theme_picker = self._app.query_one(_get_theme_picker_app_class())
            theme_picker.post_message(
                _get_theme_picker_app_class().Cancelled(
                    original_theme=self._app.config.theme
                )
            )
        except Exception:
            pass
        self.last_escape_time = None

    def handle_thinking_picker_app_escape(self) -> None:
        from chartreux.cli.textual_ui.app import _get_thinking_picker_app_class

        try:
            thinking_picker = self._app.query_one(_get_thinking_picker_app_class())
            thinking_picker.post_message(_get_thinking_picker_app_class().Cancelled())
        except Exception:
            pass
        self.last_escape_time = None

    def handle_session_picker_app_escape(self) -> None:
        from chartreux.cli.textual_ui.app import _get_session_picker_app_class

        try:
            session_picker = self._app.query_one(_get_session_picker_app_class())
            session_picker.action_cancel()
        except Exception:
            pass
        self.last_escape_time = None

    def handle_bottom_app_close_escape(self, widget_type: type[Widget]) -> None:
        try:
            cast(Any, self._app.query_one(widget_type)).action_close()
        except Exception:
            pass
        self.last_escape_time = None

    def handle_input_double_escape(self) -> None:
        """Clear the input when it has content, otherwise enter rewind mode."""
        self.last_escape_time = None
        if self._app._chat_input_container and self._app._chat_input_container.value:
            self._app._clear_input()
        else:
            self._app._start_rewind_mode()

    def try_interrupt_bottom_app_escape(self) -> bool:
        # Late-bound factory lookup: the lazy _get_*_class factories stay in
        # app.py (module-global seam); importing them at call time keeps this
        # module import-cycle-free and preserves patch interception.
        from chartreux.cli.textual_ui.app import (
            _get_mcp_app_class,
            _get_mcp_oauth_app_class,
            _get_proxy_setup_app_class,
        )

        handlers = {
            BottomApp.MCP: lambda: self.handle_bottom_app_close_escape(
                _get_mcp_app_class()
            ),
            BottomApp.MCPOAuth: lambda: self.handle_bottom_app_close_escape(
                _get_mcp_oauth_app_class()
            ),
            BottomApp.ProxySetup: lambda: self.handle_bottom_app_close_escape(
                _get_proxy_setup_app_class()
            ),
            BottomApp.Question: self.handle_question_app_escape,
            BottomApp.LogLevelPicker: self.handle_log_level_picker_app_escape,
            BottomApp.ModelPicker: self.handle_model_picker_app_escape,
            BottomApp.ThemePicker: self.handle_theme_picker_app_escape,
            BottomApp.ThinkingPicker: self.handle_thinking_picker_app_escape,
            BottomApp.SessionPicker: self.handle_session_picker_app_escape,
        }

        if handler := handlers.get(self.current_app):
            handler()
        elif self.current_app == BottomApp.Rewind:
            self._app._handle_rewind_app_escape()
            self.last_escape_time = None
        elif (
            self.current_app == BottomApp.Input
            and self.last_escape_time is not None
            and (self._clock() - self.last_escape_time) < DOUBLE_ESC_DELAY
        ):
            self.handle_input_double_escape()
        else:
            return False
        return True

    def try_interrupt(self) -> bool:
        if self._app._try_interrupt_no_job_steps():
            return True

        interrupted = self._app._try_interrupt_running_job()

        self.last_escape_time = self._clock()
        if self._app._chat_widget.is_at_bottom:
            self._app.call_after_refresh(self._app._chat_widget.anchor)
        self._app._focus_current_bottom_app()
        return interrupted
