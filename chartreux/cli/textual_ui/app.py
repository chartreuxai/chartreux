from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import aclosing, suppress
from dataclasses import dataclass, replace
from functools import partial
import gc
from importlib import import_module
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING, Any, ClassVar
from weakref import WeakKeyDictionary
import webbrowser

from rich import print as rprint
from rich.text import Text
from textual.app import App, ComposeResult, SuspendNotSupported
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, VerticalGroup, VerticalScroll
from textual.dom import NoScreen
from textual.driver import Driver
from textual.events import (
    AppBlur,
    AppFocus,
    DescendantFocus,
    MouseScrollDown,
    MouseScrollUp,
    MouseUp,
)
from textual.screen import Screen
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Static, TextArea
from textual.worker import Worker, WorkerError, WorkerFailed, WorkerState

from chartreux import __version__ as CORE_VERSION
from chartreux.app_server.config import ConfigView, ThinkingLevel
from chartreux.app_server.events import (
    AgentsUpdate,
    AppServerEvent,
    CallbackRequested,
    HistoryEntryAdded,
    HistoryEntryUpdated,
    ServerError,
    ServerWarning,
    SessionCompacted,
    SessionContextCleared,
    SessionSnapshot,
    SessionUpdated,
    StatsUpdated,
    TurnCompleted,
    TurnQueueUpdated,
    TurnStarted,
    TurnUpdated,
)
from chartreux.app_server.models import (
    ConfigIssue,
    ImageAttachment,
    PreparedPrompt,
    PublicCallbackEntry,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicHistoryEntry,
    PublicMessageEntry,
    PublicNoticeEntry,
    PublicQueuedTurn,
    PublicReasoningEntry,
    PublicTurnStatus,
    TokenUsage,
    UsageWindow,
    UsageWindowSummaries,
    UserInputCallbackOutput,
    UserQuestionRequest,
    UserQuestionResult,
    WaitingForInputNoticeDetail,
)
from chartreux.app_server.protocol import (
    AgentSummaryModel,
    AppServerResponseError,
    ConfigReadResponse,
    ConfigWriteOpWire,
    ProtocolError,
    ProtocolErrorCode,
    UsageReadResponse,
    UsageUpdatedParams,
)
from chartreux.cli.commands import Command, CommandContext, CommandRegistry
from chartreux.cli.process_start import PROCESS_START_WALLCLOCK
from chartreux.cli.textual_ui.agent_browser import (
    AgentBrowserController,
    _AgentTranscriptSource as _AgentTranscriptSource,
)
from chartreux.cli.textual_ui.bottom_panel import (
    DOUBLE_ESC_DELAY as DOUBLE_ESC_DELAY,
    BottomApp,
    BottomPanelController,
)
from chartreux.cli.textual_ui.external_editor import ExternalEditor
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.mcp_commands import (
    MCP_ADD_HELP,
    is_mcp_add_help_request,
    parse_mcp_add_args,
    parse_mcp_subcommand,
)
from chartreux.cli.textual_ui.message_queue import (
    QueueController,
    QueuePorts,
    SideChannelController,
    SideChannelPorts,
)
from chartreux.cli.textual_ui.notifications import TextualNotificationAdapter
from chartreux.cli.textual_ui.provider_services import (
    _ProviderConfigService,
    _ProviderCredentials,
    persist_api_key_for_provider as persist_api_key_for_provider,
)
from chartreux.cli.textual_ui.quit_manager import QuitConfirmKey, QuitManager
from chartreux.cli.textual_ui.root_recovery import RootRecoveryController
from chartreux.cli.textual_ui.scheduled_loop_runner import ScheduledLoopCommands
from chartreux.cli.textual_ui.session_flow import (
    SessionFlowController,
    _PickerState as _PickerState,
)
from chartreux.cli.textual_ui.startup_types import (
    AppServerBootstrap,
    AppServerSource,
    AppServerStarter,
)
from chartreux.cli.textual_ui.transcript_presenter import (
    _TRANSCRIPT_HIGH_MARK as _TRANSCRIPT_HIGH_MARK,
    TranscriptPresenter,
    _SessionTranscriptWindow,
)
from chartreux.cli.textual_ui.turn_supervisor import (
    _BENIGN_TURN_ERROR_CODES as _BENIGN_TURN_ERROR_CODES,
    _INTERRUPT_SETTLE_TIMEOUT as _INTERRUPT_SETTLE_TIMEOUT,
    _INTERRUPT_STILL_STOPPING_WARNING as _INTERRUPT_STILL_STOPPING_WARNING,
    _INTERRUPT_WAIT_TIMEOUT as _INTERRUPT_WAIT_TIMEOUT,
    _MAX_INCOMPLETE_STREAM_RETRIES as _MAX_INCOMPLETE_STREAM_RETRIES,
    _RETRYABLE_TURN_ERROR_CODES as _RETRYABLE_TURN_ERROR_CODES,
    TurnSupervisor,
    _public_entry as _public_entry,
)
from chartreux.cli.textual_ui.usage_controller import (
    _USAGE_POLL_INTERVAL_SECONDS,
    UsageController,
)
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar, agent_is_active
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.banner.banner import Banner
from chartreux.cli.textual_ui.widgets.chat_input import (
    ChatInputBody,
    ChatInputContainer,
)
from chartreux.cli.textual_ui.widgets.chat_input.input_kinds import (
    Bash,
    EmptyBash,
    Prompt,
    Skill,
    SlashCommand,
    classify,
)
from chartreux.cli.textual_ui.widgets.chat_input.text_area import ChatTextArea
from chartreux.cli.textual_ui.widgets.compact import CompactMessage
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.inline_notice import InlineNotice
from chartreux.cli.textual_ui.widgets.links import normalize_url
from chartreux.cli.textual_ui.widgets.load_more import HistoryLoadMoreRequested
from chartreux.cli.textual_ui.widgets.loading import (
    DEFAULT_LOADING_STATUS,
    INITIALIZING_LOADING_STATUS,
    LoadingWidget,
)
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    CustomToolsDeprecationMessage,
    ErrorMessage,
    ErrorRetryRequested,
    PlanFileMessage,
    ReasoningMessage,
    SlashCommandMessage,
    StreamingMessageBase,
    UserCommandMessage,
    UserMessage,
    WarningMessage,
    WhatsNewMessage,
)
from chartreux.cli.textual_ui.widgets.reload_message import ReloadConfigMessage
from chartreux.cli.textual_ui.widgets.session_status_line import (
    SessionStatusLine,
    SessionStatusState,
)
from chartreux.cli.textual_ui.widgets.tool_grouping import ToolGroupExpansionState
from chartreux.cli.textual_ui.widgets.tool_widgets import EditResultWidget
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing import HistoryLoadMoreManager, SessionWindowing
from chartreux.cli.textual_ui.windowing.transcript import TranscriptAnchor
from chartreux.config_values import FALLBACK_THEME
from chartreux.observability.logging import (
    get_effective_log_level,
    get_log_level_chain,
    logger,
    set_config_log_level,
    set_session_override,
)
from chartreux.ui.clipboard import (
    NATIVE_COPY_HINT,
    ClipboardCopyResult,
    copy_selection_to_clipboard,
    copy_text_to_clipboard,
)
from chartreux.ui.providers.graduation import GraduationStore
from chartreux.ui.theme import resolve_auto_theme, resolve_theme, resolve_theme_name
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.utils.paths import get_chartreux_home, is_dangerous_directory
from chartreux.utils.session_id import shorten_session_id

# The turn error/retry/interrupt constants moved to turn_supervisor.py (10.1a);
# app.py re-exports them, so the module-global seams (tests patch
# ``chartreux.cli.textual_ui.app._INTERRUPT_WAIT_TIMEOUT`` /
# ``_INTERRUPT_SETTLE_TIMEOUT``) keep resolving through this module.
_DIAGNOSTIC_HEARTBEAT_DRIFT_WARNING_SECONDS = 0.25

if TYPE_CHECKING:
    from chartreux.app_server.session import AppServerSession, SessionExitSummary
    from chartreux.cli.textual_ui.quit_manager import ExitConsequencesScreen
    from chartreux.cli.textual_ui.screens.settings import SettingsScreen
    from chartreux.cli.textual_ui.widgets.debug_console import DebugConsole
    from chartreux.cli.textual_ui.widgets.log_level_picker import LogLevelPickerApp
    from chartreux.cli.textual_ui.widgets.mcp_app import MCPApp
    from chartreux.cli.textual_ui.widgets.mcp_oauth_app import MCPOAuthApp
    from chartreux.cli.textual_ui.widgets.model_picker import ModelPickerApp
    from chartreux.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp
    from chartreux.cli.textual_ui.widgets.question_app import QuestionApp
    from chartreux.cli.textual_ui.widgets.rewind_app import RewindApp
    from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
    from chartreux.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp
    from chartreux.cli.textual_ui.word_selection import WordSelectScreen
    from chartreux.ui.web_search import WebSearchScreen
    from chartreux.ui.widgets.theme_picker import ThemePickerApp


def _get_mcp_app_class() -> type[MCPApp]:
    from chartreux.cli.textual_ui.widgets.mcp_app import MCPApp

    return MCPApp


def _get_mcp_oauth_app_class() -> type[MCPOAuthApp]:
    from chartreux.cli.textual_ui.widgets.mcp_oauth_app import MCPOAuthApp

    return MCPOAuthApp


def _get_session_picker_app_class() -> type[SessionPickerApp]:
    from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp

    return SessionPickerApp


def _get_model_picker_app_class() -> type[ModelPickerApp]:
    from chartreux.cli.textual_ui.widgets.model_picker import ModelPickerApp

    return ModelPickerApp


def _get_thinking_picker_app_class() -> type[ThinkingPickerApp]:
    from chartreux.cli.textual_ui.widgets.thinking_picker import ThinkingPickerApp

    return ThinkingPickerApp


def _get_log_level_picker_app_class() -> type[LogLevelPickerApp]:
    from chartreux.cli.textual_ui.widgets.log_level_picker import LogLevelPickerApp

    return LogLevelPickerApp


def _get_theme_picker_app_class() -> type[ThemePickerApp]:
    from chartreux.ui.widgets.theme_picker import ThemePickerApp

    return ThemePickerApp


def _get_settings_screen_class() -> type[SettingsScreen]:
    from chartreux.cli.textual_ui.screens.settings import SettingsScreen

    return SettingsScreen


def _get_web_search_screen_class() -> type[WebSearchScreen]:
    from chartreux.ui.web_search import WebSearchScreen

    return WebSearchScreen


def _get_proxy_setup_app_class() -> type[ProxySetupApp]:
    from chartreux.cli.textual_ui.widgets.proxy_setup_app import ProxySetupApp

    return ProxySetupApp


def _get_question_app_class() -> type[QuestionApp]:
    from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

    return QuestionApp


def _get_rewind_app_class() -> type[RewindApp]:
    from chartreux.cli.textual_ui.widgets.rewind_app import RewindApp

    return RewindApp


def _get_exit_consequences_screen_class() -> type[ExitConsequencesScreen]:
    from chartreux.cli.textual_ui.quit_manager import ExitConsequencesScreen

    return ExitConsequencesScreen


def _get_word_select_screen_class() -> type[WordSelectScreen]:
    from chartreux.cli.textual_ui.word_selection import WordSelectScreen

    return WordSelectScreen


def is_progress_event(event: AppServerEvent) -> bool:
    entry = _public_entry(event)
    return isinstance(
        entry, (PublicMessageEntry, PublicReasoningEntry, PublicEffectEntry)
    )


async def _session_selection_restrictions(
    app: ChartreuxApp,
) -> tuple[Sequence[str], dict[str, str]]:
    """The session's effective allowed_models and thinking overrides.

    ``allowed_models`` comes from the projected session config; the per-model
    thinking overrides are read from the server-side config fields, which the
    projection does not carry. Both are optional host inputs for the workbench.
    """
    allowed_models = getattr(app.config, "allowed_models", ())
    app_server = getattr(app, "app_server", None)
    if app_server is None:
        return allowed_models, {}
    response = await app_server.resources.config.read_fields()
    entry = next(
        (item for item in response.fields if item.name == "thinking_overrides"), None
    )
    value = entry.value if entry is not None else None
    overrides = (
        {str(name): str(level) for name, level in value.items()}
        if isinstance(value, dict)
        else {}
    )
    return allowed_models, overrides


# Smooth per-notch wheel scroll duration. Kept short so consecutive notches chain
# into continuous motion at the same average speed as an instant jump.
WHEEL_SCROLL_DURATION = 0.1


class ChatScroll(VerticalScroll):
    """Optimized scroll container that skips cascading style recalculations.

    Host reactions are injected as explicit callbacks instead of back-references
    into the app: ``on_scroll_activity`` fires whenever ``scroll_y`` changes and
    ``on_user_scroll`` fires when the user wheels the container. Both default to
    None, which makes the widget inert outside a host that provides them.
    """

    def __init__(
        self,
        *children: Widget,
        name: str | None = None,
        id: str | None = None,
        classes: str | None = None,
        disabled: bool = False,
        on_scroll_activity: Callable[[ChatScroll, float], None] | None = None,
        on_user_scroll: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            *children, name=name, id=id, classes=classes, disabled=disabled
        )
        self._on_scroll_activity = on_scroll_activity
        self._on_user_scroll = on_user_scroll

    @property
    def is_at_bottom(self) -> bool:
        return self.scroll_target_y >= self.max_scroll_y

    _reanchor_pending: bool = False
    _scrolling_down: bool = False
    _transcript_tracks_user_scroll: bool = True
    _transcript_engine_scroll: bool = False
    _transcript_layout_change: bool = False

    @property
    def _is_selecting(self) -> bool:
        try:
            return self.screen._selecting
        except NoScreen:
            return False

    def anchor(self, anchor: bool = True) -> None:
        if anchor and self._is_selecting:
            return
        super().anchor(anchor)

    def preserve_scroll_position(self) -> None:
        super().release_anchor()

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        if self._is_selecting and new_value < old_value:
            self._anchor_released = True
        super().watch_scroll_y(old_value, new_value)
        self._scrolling_down = new_value >= old_value
        if self._on_scroll_activity is not None:
            self._on_scroll_activity(self, new_value)

    def release_anchor(self) -> None:
        super().release_anchor()
        # Textual's MRO dispatch calls Widget._on_mouse_scroll_down AFTER
        # our override, so any re-anchor we do gets immediately undone.
        # Defer the re-check until all handlers for this event have finished.
        if not self._reanchor_pending:
            self._reanchor_pending = True
            self.call_later(self._maybe_reanchor)

    def _maybe_reanchor(self) -> None:
        self._reanchor_pending = False
        if (
            self._anchored
            and self._anchor_released
            and self.is_at_bottom
            and self._scrolling_down
        ):
            self.anchor()

    def update_node_styles(self, animate: bool = True) -> None:
        pass

    def _on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._smooth_wheel_scroll(event, self._scroll_down_for_pointer)

    def _on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._smooth_wheel_scroll(event, self._scroll_up_for_pointer)

    def _smooth_wheel_scroll(
        self, event: MouseScrollDown | MouseScrollUp, scroller: Callable[..., bool]
    ) -> None:
        # Leave ctrl/shift (horizontal) wheel to the base Widget handler, which
        # Textual's MRO dispatch still reaches because we don't prevent_default.
        if event.ctrl or event.shift or not self.allow_vertical_scroll:
            return
        # Cover the same per-notch distance as the default handler, but render it
        # as a smooth linear glide so motion passes through each line one by one
        # instead of jumping the full sensitivity in a single frame. prevent_default
        # breaks the MRO loop so the base handler's instant jump never runs on top.
        event.prevent_default()
        if self._on_user_scroll is not None:
            self._on_user_scroll()
        if scroller(animate=True, duration=WHEEL_SCROLL_DURATION, easing="linear"):
            event.stop()


def _monotonic_clock() -> float:
    """Monotonic clock seam injected into controllers (ALC §1.4)."""
    return time.monotonic()


_TEXTUAL_THEME_MAP = {"auto": None, "light": "ansi-light", "dark": "ansi-dark"}


_DEFAULT_TYPING_DEBOUNCE_MS = 1000
_TYPING_DEBOUNCE_ENV_VAR = "CHARTREUX_TYPING_GRACE_PERIOD_MS"


def _resolve_typing_debounce_s() -> float:
    try:
        ms = int(os.environ[_TYPING_DEBOUNCE_ENV_VAR])
        if ms < 0:
            raise ValueError
    except (KeyError, ValueError):
        ms = _DEFAULT_TYPING_DEBOUNCE_MS
    return ms / 1000


@dataclass(frozen=True, slots=True)
class StartupOptions:
    initial_prompt: str | None = None
    show_resume_picker: bool = False
    is_resuming_session: bool = False
    prompt_for_workspace_trust: bool = False
    startup_show_resume_picker: bool | None = None
    startup_prompt_for_workspace_trust: bool | None = None
    resume_session_id: str | None = None
    continue_latest: bool = False


def _split_app_server_source(
    source: AppServerSource,
) -> tuple[AppServerSession | None, AppServerStarter | None]:
    from chartreux.app_server.session import AppServerSession

    if isinstance(source, AppServerSession):
        return source, None
    return None, source


_REJECT_HINT_BUSY = "wait for the current job to finish."
_REJECT_HINT_PAUSED = "clear the queue first or remove this input."

_UNTRUSTED_CONFIG_WARNING_SECTION = "untrusted_config_warning"


def _usage_poll_interval_seconds() -> float:
    """Late-bound read of the re-exported usage poll-interval seam.

    Tests patch ``chartreux.cli.textual_ui.app._USAGE_POLL_INTERVAL_SECONDS``;
    reading the module global at call time keeps that patch effective.
    """
    return _USAGE_POLL_INTERVAL_SECONDS


def _interrupt_wait_timeout() -> float:
    """Late-bound read of the re-exported interrupt wait-timeout seam.

    Tests patch ``chartreux.cli.textual_ui.app._INTERRUPT_WAIT_TIMEOUT``;
    reading the module global at call time keeps that patch effective.
    """
    return _INTERRUPT_WAIT_TIMEOUT


def _interrupt_settle_timeout() -> float:
    """Late-bound read of the re-exported interrupt settle-timeout seam.

    Tests patch ``chartreux.cli.textual_ui.app._INTERRUPT_SETTLE_TIMEOUT``;
    reading the module global at call time keeps that patch effective.
    """
    return _INTERRUPT_SETTLE_TIMEOUT


class ChartreuxApp(App):  # noqa: PLR0904
    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "app.tcss"
    PAUSE_GC_ON_SCROLL: ClassVar[bool] = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("f1", "show_help", "Help", show=False),
        Binding("ctrl+c", "interrupt_or_quit", "Quit", show=False),
        Binding("ctrl+d", "delete_right_or_quit", "Quit", show=False, priority=True),
        Binding("ctrl+z", "suspend_with_message", "Suspend", show=False, priority=True),
        Binding("escape", "interrupt", "Interrupt", show=False, priority=True),
        Binding("ctrl+o", "toggle_tool", "Toggle Tool", show=False),
        Binding("ctrl+y", "copy_selection", "Copy", show=False, priority=True),
        Binding("ctrl+shift+c", "copy_selection", "Copy", show=False, priority=True),
        Binding("shift+up", "scroll_chat_up", "Scroll Up", show=False, priority=True),
        Binding(
            "shift+down", "scroll_chat_down", "Scroll Down", show=False, priority=True
        ),
        Binding(
            "ctrl+g", "open_plan_in_editor", "Edit Plan", show=False, priority=False
        ),
        Binding("ctrl+backslash", "toggle_debug_console", "Debug Console", show=False),
        Binding(
            "ctrl+shift+a", "toggle_agent_browser", "Background Agents", show=False
        ),
        # R1 (plan v0.5.6-recovery): the keyboard path of the "Retry — new
        # turn" affordance. The chat input's focus policy reclaims focus from
        # transcript widgets, so the displayed button is pointer-activated and
        # this binding is the keyboard path; both invoke the EXISTING retry
        # operation, gated exactly like the /retry command.
        Binding("ctrl+r", "retry_turn", "Retry turn", show=False),
    ]

    _main_ui_mounted: bool = False
    _mount_first: bool = False
    _initial_config_response: ConfigReadResponse | None = None

    def get_driver_class(self) -> type[Driver]:
        """Patch the platform driver to strip malformed terminal reports from input."""
        from chartreux.cli.textual_ui.terminal_input_filter import patch_driver_parser

        driver_class = super().get_driver_class()
        patch_driver_parser()
        return driver_class

    def open_url(self, url: str, *, new_tab: bool = True) -> None:
        """Normalize URLs before opening so illegal characters (e.g. a literal
        backslash from model output) can't break the platform opener. Both link
        paths — tool-output links and clicked markdown links — funnel here.
        """
        super().open_url(normalize_url(url), new_tab=new_tab)

    def __init__(
        self,
        history_file: Path,
        app_server: AppServerSource,
        *,
        startup: StartupOptions | None = None,
        current_version: str = CORE_VERSION,
        terminal_notifier: TextualNotificationAdapter | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._app_server, self._start_app_server = _split_app_server_source(app_server)
        self._client_dependencies_ready = False
        self._prepare_lock = asyncio.Lock()
        self.commands: CommandRegistry = CommandRegistry()
        self._loop_commands: ScheduledLoopCommands
        self._terminal_notifier = terminal_notifier or TextualNotificationAdapter(
            self,
            get_enabled=lambda: (
                self._app_server is not None and self.config.enable_notifications
            ),
            default_title="Chartreux",
        )
        # The turn/interrupt/retry state moved to the turn supervisor (10.2a);
        # the app forwards every field through the properties defined below.
        # Rebinding replaces the attached session and all presentation derived
        # from it. Hold newly delivered events until that presentation reset is
        # complete so the replacement session cannot render into the old UI.
        self._resume_ui_ready = asyncio.Event()
        self._resume_ui_ready.set()
        self._bash_task: asyncio.Task | None = None
        self._app_server_events_worker: Worker[None] | None = None
        self._settings_worker: Worker[None] | None = None
        self._provider_management_worker: Worker[None] | None = None
        self._web_search_worker: Worker[None] | None = None
        self._shutdown_started = False
        # Global spend outlives the attached session and its presentation resets.
        self._usage_summaries: UsageWindowSummaries | None = None
        self._usage_controller = UsageController(
            current_binding=self._usage_sync_binding,
            is_binding_current=self._is_usage_binding_current,
            status_line_segments=lambda: self.config.status_line.segments,
            on_updated=self._on_usage_updated,
            poll_interval=_usage_poll_interval_seconds,
        )
        self._usage_worker: Worker[None] | None = None
        self._app_server_event_handler_lock = asyncio.Lock()
        self._init_controllers()

        self._loading_widget: LoadingWidget | None = None
        self._secondary_returned = asyncio.Event()
        self._secondary_returned.set()

        self.event_handler: EventHandler | None = None

        self._chat_input_container: ChatInputContainer | None = None

        # The bottom panel controller owns the bottom-app surface arbitration
        # (including the current bottom app and the escape-ladder timing); the
        # app binds itself late (never constructor-captured) and injects the
        # monotonic clock through the app.py wrapper seam.
        self._bottom_panel = BottomPanelController(clock=_monotonic_clock)
        self._bottom_panel._bind_app(self)

        self.history_file = history_file

        # The transcript presenter owns the windowing state; the app binds
        # itself late (never constructor-captured) and forwards every field
        # through the properties defined next to _chat_widget below.
        self._transcript_presenter = TranscriptPresenter(
            get_show_timestamps=lambda: self.config.show_message_timestamps
        )
        self._transcript_presenter._bind_app(self)
        # The agent browser controller owns the viewer/selection state; the
        # app binds itself late (never constructor-captured) and forwards
        # every moved field through the properties defined below.
        self._agent_browser = AgentBrowserController()
        self._agent_browser._bind_app(self)
        # The session flow controller owns the picker preview/origin state and
        # the resume flow; the app binds itself late (never constructor-captured)
        # and forwards every moved field through the properties defined below.
        self._session_flow = SessionFlowController()
        self._session_flow._bind_app(self)
        # The turn supervisor owns the turn/interrupt/retry state and flow; the
        # app binds itself late (never constructor-captured) and forwards every
        # moved field through the properties defined below.
        self._turn_supervisor = TurnSupervisor(
            interrupt_wait_timeout=_interrupt_wait_timeout,
            interrupt_settle_timeout=_interrupt_settle_timeout,
        )
        self._turn_supervisor._bind_app(self)
        self._recovery_issues: dict[str, tuple[str, str]] = {}
        self._startup_issue_resolved = False
        self._pending_theme_selection: str | None = None
        self._pending_picker_reload: tuple[str, str] | None = None
        self._picker_update_pending = False
        self._proxy_save_pending = False
        self._current_version = current_version
        self._configure_startup_options(startup)
        self._quit_manager = QuitManager(self)
        self._init_cached_widgets()
        if self._app_server is not None:
            self._initialize_client_dependencies()

    def _init_cached_widgets(self) -> None:
        self._banner: Banner | None = None
        self._whats_new_message: WhatsNewMessage | None = None
        self._cached_messages_area: Widget | None = None
        self._cached_chat: ChatScroll | None = None
        self._cached_loading_area: Widget | None = None
        self._status_line: SessionStatusLine | None = None
        self._branch_identity: tuple[str, str] | None = None
        self._branch_generation = 0
        self._root_compacting = False
        self._root_context_unknown = False
        self._agent_bar: AgentBar | None = None
        self._graduation = GraduationStore()
        self._graduation_notice: InlineNotice | None = None
        self._graduation_live_attempts: dict[str, str] = {}
        self._session_only_credential_envs: set[str] = set()
        self._diagnostic_heartbeat_timer: Timer | None = None
        self._diagnostic_last_heartbeat = 0.0
        self._diagnostic_max_heartbeat_drift = 0.0
        self._debug_console: DebugConsole | None = None
        self._debug_console_focus_target: Widget | None = None
        self._queue_selected_widget: Widget | None = None
        self._fatal_init_error = False
        self._force_quit_task: asyncio.Task[None] | None = None

    def _mark_session_ready(self) -> None:
        self._session_ready.set()
        self._sync_usage_subscription()
        self._sync_terminal_title()

    def _on_session_title_changed(self, title: str) -> None:
        self._terminal_notifier.set_default_title(title)

    def _sync_terminal_title(self) -> None:
        if self._app_server is None:
            return
        # A blank title resets the tab to the default.
        title = self.app_server.resources.runtime.session_log.title
        self._terminal_notifier.set_default_title(title or "")

    @property
    def app_server(self) -> AppServerSession:
        if self._app_server is None:
            raise RuntimeError("App server has not been started")
        return self._app_server

    async def prepare(self) -> None:
        if self._client_dependencies_ready:
            return
        async with self._prepare_lock:
            if self._client_dependencies_ready:
                return
            if self._app_server is None:
                if self._start_app_server is None:
                    raise RuntimeError("App server starter is unavailable")
                self._app_server = await self._start_app_server()
            self._initialize_client_dependencies()

    def _initialize_client_dependencies(self) -> None:
        if self._client_dependencies_ready:
            return
        self.commands = self._build_command_registry()
        self._loop_commands = ScheduledLoopCommands(
            self.app_server.resources.loops,
            tools_collapsed=lambda: self._tools_collapsed,
        )
        self._client_dependencies_ready = True

    def _configure_startup_options(self, startup: StartupOptions | None) -> None:
        opts = startup or StartupOptions()
        self._initial_prompt = opts.initial_prompt
        self._show_resume_picker = opts.show_resume_picker
        self._startup_show_resume_picker = (
            opts.startup_show_resume_picker
            if opts.startup_show_resume_picker is not None
            else opts.show_resume_picker
        )
        self._is_resuming_session = opts.is_resuming_session
        self._resume_session_id = opts.resume_session_id
        self._continue_latest = opts.continue_latest
        self._startup_prompt_for_workspace_trust = (
            opts.startup_prompt_for_workspace_trust
            if opts.startup_prompt_for_workspace_trust is not None
            else opts.prompt_for_workspace_trust
        )
        self._startup_prompt_processed = False
        self._startup_command_availability_ready = asyncio.Event()
        self._initial_history_loaded = asyncio.Event()
        # Gates the turn path until the session is bound: mount-first renders the
        # input box before app_server exists, so a submit awaits this instead of
        # crashing on the unbound property.
        self._session_ready = asyncio.Event()
        # Guards against double-display of MCP/startup notices across the
        # readiness-watch and finish-resume-notices race; unrelated to picker preview.
        self._post_init_notices_shown: bool = False
        self._custom_tools_deprecation_message: CustomToolsDeprecationMessage | None = (
            None
        )
        self._custom_tools_deprecation_message_lock = asyncio.Lock()

    @property
    def config(self) -> ConfigView:
        return self.app_server.resources.config.current

    def _build_queue_ports(self) -> QueuePorts:
        return QueuePorts(
            mount_and_scroll=self._mount_and_scroll,
            current_turn_queue=lambda: self.app_server.turn_queue,
            enqueue_turn=self._enqueue_queued_turn,
            replace_queued_turn=self._replace_queued_turn,
            remove_queued_turn=lambda queue_item_id: self.app_server.remove_queued_turn(
                queue_item_id
            ),
            resume_turn_queue=lambda: self.app_server.resume_turn_queue(),
            steer_turn=self._steer_queue_prompts,
            turn_has_started=self._queued_turn_has_started,
            set_loading_queue_count=self._set_loading_queue_count,
            get_show_message_timestamps=lambda: self.config.show_message_timestamps,
            history_message=lambda entry_id: next(
                (
                    entry
                    for entry in self.app_server.history
                    if isinstance(entry, PublicMessageEntry) and entry.id == entry_id
                ),
                None,
            ),
        )

    def _init_controllers(self) -> None:
        self._queue = QueueController(self._build_queue_ports())
        self._side_channel = SideChannelController(
            SideChannelPorts(invoke_command=self._invoke_resolved_command)
        )
        # R2/OD3: the TUI half of bounded automatic root-turn recovery — an
        # in-process timer owning local cancellation via the input-generation
        # token (the app-server gate owns the authoritative decisions). The
        # app binds it late (never constructor-captured), like the turn
        # supervisor.
        self._root_recovery = RootRecoveryController()
        self._root_recovery._bind_app(self)
        # Bumped synchronously on every user input (typing, submission,
        # question action, manual retry, interrupt); a pending automatic
        # continuation is invalidated when it moves.
        self._input_generation = 0

    def _agent_job_active(self) -> bool:
        return self._turn_supervisor.agent_job_active()

    def _begin_pending_turn(self) -> None:
        self._turn_supervisor.begin_pending_turn()

    def _clear_active_turn_pins(self) -> None:
        self._active_turn_start = None
        for unit in self._transcript.units.values():
            unit.pin_reasons.discard("active-turn")

    def _set_loading_queue_count(self, count: int) -> None:
        if self._loading_widget is not None:
            self._loading_widget.set_queue_count(count)

    async def _enqueue_queued_turn(
        self,
        content: str,
        *,
        images: list[ImageAttachment] | None = None,
        message_entry_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> PublicQueuedTurn:
        return await self.app_server.enqueue(
            content,
            images=images,
            message_entry_id=message_entry_id,
            idempotency_key=idempotency_key,
        )

    async def _replace_queued_turn(
        self,
        queue_item_id: str,
        content: str,
        *,
        images: list[ImageAttachment] | None = None,
        message_entry_id: str | None = None,
    ) -> PublicQueuedTurn | None:
        return await self.app_server.replace_queued_turn(
            queue_item_id, content, images=images, message_entry_id=message_entry_id
        )

    async def _steer_queue_prompts(
        self,
        content: str,
        images: list[ImageAttachment] | None = None,
        client_message_id: str | None = None,
        *,
        require_waiting_only: bool = False,
        expected_turn_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> None:
        # Fail closed when the turn already ended: inject_user_context would
        # otherwise write idle session context and steer_pending would drop the
        # queue. Raising here re-enqueues the block as the next turn.
        await self.app_server.inject_user_context(
            content,
            as_message=True,
            inject_invoked_skill=True,
            images=images,
            client_message_id=client_message_id,
            require_active_turn=True,
            require_waiting_only=require_waiting_only,
            expected_turn_id=expected_turn_id,
            idempotency_key=idempotency_key,
        )

    def _queued_turn_has_started(self, queue_item_id: str) -> bool:
        return any(
            turn.queue_item_id == queue_item_id
            for turn in self.app_server.state.turns or []
        )

    async def _run_settings_update(
        self, description: str, update: Callable[[], Awaitable[None]]
    ) -> tuple[bool, str]:
        try:
            await update()
        except Exception as exc:
            logger.warning("Settings update failed: %s", description, exc_info=exc)
            return False, str(exc)
        return True, ""

    def _build_command_registry(self) -> CommandRegistry:
        return CommandRegistry(context=self._command_context())

    def _command_context(self) -> CommandContext:
        return CommandContext(
            retry_available=bool(
                self.event_handler is not None
                and self.event_handler._retry_presentation is not None
                and not self.event_handler._retry_presentation.active
            )
        )

    def _refresh_command_registry(self) -> None:
        self.commands.refresh(self._command_context())

    def _sync_retry_command_availability(self) -> None:
        if (
            self.commands.has_command("retry")
            != self._command_context().retry_available
        ):
            self._refresh_command_registry()

    async def on_load(self) -> None:
        if (
            self._mount_first
            and self._app_server is None
            and self._start_app_server is not None
        ):
            return
        await self.prepare()

    async def _refresh_config_from_disk(self) -> None:
        await self.app_server.resources.config.reload(reload_runtime=False)
        self._refresh_command_registry()

    def get_default_screen(self) -> Screen:
        return _get_word_select_screen_class()(id="_default")

    def compose(self) -> ComposeResult:
        yield from self._compose_main_ui()

    def _compose_main_ui(self) -> ComposeResult:
        has_session = self._app_server is not None
        init = self._initial_config_response
        with ChatScroll(
            id="chat",
            on_scroll_activity=self._on_chat_scroll_activity,
            on_user_scroll=self._request_scroll_driven_reconcile,
        ):
            self._banner = Banner(
                config=self.config if has_session else (init.config if init else None),
                skills_count=(
                    self.app_server.resources.runtime.custom_skills_count
                    if has_session
                    else (init.skills_count if init else 0)
                ),
                mcp=self.app_server.resources.runtime.mcp if has_session else None,
                mcp_servers_total=(
                    init.mcp_servers_total if init and not has_session else 0
                ),
                mcp_servers_enabled=(
                    init.mcp_servers_enabled if init and not has_session else 0
                ),
                hooks_count=(
                    self.app_server.resources.runtime.hooks_count
                    if has_session
                    else (init.hooks_count if init else 0)
                ),
            )
            yield self._banner
            yield VerticalGroup(id="messages")

        self._recovery_notice = InlineNotice(id="recovery-notice")
        self._recovery_notice.styles.width = "100%"
        self._recovery_notice.styles.height = "auto"
        yield self._recovery_notice

        with Horizontal(id="loading-area"):
            yield Static(id="loading-area-content")
            self._inline_notice = InlineNotice(id="inline-notice")
            yield self._inline_notice

        self._turn_outcome_notice = InlineNotice(id="turn-outcome-notice")
        self._turn_outcome_notice.styles.width = "100%"
        self._turn_outcome_notice.styles.height = 1
        yield self._turn_outcome_notice

        self._graduation_notice = InlineNotice(id="graduation-notice")
        self._graduation_notice.styles.height = "auto"
        yield self._graduation_notice

        self._agent_bar = AgentBar()
        yield self._agent_bar

        with Static(id="bottom-app-container"):
            yield ChatInputContainer(
                history_file=self.history_file,
                command_registry=self.commands,
                id="input-container",
                skill_entries_getter=self._get_skill_entries,
                file_watcher_for_autocomplete_getter=self._is_file_watcher_enabled,
                queue_edit_active_getter=self._is_queue_edit_active,
                queue_items_getter=self._queue.queue_item_texts,
                queue_selected_index_getter=self._queue_selected_queue_index,
            )
            pending = NoMarkupStatic(
                "! Action pending · finish this task, then Esc to answer",
                id="bottom-pending-action",
            )
            pending.display = False
            yield pending

        with Horizontal(id="bottom-bar"):
            config = self.config if has_session else (init.config if init else None)
            self._status_line = SessionStatusLine(
                SessionStatusState(
                    cwd=self.app_server.cwd if has_session else Path.cwd(),
                    pid=os.getpid(),
                ),
                config.status_line if config is not None else None,
                id="session-status-line",
            )
            yield self._status_line

    @property
    def _messages_area(self) -> Widget:
        if self._cached_messages_area is None:
            self._cached_messages_area = self.query_one("#messages")
        return self._cached_messages_area

    @property
    def _chat_widget(self) -> ChatScroll:
        if self._cached_chat is None:
            self._cached_chat = self.query_one("#chat", ChatScroll)
        return self._cached_chat

    # -- Transcript presenter forwarding properties ---------------------------
    # TranscriptPresenter owns this state; each field forwards so
    # direct-attribute consumers (including tests) see no change. A setter
    # exists exactly where app code assigns the field.

    @property
    def _transcript(self) -> _SessionTranscriptWindow:
        return self._transcript_presenter._transcript

    @property
    def _windowing(self) -> SessionWindowing:
        return self._transcript_presenter._windowing

    @property
    def _load_more(self) -> HistoryLoadMoreManager:
        return self._transcript_presenter._load_more

    @_load_more.setter
    def _load_more(self, value: HistoryLoadMoreManager) -> None:
        self._transcript_presenter._load_more = value

    @property
    def _history_widget_indices(self) -> WeakKeyDictionary[Widget, int]:
        return self._transcript_presenter._history_widget_indices

    @_history_widget_indices.setter
    def _history_widget_indices(self, value: WeakKeyDictionary[Widget, int]) -> None:
        self._transcript_presenter._history_widget_indices = value

    @property
    def _tools_collapsed(self) -> bool:
        return self._transcript_presenter._tools_collapsed

    @_tools_collapsed.setter
    def _tools_collapsed(self, value: bool) -> None:
        self._transcript_presenter._tools_collapsed = value

    @property
    def _tool_group_expansion_state(self) -> ToolGroupExpansionState:
        return self._transcript_presenter._tool_group_expansion_state

    @property
    def _preview_tool_group_expansion_state(self) -> ToolGroupExpansionState:
        return self._transcript_presenter._preview_tool_group_expansion_state

    @_preview_tool_group_expansion_state.setter
    def _preview_tool_group_expansion_state(
        self, value: ToolGroupExpansionState
    ) -> None:
        self._transcript_presenter._preview_tool_group_expansion_state = value

    @property
    def _entry_expansion_state(self) -> EntryExpansionState:
        return self._transcript_presenter._entry_expansion_state

    @property
    def _preview_entry_expansion_state(self) -> EntryExpansionState:
        return self._transcript_presenter._preview_entry_expansion_state

    @_preview_entry_expansion_state.setter
    def _preview_entry_expansion_state(self, value: EntryExpansionState) -> None:
        self._transcript_presenter._preview_entry_expansion_state = value

    @property
    def _transcript_under_budget_key(self) -> tuple[int, int, int] | None:
        return self._transcript_presenter._transcript_under_budget_key

    @_transcript_under_budget_key.setter
    def _transcript_under_budget_key(self, value: tuple[int, int, int] | None) -> None:
        self._transcript_presenter._transcript_under_budget_key = value

    @property
    def _transcript_following(self) -> bool:
        return self._transcript_presenter._transcript_following

    @_transcript_following.setter
    def _transcript_following(self, value: bool) -> None:
        self._transcript_presenter._transcript_following = value

    @property
    def _resize_anchor_task(self) -> asyncio.Task[None] | None:
        return self._transcript_presenter._resize_anchor_task

    @_resize_anchor_task.setter
    def _resize_anchor_task(self, value: asyncio.Task[None] | None) -> None:
        self._transcript_presenter._resize_anchor_task = value

    @property
    def _last_transcript_anchor(self) -> TranscriptAnchor | None:
        return self._transcript_presenter._last_transcript_anchor

    @_last_transcript_anchor.setter
    def _last_transcript_anchor(self, value: TranscriptAnchor | None) -> None:
        self._transcript_presenter._last_transcript_anchor = value

    @property
    def _has_older_history(self) -> bool:
        return self._transcript_presenter.has_older_history

    @property
    def _history_backfill_remaining(self) -> int | None:
        return self._transcript_presenter.history_backfill_remaining

    # -- Bottom panel controller forwarding properties ------------------------
    # BottomPanelController owns the current bottom app; the property forwards
    # so direct-attribute consumers (including tests) see no change. The setter
    # exists because app code assigns the field (rewind mode entry).

    @property
    def _current_bottom_app(self) -> BottomApp:
        return self._bottom_panel.current_app

    @_current_bottom_app.setter
    def _current_bottom_app(self, value: BottomApp) -> None:
        self._bottom_panel.current_app = value

    # The settings-return routing state moved to the controller (7.3b); the
    # app's worker finally-blocks and _wait_for_settings read/write it directly.

    @property
    def _settings_return_bottom_app(self) -> BottomApp | None:
        return self._bottom_panel.settings_return_bottom_app

    @_settings_return_bottom_app.setter
    def _settings_return_bottom_app(self, value: BottomApp | None) -> None:
        self._bottom_panel.settings_return_bottom_app = value

    @property
    def _settings_return_providers(self) -> bool:
        return self._bottom_panel.settings_return_providers

    @_settings_return_providers.setter
    def _settings_return_providers(self, value: bool) -> None:
        self._bottom_panel.settings_return_providers = value

    @property
    def _settings_return_web_search(self) -> bool:
        return self._bottom_panel.settings_return_web_search

    @_settings_return_web_search.setter
    def _settings_return_web_search(self, value: bool) -> None:
        self._bottom_panel.settings_return_web_search = value

    @property
    def _settings_return_state(self) -> tuple[str, str | None, int, str | None] | None:
        return self._bottom_panel.settings_return_state

    @_settings_return_state.setter
    def _settings_return_state(
        self, value: tuple[str, str | None, int, str | None] | None
    ) -> None:
        self._bottom_panel.settings_return_state = value

    # The escape-ladder timing state moved to the controller (7.4a); staying
    # app code (no-job steps, action_interrupt) and tests write/read it.

    @property
    def _last_escape_time(self) -> float | None:
        return self._bottom_panel.last_escape_time

    @_last_escape_time.setter
    def _last_escape_time(self, value: float | None) -> None:
        self._bottom_panel.last_escape_time = value

    # The agent browser controller owns the viewer/selection state (8.2a);
    # staying app code (heartbeat log, _show_no_output_outcome, config
    # propagation, check_action/action_interrupt), the bottom-panel
    # controller, the transcript presenter's retained set, and tests
    # read/write these fields through the forwarders.

    @property
    def _agent_summaries(self) -> list[AgentSummaryModel]:
        return self._agent_browser.agent_summaries

    @_agent_summaries.setter
    def _agent_summaries(self, value: list[AgentSummaryModel]) -> None:
        self._agent_browser.agent_summaries = value

    @property
    def _agent_transcript_viewer(self) -> AgentTranscriptViewer | None:
        return self._agent_browser.agent_transcript_viewer

    @property
    def _agent_transcript_focus_target(self) -> Widget | None:
        return self._agent_browser.agent_transcript_focus_target

    @property
    def _agent_selection_target(self) -> str | None:
        return self._agent_browser.selection_target

    @property
    def _agent_selection_generation(self) -> int:
        return self._agent_browser.selection_generation

    @property
    def _agent_transition_task(self) -> asyncio.Task[None] | None:
        return self._agent_browser.transition_task

    # -- Session flow controller forwarding properties ------------------------
    # SessionFlowController owns the picker preview/origin state (9.1a), the
    # resume flow (9.3a), the adoption + recovery notices (9.3b), and the
    # rewind mode state (9.5a). The app keeps the app._picker alias plus
    # origin-field forwarders: tests mutate app._picker fields and the
    # transcript presenter reads the preview flag and the origin
    # range/restoring flag. No app code assigns any picker field any more (the
    # controller's show_session_picker writes the origin fields internally
    # since 9.3a), so the origin forwarders are get-only; _picker itself is
    # never rebound, so it is get-only too. The _resume_adopted forwarder
    # (9.3b) keeps the app-visible adoption flag reachable; the controller
    # reads and writes it internally. The rewind forwarders (9.5a) are get-only
    # as well: the presenter and the tests only ever read the rewind mode and
    # the highlighted widget, and the controller's rewind helpers write the
    # state internally.

    @property
    def _picker(self) -> _PickerState:
        return self._session_flow.picker

    @property
    def _picker_origin_anchor(self) -> TranscriptAnchor | None:
        return self._session_flow.picker_origin_anchor

    @property
    def _picker_origin_restoring(self) -> bool:
        return self._session_flow.picker_origin_restoring

    @property
    def _picker_origin_range(self) -> tuple[int, int] | None:
        return self._session_flow.picker_origin_range

    @property
    def _picker_origin_focus(self) -> Widget | None:
        return self._session_flow.picker_origin_focus

    @property
    def _resume_adopted(self) -> bool:
        return self._session_flow.resume_adopted

    @_resume_adopted.setter
    def _resume_adopted(self, value: bool) -> None:
        self._session_flow.resume_adopted = value

    @property
    def _rewind_mode(self) -> bool:
        return self._session_flow.rewind_mode

    @property
    def _rewind_highlighted_widget(self) -> UserMessage | None:
        return self._session_flow.rewind_highlighted_widget

    # -- Turn supervisor forwarding properties --------------------------------
    # TurnSupervisor owns the turn/interrupt/retry state (10.2a). The app keeps
    # underscore-named forwarding properties so direct-attribute consumers
    # (staying app code, the other controllers, tests) see no change. A setter
    # exists exactly where app code or tests assign the field; the four
    # get-only forwarders are never rebound (the owning objects are created
    # once in the controller's __init__ and only mutated in place).

    @property
    def _pending_turn(self) -> bool:
        return self._turn_supervisor.pending_turn

    @_pending_turn.setter
    def _pending_turn(self, value: bool) -> None:
        self._turn_supervisor.pending_turn = value

    @property
    def _turn_started_event(self) -> asyncio.Event:
        return self._turn_supervisor.turn_started_event

    @property
    def _interrupt_requested(self) -> bool:
        return self._turn_supervisor.interrupt_requested

    @_interrupt_requested.setter
    def _interrupt_requested(self, value: bool) -> None:
        self._turn_supervisor.interrupt_requested = value

    @property
    def _interrupt_pending(self) -> bool:
        return self._turn_supervisor.interrupt_pending

    @_interrupt_pending.setter
    def _interrupt_pending(self, value: bool) -> None:
        self._turn_supervisor.interrupt_pending = value

    @property
    def _interrupt_settled(self) -> asyncio.Event:
        return self._turn_supervisor.interrupt_settled

    @property
    def _interrupt_operation(self) -> asyncio.Task[None] | None:
        return self._turn_supervisor.interrupt_operation

    @_interrupt_operation.setter
    def _interrupt_operation(self, value: asyncio.Task[None] | None) -> None:
        self._turn_supervisor.interrupt_operation = value

    @property
    def _deferred_submit_lock(self) -> asyncio.Lock:
        return self._turn_supervisor.deferred_submit_lock

    @property
    def _deferred_submits(self) -> int:
        return self._turn_supervisor.deferred_submits

    @_deferred_submits.setter
    def _deferred_submits(self, value: int) -> None:
        self._turn_supervisor.deferred_submits = value

    @property
    def _agent_task(self) -> asyncio.Task | None:
        return self._turn_supervisor.agent_task

    @_agent_task.setter
    def _agent_task(self, value: asyncio.Task | None) -> None:
        self._turn_supervisor.agent_task = value

    @property
    def _turn_ui_generation(self) -> int:
        return self._turn_supervisor.turn_ui_generation

    @_turn_ui_generation.setter
    def _turn_ui_generation(self, value: int) -> None:
        self._turn_supervisor.turn_ui_generation = value

    @property
    def _turn_ui_mutex(self) -> asyncio.Lock | None:
        return self._turn_supervisor.turn_ui_mutex

    @_turn_ui_mutex.setter
    def _turn_ui_mutex(self, value: asyncio.Lock | None) -> None:
        self._turn_supervisor.turn_ui_mutex = value

    @property
    def _turn_has_final_prose(self) -> bool:
        return self._turn_supervisor.turn_has_final_prose

    @_turn_has_final_prose.setter
    def _turn_has_final_prose(self, value: bool) -> None:
        self._turn_supervisor.turn_has_final_prose = value

    @property
    def _active_callback(self) -> PublicCallbackEntry | None:
        return self._turn_supervisor.active_callback

    @_active_callback.setter
    def _active_callback(self, value: PublicCallbackEntry | None) -> None:
        self._turn_supervisor.active_callback = value

    @property
    def _pending_callbacks(self) -> deque[PublicCallbackEntry]:
        return self._turn_supervisor.pending_callbacks

    @property
    def _callback_submitting(self) -> bool:
        return self._turn_supervisor.callback_submitting

    @_callback_submitting.setter
    def _callback_submitting(self, value: bool) -> None:
        self._turn_supervisor.callback_submitting = value

    @property
    def _callback_delivery_uncertain(self) -> bool:
        return self._turn_supervisor.callback_delivery_uncertain

    @_callback_delivery_uncertain.setter
    def _callback_delivery_uncertain(self, value: bool) -> None:
        self._turn_supervisor.callback_delivery_uncertain = value

    @property
    def _pending_local_question(self) -> asyncio.Future[UserQuestionResult] | None:
        return self._turn_supervisor.pending_local_question

    @_pending_local_question.setter
    def _pending_local_question(
        self, value: asyncio.Future[UserQuestionResult] | None
    ) -> None:
        self._turn_supervisor.pending_local_question = value

    @property
    def _active_turn_start(self) -> int | None:
        return self._turn_supervisor.active_turn_start

    @_active_turn_start.setter
    def _active_turn_start(self, value: int | None) -> None:
        self._turn_supervisor.active_turn_start = value

    @property
    def _loading_area(self) -> Widget:
        if self._cached_loading_area is None:
            self._cached_loading_area = self.query_one("#loading-area-content")
        return self._cached_loading_area

    async def on_mount(self) -> None:
        # /log-level changes after mount affect the handler immediately, but
        # heartbeat creation is mount-time only; remount to change timer state.
        if get_effective_log_level() == "DEBUG":
            self._diagnostic_last_heartbeat = time.monotonic()
            self._diagnostic_heartbeat_timer = self.set_interval(
                0.5, self._record_diagnostic_heartbeat
            )
        if self._app_server is None and self._start_app_server is not None:
            init = self._initial_config_response
            initial_theme = init.config.theme if init is not None else FALLBACK_THEME
            await self._apply_theme(initial_theme)
            if init is not None and init.startup_issue is not None:
                self._show_config_issue(init.startup_issue)
                self.call_after_refresh(self._start_bootstrap_session)
            else:
                self._start_bootstrap_session()
            return
        await self._mount_after_session_ready()

    def _record_diagnostic_heartbeat(self) -> None:
        now = time.monotonic()
        drift = now - self._diagnostic_last_heartbeat - 0.5
        self._diagnostic_last_heartbeat = now
        self._diagnostic_max_heartbeat_drift = max(
            self._diagnostic_max_heartbeat_drift, drift
        )
        if drift > _DIAGNOSTIC_HEARTBEAT_DRIFT_WARNING_SECONDS:
            logger.debug(
                "App heartbeat drift: %.3fs selection_generation=%d target=%s",
                drift,
                self._agent_selection_generation,
                self._agent_selection_target,
            )

    def on_unmount(self) -> None:
        # R2/OD3: app shutdown cancels the local recovery timer and drops
        # its references (an admitted successor in flight settles via normal
        # interruption rules — never cancelled as rollback).
        self._root_recovery.shutdown()
        if self._diagnostic_heartbeat_timer is not None:
            self._diagnostic_heartbeat_timer.stop()
            self._diagnostic_heartbeat_timer = None
            logger.debug(
                "App max heartbeat drift: %.3fs", self._diagnostic_max_heartbeat_drift
            )

    def _start_bootstrap_session(self) -> None:
        self.run_worker(self._bootstrap_session(), exclusive=False)

    async def _mount_after_session_ready(self) -> None:
        await self._apply_theme(self.config.theme)
        self.app_server.resources.config.subscribe(self._on_config_changed)
        set_config_log_level(self.config.log_level)
        self._terminal_notifier.restore()
        if self._chat_input_container is not None:
            self._chat_input_container.replace_command_registry(self.commands)
            self._refresh_command_registry()
        self._refresh_banner()
        self._refresh_context_progress()
        # Ready now unless a resume/continue/picker flow is pending — those mark
        # ready at their own return-to-input points to avoid dispatching against
        # a half-rebound session.
        if not (
            self._show_resume_picker
            or self._resume_session_id is not None
            or self._continue_latest
        ):
            self._mark_session_ready()
        self.run_worker(self._complete_mount(), exclusive=False)

    async def _bootstrap_session(self) -> None:
        if self._start_app_server is None:
            raise RuntimeError("App server starter is unavailable")
        try:
            session = await self._start_app_server()
            self._app_server = session
            self._initialize_client_dependencies()
        except Exception as e:
            self._app_server = None
            await self._show_bootstrap_error(e)
            return
        await self._transition_to_main_ui()

    async def _transition_to_main_ui(self) -> None:
        # _main_ui_mounted now means "session-ready refresh has run" — compose()
        # already mounted the UI. Kept the name to avoid touching call sites.
        if self._main_ui_mounted:
            return
        self._main_ui_mounted = True
        await self._mount_after_session_ready()

    async def _show_bootstrap_error(self, error: Exception) -> None:
        await self._mount_and_scroll(Static(f"Failed to start session: {error}"))
        try:
            container = self.query_one(ChatInputContainer)
        except Exception:
            container = None
        if container is not None:
            container.disabled = True
            container.display = False
        self._fatal_init_error = True

    def _on_config_changed(self, config: ConfigView) -> None:
        resolved_theme = resolve_theme(resolve_theme_name(config.theme))
        textual_theme = _TEXTUAL_THEME_MAP[resolved_theme]
        if textual_theme != self.theme:
            self.run_worker(self._apply_theme(config.theme))
        set_config_log_level(config.log_level)
        self._refresh_banner()
        self._refresh_context_progress()
        for widget in [
            *self.query(UserMessage),
            *self.query(AssistantMessage),
            *self.query(ToolGroup),
            *self.query(ToolCallMessage),
            *self.query(ToolResultMessage),
        ]:
            widget.set_show_message_timestamps(config.show_message_timestamps)
        if self._agent_transcript_viewer is not None:
            self._agent_transcript_viewer.set_show_message_timestamps(
                config.show_message_timestamps
            )

    async def _complete_mount(self) -> None:
        self.event_handler = EventHandler(
            mount_callback=self._mount_and_scroll,
            get_tools_collapsed=lambda: self._tools_collapsed,
            get_show_thinking=lambda: self.config.show_thinking_nodes,
            get_show_message_timestamps=lambda: self.config.show_message_timestamps,
            on_user_message=self._queue.reconcile_history_entry,
            mounted_entry_widgets=self._transcript_presenter.mounted_timing_widgets,
            update_retained_entry=self._transcript.update_entry,
            on_context_cleared=self._on_context_cleared,
            on_session_title_changed=self._on_session_title_changed,
            entry_expansion_state=self._entry_expansion_state,
            group_expansion_state=self._tool_group_expansion_state,
        )

        self._chat_input_container = self.query_one(ChatInputContainer)
        self._chat_input_container.replace_command_registry(self.commands)
        self._refresh_command_registry()

        chat_input_container = self.query_one(ChatInputContainer)
        chat_input_container.focus_input()
        await self._show_dangerous_directory_warning()
        self.run_worker(self._deferred_resume_and_start(), exclusive=False)
        # Non-critical: runs off the mount path so its app-server round-trip
        # never delays history resume into a fast-exit teardown window.
        self.run_worker(self._show_untrusted_config_warning(), exclusive=False)

        self.call_after_refresh(self._start_post_ready_startup)
        self._show_config_issues()

        self.run_worker(self._watch_init_completion(), exclusive=False)

        if self._show_resume_picker:
            self.run_worker(self._show_session_picker(), exclusive=False)
        elif self._resume_session_id is not None or self._continue_latest:
            self.run_worker(self._auto_resume_on_startup(), exclusive=False)
        else:
            # Fresh session: no later set-point, so the input is ready now.
            self._mark_session_ready()

        gc.collect()
        gc.freeze()

    def _update_context_progress(self, event: StatsUpdated) -> None:
        self._root_context_unknown = False
        self._refresh_context_progress()
        if self._status_line is not None:
            self._status_line.set_state(
                replace(
                    self._status_line.state,
                    context_tokens=event.params.stats.context_tokens,
                    auto_compact_threshold=event.params.context_window,
                )
            )
            self._refresh_main_agent_details()

    def _start_post_ready_startup(self) -> None:
        self.run_worker(self._complete_post_ready_startup(), exclusive=False)

    async def _complete_post_ready_startup(self) -> None:
        self._refresh_command_registry()
        self._startup_command_availability_ready.set()
        if (
            not self._show_resume_picker
            and self._resume_session_id is None
            and not self._continue_latest
        ):
            await self._show_custom_tools_deprecation_warning_after_initial_history()
        self._refresh_banner()
        if self._show_resume_picker:
            return
        if self._resume_session_id is not None or self._continue_latest:
            return
        self._process_startup_prompt()

    async def _process_startup_prompt_when_available(self) -> None:
        await self._startup_command_availability_ready.wait()
        self._process_startup_prompt()

    def _process_startup_prompt(self) -> None:
        if self._startup_prompt_processed:
            return
        self._startup_prompt_processed = True
        if self._initial_prompt:
            self._process_initial_prompt()

    def _set_recovery_issue(
        self, key: str, message: str | None, *, severity: str = "warning"
    ) -> None:
        if message is None:
            self._recovery_issues.pop(key, None)
        else:
            self._recovery_issues[key] = (severity, message)
        if not self._recovery_issues:
            self._recovery_notice.hide()
            return
        issues = list(self._recovery_issues.values())
        combined = " · ".join(text for _, text in issues)
        self._recovery_notice.show(
            combined,
            severity="error"
            if any(kind == "error" for kind, _ in issues)
            else "warning",
            timeout=None,
        )

    def _show_config_issues(self) -> None:
        issues = self.app_server.resources.runtime.issues
        warnings = self.app_server.resources.config.current.validation_warnings
        current = {
            f"config:{issue.file}:{issue.message}": (
                f"Config {issue.file}: {issue.message}. Fix the file and run /reload."
            )
            for issue in issues
        }
        startup_issue = (
            self._initial_config_response.startup_issue
            if self._initial_config_response is not None
            and not self._startup_issue_resolved
            else None
        )
        if startup_issue is not None:
            current[f"config:{startup_issue.file}:{startup_issue.message}"] = (
                f"Config {startup_issue.file}: {startup_issue.message}. "
                "Fix the file and run /reload."
            )
        current.update({
            f"config-warning:{warning}": (
                f"Config: {warning}"
                if "/web-search" in warning
                else f"Config: {warning}. Review the config and run /reload."
            )
            for warning in warnings
        })
        for key in list(self._recovery_issues):
            if key.startswith(("config:", "config-warning:")) and key not in current:
                self._set_recovery_issue(key, None)
        for key, message in current.items():
            self._set_recovery_issue(key, message)

    def _show_config_issue(self, issue: ConfigIssue) -> None:
        self._set_recovery_issue(
            f"config:{issue.file}:{issue.message}",
            f"Config {issue.file}: {issue.message}. Fix the file and run /reload.",
        )

    async def _watch_init_completion(self) -> None:
        """Show 'Initializing' loading indicator until background init finishes."""
        init_widget = None
        try:
            if not self.app_server.resources.runtime.ready:
                await self._ensure_loading_widget(
                    INITIALIZING_LOADING_STATUS, show_hint=False
                )
                init_widget = self._loading_widget
            await self.app_server.resources.runtime.wait_until_ready()
            await self._show_post_init_notices_once()
        except Exception as e:
            if isinstance(e, AppServerResponseError) and e.error.code in {
                ProtocolErrorCode.CONFLICT,
                ProtocolErrorCode.NOT_FOUND,
            }:
                # The fresh session's readiness watch is superseded by a resume:
                # CONFLICT while the resume holds the lifecycle reservation, or
                # NOT_FOUND once the rebind re-attaches the root to the resumed id.
                # Neither is a real init failure — the resume owns readiness/UI.
                logger.info("Init readiness watch superseded by a session resume")
                return
            logger.exception("Background initialization failed")
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Background initialization failed: {e}",
                    collapsed=self._tools_collapsed,
                )
            )
            await self._mount_and_scroll(
                Static("Press any key to exit...", classes="error-hint")
            )
            if self._chat_input_container:
                self._chat_input_container.disabled = True
                self._chat_input_container.display = False
            self._fatal_init_error = True
        finally:
            # A submit during init reuses this same widget as the turn spinner
            # (via _ensure_loading_widget, which flips the label to "Generating"),
            # so only tear it down if it is still the idle init indicator.
            # Otherwise the turn lifecycle owns it now and removing it would drop
            # the first-turn spinner. Checking the widget's own label keeps this
            # path decoupled from turn state.
            if (
                init_widget is not None
                and self._loading_widget is init_widget
                and init_widget.base_status == INITIALIZING_LOADING_STATUS
            ):
                await self._remove_loading_widget()
            self._refresh_banner()
            try:
                self.query_one(_get_mcp_app_class()).refresh_index()
            except Exception:
                pass

    async def _show_post_init_notices_once(self) -> None:
        # Shown by whichever of the readiness watch or a resume gets there first: a
        # resume supersedes the watch, but reuses the same MCP pool, so the notices
        # still apply.
        if self._post_init_notices_shown:
            return
        self._post_init_notices_shown = True
        self._show_mcp_discovery_failures()
        await self._show_mcp_auth_required_notice()

    def _is_cold_start(self) -> bool | None:
        """True if this process paid first-run startup cost (cold), False if it
        resumed a warm cache (warm), None when the signal is unavailable.
        """
        if sys.dont_write_bytecode:
            return None
        from importlib.util import cache_from_source

        pyc = Path(cache_from_source(__file__))
        try:
            return pyc.stat().st_mtime >= PROCESS_START_WALLCLOCK
        except OSError:
            return None

    def _show_mcp_discovery_failures(self) -> None:
        errors = self.app_server.resources.runtime.mcp.discovery_errors
        for key in list(self._recovery_issues):
            if key.startswith("mcp:") and key.removeprefix("mcp:") not in errors:
                self._set_recovery_issue(key, None)
        for server_name, error in sorted(errors.items()):
            self._set_recovery_issue(
                f"mcp:{server_name}",
                f"MCP server '{server_name}' failed to connect: {error}. "
                "Other servers remain usable; run /mcp to inspect or reconnect.",
            )

    async def _show_mcp_auth_required_notice(self) -> None:
        """Show a notice if any enabled MCP servers require OAuth authentication."""
        aliases = self.app_server.resources.runtime.mcp.needs_auth
        if not aliases:
            return
        command = f"/mcp login {aliases[0]}"
        if len(aliases) > 1:
            detail = ", ".join(aliases)
            message = (
                "MCP servers need OAuth authentication: "
                f"{detail}. Run `{command}` to start with {aliases[0]!r}."
            )
        else:
            message = (
                f"MCP server {aliases[0]!r} needs OAuth authentication. "
                f"Run `{command}` to authenticate."
            )
        await self._mount_and_scroll(UserCommandMessage(message))

    def _process_initial_prompt(self) -> None:
        if self._initial_prompt:
            self.run_worker(
                self._handle_user_message(self._initial_prompt), exclusive=False
            )

    def _is_file_watcher_enabled(self) -> bool:
        return (
            self._app_server is not None and self.config.file_watcher_for_autocomplete
        )

    def on_key(self) -> None:
        if self._fatal_init_error:
            self.exit()

    async def on_chat_input_container_submitted(
        self, event: ChatInputContainer.Submitted
    ) -> None:
        await self._submit_or_defer(event.value)

    def _invalidate_user_input(self) -> None:
        # R2/OD3: the input-generation token behind the bounded automatic
        # root-turn recovery schedule. Bumped synchronously on every user
        # input action (typing, submission, question action, manual retry,
        # interrupt, session change); a pending automatic continuation is
        # cancelled when it moves — the user took over, so no unattended
        # turn is admitted.
        self._input_generation += 1
        self._root_recovery.cancel_local(reason="user_input")

    def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        # Typing is user input: invalidate any pending automatic continuation
        # synchronously (the generation token's primary half). ChatTextArea
        # does not stop this message, so the app observes every edit.
        self._invalidate_user_input()

    async def _submit_or_defer(self, raw_value: str) -> None:
        self._invalidate_user_input()
        await self._turn_supervisor.submit_or_defer(raw_value)

    async def _dispatch_submitted_value(self, raw_value: str) -> None:
        value = raw_value.strip()
        input_widget = self.query_one(ChatInputContainer)

        if not value and not self._queue.paused:
            # Enter on an empty input steers queued messages into the running
            # turn (same as Ctrl+Enter); otherwise it stays a no-op.
            await self._steer_queued_now()
            return

        if self._banner:
            self._banner.freeze_animation()

        if self._whats_new_message:
            await self._whats_new_message.remove()
            self._whats_new_message = None

        if self._queue.paused:
            if not await self._try_side_channel_command(value, input_widget):
                if not await self._handle_paused_submit(value):
                    self._restore_input_if_empty(input_widget, value)
            return

        if self._is_busy():
            if not await self._try_side_channel_command(value, input_widget):
                if not await self._handle_queue_submit(
                    value, reject_hint=_REJECT_HINT_BUSY, allow_waiting_only=True
                ):
                    self._restore_input_if_empty(input_widget, value)
            return

        await self._dispatch_idle_input(value)

    async def on_chat_input_container_queue_edit_submitted(
        self, event: ChatInputContainer.QueueEditSubmitted
    ) -> None:
        event.stop()
        # Resolve the edited item by widget identity: app-server promotion is
        # FIFO, so an older item starting shifts this one to a lower index. A
        # cached index could then miss it or update the wrong prompt.
        current_index = self._queue_selected_queue_index()
        if current_index is None:
            # Promoted despite the body's guard (race): copy on write.
            await self._enqueue_edited_copy(event.value)
            return
        if self._queue.is_unresolved_prompt(current_index):
            await self._report_unresolved_queue_edit()
            return
        widget = self._queue_selected_widget
        prepared = await self._prepare_prompt_or_abort(event.value)
        if prepared is None:
            return
        # Re-resolve the index of the item captured at submit time (by widget
        # identity), not the live highlight: the body returns to selection
        # mode before this await, so Up/Down/Esc can move ``_queue_selected_widget``
        # away from the edited item. Tracking the captured widget avoids writing
        # the edit onto a different item; if it was promoted, copy on write.
        current_index = self._queue_index_of_widget(widget)
        if current_index is None:
            await self._enqueue_edited_copy(event.value)
            return
        updated = await self._queue.update_prompt(
            current_index, event.value, prepared_prompt=prepared
        )
        if not updated:
            if self._queue.is_unresolved_prompt(current_index):
                await self._report_unresolved_queue_edit()
            else:
                await self._enqueue_edited_copy(event.value)

    async def _report_unresolved_queue_edit(self) -> None:
        await self._mount_and_scroll(
            ErrorMessage(
                "Steering delivery is unresolved; the original prompt was retained. "
                "Retry the delivery or remove it from the queue instead of editing it.",
                collapsed=self._tools_collapsed,
            )
        )

    async def _enqueue_edited_copy(self, value: str) -> None:
        """Queue an edited prompt again if the original started meanwhile."""
        await self._enqueue_prompt_with_resources(value)

    async def on_chat_input_container_queue_edit_consumed(
        self, event: ChatInputContainer.QueueEditConsumed
    ) -> None:
        event.stop()
        await self._enqueue_edited_copy(event.value)

    async def on_chat_input_container_queue_remove_requested(
        self, event: ChatInputContainer.QueueRemoveRequested
    ) -> None:
        event.stop()
        # Resolve the target by widget identity, like the edit path: a cached
        # index goes stale if an older prompt starts between the body's resync
        # and this handler. Once the highlighted widget is gone there is
        # nothing left to remove.
        current_index = self._queue_selected_queue_index()
        if current_index is None:
            return
        self._queue_selected_widget = None
        await self._queue.pop_at(current_index)

    async def on_chat_input_container_queue_selection_scroll(
        self, event: ChatInputContainer.QueueSelectionScroll
    ) -> None:
        event.stop()
        widgets = self._queue.widgets
        if event.queue_index >= len(widgets):
            return
        widget = widgets[event.queue_index]

        if self._queue_selected_widget is not None:
            self._queue_selected_widget.remove_class("queue-selected")
        widget.add_class("queue-selected")
        self._queue_selected_widget = widget

        chat = self._messages_area
        self.call_after_refresh(chat.scroll_to_widget, widget, animate=False, top=True)

    def _queue_selected_queue_index(self) -> int | None:
        """Live queue index of the highlighted widget, or None if consumed.

        Used by the body to detect app-server promotion by widget identity. The
        cached queue index is unreliable once older prompts start.
        """
        return self._queue_index_of_widget(self._queue_selected_widget)

    def _queue_index_of_widget(self, widget: object | None) -> int | None:
        """Live queue index of a specific widget, or None if it's been removed.

        Resolves by identity (``is``) rather than the current highlight, so a
        queued edit can re-resolve the exact item it captured at submit time
        even if the user navigates the highlight during the prepare await.
        """
        if widget is None:
            return None
        for i, w in enumerate(self._queue.widgets):
            if w is widget:
                return i
        return None

    def _clear_queue_selection(self) -> None:
        if self._queue_selected_widget is not None:
            self._queue_selected_widget.remove_class("queue-selected")
            self._queue_selected_widget = None

    async def on_chat_input_container_queue_mode_exited(
        self, _event: ChatInputContainer.QueueModeExited
    ) -> None:
        self._clear_queue_selection()

    @staticmethod
    def _restore_input_if_empty(input_widget: ChatInputContainer, value: str) -> None:
        if not input_widget.value:
            input_widget.value = value

    async def _empty_bash_error(self) -> None:
        await self._mount_and_scroll(
            ErrorMessage(
                "No command provided after '!'", collapsed=self._tools_collapsed
            )
        )

    def _warn_not_queueable(self, message: str) -> None:
        self.notify(message, severity="warning", markup=False)

    async def _try_side_channel_command(
        self, value: str, input_widget: ChatInputContainer
    ) -> bool:
        resolved = self.commands.parse_command(value)
        if resolved is None:
            return False
        cmd_name, command, cmd_args = resolved
        if not command.side_channel:
            return False
        display = self._command_display(value, cmd_name)
        if not self._side_channel.enqueue(cmd_name, command, cmd_args, display):
            self._warn_not_queueable(
                "A slash command is already running — wait for it to finish."
            )
            self._restore_input_if_empty(input_widget, value)
        return True

    async def _dispatch_idle_input(self, value: str) -> None:
        # Mount-first renders an interactive input before the session is bound.
        # Every input kind eventually touches session-owned state, including
        # classification of skills, so hold dispatch at this common boundary.
        await self._session_ready.wait()
        match classify(
            value, commands=self.commands, resolve_skill=self._resolve_skill
        ):
            case SlashCommand():
                await self._handle_command(value)
            case Skill(command=command):
                await self._handle_user_message(command)
            case Bash(command=command):
                self._bash_task = asyncio.create_task(
                    self._handle_bash_command(command)
                )
                self._queue.notify_busy_changed()
            case EmptyBash():
                await self._empty_bash_error()
            case Prompt(text=text):
                await self._handle_user_message(text)

    async def _handle_paused_submit(self, value: str) -> bool:
        if value and not await self._handle_queue_submit(
            value, reject_hint=_REJECT_HINT_PAUSED
        ):
            return False
        await self._queue.resume()
        return True

    async def _handle_queue_submit(
        self, value: str, *, reject_hint: str, allow_waiting_only: bool = False
    ) -> bool:
        if self._bash_task is not None and not self._bash_task.done():
            rejection = f"Input cannot be queued while a shell command is running — {reject_hint}"
        else:
            rejection = None
            match classify(
                value, commands=self.commands, resolve_skill=self._resolve_skill
            ):
                case SlashCommand():
                    rejection = f"Slash commands cannot be queued — {reject_hint}"
                case Skill(command=command, name=name):
                    return await self._enqueue_prompt_with_resources(
                        command, skill_name=name, allow_waiting_only=allow_waiting_only
                    )
                case Bash():
                    rejection = f"Shell commands cannot be queued — {reject_hint}"
                case EmptyBash():
                    await self._empty_bash_error()
                    return False
                case Prompt(text=text):
                    return await self._enqueue_prompt_with_resources(
                        text, allow_waiting_only=allow_waiting_only
                    )
        if rejection is not None:
            self._warn_not_queueable(rejection)
            return False
        return True

    async def _enqueue_prompt_with_resources(
        self,
        content: str,
        *,
        skill_name: str | None = None,
        optimistic_start: bool = False,
        allow_waiting_only: bool = False,
    ) -> bool:
        # Capture before preparation: never retarget a successor after an await.
        waiting_turn = (
            next(
                (
                    turn.id
                    for turn in self.app_server.state.turns or []
                    if turn.waiting_only
                ),
                None,
            )
            if allow_waiting_only and self.app_server.waiting_only
            else None
        )
        with self._queue.reserve_enqueue():
            prepared = await self._prepare_prompt_or_abort(content)
            if prepared is None:
                return False
            if waiting_turn is not None:
                await self._steer_queued_now(
                    content=content,
                    prepared_prompt=prepared,
                    skill_name=skill_name,
                    expected_turn_id=waiting_turn,
                )
                # Delivery owns the content even on rejection/ambiguity; do not
                # restore raw input and create a second submission.
                return True
            try:
                await self._queue.enqueue_prompt(
                    content,
                    skill_name=skill_name,
                    prepared_prompt=prepared,
                    optimistic_start=optimistic_start,
                )
            except AppServerResponseError as error:
                await self._mount_and_scroll(
                    ErrorMessage(str(error), collapsed=self._tools_collapsed)
                )
                return False
        return True

    def _is_busy(self) -> bool:
        if self._agent_job_active():
            return True
        if self._queue.has_server_work:
            return True
        if self._bash_task is not None and not self._bash_task.done():
            return True
        return False

    def _is_queue_edit_active(self) -> bool:
        if not self._queue:
            return False
        return (
            self._queue.has_unresolved_steering
            or self._agent_job_active()
            or (self._bash_task is not None and not self._bash_task.done())
        )

    async def on_question_app_answered(self, message: QuestionApp.Answered) -> None:
        # R2/OD3: a question action is user input — invalidate any pending
        # automatic continuation before the answer flows.
        self._invalidate_user_input()
        await self._turn_supervisor.question_answered(message)

    async def on_question_app_cancelled(self, message: QuestionApp.Cancelled) -> None:
        self._invalidate_user_input()
        await self._turn_supervisor.question_cancelled(message)

    async def on_question_app_keep_waiting(
        self, message: QuestionApp.KeepWaiting
    ) -> None:
        self._invalidate_user_input()
        await self._turn_supervisor.question_keep_waiting(message)

    async def on_error_retry_requested(self, _: ErrorRetryRequested) -> None:
        # R1 (plan v0.5.6-recovery): the "Retry — new turn" affordance on a
        # terminal retryable root failure invokes the EXISTING retry operation
        # — one injected new turn with the user-initiated cooldown reset —
        # never resume_agent, never an automatic continuation. The operation
        # self-gates: no retry action while stopping, active execution, or an
        # unresolved runtime approval is present.
        await self._retry()

    async def action_retry_turn(self) -> None:
        # R1: the keyboard path (Ctrl+R) of the "Retry — new turn" affordance.
        # Gated exactly like the /retry command (a retry presentation is
        # offered and not yet active), then invokes the EXISTING retry
        # operation — never resume_agent, never automatic.
        if not self._command_context().retry_available:
            return
        await self._retry()

    async def _remove_loading_widget(self) -> None:
        if self._loading_widget and self._loading_widget.parent:
            await self._loading_widget.remove()
            self._loading_widget = None

    async def _prepare_prompt_or_abort(self, message: str) -> PreparedPrompt | None:
        # Block until the session is bound before touching app_server (instant
        # once ready).
        await self._session_ready.wait()
        try:
            return await self.app_server.resources.workspace.prepare_prompt(message)
        except AppServerResponseError as exc:
            await self._remove_loading_widget()
            await self._mount_and_scroll(
                ErrorMessage(str(exc), collapsed=self._tools_collapsed)
            )
            return None

    def _input_in_queue_mode(self) -> bool:
        containers = self.query(ChatInputContainer)
        if not containers:
            return False
        body = containers[0]._body
        return body is not None and body.in_queue_mode

    async def on_chat_text_area_steer_queue_requested(
        self, _event: ChatTextArea.SteerQueueRequested
    ) -> None:
        # Ctrl+Enter is the same action as empty Enter, including the
        # interrupt-settle gate. Steering immediately would remove the queued
        # block and inject it into a turn that Escape is already cancelling.
        await self._submit_or_defer("")

    async def _steer_queued_now(  # noqa: PLR0911
        self,
        *,
        content: str | None = None,
        prepared_prompt: PreparedPrompt | None = None,
        skill_name: str | None = None,
        expected_turn_id: str | None = None,
    ) -> bool:
        """Send queued messages into the running turn as steering.

        Returns True when a steer was sent. No-op (returns False) unless a turn
        is actually running and there is a promotable queued block; otherwise
        the queue promotes on its own as usual.
        """
        from chartreux.app_server import AppServerConnectionClosed

        if prepared_prompt is not None:
            assert content is not None and expected_turn_id is not None
            try:
                return await self._queue.steer_prepared(
                    content,
                    prepared_prompt=prepared_prompt,
                    skill_name=skill_name,
                    expected_turn_id=expected_turn_id,
                )
            except Exception as error:
                await self._mount_and_scroll(
                    ErrorMessage(
                        f"Steering admission failed; your prompt was kept. {error}",
                        collapsed=self._tools_collapsed,
                    )
                )
                return False
        if self._input_in_queue_mode():
            # Empty Enter while selecting/editing a queued item is an edit
            # action, not a steer (Ctrl+Enter already skips queue mode).
            return False
        if self._queue.has_unresolved_steering:
            try:
                return await self._queue.retry_unresolved_steering()
            except Exception as error:
                await self._mount_and_scroll(
                    ErrorMessage(str(error), collapsed=self._tools_collapsed)
                )
                return False
        if not self._queue.has_removable or not self.app_server.turn_active:
            return False
        turn_id = next(
            (
                turn.id
                for turn in self.app_server.state.turns or []
                if turn.status is PublicTurnStatus.IN_PROGRESS
            ),
            None,
        )
        if turn_id is None:
            return False
        try:
            return await self._queue.steer_pending(expected_turn_id=turn_id)
        except AppServerConnectionClosed:
            # A failed steer may have removed the server item before reconnect
            # could re-enqueue it. QueueController retains the prompt in that
            # case, so surface the unresolved recovery rather than losing it.
            if not self._shutdown_started:
                await self._mount_and_scroll(
                    ErrorMessage(
                        "Connection lost while steering; your queued prompt was kept.",
                        collapsed=self._tools_collapsed,
                    )
                )
            return False
        except AppServerResponseError as error:
            # The turn ended between remove and steer: steer_pending already
            # put the block back on the queue so it can promote as the next
            # turn. Do not flash that recovery as an error.
            if error.error.code is ProtocolErrorCode.STALE_TURN or (
                error.error.code is ProtocolErrorCode.CONFLICT
                and error.error.message == "No active turn"
            ):
                return False
            await self._mount_and_scroll(
                ErrorMessage(str(error), collapsed=self._tools_collapsed)
            )
            return False
        except Exception as error:
            await self._mount_and_scroll(
                ErrorMessage(str(error), collapsed=self._tools_collapsed)
            )
            return False

    def on_chat_text_area_clipboard_image_pasted(
        self, message: ChatTextArea.ClipboardImagePasted
    ) -> None:
        from chartreux.cli.textual_ui.widgets.chat_input.paste_image import (
            handle_clipboard_image_paste,
        )

        self.run_worker(
            handle_clipboard_image_paste(
                self, notify_when_empty=message.notify_when_empty
            ),
            exclusive=False,
        )

    async def _paste_clipboard_image_command(self, **_kwargs: Any) -> None:
        from chartreux.cli.textual_ui.widgets.chat_input.paste_image import (
            handle_clipboard_image_paste,
        )

        await handle_clipboard_image_paste(self, notify_when_empty=True)

    async def _persist_config_changes(
        self, changes: Mapping[str, str | bool | None]
    ) -> None:
        await self.app_server.resources.config.update(changes)

    async def _persist_proxy(self, changes: dict[str, str | None]) -> None:
        await self.app_server.resources.config.update_proxy(changes)
        await self._mount_and_scroll(
            UserCommandMessage(
                "Proxy settings saved. Restart the CLI for changes to take effect."
            )
        )

    async def _remove_config_field(self, field: str) -> None:
        response = await self.app_server.resources.config.write(
            [ConfigWriteOpWire(op="remove", path=f"/{field}")],
            reason="app-server config update",
        )
        if response.rejected:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INVALID_PARAMS,
                    message="Invalid configuration edit",
                )
            )
        if response.failures:
            raise AppServerResponseError(
                ProtocolError(
                    code=ProtocolErrorCode.INTERNAL_ERROR,
                    message="; ".join(response.failures),
                )
            )

    async def _ensure_loading_widget(
        self, status: str = DEFAULT_LOADING_STATUS, *, show_hint: bool = True
    ) -> None:
        if self._loading_widget and self._loading_widget.parent:
            self._loading_widget.set_status(status)
            return

        try:
            loading_area = self._loading_area
        except Exception:
            return
        loading = LoadingWidget(status=status, show_hint=show_hint)
        loading.set_waiting_only(
            self._app_server is not None and self._app_server.waiting_only
        )
        loading.set_hint_suppressed(self._current_bottom_app == BottomApp.Question)
        self._loading_widget = loading
        await loading_area.mount(loading)

    def _apply_thinking_visibility(self) -> None:
        show = self.config.show_thinking_nodes
        for node in self._messages_area.query(ReasoningMessage):
            node.display = show
        for tc in self._messages_area.query(ToolCallMessage):
            tc.recompute_gap()
        for tr in self._messages_area.query(ToolResultMessage):
            tr.recompute_gap()
        # A group holding only a now-hidden thinking node must collapse so it
        # doesn't leave a stray blank line.
        for group in self._messages_area.query(ToolGroup):
            group.sync_visibility()

    async def on_model_picker_app_model_selected(
        self, message: ModelPickerApp.ModelSelected
    ) -> None:
        if self._current_bottom_app != BottomApp.ModelPicker:
            return
        if await self._is_active_model_enforced():
            self.notify(
                "'active_model' is enforced by your administrator. "
                "Contact your admin to change it.",
                severity="warning",
                markup=False,
            )
            await self._switch_to_input_app()
            return
        picker = self.query_one(_get_model_picker_app_class())
        if await self._apply_picker_update(
            "model", message.alias, partial(self._persist_model, message.alias), picker
        ):
            await self._switch_to_input_app()

    async def _persist_model(self, alias: str) -> None:
        await self.app_server.resources.config.update({"active_model": alias})

    async def on_model_picker_app_cancelled(
        self, _event: ModelPickerApp.Cancelled
    ) -> None:
        self._pending_picker_reload = None
        await self._switch_to_input_app()

    async def on_thinking_picker_app_thinking_selected(
        self, message: ThinkingPickerApp.ThinkingSelected
    ) -> None:
        if self._current_bottom_app != BottomApp.ThinkingPicker:
            return
        picker = self.query_one(_get_thinking_picker_app_class())
        if await self._apply_picker_update(
            "thinking",
            message.level,
            partial(self._persist_thinking, message.level),
            picker,
        ):
            await self._switch_to_input_app()

    async def _persist_thinking(self, level: ThinkingLevel) -> None:
        await self.app_server.resources.config.set_thinking(level)

    async def _apply_picker_update(
        self,
        kind: str,
        value: str,
        update: Callable[[], Awaitable[None]],
        picker: ModelPickerApp | ThinkingPickerApp,
    ) -> bool:
        if self._picker_update_pending:
            return False
        self._picker_update_pending = True
        try:
            picker.clear_error()
            if self._pending_picker_reload != (kind, value):
                saved, error = await self._run_settings_update(
                    f"{kind} {value}", update
                )
                if not saved:
                    picker.show_error(
                        f"Could not save for this session: {error}. "
                        "Enter retries; Esc closes."
                    )
                    return False
                self._pending_picker_reload = (kind, value)
            reload_error = await self._reload_config()
            if reload_error is not None:
                picker.show_error(
                    "Setting saved, but this session may still use its previous "
                    f"value: runtime reload failed ({reload_error}). "
                    "Enter retries reload; Esc closes."
                )
                return False
            self._pending_picker_reload = None
            return True
        finally:
            self._picker_update_pending = False

    async def on_thinking_picker_app_cancelled(
        self, _event: ThinkingPickerApp.Cancelled
    ) -> None:
        self._pending_picker_reload = None
        await self._switch_to_input_app()

    async def on_log_level_picker_app_cancelled(
        self, _event: LogLevelPickerApp.Cancelled
    ) -> None:
        await self._switch_to_input_app()

    async def on_log_level_picker_app_applied(
        self, message: LogLevelPickerApp.Applied
    ) -> None:
        parts: list[str] = []
        previous_chain = get_log_level_chain()
        if message.session_level:
            set_session_override(message.session_level)
            parts.append(f"session override → {message.session_level}")
        else:
            set_session_override(None)
            if previous_chain.session:
                parts.append("session override cleared")
        config_feedback: str | None = None
        try:
            if message.config_level:
                await self._persist_log_level_config(
                    level=message.config_level, clear=False
                )
                config_feedback = f"config.toml → {message.config_level}"
            elif message.config_cleared:
                await self._persist_log_level_config(level=None, clear=True)
                config_feedback = "config.toml cleared"
        except Exception as exc:
            logger.warning("Failed to persist log-level config", exc_info=exc)
            chain = get_log_level_chain()
            session = (
                f"session override {chain.session} applied"
                if chain.session
                else "session override cleared"
            )
            self._set_recovery_issue(
                "log-level-save",
                f"Log-level config.toml save failed: {exc}; {session} "
                f"(effective: {chain.effective}). Config change was not saved; "
                "run /log-level to retry the config change.",
                severity="error",
            )
            await self._switch_to_input_app()
            return
        self._set_recovery_issue("log-level-save", None)
        if config_feedback is not None:
            parts.append(config_feedback)
        chain = get_log_level_chain()
        feedback = (
            "  ".join(parts) + f"  (effective: {chain.effective})"
            if parts
            else f"Log level unchanged  (effective: {chain.effective})"
        )
        await self._switch_to_input_app()
        await self._mount_and_scroll(UserCommandMessage(feedback))

    async def _persist_log_level_config(
        self, *, level: str | None, clear: bool
    ) -> None:
        if clear:
            await self._remove_config_field("log_level")
        else:
            await self._persist_config_changes({"log_level": level})

    async def on_theme_picker_app_theme_previewed(
        self, message: ThemePickerApp.ThemePreviewed
    ) -> None:
        await self._apply_theme(message.theme)

    async def on_theme_picker_app_theme_selected(
        self, message: ThemePickerApp.ThemeSelected
    ) -> None:
        await self._apply_theme(message.theme)
        if not await self._persist_theme(message.theme):
            # Keep the selected row and picker alive so Enter can retry; only
            # the speculative visual preview is reverted to the saved theme.
            return
        self._pending_theme_selection = None
        self._set_recovery_issue("theme-save", None)
        await self._switch_to_input_app()

    async def _persist_theme(self, theme: str) -> bool:
        try:
            await self.app_server.resources.config.update({"theme": theme})
        except Exception as exc:
            logger.warning("Failed to persist theme %s", theme, exc_info=exc)
            self._pending_theme_selection = theme
            await self._apply_theme(self.config.theme)
            self._set_recovery_issue(
                "theme-save",
                f"Theme '{theme}' save failed: {exc}. The saved theme "
                f"'{self.config.theme}' is restored and usable; selection is "
                "retained in the picker. Press Enter to retry or Esc to leave; "
                "run /theme to reopen it later.",
                severity="error",
            )
            return False
        return True

    async def on_theme_picker_app_cancelled(
        self, message: ThemePickerApp.Cancelled
    ) -> None:
        await self._apply_theme(message.original_theme)
        await self._switch_to_input_app()

    def _restyle_diff_widgets(self, *, ansi: bool, dark: bool) -> None:
        # Diff content bakes in ANSI-vs-truecolor styling, so it must be rebuilt.
        for widget in self.query(EditResultWidget):
            widget.request_diff_render(ansi=ansi, dark=dark)

    async def on_mcpapp_mcpclosed(self, _message: MCPApp.MCPClosed) -> None:
        await self._mount_and_scroll(UserCommandMessage("MCP servers closed."))
        await self._switch_to_input_app()

    async def on_mcpapp_mcptoggled(self, message: MCPApp.MCPToggled) -> None:
        try:
            await self.app_server.resources.mcp.toggle(
                name=message.name,
                disabled=message.disabled,
                tool_name=message.tool_name,
            )
        except AppServerResponseError as exc:
            # Nothing above a message handler catches, so propagating takes the
            # app down over one keypress. The row was flipped optimistically
            # before the request; the refresh below reads the server over it.
            logger.warning("MCP toggle for %r was rejected: %s", message.name, exc)
            self.notify(str(exc), severity="warning", markup=False)
        self.query_one(_get_mcp_app_class()).refresh_index()
        self._refresh_banner()

    async def on_mcpapp_mcpoauth_requested(
        self, message: MCPApp.MCPOAuthRequested
    ) -> None:
        await self._switch_to_input_app(return_to_settings=False)
        await self._switch_from_input(
            _get_mcp_oauth_app_class()(
                server_name=message.server_name, mcp=self.app_server.resources.mcp
            )
        )

    async def on_mcpoauth_app_mcpoauth_closed(
        self, message: MCPOAuthApp.MCPOAuthClosed
    ) -> None:
        if message.refreshed:
            await self._refresh_mcp_browser()
        await self._switch_to_input_app(return_to_settings=False)
        await self._show_mcp(cmd_args=message.server_name if message.refreshed else "")

    async def on_proxy_setup_app_proxy_setup_closed(
        self, message: ProxySetupApp.ProxySetupClosed
    ) -> None:
        if self._proxy_save_pending:
            return
        if not message.saved:
            await self._mount_and_scroll(UserCommandMessage("Proxy setup cancelled."))
        else:
            self._proxy_save_pending = True
            try:
                await self._persist_proxy(message.changes)
            except Exception as exc:
                logger.warning("Proxy settings update was rejected: %s", exc)
                proxy = self.query_one(_get_proxy_setup_app_class())
                proxy.finish_save(success=False)
                proxy.show_error(str(exc))
                self._proxy_save_pending = False
                return

        await self._switch_to_input_app()

    async def _handle_command(self, user_input: str) -> bool:
        self._refresh_command_registry()
        if user_input.strip().split(maxsplit=1)[0:1] == [
            "/retry"
        ] and not self.commands.has_command("retry"):
            await self._mount_and_scroll(
                UserCommandMessage("No interrupted response is available to retry.")
            )
            return True
        resolved = self.commands.parse_command(user_input)
        if not resolved:
            return False
        cmd_name, command, cmd_args = resolved
        display = self._command_display(user_input, cmd_name)
        return await self._invoke_resolved_command(cmd_name, command, cmd_args, display)

    @staticmethod
    def _command_display(user_input: str, cmd_name: str) -> str:
        command_text = user_input.strip()
        return (
            command_text.removeprefix("/") if command_text.startswith("/") else cmd_name
        )

    async def _invoke_resolved_command(
        self, cmd_name: str, command: Command, cmd_args: str, display_text: str
    ) -> bool:
        command_message = SlashCommandMessage(display_text)
        await self._mount_and_scroll(command_message)
        handler = getattr(self, command.handler)
        if asyncio.iscoroutinefunction(handler):
            await handler(cmd_args=cmd_args, command_message=command_message)
        else:
            handler(cmd_args=cmd_args, command_message=command_message)
        return True

    def _get_skill_entries(self) -> list[tuple[str, str]]:
        if self._app_server is None:
            return []
        return [
            (f"/{skill.name}", skill.description)
            for skill in self.app_server.resources.runtime.skills
            if skill.user_invocable
        ]

    def _resolve_skill(self, user_input: str) -> Skill | None:
        stripped = user_input.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped[1:].split(None, 1)
        if not parts:
            return None
        name = parts[0].lower()
        skill = self.app_server.resources.runtime.get_skill(name)
        if skill is None or not skill.user_invocable:
            return None
        return Skill(command=user_input, name=skill.name)

    async def _handle_bash_command(self, command: str) -> None:
        try:
            await self._handle_bash_command_inner(command)
        finally:
            current = asyncio.current_task()
            if self._bash_task is current:
                self._bash_task = None
            self._queue.notify_busy_changed()

    async def _handle_bash_command_inner(self, command: str) -> None:
        if not command:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No command provided after '!'", collapsed=self._tools_collapsed
                )
            )
            return

        await self._ensure_loading_widget("Running command")
        self._queue.notify_busy_changed()
        bash_loading_widget = self._loading_widget

        try:
            async for event in self.app_server.resources.shell.run(command):
                if self.event_handler is not None:
                    await self.event_handler.handle_event(
                        event, loading_widget=self._loading_widget
                    )
        except asyncio.CancelledError:
            pass
        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(f"Command failed: {e}", collapsed=self._tools_collapsed)
            )
        finally:
            if self._loading_widget is bash_loading_widget:
                await self._remove_loading_widget()

    async def _handle_user_message(self, message: str) -> None:
        # Enter the "turn in flight" state: the unified backend runs the turn
        # server-side, so nothing looks active until TurnStarted arrives. Marking
        # it here (rather than waiting for the server) makes the spinner instant,
        # keeps the prompt interruptible during the enqueue -> promote gap, and
        # renders it as a normal message (not "queued") -- matching the legacy
        # local-turn behavior. _begin_unsolicited_turn clears it on TurnStarted.
        self._begin_pending_turn()
        await self._ensure_loading_widget()
        try:
            if await self._enqueue_prompt_with_resources(
                message, optimistic_start=True
            ):
                return
        except BaseException:
            self._turn_supervisor._clear_pending_turn()
            self._clear_active_turn_pins()
            raise
        # Enqueue was rejected (e.g. prompt prep aborted): leave the in-flight
        # state so the UI returns to idle instead of a stuck spinner.
        self._turn_supervisor._clear_pending_turn()
        self._clear_active_turn_pins()
        await self._remove_loading_widget()
        input_widget = self.query_one(ChatInputContainer)
        if not input_widget.value:
            input_widget.value = message

    def _reset_ui_state(self) -> None:
        self._transcript_presenter.reset_ui_state()

    async def _rebuild_transcript_from_current_session(self) -> None:
        await self._transcript_presenter.rebuild_transcript_from_current_session()

    async def _deferred_resume_and_start(self) -> None:
        logger.debug("Startup resume stage=initial-history-started")
        try:
            await self._resume_history_from_messages()
            logger.debug("Startup resume stage=initial-transcript-rebuilt")
            await self._queue.sync_server_queue(self.app_server.turn_queue)
            logger.debug("Startup resume stage=initial-queue-synced")
        finally:
            self._initial_history_loaded.set()
            logger.debug("Startup resume stage=initial-history-ready")
        self._app_server_events_worker = self.run_worker(
            self._listen_app_server_events(), exclusive=False
        )

    async def _resume_history_from_messages(self) -> None:
        await self._transcript_presenter.resume_history_from_messages()

    async def _mount_history_batch(
        self,
        batch: list[PublicHistoryEntry],
        messages_area: Widget,
        *,
        start_index: int,
        before: Widget | int | None = None,
        after: Widget | None = None,
    ) -> None:
        await self._transcript_presenter.mount_history_batch(
            batch, messages_area, start_index=start_index, before=before, after=after
        )

    async def _restore_reading_anchor(
        self, reading_anchor: tuple[str, int, int], messages_area: Widget
    ) -> None:
        await self._transcript_presenter.restore_reading_anchor(
            reading_anchor, messages_area
        )

    def _admit_live_history(self) -> None:
        self._transcript_presenter.admit_live_history()

    def _on_chat_scroll_activity(self, chat: ChatScroll, new_value: float) -> None:
        """Host reaction to ChatScroll scroll activity (injected into the widget)."""
        self._transcript_presenter.on_chat_scroll_activity(chat, new_value)

    def _request_transcript_reconcile(self, *, scroll_driven: bool = False) -> None:
        self._transcript_presenter.request_transcript_reconcile(
            scroll_driven=scroll_driven
        )

    def _request_scroll_driven_reconcile(self) -> None:
        """Kick a scroll-driven reconcile tick (injected into ChatScroll)."""
        self._transcript_presenter.request_scroll_driven_reconcile()

    def _remember_transcript_anchor(self) -> None:
        self._transcript_presenter.remember_transcript_anchor()

    def on_resize(self) -> None:
        self._transcript_presenter.handle_resize()

    def _is_tool_enabled_in_main_agent(self, tool: str) -> bool:
        return self.app_server.resources.runtime.has_tool(tool)

    async def _wait_for_typing_pause(self) -> None:
        await self._turn_supervisor.wait_for_typing_pause()

    async def _request_local_user_input(
        self, request: UserQuestionRequest
    ) -> UserQuestionResult:
        return await self._turn_supervisor.request_local_user_input(request)

    def _secondary_surface_active(self) -> bool:
        return self._bottom_panel.secondary_surface_active()

    def _indicate_pending_action(self) -> None:
        self._bottom_panel.indicate_pending_action()

    async def _present_pending_callback(self) -> None:
        await self._turn_supervisor.present_pending_callback()

    async def _show_callback(self, callback: PublicCallbackEntry) -> None:
        await self._turn_supervisor.show_callback(callback)

    async def _respond_to_active_callback(
        self, output: UserInputCallbackOutput
    ) -> None:
        await self._turn_supervisor.respond_to_active_callback(output)

    async def _ensure_runtime_ready(self) -> None:
        await self._turn_supervisor.ensure_runtime_ready()

    def _show_no_output_outcome(self) -> None:
        self._turn_supervisor.show_no_output_outcome()

    def _refresh_status_for_event(self, event: AppServerEvent) -> None:
        if isinstance(event, AgentsUpdate):
            self._track_graduation_attempts(event)
            self.call_after_refresh(self._maybe_show_graduation)
        if isinstance(event, SessionCompacted):
            self._graduation.state.compacted()
            self.call_after_refresh(self._maybe_show_graduation)
            self._root_context_unknown = True
        elif isinstance(event, SessionContextCleared):
            self._root_context_unknown = False
        if isinstance(event, (SessionSnapshot, SessionUpdated)):
            self._refresh_context_progress()
        if isinstance(event, (TurnCompleted, SessionCompacted, SessionContextCleared)):
            self._root_compacting = False
            self._refresh_context_progress()
        if isinstance(event, TurnStarted):
            self._refresh_context_progress()
        if isinstance(event, TurnCompleted):
            self._schedule_branch_refresh()
        if isinstance(event, SessionCompacted) and self._status_line is not None:
            self._status_line.set_state(
                replace(self._status_line.state, context_tokens=None, compacting=False)
            )
            self._refresh_main_agent_details()
        if isinstance(event, (HistoryEntryAdded, HistoryEntryUpdated)):
            checkpoint = _public_entry(event)
            if (
                isinstance(checkpoint, PublicCheckpointEntry)
                and checkpoint.kind == "compaction"
            ):
                if (
                    checkpoint.generation_status
                    is PublicEntryGenerationStatus.COMPLETED
                ):
                    # Completion is published before the post-compaction stats.
                    self._root_context_unknown = True
                self._root_compacting = (
                    checkpoint.generation_status
                    is PublicEntryGenerationStatus.IN_PROGRESS
                )
                self._refresh_context_progress()

    async def _handle_agents_update(
        self, event: AgentsUpdate, received_at: float
    ) -> None:
        await self._agent_browser.handle_agents_update(event, received_at)

    async def _handle_turn_event(self, event: AppServerEvent) -> None:
        received_at = time.monotonic()
        self._turn_supervisor._track_turn_outcome(event)
        self._refresh_status_for_event(event)
        if (
            isinstance(event, (TurnStarted, TurnUpdated, TurnCompleted))
            and self._loading_widget is not None
        ):
            self._loading_widget.set_waiting_only(event.turn.waiting_only)
        if isinstance(event, AgentsUpdate):
            await self._handle_agents_update(event, received_at)
            return
        if isinstance(event, TurnQueueUpdated):
            await self._queue.sync_server_queue(event.queue)
            return
        if isinstance(event, ServerWarning):
            self.notify(event.params.warning.message, severity="warning")
            return
        if isinstance(event, ServerError):
            self.notify(event.params.error.message, severity="error")
            return
        if isinstance(event, CallbackRequested):
            await self._show_callback(event.callback)
            return
        if isinstance(event, StatsUpdated):
            self._update_context_progress(event)
            return
        entry = _public_entry(event)
        if (
            isinstance(event, HistoryEntryUpdated)
            and isinstance(entry, PublicCallbackEntry)
            and entry.state.status == "expired"
        ):
            # The server expired this question unanswered (plan U/OD1): close
            # the live question UI, preserve the draft, and continue. The
            # entry still flows through the generic handler so history shows
            # the terminal expired state.
            self._invalidate_user_input()
            await self._turn_supervisor.question_expired(entry)
        if isinstance(entry, PublicNoticeEntry) and isinstance(
            entry.detail, WaitingForInputNoticeDetail
        ):
            await self._remove_loading_widget()
        elif self._loading_widget is None and is_progress_event(event):
            await self._ensure_loading_widget()
        if self.event_handler:
            await self.event_handler.handle_event(
                event, loading_widget=self._loading_widget
            )
            self._sync_retry_command_availability()

    async def _shutdown(self) -> None:
        # Stop consuming app-server events before Textual tears down the DOM.
        # An in-flight turn (e.g. a queued skill) can still be streaming effects
        # that _listen_app_server_events mounts; Textual only cancels workers
        # after _shutdown has closed the screen, so the worker would otherwise
        # mount into a closing tree and raise MountError. This runs on every
        # shutdown path (App.run_async and Pilot.run_test both call _shutdown).
        self._transcript.reset()
        await self._stop_app_server_event_listener()
        await super()._shutdown()

    async def _stop_app_server_event_listener(self) -> None:
        await self._begin_shutdown()
        worker = self._app_server_events_worker
        if worker is None:
            return
        self._app_server_events_worker = None
        async with self._app_server_event_handler_lock:
            worker.cancel()
            # wait() re-raises the cancellation (and any error) the worker unwound
            # with; we only need it fully stopped before the DOM is torn down.
            with suppress(WorkerError, asyncio.CancelledError):
                await worker.wait()

    async def _listen_app_server_events(self) -> None:
        from chartreux.app_server import AppServerConnectionClosed

        try:
            async with aclosing(self.app_server.events()) as events:
                async for event in events:
                    await self._resume_ui_ready.wait()
                    async with self._app_server_event_handler_lock:
                        if isinstance(event, TurnStarted):
                            self._active_turn_start = (
                                self._active_turn_start
                                if self._active_turn_start is not None
                                else self._transcript.admitted_end_index
                            )
                            # A server-started turn updates the transcript behind the
                            # current task; it must not change inspection or focus.
                            # The in-flight turn surfaced: leave the pending state and
                            # wake any interrupt waiting for it.
                            self._turn_supervisor._clear_pending_turn()
                            await self._queue.turn_started(event.turn.queue_item_id)
                            await self._turn_supervisor._begin_unsolicited_turn()
                        await self._handle_turn_event(event)
                        if isinstance(event, TurnCompleted):
                            await self._turn_supervisor._complete_unsolicited_turn(
                                event
                            )
                        self._maybe_settle_interrupt()
        except AppServerConnectionClosed:
            if not self._shutdown_started:
                raise

    def _turn_ui_lock(self) -> asyncio.Lock:
        if self._turn_ui_mutex is None:
            self._turn_ui_mutex = asyncio.Lock()
        return self._turn_ui_mutex

    async def _handle_turn(
        self,
        prompt: str,
        *,
        prepared_prompt: PreparedPrompt | None = None,
        client_message_id: str | None = None,
        injected: bool = False,
        user_initiated_retry: bool = False,
        incomplete_stream_retries: int = 0,
        idempotency_key: str | None = None,
    ) -> None:
        await self._turn_supervisor.handle_turn(
            prompt,
            prepared_prompt=prepared_prompt,
            client_message_id=client_message_id,
            injected=injected,
            user_initiated_retry=user_initiated_retry,
            incomplete_stream_retries=incomplete_stream_retries,
            idempotency_key=idempotency_key,
        )

    async def _auto_retry_incomplete_stream(
        self, incomplete_stream_retries: int, *, idempotency_key: str | None = None
    ) -> None:
        await self._turn_supervisor.auto_retry_incomplete_stream(
            incomplete_stream_retries, idempotency_key=idempotency_key
        )

    async def _retry(
        self,
        cmd_args: str = "",
        command_message: SlashCommandMessage | None = None,
        incomplete_stream_retries: int = 0,
        **_kwargs: Any,
    ) -> None:
        # R2/OD3: a MANUAL retry is a user-directed new episode — it
        # invalidates any pending automatic continuation (and keeps its
        # intentional cooldown reset: it starts a NEW user-directed episode,
        # never replenishing the automatic one).
        self._invalidate_user_input()
        await self._turn_supervisor.retry(
            cmd_args,
            command_message=command_message,
            incomplete_stream_retries=incomplete_stream_retries,
            **_kwargs,
        )

    async def _finalize_turn_ui(
        self, *, notify_complete: bool = True, turn_ui_generation: int | None = None
    ) -> None:
        await self._turn_supervisor.finalize_turn_ui(
            notify_complete=notify_complete, turn_ui_generation=turn_ui_generation
        )

    async def _interrupt_turn(self) -> None:
        await self._turn_supervisor.interrupt_turn()

    async def _finish_interrupt(self, interrupt: Awaitable[None]) -> None:
        await self._turn_supervisor.finish_interrupt(interrupt)

    async def action_show_help(self) -> None:
        await self._show_help()

    async def _show_help(self, **kwargs: Any) -> None:
        self._refresh_command_registry()
        help_text = self.commands.get_help_text()
        await self._mount_and_scroll(UserCommandMessage(help_text))

    def _get_last_assistant_message_text(self) -> str | None:
        for child in reversed(self._messages_area.children):
            if not isinstance(child, AssistantMessage):
                continue
            if not (content := child.get_content().strip()):
                continue
            return content
        return None

    async def _copy_last_agent_message(self, **kwargs: Any) -> None:
        if (content := self._get_last_assistant_message_text()) is None:
            self.notify(
                "No agent message available to copy", severity="warning", timeout=3
            )
            return

        copy_text_to_clipboard(
            self, content, success_message="Last agent message copied to clipboard"
        )

    async def _refresh_mcp_browser(self) -> str:
        # Wait for deferred init before the destructive force-refresh, otherwise
        # clearing the registries mid-initialization briefly empties the list
        # (the panel collapses then expands once discovery repopulates it).
        await self.app_server.resources.runtime.wait_until_ready()
        await self.app_server.resources.mcp.refresh()
        await self.app_server.resources.mcp.read()
        self._refresh_banner()
        return "Refreshed."

    async def _maybe_handle_mcp_subcommand(self, cmd_args: str) -> bool:
        parsed = parse_mcp_subcommand(cmd_args)
        if parsed is None:
            return False

        match parsed.name:
            case "add":
                await self._mcp_add(parsed.args)
            case "status":
                if parsed.args:
                    await self._mount_and_scroll(
                        ErrorMessage("Usage: /mcp status", collapsed=True)
                    )
                    return True
                await self._show_mcp_status()
            case "login":
                await self._mcp_login(parsed.args)
            case "logout":
                await self._mcp_logout(parsed.args)
        return True

    async def _show_mcp_status(self) -> None:
        await self.app_server.resources.runtime.wait_until_ready()
        statuses = (await self.app_server.resources.mcp.read()).statuses
        if not statuses:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "No MCP servers configured. Run `/mcp add <url>` to add one."
                )
            )
            return
        lines = ["### MCP auth status", ""]
        for alias, status in sorted(statuses.items()):
            lines.append(f"- `{alias}`: `{status}`")
        await self._mount_and_scroll(UserCommandMessage("\n".join(lines)))

    async def _mcp_login(self, alias: str) -> None:
        if not alias:
            await self._mount_and_scroll(
                ErrorMessage("Usage: /mcp login <alias>", collapsed=True)
            )
            return

        try:
            async for event in self.app_server.resources.mcp.login(alias):
                await self._mount_and_scroll(
                    UserCommandMessage(
                        f"Open this URL in your browser:\n\n  {event.url}"
                    )
                )
                try:
                    webbrowser.open(event.url)
                except Exception as exc:
                    logger.debug("Failed to open MCP OAuth URL in browser: %s", exc)
        except AppServerResponseError as exc:
            await self._mount_and_scroll(
                ErrorMessage(exc.error.message, collapsed=True)
            )
            return

        await self._mount_and_scroll(
            UserCommandMessage(f"MCP server `{alias}` authenticated.")
        )

    async def _mcp_logout(self, alias: str) -> None:
        if not alias:
            await self._mount_and_scroll(
                ErrorMessage("Usage: /mcp logout <alias>", collapsed=True)
            )
            return

        try:
            await self.app_server.resources.mcp.logout(alias)
        except AppServerResponseError as exc:
            await self._mount_and_scroll(
                ErrorMessage(exc.error.message, collapsed=True)
            )
            return

        await self._mount_and_scroll(
            UserCommandMessage(f"MCP server `{alias}` logged out.")
        )

    async def _mcp_add(self, raw_args: str) -> None:
        if is_mcp_add_help_request(raw_args):
            await self._mount_and_scroll(UserCommandMessage(MCP_ADD_HELP))
            return

        try:
            args = parse_mcp_add_args(raw_args)
        except ValueError as exc:
            await self._mount_and_scroll(ErrorMessage(str(exc), collapsed=True))
            return

        try:
            result = await self.app_server.resources.mcp.add(
                url=args.url,
                name=args.name,
                scopes=args.scopes,
                transport=args.transport,
            )
        except AppServerResponseError as exc:
            await self._mount_and_scroll(
                ErrorMessage(exc.error.message, collapsed=True)
            )
            return

        head = (
            f"Added OAuth MCP server `{result.name}`."
            if result.created
            else f"OAuth MCP server `{result.name}` is already configured."
        )
        tail = (
            "Starting OAuth login..."
            if args.login
            else (
                f"Run `/mcp login {result.name}` to authenticate, "
                "or `/mcp status` to inspect it."
            )
        )
        await self._mount_and_scroll(UserCommandMessage(f"{head}\n{tail}"))

        if args.login:
            await self._mcp_login(result.name)

    async def _show_mcp(self, cmd_args: str = "", **kwargs: Any) -> None:
        if await self._maybe_handle_mcp_subcommand(cmd_args):
            return

        state = await self.app_server.resources.mcp.read()
        if not state.sources:
            await self._mount_and_scroll(
                UserCommandMessage(
                    "No MCP servers configured. Run `/mcp add <url>` to add one."
                )
            )
            return

        if self._current_bottom_app == BottomApp.MCP:
            return
        name = cmd_args.strip()
        all_names = [source.name for source in state.sources]
        if name and name not in all_names:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Unknown MCP server: {name}. Known: " + ", ".join(all_names),
                    collapsed=self._tools_collapsed,
                )
            )
            return
        mcp_app_class = _get_mcp_app_class()
        await self._mount_and_scroll(UserCommandMessage("MCP servers opened..."))
        await self._switch_from_input(
            mcp_app_class(
                state=state,
                initial_source=name,
                state_getter=lambda: self.app_server.resources.mcp.state,
                refresh_callback=self._refresh_mcp_browser,
            )
        )

    async def _show_status(self, **kwargs: Any) -> None:
        stats = self.app_server.resources.runtime.stats
        session_cached = (
            f" _(including {stats.session_cached_tokens:,} cached)_"
            if stats.session_cached_tokens > 0
            else ""
        )
        last_turn_cached = (
            f" _(including {stats.last_turn_cached_tokens:,} cached)_"
            if stats.last_turn_cached_tokens > 0
            else ""
        )
        status_text = f"""## Agent Statistics

- **Steps**: {stats.steps:,}
- **Session Prompt Tokens**: {stats.session_prompt_tokens:,}{session_cached}
- **Session Completion Tokens**: {stats.session_completion_tokens:,}
- **Session Total LLM Tokens**: {stats.session_total_llm_tokens:,}
- **Last Turn Tokens**: {stats.last_turn_total_tokens:,}{last_turn_cached}
- **Cost**: ${stats.session_cost:.4f}
"""
        try:
            roots = await self.app_server.resources.config.read_roots()
            effective = (
                "\n".join(f"  - `{root}`" for root in roots.effective_roots)
                or "  - None"
            )
            saved = (
                "unavailable"
                if roots.saved_roots is None
                else "\n".join(f"  - `{root}`" for root in roots.saved_roots)
                or "  - None"
            )
            status_text += (
                f"\n## Authorized Roots\n\nProject: `{roots.project}`\n\n"
                f"**Effective roots**\n{effective}\n\n**Saved roots**\n{saved}\n"
            )
        except Exception:
            status_text += "\n## Authorized Roots\n\nProject roots unavailable.\n"
        await self._mount_and_scroll(UserCommandMessage(status_text))

    async def _show_model(self, **kwargs: Any) -> None:
        """Switch to the model picker in the bottom panel."""
        if self._current_bottom_app == BottomApp.ModelPicker:
            return
        await self._switch_to_model_picker_app()

    async def _show_thinking(self, **kwargs: Any) -> None:
        """Switch to the thinking level picker in the bottom panel."""
        if self._current_bottom_app == BottomApp.ThinkingPicker:
            return
        await self._switch_to_thinking_picker_app()

    async def _show_theme(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ThemePicker:
            return
        await self._switch_to_theme_picker_app()

    async def _show_usage(self, **kwargs: Any) -> None:
        if self._shutdown_started or (
            self._usage_worker is not None and not self._usage_worker.is_finished
        ):
            return
        from chartreux.cli.textual_ui.screens.usage import UsageScreen

        if any(isinstance(screen, UsageScreen) for screen in self.screen_stack):
            return
        self._usage_worker = self.run_worker(
            self._wait_for_usage(), exclusive=False, name="usage"
        )

    async def _wait_for_usage(self) -> None:
        from chartreux.cli.textual_ui.screens.usage import UsageScreen

        try:
            read_provider, subscribe = self._usage_screen_providers()
            # Push before reading: the browser owns loading and read-error states.
            await self.push_screen_wait(UsageScreen(read_provider, subscribe))
        finally:
            self._usage_worker = None
            if not self._shutdown_started:
                await self._present_pending_callback()

    async def _show_settings(self, **kwargs: Any) -> None:
        if (
            self._settings_worker is not None and not self._settings_worker.is_finished
        ) or any(
            isinstance(screen, _get_settings_screen_class())
            for screen in self.screen_stack
        ):
            return
        self._settings_worker = self.run_worker(
            self._wait_for_settings(), exclusive=False, name="settings"
        )

    async def _show_proxy_setup(self, **kwargs: Any) -> None:
        if self._current_bottom_app == BottomApp.ProxySetup:
            return
        await self._switch_to_proxy_setup_app()

    async def _show_web_search(self, **kwargs: Any) -> None:
        """Open the dedicated web search settings editor while idle."""
        if self._is_busy():
            await self._mount_and_scroll(
                ErrorMessage(
                    "Web search settings are available while no turn is active.",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        if (
            self._web_search_worker is not None
            and not self._web_search_worker.is_finished
        ):
            return
        self._web_search_worker = self.run_worker(
            self._wait_for_web_search_from_settings(),
            exclusive=False,
            name="web-search-settings",
        )

    async def _wait_for_web_search_from_settings(self) -> None:
        try:
            from chartreux.ui.settings_service import SettingsService

            service = SettingsService(
                self.app_server.resources.config, apply_ui=self._apply_config_to_ui
            )
            snapshot = await service.read()
            if snapshot.web_search is None:
                raise RuntimeError("Web search settings are unavailable")
            await self.push_screen_wait(
                _get_web_search_screen_class()(
                    service, snapshot, credentials=_ProviderCredentials()
                )
            )
        except Exception as exc:
            logger.warning("Could not open web search settings", exc_info=True)
            await self._mount_and_scroll(
                ErrorMessage(f"Could not open web search settings: {exc}")
            )
        finally:
            self._web_search_worker = None
            if self._settings_return_web_search:
                self._settings_return_web_search = False
                await self._show_settings()
            else:
                await self._present_pending_callback()

    def _track_graduation_attempts(self, event: AgentsUpdate) -> None:
        for agent in event.agents:
            run_id = agent.current_run_id or agent.latest_run_id
            task = agent.current_task_summary
            if not run_id or not task:
                continue
            # Only attempts observed running in this UI are live evidence. A
            # resumed terminal summary is not a newly consumed attempt.
            key = f"{self.app_server.session_id}:{agent.agent_id}:{run_id}"
            if agent.current_run_status == "running":
                if "implementation" in agent.launch_purposes:
                    self._graduation_live_attempts[key] = (
                        f"{self.app_server.session_id}:{agent.agent_id}:{task}"
                    )
            elif key in self._graduation_live_attempts:
                # ERROR does not distinguish provider exhaustion from an
                # implementation failure. Only the structured budget outcome
                # supplies unambiguous consumed-attempt evidence here.
                # Conservative v1: budget-exceeded runs with retained identity only;
                # generic implementation-failure coverage is a follow-up.
                # Accepted v1: fresh workers retrying the same task have separate
                # agent-identity buckets; cross-worker aggregation is not inferred.
                if agent.stop_reason == "budget_exceeded":
                    self._graduation.state.implementation_failed(
                        self._graduation_live_attempts[key], key, attempt_budgeted=True
                    )
                # Idle (retained) and closed (ephemeral terminal) agents both
                # end the live attempt; stale keys must not accumulate. WP2:
                # a suspended/restoring agent's failed run is terminal too —
                # the resumed successor is a NEW run identity tracked fresh.
                if agent.availability in {"idle", "closed", "suspended", "restoring"}:
                    self._graduation_live_attempts.pop(key, None)

    def _graduation_models_saved(
        self,
        roster_before: frozenset[str],
        roster_after: frozenset[str],
        saved_models: frozenset[str],
    ) -> None:
        # The dispatch roster is retired with package 16b: the callback still
        # receives the pre/post rosters (empty — the workbench no longer
        # computes them) and the graduation handler records the transition.
        self._graduation.state.model_saved(roster_before, roster_after, saved_models)
        self._graduation.save()

    def _maybe_show_graduation(self) -> None:
        notice = self._graduation_notice
        if notice is None:
            return
        idle = (
            not self._is_busy()
            and len(self.screen_stack) == 1
            and self._active_callback is None
            and not any(agent_is_active(agent) for agent in self._agent_summaries)
        )
        if not self._graduation.state.eligible(idle=idle, headless=False):
            return
        self._graduation.state.shown = True
        if not self._graduation.save():
            return
        notice.update(
            Text.from_markup(
                "Budgeted implementation attempts failed repeatedly. "
                "[@click=app.graduation_customize]Customize[/] · "
                "[@click=app.graduation_dismiss]Dismiss[/] (applies next session)"
            )
        )
        notice.display = True

    def action_graduation_dismiss(self) -> None:
        self._graduation.state.dismissed = True
        self._graduation.save()
        if self._graduation_notice is not None:
            self._graduation_notice.hide()

    async def action_graduation_customize(self) -> None:
        if self._is_busy():
            return
        self.action_graduation_dismiss()
        await self._show_providers(initial_view="presets")

    async def _show_providers(self, **kwargs: Any) -> None:
        """Open Provider Settings only while the session is idle."""
        if self._is_busy():
            await self._mount_and_scroll(
                ErrorMessage(
                    "Provider Settings is only available while no turn is active.",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        if (
            self._provider_management_worker is not None
            and not self._provider_management_worker.is_finished
        ):
            await self._mount_and_scroll(
                ErrorMessage("Provider Settings is already open.")
            )
            return
        self._provider_management_worker = self.run_worker(
            self._wait_for_provider_management_from_settings(
                initial_view=kwargs.get("initial_view", "providers")
            ),
            exclusive=False,
            name="provider-management",
        )

    async def _wait_for_provider_management_from_settings(
        self, *, initial_view: str = "providers"
    ) -> None:
        try:
            await self._wait_for_provider_management(initial_view=initial_view)
        finally:
            if self._settings_return_providers:
                self._settings_return_providers = False
                await self._show_settings()
            else:
                await self._present_pending_callback()
                self._maybe_show_graduation()

    async def _wait_for_provider_management(
        self, *, initial_view: str = "providers"
    ) -> None:
        """Wait outside the dismissed screen, then explicitly report adoption."""
        try:
            workbench = import_module("chartreux.ui.providers.workbench")
            contracts = import_module("chartreux.core.model_catalog.contracts")
            catalog_loader = import_module("chartreux.core.model_catalog.loader")
            discovery_module = import_module("chartreux.core.model_catalog.discovery")
            credentials = _ProviderCredentials()
            config = _ProviderConfigService(self)
            allowed_models, thinking_overrides = await _session_selection_restrictions(
                self
            )
            result = await self.push_screen_wait(
                workbench.ProviderWorkbenchScreen(
                    discovery=discovery_module.discover_models,
                    catalog_writer=catalog_loader.CatalogStore(),
                    credentials=credentials,
                    credential_resolver=credentials.resolve_key,
                    config=config,
                    snapshot=catalog_loader.load_catalog(),
                    initial_view=initial_view,
                    allowed_models=allowed_models,
                    thinking_overrides=thinking_overrides,
                    on_models_saved=self._graduation_models_saved,
                    session_only_credential_envs=self._session_only_credential_envs,
                    tls=contracts.TLSConfig(
                        enable_system_trust_store=self.config.enable_system_trust_store
                    ),
                )
            )
            if not isinstance(result, contracts.ProviderWorkbenchResult):
                raise TypeError("Provider workbench returned an invalid result")
        except Exception:
            logger.warning("Could not open provider management", exc_info=True)
            await self._mount_and_scroll(
                ErrorMessage("Could not open Provider Settings. Please try again.")
            )
            return
        if result.warning:
            self.notify(result.warning, severity="warning", markup=False)
        if result.changed:
            reload_result = await _ProviderConfigService(
                self
            ).reload_catalog_and_config()
            if reload_result.snapshot is None:
                await self._mount_and_scroll(
                    ErrorMessage(
                        reload_result.message
                        or "Provider changes were saved but could not be adopted. Retry /providers."
                    )
                )
                return
        if result.status == "completed":
            await self._mount_and_scroll(
                UserCommandMessage("Provider changes adopted.")
            )
        else:
            await self._mount_and_scroll(
                UserCommandMessage("Provider Settings closed.")
            )

    async def _log_level_command(self, **kwargs: Any) -> None:
        await self._switch_to_log_level_picker_app()

    async def _show_session_picker(
        self, command_message: SlashCommandMessage | None = None, **kwargs: Any
    ) -> None:
        await self._session_flow.show_session_picker(command_message, **kwargs)

    async def on_session_picker_app_session_highlighted(
        self, event: SessionPickerApp.SessionHighlighted
    ) -> None:
        await self._session_flow.on_session_picker_app_session_highlighted(event)

    async def _apply_picker_preview(
        self, session_id: str, history: list[PublicHistoryEntry]
    ) -> None:
        await self._session_flow.apply_picker_preview(session_id, history)

    def _exit_picker_preview(self) -> bool:
        return self._session_flow.exit_picker_preview()

    async def on_session_picker_app_session_selected(
        self, event: SessionPickerApp.SessionSelected
    ) -> None:
        await self._session_flow.on_session_picker_app_session_selected(event)

    async def on_session_picker_app_session_delete_requested(
        self, event: SessionPickerApp.SessionDeleteRequested
    ) -> None:
        await self._session_flow.on_session_picker_app_session_delete_requested(event)

    async def _exit_picker_to_input(self) -> None:
        await self._session_flow.exit_picker_to_input()

    async def on_session_picker_app_cancelled(
        self, event: SessionPickerApp.Cancelled
    ) -> None:
        await self._session_flow.on_session_picker_app_cancelled(event)

    async def _resume_local_session(self, session_id: str) -> None:
        # R2/OD3: a session change invalidates any pending automatic
        # continuation synchronously (the schedule belongs to the old
        # session's episode; the gate is invalidated server-side too).
        self._root_recovery.cancel_local(reason="session_changed")
        await self._session_flow.resume_local_session(session_id)

    async def _reset_presentation_after_resume(self) -> None:
        await self._session_flow.reset_presentation_after_resume()

    async def _finish_resume_notices(self) -> None:
        await self._session_flow.finish_resume_notices()

    async def _auto_resume_on_startup(self) -> None:
        await self._session_flow.auto_resume_on_startup()

    async def _rename_session(self, cmd_args: str = "", **kwargs: Any) -> None:
        await self._session_flow.rename_session(cmd_args, **kwargs)

    async def _branch_session(self, cmd_args: str = "", **kwargs: Any) -> None:
        await self._session_flow.branch_session(cmd_args, **kwargs)

    def _start_rewind_mode(self, **kwargs: Any) -> None:
        # The old body was a one-line alias for action_rewind_prev; the delegate
        # forwards straight to the controller's rewind_prev (PLR0904 keeps the
        # controller at the public-method cap, so no alias layer is kept).
        self._session_flow.rewind_prev()

    def action_rewind_prev(self) -> None:
        self._session_flow.rewind_prev()

    def action_rewind_next(self) -> None:
        self._session_flow.rewind_next()

    async def _apply_config_to_ui(self) -> None:
        await self._apply_theme(self.config.theme)

        if self._banner:
            self._banner.set_state(
                self.app_server.resources.config.current,
                self.app_server.resources.runtime.custom_skills_count,
                mcp=self.app_server.resources.runtime.mcp,
                hooks_count=self.app_server.resources.runtime.hooks_count,
            )
        self._show_config_issues()
        self._show_mcp_discovery_failures()

    async def _reload_config(self, **kwargs: Any) -> str | None:
        from chartreux.app_server import AppServerConnectionClosed

        reload_message = ReloadConfigMessage()
        try:
            await self._mount_and_scroll(reload_message)
            self._reset_ui_state()
            await self._load_more.hide()
            reload_result = await self.app_server.resources.config.reload(
                reload_runtime=True
            )
            self._startup_issue_resolved = True
            await self._apply_config_to_ui()
            reload_message.set_complete()
            if not reload_result.launch_metadata_persisted:
                await self._mount_and_scroll(
                    WarningMessage(
                        "Configuration was applied for this session, but its launch "
                        "settings could not be saved. Resuming this session may use "
                        "the previous configuration."
                    )
                )
            stripped_count = reload_result.stripped_history_images
            if stripped_count > 0:
                model_name = self.config.active_model.display_name
                noun = "image" if stripped_count == 1 else "images"
                await self._mount_and_scroll(
                    WarningMessage(
                        f"{stripped_count} {noun} from earlier turns will be omitted "
                        f"when sending to {model_name} (no vision support)."
                    )
                )
        except (asyncio.CancelledError, AppServerConnectionClosed):
            if reload_message.parent:
                await reload_message.remove()
            raise
        except Exception as e:
            reload_message.set_error(str(e))
            return str(e)
        return None

    async def _config_command(self, **kwargs: Any) -> None:
        """Open the user config file in an external editor, then reload on change."""
        config_path = get_chartreux_home() / "config.toml"
        try:
            config_path.parent.mkdir(parents=True, exist_ok=True)
            if config_path.parent.resolve(strict=True) != config_path.parent:
                raise OSError("Config directory changed")
            directory_fd = os.open(
                config_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            )
            try:
                opened_parent = os.fstat(directory_fd)
                current_parent = os.stat(config_path.parent, follow_symlinks=False)
                if (opened_parent.st_dev, opened_parent.st_ino) != (
                    current_parent.st_dev,
                    current_parent.st_ino,
                ):
                    raise OSError("Config directory changed")
                try:
                    fd = os.open(
                        config_path.name,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=directory_fd,
                    )
                except FileExistsError:
                    pass  # Another writer (or an existing user file) wins.
                else:
                    with os.fdopen(fd, "w", encoding="utf-8") as config_file:
                        config_file.write(
                            import_module(
                                "chartreux.core.config.settings_catalog"
                            ).render_initial_user_config()
                        )
            finally:
                os.close(directory_fd)
            if config_path.is_symlink():
                raise OSError("Config file is a symbolic link")
            before = config_path.read_text("utf-8")
            with self.suspend():
                ExternalEditor.edit_file(config_path, check=True)
            changed = config_path.read_text("utf-8") != before
        except SuspendNotSupported:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Terminal does not support suspending to an editor",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        except subprocess.CalledProcessError as exc:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Editor exited with an error (code {exc.returncode}): "
                    f"`{config_path}`",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        except UnicodeDecodeError:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Config file is not valid UTF-8: `{config_path}`",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        except ValueError:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Invalid editor command: check the quotes in $VISUAL or $EDITOR",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        except OSError:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Could not open or read config file: `{config_path}`",
                    collapsed=self._tools_collapsed,
                )
            )
            return
        if changed:
            await self._reload_config()
        else:
            await self._mount_and_scroll(
                UserCommandMessage(f"Config file unchanged: `{config_path}`")
            )

    async def _reset_message_widgets(self) -> None:
        """Tear down the on-screen conversation widgets and UI state.

        Shared by ``/clear`` and the clear-context-on-plan-accept flow. Does not
        touch the agent loop's message history — callers decide whether the core
        history also needs clearing.
        """
        self._entry_expansion_state.reset(default_collapsed=self._tools_collapsed)
        self._tool_group_expansion_state.reset(default_collapsed=self._tools_collapsed)
        self._reset_ui_state()
        if self._chat_input_container:
            self._chat_input_container.set_custom_border(None)
        if self.event_handler:
            await self.event_handler.finalize_streaming()
        await self._messages_area.remove_children()

    async def _clear_history(self, cmd_args: str = "", **kwargs: Any) -> None:
        # R2/OD3: clearing starts a NEW session — invalidate any pending
        # automatic continuation synchronously (session change).
        self._root_recovery.cancel_local(reason="session_changed")
        await self._request_agent_transcript_close(restore_focus=False)
        old_session_id = self.app_server.session_id
        session_log = self.app_server.resources.runtime.session_log
        resumable = session_log.enabled and session_log.persisted
        prompt = cmd_args.strip()
        try:
            await self.app_server.clear_history()
            await self._queue.clear_server_queue()
            # The gauge otherwise waits for the next `session/statsUpdated`, and
            # the replacement session has nothing to report until its first model
            # call -- so it would keep showing the context the clear discarded.
            self._refresh_context_progress()
            self._refresh_banner()
            await self._reset_message_widgets()

            await self._messages_area.mount(SlashCommandMessage("clear"))
            if resumable and old_session_id is not None:
                short_old = shorten_session_id(old_session_id)
                await self._mount_and_scroll(
                    UserCommandMessage(
                        "New conversation started.\n\n"
                        f"Previous session: `{short_old}`\n"
                        f"To resume it later, run: `chartreux --resume {short_old}`"
                    )
                )
            else:
                await self._mount_and_scroll(
                    UserCommandMessage("New conversation started.")
                )
            self._chat_widget.scroll_home(animate=False)
            self._sync_terminal_title()

        except Exception as e:
            await self._mount_and_scroll(
                ErrorMessage(
                    f"Failed to clear history: {e}", collapsed=self._tools_collapsed
                )
            )
            return

        if prompt:
            await self._handle_user_message(prompt)

    async def _on_context_cleared(self, plan_file_path: Path | None = None) -> None:
        """React to a session handoff emitted during plan accept.

        Core already cleared the agent loop's history, so this only resets the
        on-screen widgets and posts a notice that implementation is starting. The
        approved plan is re-mounted so it stays visible in the discussion.
        """
        await self._reset_message_widgets()
        if plan_file_path is not None:
            await self._mount_and_scroll(PlanFileMessage(file_path=plan_file_path))
        await self._mount_and_scroll(
            UserCommandMessage("Context cleared. Implementing the approved plan...")
        )
        self._chat_widget.scroll_home(animate=False)

    async def _show_log_path(self, **kwargs: Any) -> None:
        session_log = await self.app_server.resources.sessions.read_log()
        if not session_log.enabled:
            await self._mount_and_scroll(
                ErrorMessage(
                    "Session logging is disabled in configuration.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not session_log.persisted:
            await self._mount_and_scroll(
                ErrorMessage(
                    "The current session has not been persisted yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        await self._mount_and_scroll(
            UserCommandMessage(
                "## Current Log Directory\n\n"
                f"`{session_log.path}`\n\n"
                "You can send this directory to share your interaction."
            )
        )

    async def _loop_command(self, cmd_args: str = "", **kwargs: Any) -> None:
        widget = await self._loop_commands.handle_command(cmd_args)
        await self._mount_and_scroll(widget)

    async def _compact_history(self, cmd_args: str = "", **kwargs: Any) -> None:
        if self._agent_job_active():
            await self._mount_and_scroll(
                ErrorMessage(
                    "Cannot compact while agent loop is processing. Please wait.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.app_server.history:
            await self._mount_and_scroll(
                ErrorMessage(
                    "No conversation history to compact yet.",
                    collapsed=self._tools_collapsed,
                )
            )
            return

        if not self.event_handler:
            return

        compact_msg = CompactMessage()
        self.event_handler.current_compact = compact_msg
        await self._mount_and_scroll(compact_msg)

        self._agent_task = asyncio.create_task(
            self._run_compact(compact_msg, cmd_args.strip())
        )

    async def _run_compact(
        self, compact_msg: CompactMessage, extra_instructions: str = ""
    ) -> None:
        try:
            await self.app_server.compact(extra_instructions=extra_instructions)
            await self._queue.clear_server_queue()
            # Same reason as `/clear`: the summary that survives is unmeasured
            # until the next model call, so nothing pushes the gauge down on its
            # own and it would keep showing the context compaction replaced.
            self._refresh_context_progress()
            compact_msg.set_complete()

        except asyncio.CancelledError:
            compact_msg.set_error("Compaction interrupted")
            raise
        except Exception as e:
            compact_msg.set_error(str(e))
        finally:
            self._agent_task = None
            if self.event_handler:
                self.event_handler.current_compact = None

    def _get_session_exit_summary(self) -> SessionExitSummary:
        from chartreux.app_server.session import SessionExitSummary

        if self._mount_first and self._app_server is None:
            return SessionExitSummary(session_id=None, usage=TokenUsage())
        return self.app_server.exit_summary()

    async def _exit_app(self, **kwargs: Any) -> None:
        self._request_intentional_exit("/exit")

    def _exit_consequences(self) -> tuple[bool, str]:
        if self._app_server is None:
            return False, "No session is attached."
        main = self._agent_job_active() or (
            self._bash_task is not None and not self._bash_task.done()
        )
        agents = any(agent_is_active(agent) for agent in self._agent_summaries)
        decisions = bool(
            self._active_callback
            or self._pending_callbacks
            or self._pending_local_question is not None
        )
        queued = self._queue.has_removable or self._queue.has_server_work
        return main or agents or decisions or queued, (
            f"Main work: {'shutdown requests its stop' if main else 'none active'}; "
            f"background agents: {'shutdown requests their stop' if agents else 'none running'}; "
            f"pending decisions: {'left unanswered' if decisions else 'none'}; "
            f"queued input: {'not run after exit' if queued else 'none'}. "
            "Completed effects are not rolled back."
        )

    def _request_intentional_exit(self, key: QuitConfirmKey) -> None:
        consequential, consequences = self._exit_consequences()
        if consequential and self.is_running:
            if not any(
                isinstance(screen, _get_exit_consequences_screen_class())
                for screen in self.screen_stack
            ):
                self.push_screen(
                    _get_exit_consequences_screen_class()(consequences),
                    callback=lambda confirmed: (
                        self._force_quit() if confirmed else None
                    ),
                )
            return
        if self._quit_manager.is_confirmed(key):
            self._quit_manager.cancel_confirmation()
            self._force_quit()
            return
        if consequential or (
            key != "/exit"
            and self._app_server is not None
            and self.config.ask_confirmation_on_exit
        ):
            self._quit_manager.request_confirmation(key, consequences)
            return
        self._force_quit()

    async def _switch_from_input(self, widget: Widget, scroll: bool = False) -> None:
        await self._bottom_panel.switch_from_input(widget, scroll)

    async def _replace_bottom_app(self, widget: Widget, scroll: bool = False) -> None:
        await self._bottom_panel.replace_bottom_app(widget, scroll)

    async def _is_active_model_enforced(self) -> bool:
        response = await self.app_server.resources.config.read_fields()
        field = next((f for f in response.fields if f.name == "active_model"), None)
        return field is not None and field.origin == "admin"

    async def _switch_to_model_picker_app(self) -> None:
        await self._bottom_panel.switch_to_model_picker_app()

    async def _switch_to_thinking_picker_app(self) -> None:
        await self._bottom_panel.switch_to_thinking_picker_app()

    async def _switch_to_log_level_picker_app(self) -> None:
        await self._bottom_panel.switch_to_log_level_picker_app()

    async def _switch_to_theme_picker_app(self) -> None:
        await self._bottom_panel.switch_to_theme_picker_app()

    async def _apply_theme(self, theme: str) -> None:
        resolved_theme = resolve_theme(resolve_theme_name(theme))
        textual_theme = _TEXTUAL_THEME_MAP[resolved_theme]
        if textual_theme is None or textual_theme == self.theme:
            return

        previous_diff_theme = (self.native_ansi_color, self.current_theme.dark)
        self.theme = textual_theme
        current_diff_theme = (self.native_ansi_color, self.current_theme.dark)
        if current_diff_theme != previous_diff_theme:
            self._restyle_diff_widgets(
                ansi=current_diff_theme[0], dark=current_diff_theme[1]
            )

    async def _wait_for_settings(self) -> None:
        from chartreux.ui.settings_service import SettingsService
        from chartreux.ui.shell_permissions import ShellPermissionsService

        service = SettingsService(
            self.app_server.resources.config, apply_ui=self._apply_config_to_ui
        )
        shell_permissions = ShellPermissionsService(self.app_server.resources.config)
        try:
            snapshot = await service.read()
            screen = _get_settings_screen_class()(
                service, snapshot, shell_permissions=shell_permissions
            )
            screen.return_state = self._settings_return_state
            self._settings_return_state = None
            command_name = await self.push_screen_wait(screen)
            if command_name:
                self._settings_return_state = screen.return_state
        except Exception as exc:
            self._settings_worker = None
            await self._mount_and_scroll(
                ErrorMessage(f"Failed to open settings: {exc}")
            )
            return
        self._settings_worker = None
        if command_name:
            command = self.commands.parse_command(command_name)
            if command is not None:
                name, resolved, args = command
                await self._bottom_panel.record_settings_return(name, resolved, args)
        else:
            await self._present_pending_callback()

    async def _switch_to_proxy_setup_app(self) -> None:
        await self._bottom_panel.switch_to_proxy_setup_app()

    async def _switch_to_question_app(self, args: UserQuestionRequest) -> None:
        await self._bottom_panel.switch_to_question_app(args)

    async def _switch_to_input_app(self, *, return_to_settings: bool = True) -> None:
        await self._bottom_panel.switch_to_input_app(
            return_to_settings=return_to_settings
        )

    def _focus_current_bottom_app(self) -> None:
        self._bottom_panel.focus_current_bottom_app()

    def _try_interrupt_bottom_app_escape(self) -> bool:
        return self._bottom_panel.try_interrupt_bottom_app_escape()

    def _try_interrupt(self) -> bool:
        return self._bottom_panel.try_interrupt()

    def _handle_input_double_escape(self) -> None:
        self._bottom_panel.handle_input_double_escape()

    # --- Rewind mode ---

    def _handle_rewind_app_escape(self) -> None:
        try:
            rewind_app = self.query_one(_get_rewind_app_class())
        except Exception:
            rewind_app = None
        if rewind_app is not None and rewind_app.go_back():
            return
        self.action_rewind_prev()

    async def on_rewind_app_rewind_confirmed(
        self, message: RewindApp.RewindConfirmed
    ) -> None:
        await self._session_flow.execute_rewind(
            restore_files=message.restore_files, inplace=message.inplace
        )

    def on_rewind_app_edit_prev(self, message: RewindApp.EditPrev) -> None:
        self._session_flow.rewind_prev()

    def on_rewind_app_edit_next(self, message: RewindApp.EditNext) -> None:
        self._session_flow.rewind_next()

    async def on_rewind_app_quit(self, message: RewindApp.Quit) -> None:
        await self._session_flow.exit_rewind_mode()

    # --- End rewind mode ---

    def _clear_input(self) -> None:
        try:
            input_widget = self.query_one(ChatInputContainer)
            input_widget.value = ""
        except Exception:
            pass

    def _handle_agent_running_escape(self) -> None:
        # R2/OD3: an interrupt is user input — invalidate any pending
        # automatic continuation before the interrupt settles.
        self._invalidate_user_input()
        self._turn_supervisor.handle_agent_running_escape()

    def _begin_interrupt_settle(self) -> None:
        self._turn_supervisor.begin_interrupt_settle()

    def _settle_interrupt(self) -> None:
        self._turn_supervisor.settle_interrupt()

    def _maybe_settle_interrupt(self) -> None:
        self._turn_supervisor.maybe_settle_interrupt()

    async def _await_interrupt_settled(self) -> None:
        await self._turn_supervisor.await_interrupt_settled()

    def _try_interrupt_no_job_steps(self) -> bool:
        if self._chat_input_container:
            dismissed = self._chat_input_container.dismiss_completion()
            if dismissed:
                self._last_escape_time = None
                return True

            if self._chat_input_container.value.startswith("/"):
                self._chat_input_container.value = ""
                self._last_escape_time = None
                return True

        if self._try_interrupt_bottom_app_escape():
            return True

        return False

    def _try_interrupt_running_job(self) -> bool:
        interrupted = False
        if self._bash_task and not self._bash_task.done():
            self._bash_task.cancel()
            interrupted = True
        if self._agent_job_active():
            self._handle_agent_running_escape()
            interrupted = True
        return interrupted

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Let Escape fall through while the input queue selection is active."""
        if action != "interrupt":
            return True
        if (
            self.screen.id
            in {
                "settings-screen",
                "usage-screen",
                "websearch-screen",
                "status-line-settings-screen",
                "shell-permissions-screen",
            }
            or self.screen.__class__.__name__ == "ProviderWorkbenchScreen"
        ):
            return False
        if (
            isinstance(self.screen, _get_exit_consequences_screen_class())
            or self._agent_transcript_viewer is not None
            or (
                self._debug_console is not None and self._debug_console.owns_interaction
            )
        ):
            # Route priority Escape to the active overlay rather than the
            # parent interrupt action (even while queue selection is active).
            return True
        containers = self.query(ChatInputContainer)
        if not containers:
            return True
        body = containers[0]._body
        if body is not None and body.in_queue_mode:
            return False
        return True

    def action_interrupt(self) -> None:
        if self.is_running and isinstance(
            self.screen, _get_exit_consequences_screen_class()
        ):
            self.screen.action_cancel()
            return
        if viewer := self._agent_transcript_viewer:
            viewer.action_close()
            return
        # Only the focused console owns Escape; a visible dock is not an overlay.
        # Keep modal/transcript owners above it and agent browsing below it.
        if self._debug_console is not None and self._debug_console.owns_interaction:
            self._last_escape_time = None
            self._debug_console.action_close()
            return
        if self._agent_bar is not None and self._agent_bar.expanded:
            self._agent_bar.action_collapse()
            return
        if self._app_server is None:
            return
        self._try_interrupt()

    async def on_history_load_more_requested(self, _: HistoryLoadMoreRequested) -> None:
        await self._transcript_presenter.on_load_more_requested(_)

    async def action_toggle_tool(self) -> None:
        await self._transcript_presenter.toggle_tool()

    def _usage_sync_binding(self) -> tuple[AppServerSession, str] | None:
        """Current usage subscription binding, or None while sync must not run."""
        if (
            self._shutdown_started
            or self._app_server is None
            or not self._session_ready.is_set()
            or not self._resume_ui_ready.is_set()
        ):
            return None
        session = self.app_server
        return (session, session.session_id)

    def _is_usage_binding_current(self, binding: tuple[AppServerSession, str]) -> bool:
        """Whether a subscription binding still matches the live session."""
        session = binding[0]
        return (
            self._app_server is session
            and session.session_id == binding[1]
            and not self._shutdown_started
        )

    @property
    def _usage_poll_task(self) -> asyncio.Task[None] | None:
        return self._usage_controller.poll_task

    @property
    def _usage_tasks(self) -> set[asyncio.Task[None]]:
        return self._usage_controller.tasks

    def _sync_usage_subscription(self) -> None:
        self._usage_controller.sync()

    def _stop_usage_subscription(self) -> None:
        self._usage_controller.stop()

    def _usage_screen_providers(
        self,
    ) -> tuple[
        Callable[[UsageWindow, str | None], Awaitable[UsageReadResponse]],
        Callable[[Callable[[UsageUpdatedParams], None]], Callable[[], None]],
    ]:
        """Resource-backed constructor arguments for UsageScreen, without opening it."""
        usage = self.app_server.resources.usage
        return usage.read, usage.subscribe

    def _on_usage_updated(self, snapshot: UsageUpdatedParams) -> None:
        self._usage_summaries = snapshot.summaries
        if self._status_line is not None:
            self._status_line.set_state(
                replace(
                    self._status_line.state,
                    usage_day=snapshot.summaries.day,
                    usage_week=snapshot.summaries.week,
                    usage_month=snapshot.summaries.month,
                )
            )

    def _refresh_context_progress(self) -> None:
        self._sync_usage_subscription()
        if self._status_line is None or self._app_server is None:
            return
        runtime = self.app_server.resources.runtime
        identity = (self.app_server.session_id, self.app_server.cwd)
        if identity != self._branch_identity:
            self._root_context_unknown = False
            self._root_compacting = any(
                isinstance(entry, PublicCheckpointEntry)
                and entry.kind == "compaction"
                and entry.generation_status is PublicEntryGenerationStatus.IN_PROGRESS
                for entry in self.app_server.history
            )
        self._status_line.set_config(self.config.status_line)
        self._status_line.set_state(
            replace(
                self._status_line.state,
                cwd=self.app_server.cwd,
                model_identity=self.config.active_model.display_name,
                context_tokens=(
                    None if self._root_context_unknown else runtime.stats.context_tokens
                ),
                auto_compact_threshold=runtime.context_window,
                compacting=self._root_compacting,
                ascii_chrome=self.config.ascii_chrome,
                active_background_job_count=self.app_server.state.active_background_job_count,
                usage_day=self._usage_summaries.day if self._usage_summaries else None,
                usage_week=self._usage_summaries.week
                if self._usage_summaries
                else None,
                usage_month=self._usage_summaries.month
                if self._usage_summaries
                else None,
            )
        )
        self._refresh_main_agent_details()
        identity = (self.app_server.session_id, self.app_server.cwd)
        if identity != self._branch_identity:
            self._branch_identity = identity
            self._status_line.set_state(
                replace(self._status_line.state, branch=None, branch_status="unknown")
            )
            self._schedule_branch_refresh()

    def _refresh_main_agent_details(self) -> None:
        if self._agent_bar is None or self._status_line is None:
            return
        state = self._status_line.state
        self._agent_bar.update_main_details(
            {
                "Session": self.app_server.session_id,
                "Directory": self.app_server.cwd,
                "Model": state.model_identity or "—",
                "State": "Running" if self._agent_job_active() else "Idle",
            },
            context_tokens=state.context_tokens,
            auto_compact_threshold=state.auto_compact_threshold,
            compacting=state.compacting,
        )

    def _schedule_branch_refresh(self) -> None:
        if self._app_server is None:
            return
        self._branch_generation += 1
        generation = self._branch_generation
        session = self.app_server
        identity = (session.session_id, session.cwd)
        self.run_worker(
            self._refresh_branch(session, identity, generation), exclusive=False
        )

    async def _refresh_branch(
        self, session: AppServerSession, identity: tuple[str, str], generation: int
    ) -> None:
        try:
            branch = await session.resources.workspace.read_branch(refresh=True)
        except Exception as exc:
            logger.debug("Branch refresh failed", exc_info=exc)
            return
        if (
            branch is None
            or generation != self._branch_generation
            or self._app_server is not session
            or identity != (session.session_id, session.cwd)
            or self._status_line is None
        ):
            return
        self._status_line.set_state(
            replace(
                self._status_line.state,
                branch=branch.branch,
                branch_status=branch.status,
            )
        )

    def _refresh_banner(self) -> None:
        if self._banner:
            self._banner.set_state(
                self.config,
                self.app_server.resources.runtime.custom_skills_count,
                mcp=self.app_server.resources.runtime.mcp,
                hooks_count=self.app_server.resources.runtime.hooks_count,
            )

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        self._agent_browser.on_descendant_focus(event)

    async def on_agent_bar_closed(self, message: AgentBar.Closed) -> None:
        await self._agent_browser.on_agent_bar_closed(message)

    async def action_toggle_agent_browser(self, **kwargs: Any) -> None:
        await self._agent_browser.toggle_agent_browser(**kwargs)

    def on_agent_bar_stop_requested(self, message: AgentBar.StopRequested) -> None:
        self._agent_browser.on_agent_bar_stop_requested(message)

    def on_agent_bar_release_requested(
        self, message: AgentBar.ReleaseRequested
    ) -> None:
        self._agent_browser.on_agent_bar_release_requested(message)

    def on_agent_bar_selection_requested(
        self, message: AgentBar.SelectionRequested
    ) -> None:
        self._agent_browser.on_agent_bar_selection_requested(message)

    def _agent_is_active(self, agent: AgentSummaryModel) -> bool:
        """Late-bound read of the agent_is_active seam.

        Tests patch ``chartreux.cli.textual_ui.app.agent_is_active``; reading
        the module global at call time keeps that patch effective.
        """
        return agent_is_active(agent)

    def _submit_agent_selection(self, agent_id: str | None) -> None:
        self._agent_browser.submit_agent_selection(agent_id)

    async def _request_agent_transcript_close(
        self, *, restore_focus: bool = True
    ) -> None:
        await self._agent_browser.request_agent_transcript_close(
            restore_focus=restore_focus
        )

    async def on_agent_transcript_viewer_closed(
        self, message: AgentTranscriptViewer.Closed
    ) -> None:
        await self._agent_browser.on_agent_transcript_viewer_closed(message)

    async def _close_agent_transcript_viewer(
        self, *, restore_focus: bool, show_chat: bool = True
    ) -> None:
        await self._agent_browser.close_agent_transcript_viewer(
            restore_focus=restore_focus, show_chat=show_chat
        )

    async def on_debug_console_closed(self, _message: DebugConsole.Closed) -> None:
        await self._close_debug_console()

    async def _close_debug_console(self) -> None:
        console = self._debug_console
        if console is None:
            return
        self._debug_console = None
        await console.remove()
        target = self._debug_console_focus_target
        self._debug_console_focus_target = None
        if (
            target is not None
            and target.is_mounted
            and target.display
            and target.can_focus
        ):
            self.call_after_refresh(target.focus)
        else:
            self.call_after_refresh(self._focus_current_bottom_app)

    async def action_toggle_debug_console(self, **kwargs: Any) -> None:
        if self._app_server is None:
            return
        if self._debug_console is not None:
            await self._close_debug_console()
        else:
            from chartreux.cli.textual_ui.widgets.debug_console import DebugConsole

            self._debug_console_focus_target = self.screen.focused
            self._debug_console = DebugConsole(
                log_source=self.app_server.resources.runtime
            )
            await self.mount(self._debug_console)
            if self._debug_console.has_class("-fullscreen"):
                self.call_after_refresh(self._debug_console._focus_fullscreen_log)

    def _get_chat_input(self) -> ChatInputContainer | None:
        input_widgets = self.query(ChatInputContainer)
        if input_widgets:
            return input_widgets.first()
        return None

    def action_interrupt_or_quit(self) -> None:  # noqa: PLR0911
        if self.is_running and isinstance(
            self.screen, _get_exit_consequences_screen_class()
        ):
            return
        # Ctrl+C priority ladder: clear input → second-press quit → bottom-app/etc
        # no-op steps → pop last queued item (LIFO) → cancel running job → request quit.
        if self._app_server is None:
            self._request_intentional_exit("Ctrl+C")
            return
        if (container := self._get_chat_input()) and container.value:
            container.value = ""
            return
        if self._quit_manager.is_confirmed("Ctrl+C"):
            self._request_intentional_exit("Ctrl+C")
            return
        if self._try_interrupt_no_job_steps():
            return
        if self._queue.has_removable:
            self.run_worker(self._queue.pop_last(), exclusive=False)
            return
        if self._try_interrupt_running_job():
            return
        self._request_intentional_exit("Ctrl+C")

    def action_delete_right_or_quit(self) -> None:
        if self.is_running and isinstance(
            self.screen, _get_exit_consequences_screen_class()
        ):
            return
        if self._app_server is None:
            self._request_intentional_exit("Ctrl+D")
            return
        if (container := self._get_chat_input()) and container.value:
            if container.input_widget:
                container.input_widget.action_delete_right()
            return

        self._request_intentional_exit("Ctrl+D")

    async def _begin_shutdown(self) -> None:
        self._shutdown_started = True
        self._stop_usage_subscription()
        if self._usage_tasks:
            await asyncio.gather(*self._usage_tasks, return_exceptions=True)
        await self._request_agent_transcript_close(restore_focus=False)
        if self._app_server is not None:
            # Signal shutdown before waiting for or cancelling UI work, so a
            # pending attach or runtime refresh treats a dropped connection as
            # expected teardown.
            self._app_server.begin_close()
        await self._side_channel.shutdown()

    def _force_quit(self) -> None:
        if self._force_quit_task is not None and not self._force_quit_task.done():
            return
        self._force_quit_task = asyncio.create_task(self._force_quit_async())

    async def _force_quit_async(self) -> None:
        try:
            await self._begin_shutdown()
            if self._agent_task and not self._agent_task.done():
                self._agent_task.cancel()
            if self._bash_task and not self._bash_task.done():
                self._bash_task.cancel()
        finally:
            self.exit(result=self._get_session_exit_summary())

    async def shutdown_cleanup(self) -> None:
        with suppress(Exception):
            await self._begin_shutdown()
        tasks = tuple(
            task
            for task in (self._agent_task, self._bash_task, self._interrupt_operation)
            if task is not None
        )
        for task in tasks:
            if not task.done():
                task.cancel()
        for task in tasks:
            if task.done():
                continue
            with suppress(asyncio.CancelledError, Exception):
                await task
        if self._app_server is not None:
            with suppress(Exception):
                await self._app_server.close()

    def action_scroll_chat_up(self) -> None:
        try:
            self._request_transcript_reconcile(scroll_driven=True)
            self._chat_widget.scroll_relative(y=-5, animate=False)
        except Exception:
            pass

    def action_scroll_chat_down(self) -> None:
        try:
            self._request_transcript_reconcile(scroll_driven=True)
            self._chat_widget.scroll_relative(y=5, animate=False)
        except Exception:
            pass

    async def _show_dangerous_directory_warning(self) -> None:
        is_dangerous, reason = is_dangerous_directory()
        if is_dangerous:
            warning = (
                f"⚠ WARNING: {reason}\n\nRunning in this location is not recommended."
            )
            await self._mount_and_scroll(WarningMessage(warning, show_border=False))

    async def _show_untrusted_config_warning(self) -> None:
        from chartreux.utils.cache_store import FileSystemCacheStore

        try:
            response = await self.app_server.resources.workspace.untrusted_config_dirs()
            if not response.dirs:
                return
            cache = FileSystemCacheStore()
            section = await asyncio.to_thread(
                cache.read_section, _UNTRUSTED_CONFIG_WARNING_SECTION
            )
            acknowledged = section.get("dirs")
            acknowledged = (
                set(acknowledged) if isinstance(acknowledged, list) else set()
            )
            # A folder can be intentionally left untrusted; warn once per folder
            # instead of nagging on every launch, but still surface new ones.
            if set(response.dirs) <= acknowledged:
                return
            folders = "\n".join(f"  • {d}" for d in response.dirs)
            warning = (
                "⚠ Untrusted local config folders are being ignored:\n\n"
                f"{folders}\n\n"
                f'If you want them loaded, remove them from "untrusted" in '
                f"{response.settings_path}, or ask Chartreux to do it."
            )
            await self._mount_and_scroll(WarningMessage(warning, show_border=False))
            await asyncio.to_thread(
                cache.write_section,
                _UNTRUSTED_CONFIG_WARNING_SECTION,
                {"dirs": sorted(acknowledged | set(response.dirs))},
            )
        except Exception:
            logger.warning(
                "Failed to check for untrusted config folders", exc_info=True
            )

    async def _show_custom_tools_deprecation_warning_after_initial_history(
        self,
    ) -> None:
        await self._initial_history_loaded.wait()
        await self._show_custom_tools_deprecation_warning()

    async def _show_custom_tools_deprecation_warning(self) -> None:
        async with self._custom_tools_deprecation_message_lock:
            if self._custom_tools_deprecation_message is not None:
                return
            tool_names = [
                tool.name
                for tool in self.app_server.resources.runtime.tools
                if tool.is_custom
            ]
            if not tool_names:
                return
            message = CustomToolsDeprecationMessage(tool_names)
            await self._mount_and_scroll(message)
            self._custom_tools_deprecation_message = message

    async def _mount_and_scroll(
        self,
        widget: Widget,
        after: Widget | None = None,
        before: Widget | None = None,
        *,
        container: Widget | None = None,
    ) -> None:
        messages_area = self._messages_area
        is_user_initiated = isinstance(widget, (UserMessage, UserCommandMessage))
        should_anchor = is_user_initiated or self._chat_widget.is_at_bottom

        pin_anchor: Widget | None = None
        if after is None and container is None:
            pin_anchor = self._queue.pin_target(messages_area)

        before_parent = before.parent if before is not None else None
        after_parent = after.parent if after is not None else None
        with self.batch_update():
            if isinstance(before_parent, Widget):
                await before_parent.mount(widget, before=before)
            elif isinstance(after_parent, Widget):
                await after_parent.mount(widget, after=after)
            elif container is not None:
                await container.mount(widget)
            elif pin_anchor is not None:
                await messages_area.mount(widget, before=pin_anchor)
            else:
                await messages_area.mount(widget)
            if not widget.is_mounted:
                return
            if isinstance(widget, StreamingMessageBase):
                await widget.write_initial_content()

        self._request_transcript_reconcile()
        if should_anchor:
            self._chat_widget.anchor()

    async def _refresh_windowing_from_history(self) -> None:
        await self._transcript_presenter.refresh_windowing_from_history()

    def _clipboard_notice_message(self, copy_result: ClipboardCopyResult) -> str:
        if copy_result.verified:
            return "Copied to clipboard"
        return f"Copied · {NATIVE_COPY_HINT}"

    def on_chat_input_body_inline_notice_requested(
        self, event: ChatInputBody.InlineNoticeRequested
    ) -> None:
        self._inline_notice.show(event.message, timeout=event.timeout)

    def on_chat_input_body_inline_notice_cleared(
        self, event: ChatInputBody.InlineNoticeCleared
    ) -> None:
        self._inline_notice.hide()

    def action_copy_selection(self) -> None:
        copy_result = copy_selection_to_clipboard(self, show_toast=False)
        if copy_result is None:
            return
        self._inline_notice.show(self._clipboard_notice_message(copy_result))

    def on_mouse_up(self, event: MouseUp) -> None:
        if self._app_server is None or not self.config.autocopy_to_clipboard:
            return
        copy_result = copy_selection_to_clipboard(self, show_toast=False)
        if copy_result is None:
            return
        self._inline_notice.show(self._clipboard_notice_message(copy_result))

    def on_app_blur(self, event: AppBlur) -> None:
        self._terminal_notifier.on_blur()
        if self._chat_input_container and self._chat_input_container.input_widget:
            self._chat_input_container.input_widget.set_app_focus(False)

    def on_app_focus(self, event: AppFocus) -> None:
        self._terminal_notifier.on_focus()
        if self._chat_input_container and self._chat_input_container.input_widget:
            self._chat_input_container.input_widget.set_app_focus(True)

    def action_open_plan_in_editor(self) -> None:
        if self.event_handler is None:
            return

        if plan_file_message := self.event_handler.plan_file_message:
            plan_file_message.open_in_editor()

    def action_suspend_with_message(self) -> None:
        if self._driver is None or not self._driver.can_suspend:
            return
        with self.suspend():
            rprint(
                "Chartreux has been suspended. Run [bold cyan]fg[/bold cyan] to bring Chartreux back."
            )
            os.kill(os.getpid(), signal.SIGTSTP)

    def _on_driver_signal_resume(self, event: Driver.SignalResume) -> None:
        # Textual doesn't repaint after resuming from Ctrl+Z (SIGTSTP);
        # force a full layout refresh so the UI isn't garbled.
        self.refresh(layout=True)

    def _handle_exception(self, error: Exception) -> None:
        if not isinstance(error, WorkerFailed):
            logger.error("Textual application failed", exc_info=error)
        return super()._handle_exception(error)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        error = event.worker.error
        if event.state == WorkerState.ERROR and error:
            logger.error("Textual worker %s failed", event.worker.name, exc_info=error)


async def _run_app_with_cleanup(app: ChartreuxApp) -> SessionExitSummary | None:
    from chartreux.cli.stderr_guard import stderr_guard

    loop = asyncio.get_running_loop()
    try:

        def _sigterm_handler() -> None:
            loop.remove_signal_handler(signal.SIGTERM)
            app._force_quit()

        loop.add_signal_handler(signal.SIGTERM, _sigterm_handler)
    except (NotImplementedError, OSError):
        pass

    try:
        with stderr_guard():
            return await app.run_async()
    finally:
        try:
            loop.remove_signal_handler(signal.SIGTERM)
        except (NotImplementedError, OSError):
            pass
        sys.stderr.write("Closing\u2026\r")
        sys.stderr.flush()
        try:
            await app.shutdown_cleanup()
        finally:
            sys.stderr.write("\033[2K\r")
            sys.stderr.flush()


def run_textual_ui(
    start_app_server: AppServerBootstrap,
    history_file: Path,
    startup: StartupOptions | None = None,
    *,
    close_app_server: Callable[[], Awaitable[None]] | None = None,
) -> SessionExitSummary | None:
    resolve_auto_theme()

    async def run_owned() -> SessionExitSummary | None:
        from chartreux.app_server.host import AppServerHost

        app_server = await start_app_server()
        effective_startup = startup or StartupOptions()
        if isinstance(app_server, AppServerHost):
            from chartreux.cli.textual_ui.startup import (
                _execute_session_open_plan,
                resolve_session_open_plan,
            )

            host = app_server
            plan = await resolve_session_open_plan(
                host,
                prompt_for_workspace_trust=effective_startup.prompt_for_workspace_trust,
                show_resume_picker=effective_startup.show_resume_picker,
                initially_resuming=effective_startup.is_resuming_session,
                resume_session_id=effective_startup.resume_session_id,
            )
            if plan is None:
                return None
            effective_startup = replace(
                effective_startup,
                show_resume_picker=effective_startup.show_resume_picker,
                is_resuming_session=False,
                prompt_for_workspace_trust=False,
                startup_show_resume_picker=plan.showed_resume_picker,
                startup_prompt_for_workspace_trust=plan.showed_trust_prompt,
                resume_session_id=plan.resume_session_id,
                continue_latest=plan.continue_latest,
            )

            async def _session_starter() -> AppServerSession:
                try:
                    return await _execute_session_open_plan(host, plan)
                except BaseException:
                    with suppress(BaseException):
                        await host.close()
                    raise

            initial_config_response = await host.read_config()
            app = ChartreuxApp(
                app_server=_session_starter,
                history_file=history_file,
                startup=effective_startup,
            )
            app._initial_config_response = initial_config_response
            app._mount_first = True
        else:
            app = ChartreuxApp(
                app_server=app_server,
                history_file=history_file,
                startup=effective_startup,
            )
        return await _run_app_with_cleanup(app)

    async def run() -> SessionExitSummary | None:
        try:
            return await run_owned()
        finally:
            if close_app_server is not None:
                await close_app_server()

    return asyncio.run(run())
