#!/usr/bin/env python3
"""Capture real, offline Textual review fixtures and their provenance.

Run from the repository root with ``uv run python docs/reviews/tui-affordances/capture_review.py``.
The application, tests, and baseline snapshots are imported without modification.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Protocol, cast
from unittest.mock import patch

# Rich/Textual inspect NO_COLOR during import. Select the terminal mode first.
if "--no-color" in sys.argv:
    os.environ["NO_COLOR"] = "1"
else:
    os.environ.pop("NO_COLOR", None)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from extra_fixtures import extra_fixtures
from provider_fixtures import provider_fixtures
from rich.console import Console
from textual.app import App
from textual.pilot import Pilot
from textual.widgets import Button, OptionList

from chartreux.app_server.protocol import MCPAuthUrlParams
from chartreux.cli.textual_ui.screens.settings import SettingsScreen
from chartreux.cli.textual_ui.widgets.debug_console import _LogView
from chartreux.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from scripts.capture_tui import (
    APP_CSS_PATH,
    ModelHost,
    PickerHost,
    SessionHost,
    SettingsHost,
    _provider_captures,
    _submit_chat_prompt,
)
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.test_ui_snapshot_basic_conversation import (
    SnapshotTestAppWithConversation,
)
from tests.snapshots.test_ui_snapshot_parallel_tool_calls import ParallelToolCallsApp

Prepare = Callable[[Pilot], Awaitable[None]]


class _ReviewConfig(Protocol):
    theme: str
    ascii_chrome: bool


class _ReviewApp(Protocol):
    ascii_chrome: bool
    config: Any


class _SegmentFrame(Protocol):
    def render_segments(self, console: Console) -> str: ...


@dataclass(frozen=True)
class Fixture:
    name: str
    family: str
    route: str
    factory: Callable[[], App]
    prepare: Prepare | None = None
    actions: tuple[str, ...] = ()


def _factory(module: str, name: str) -> Callable[[], App]:
    return getattr(importlib.import_module(f"tests.snapshots.{module}"), name)


def keys(*sequence: str) -> Prepare:
    async def prepare(pilot: Pilot) -> None:
        await pilot.press(*sequence)
        await pilot.pause(0.15)

    return prepare


async def _mcp(pilot: Pilot) -> None:
    await pilot.pause(0.1)
    await pilot.press(*"/mcp", "enter")
    await pilot.pause(0.2)
    pilot.app.set_focus(None)


async def _popup(pilot: Pilot) -> None:
    from tests.snapshots.test_ui_snapshot_completion_popup import (
        LONG_SLASH_COMMAND_SUGGESTIONS,
        _show,
    )

    await _show(pilot, LONG_SLASH_COMMAND_SUGGESTIONS)


async def _full_path_click(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.chat_input import (
        ChatInputContainer,
        ChatTextArea,
    )
    from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import (
        CompletionPopup,
        _CompletionRow,
    )

    Path("alpha.txt").write_text("a")
    Path("alpine.txt").write_text("b")
    await pilot.press(*"look @al")
    popup = pilot.app.query_one(CompletionPopup)
    deadline = asyncio.get_running_loop().time() + 2
    expected_rows = 2
    while len(popup.query(_CompletionRow)) < expected_rows:
        assert asyncio.get_running_loop().time() < deadline
        await pilot.pause(0.01)
    expected = popup._suggestions[1].label
    row = list(popup.query(_CompletionRow))[1]
    await pilot.click(row, offset=(5, 0))
    await pilot.pause(0.1)
    assert pilot.app.query_one(ChatInputContainer).value == f"look {expected} "
    assert not popup._suggestions
    assert pilot.app.focused is pilot.app.query_one(ChatTextArea)
    assert not cast(BaseSnapshotTestApp, pilot.app)._agent_job_active()


async def _settings_enum(pilot: Pilot) -> None:
    from chartreux.app_server.protocol import SettingDescriptorWire, SettingLeafWire

    screen = cast(SettingsScreen, pilot.app.screen)
    screen.catalog.append(
        SettingDescriptorWire(
            path="ui_color_scheme",
            label="Theme",
            description="Choose a theme.",
            kind="enum",
            group="Interface",
            choices=("light", "dark"),
        )
    )
    screen.fields["ui_color_scheme"] = SettingLeafWire(
        path="ui_color_scheme",
        effective_value="dark",
        origin="default",
        saved_explicit=False,
    )
    screen._refresh_options()
    await pilot.press(*"ui_color_scheme", "enter")
    await pilot.pause(0.1)


async def _settings_checklist(pilot: Pilot) -> None:
    from chartreux.app_server.protocol import InventoryItemStateWire

    screen = cast(SettingsScreen, pilot.app.screen)
    screen.snapshot.inventories["tools"] = ["bash", "read_file", "write_file"]
    screen.snapshot.inventory_states["tools"] = {
        name: InventoryItemStateWire(
            effective=name == "bash", default_effective=True, pattern_driven=False
        )
        for name in ("bash", "read_file", "write_file")
    }
    screen._refresh_options()
    await pilot.press(*"inventory_tools", "enter")
    await pilot.pause(0.1)


async def _agent(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
    from chartreux.cli.textual_ui.widgets.messages import UserMessage

    viewer = pilot.app.query_one(AgentTranscriptViewer)
    viewer.action_refresh()
    for _ in range(25):
        await pilot.pause(0.1)
        if len(viewer.query(UserMessage)) == 1:
            return
    raise RuntimeError("agent transcript fixture did not populate")


async def _proxy_error(pilot: Pilot) -> None:
    await pilot.pause(0.2)
    await pilot.press(*"http://proxy.example.com:8080", "tab")
    await pilot.press(*"invalid-proxy:8443")
    pilot.app.query_one("#proxysetup-save", Button).press()
    await pilot.pause(0.25)


async def _tools(pilot: Pilot) -> None:
    app = cast(ParallelToolCallsApp, pilot.app)
    await app.emit_all_tool_calls()
    await pilot.pause(0.3)
    app.freeze_spinners()


async def _debug(pilot: Pilot) -> None:
    await pilot.pause(0.1)
    await pilot.press("ctrl+backslash")
    await pilot.pause(0.4)


async def _debug_selected(pilot: Pilot) -> None:
    await _debug(pilot)
    log = pilot.app.query_one("#debug-console-log", _LogView)
    assert log._lines, "debug fixture has no selectable log rows"
    await pilot.click(log, offset=(5, 0))
    await pilot.pause(0.1)
    assert log._selected_line is not None


async def _debug_copied(pilot: Pilot) -> None:
    await _debug_selected(pilot)
    with patch.object(pilot.app, "copy_to_clipboard") as copy:
        await pilot.press("c")
        await pilot.pause(0.1)
        copy.assert_called_once()


async def _reasoning(pilot: Pilot) -> None:
    await pilot.press(*"What is the answer?", "enter")
    await pilot.pause(0.55)


async def _reasoning_expanded(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.messages import ReasoningMessage
    from chartreux.cli.textual_ui.widgets.tools import ToolGroupHeader

    await _reasoning(pilot)
    groups = list(pilot.app.query(ToolGroupHeader))
    if groups:
        await pilot.click(groups[0])
        await pilot.pause(0.1)
    message = pilot.app.query_one(ReasoningMessage)
    assert message._header_widget is not None
    await pilot.click(message._header_widget)
    await pilot.pause(0.1)
    assert not message.collapsed, "reasoning disclosure did not expand"


async def _reasoning_body_collapse(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.messages import ReasoningMessage
    from chartreux.cli.textual_ui.widgets.tools import ToolGroupHeader

    await _reasoning(pilot)
    groups = list(pilot.app.query(ToolGroupHeader))
    if groups:
        await pilot.click(groups[0])
        await pilot.pause(0.1)
    message = pilot.app.query_one(ReasoningMessage)
    assert message._header_widget is not None
    await pilot.click(message._header_widget)
    await pilot.pause(0.1)
    assert message._markdown is not None and message._markdown.display
    await pilot.click(message._markdown, offset=(5, 0))
    await pilot.pause(0.1)
    assert message.collapsed, "body click did not collapse reasoning"


async def _reasoning_body_after(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.messages import ReasoningMessage

    await _reasoning_expanded(pilot)
    message = pilot.app.query_one(ReasoningMessage)
    assert message._markdown is not None and message._markdown.display
    await pilot.click(message._markdown, offset=(5, 0))
    await pilot.pause(0.1)
    assert not message.collapsed, "reasoning body click collapsed the disclosure"


async def _rewind(pilot: Pilot) -> None:
    for message in ("first message", "second message", "third message"):
        await pilot.press(*message, "enter")
        await pilot.pause(0.45)

    async def has_changes(entry_id: str) -> bool:
        _ = entry_id
        return True

    cast(
        BaseSnapshotTestApp, pilot.app
    ).app_server.resources.sessions.rewind_has_file_changes = has_changes
    await pilot.press("escape", "escape")
    await pilot.pause(0.3)


async def _rewind_mouse_inert(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.rewind_app import RewindApp

    await _rewind(pilot)
    rewind = pilot.app.query_one(RewindApp)
    await pilot.click(rewind.option_widgets[2], offset=(5, 0))
    await pilot.pause(0.1)


async def _rewind_double_click_after(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.rewind_app import RewindApp

    await _rewind(pilot)
    rewind = pilot.app.query_one(RewindApp)
    await pilot.click(rewind.option_widgets[2], offset=(5, 0), times=2)
    await pilot.pause(0.1)
    assert pilot.app.query_one(RewindApp) is rewind
    assert rewind._step == "persistence"


async def _rewind_keyboard_persistence(pilot: Pilot) -> None:
    await _rewind(pilot)
    await pilot.press("down", "down", "enter")
    await pilot.pause(0.1)


async def _exit_consequences(pilot: Pilot) -> None:
    app = cast(BaseSnapshotTestApp, pilot.app)
    app._pending_turn = True
    await app._exit_app()
    await pilot.pause(0.1)


class FullQuestionHost(BaseSnapshotTestApp):
    CSS_PATH = APP_CSS_PATH

    async def on_mount(self) -> None:
        from tests.snapshots.test_ui_snapshot_question_app import single_question_args

        await super().on_mount()
        await self._switch_to_question_app(single_question_args())


async def _question_other_text(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.question_app import QuestionApp

    question = pilot.app.query_one(QuestionApp)
    question.selected_option = question._other_option_idx
    await pilot.pause(0.1)
    assert question.selected_option == question._other_option_idx, (
        f"selected={question.selected_option}; other={question._other_option_idx}"
    )
    assert question.other_input is not None, "Other input not mounted"
    assert question.other_input.display, "Other input not visible"
    question.other_input.focus()
    await pilot.pause(0.05)
    assert question.other_input.has_focus, "Other input not focused"
    await pilot.press(*"SQLite")
    await pilot.pause(0.1)
    assert question.other_input is not None, "Other input was not mounted"
    assert question.other_input.value == "SQLite", (
        f"Other input contains {question.other_input.value!r}"
    )


class FullThinkingHost(BaseSnapshotTestApp):
    CSS_PATH = APP_CSS_PATH

    async def on_mount(self) -> None:
        await super().on_mount()
        await self._switch_to_thinking_picker_app()


class FullThemeHost(BaseSnapshotTestApp):
    CSS_PATH = APP_CSS_PATH

    async def on_mount(self) -> None:
        await super().on_mount()
        await self._switch_to_theme_picker_app()


class FullLogLevelHost(BaseSnapshotTestApp):
    CSS_PATH = APP_CSS_PATH

    async def on_mount(self) -> None:
        await super().on_mount()
        await self._switch_to_log_level_picker_app()


class OAuthFake:
    async def login(self, name: str) -> AsyncGenerator[MCPAuthUrlParams, None]:
        yield MCPAuthUrlParams(name=name, url="https://auth.example.invalid/oauth")
        await asyncio.Event().wait()


class OAuthFailFake:
    async def login(self, name: str) -> AsyncGenerator[MCPAuthUrlParams, None]:
        if False:
            yield MCPAuthUrlParams(name=name, url="https://auth.example.invalid/oauth")
        raise RuntimeError("fake authentication denied")


class OAuthHost(PickerHost):
    def __init__(self) -> None:
        from chartreux.cli.textual_ui.widgets.mcp_oauth_app import MCPOAuthApp

        super().__init__()
        self.picker = MCPOAuthApp("fake-oauth", OAuthFake())


class OAuthFailHost(PickerHost):
    def __init__(self) -> None:
        from chartreux.cli.textual_ui.widgets.mcp_oauth_app import MCPOAuthApp

        super().__init__()
        self.picker = MCPOAuthApp("fake-oauth", OAuthFailFake())


async def _oauth_wait(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.mcp_oauth_app import MCPOAuthApp

    for _ in range(30):
        await pilot.pause(0.05)
        if pilot.app.query_one(MCPOAuthApp)._auth_url:
            return
    raise RuntimeError("fake OAuth URL did not appear")


async def _oauth_open_failed(pilot: Pilot) -> None:
    await _oauth_wait(pilot)
    with patch(
        "chartreux.cli.textual_ui.widgets.mcp_oauth_app.webbrowser.open",
        return_value=False,
    ):
        await pilot.press("enter")
        await pilot.pause(0.1)


async def _oauth_show(pilot: Pilot) -> None:
    await _oauth_wait(pilot)
    await pilot.press("down", "down", "enter")
    await pilot.pause(0.1)


async def _oauth_failed(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.mcp_oauth_app import MCPOAuthApp

    for _ in range(30):
        await pilot.pause(0.05)
        if "Failed" in str(pilot.app.query_one(MCPOAuthApp)._status_message):
            return
    raise RuntimeError("fake OAuth failure did not appear")


async def _oauth_retried(pilot: Pilot) -> None:
    await _oauth_failed(pilot)
    await pilot.press("r")
    await _oauth_failed(pilot)


async def _proxy_discard(pilot: Pilot) -> None:
    await pilot.press(*"http://proxy.example.invalid:8080", "escape")
    await pilot.pause(0.1)


async def _trust_click_second(pilot: Pilot) -> None:
    from chartreux.setup.trusted_folders.trust_folder_dialog import TrustFolderDialog

    dialog = pilot.app.query_one(TrustFolderDialog)
    await pilot.click(dialog.option_widgets[1])
    await pilot.pause(0.1)


async def _trust_down(pilot: Pilot) -> None:
    await pilot.press("down")
    await pilot.pause(0.1)


async def _provider_single_click(pilot: Pilot) -> None:
    browser = pilot.app.screen.query_one("#wb-providers")
    await pilot.click(browser, offset=(15, 2))
    await pilot.pause(0.1)


async def _provider_double_click(pilot: Pilot) -> None:
    browser = pilot.app.screen.query_one("#wb-providers")
    await pilot.click(browser, offset=(15, 2), times=2)
    await pilot.pause(0.1)


async def _session_delete_escape(pilot: Pilot) -> None:
    await pilot.press("d", "escape")
    await pilot.pause(0.1)


async def _session_after_down(pilot: Pilot) -> None:
    await pilot.press("down")
    await pilot.pause(0.1)
    help_widget = pilot.app.screen.query_one("#sessionpicker-help")
    help_text = str(help_widget.render())
    assert "ID" in help_text and "Copy ID" in help_text, help_text


async def _session_modal_i(pilot: Pilot) -> None:
    await pilot.press("d")
    await pilot.pause(0.05)
    confirmation = pilot.app.screen.query_one("#sessionpicker-delete")
    assert confirmation.display
    await pilot.press("i")
    await pilot.pause(0.1)
    assert confirmation.display
    assert not pilot.app.screen.query_one("#sessionpicker-id-detail").display


async def _session_delete_escape_attempt(pilot: Pilot) -> None:
    await _session_delete_escape(pilot)
    await pilot.press("down", "enter")
    await pilot.pause(0.1)


class EmptySessionHost(PickerHost):
    def __init__(self) -> None:
        from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp

        super().__init__()
        self.picker = SessionPickerApp(
            sessions=[], latest_messages={}, cwd="/test/workdir"
        )


async def _session_loading(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp

    pilot.app.query_one(SessionPickerApp).set_loading(True)
    await pilot.pause(0.1)


async def _session_error(pilot: Pilot) -> None:
    from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp

    pilot.app.query_one(SessionPickerApp).set_loading(
        False, error="Fake local session index failed"
    )
    await pilot.pause(0.1)


async def _model_save_failed(pilot: Pilot) -> None:
    from unittest.mock import AsyncMock

    cast(BaseSnapshotTestApp, pilot.app).app_server.resources.config.update = AsyncMock(
        side_effect=RuntimeError("fake disk denied")
    )
    await pilot.press(*"/model", "enter")
    await pilot.pause(0.2)
    await pilot.press("down", "enter")
    await pilot.pause(0.3)


def fixtures() -> list[Fixture]:
    result = [
        Fixture("settings-browser", "settings", "SettingsScreen", SettingsHost),
        Fixture(
            "settings-inline",
            "settings",
            "SettingsScreen",
            SettingsHost,
            keys(*"displayed_workdir", "enter"),
            ("type displayed_workdir", "Enter"),
        ),
        Fixture(
            "settings-confirm",
            "settings",
            "SettingsScreen",
            SettingsHost,
            keys(*"enable_system_trust_store", "enter"),
            ("type enable_system_trust_store", "Enter"),
        ),
        Fixture(
            "settings-detail",
            "settings",
            "SettingsScreen",
            SettingsHost,
            keys("f1"),
            ("F1",),
        ),
        Fixture(
            "settings-no-match",
            "settings",
            "SettingsScreen",
            SettingsHost,
            keys(*"unfindable_setting_zzzz"),
            ("type unmatched search",),
        ),
        Fixture(
            "settings-enum",
            "settings",
            "SettingsScreen",
            SettingsHost,
            _settings_enum,
            ("add fake enum setting", "type ui_color_scheme", "Enter"),
        ),
        Fixture(
            "settings-checklist",
            "settings",
            "SettingsScreen",
            SettingsHost,
            _settings_checklist,
            ("seed fake tool inventory", "type inventory_tools", "Enter"),
        ),
        Fixture(
            "settings-invalid",
            "settings",
            "SettingsScreen",
            SettingsHost,
            keys(*"api_timeout", "enter", "ctrl+u", "x", "enter"),
            ("type api_timeout", "Enter", "replace with x", "Enter"),
        ),
        Fixture("model-picker", "pickers", "ModelPickerApp", ModelHost),
        Fixture(
            "model-picker-full-host",
            "pickers",
            "ModelPickerApp",
            _factory("test_ui_snapshot_model_picker", "ModelPickerTestApp"),
        ),
        Fixture(
            "model-save-failed",
            "pickers",
            "ChartreuxApp",
            SnapshotTestAppWithConversation,
            _model_save_failed,
            (
                "patch fake config update -> RuntimeError",
                "type /model",
                "Enter",
                "Down",
                "Enter",
                "wait for error",
            ),
        ),
        Fixture("session-picker", "sessions", "SessionPickerApp", SessionHost),
        Fixture(
            "session-after-down",
            "sessions",
            "SessionPickerApp after navigation",
            SessionHost,
            _session_after_down,
            ("Down to second session",),
        ),
        Fixture(
            "session-delete-modal-i",
            "sessions",
            "SessionPickerApp modal ID shortcut",
            SessionHost,
            _session_modal_i,
            ("d opens delete confirmation", "press i", "verify ID detail stays hidden"),
        ),
        Fixture(
            "session-delete-confirm",
            "sessions",
            "SessionPickerApp",
            SessionHost,
            keys("d"),
            ("d",),
        ),
        Fixture(
            "session-delete-esc-focus",
            "sessions",
            "SessionPickerApp",
            SessionHost,
            _session_delete_escape,
            ("d", "Escape"),
        ),
        Fixture(
            "session-delete-esc-attempt",
            "sessions",
            "SessionPickerApp",
            SessionHost,
            _session_delete_escape_attempt,
            ("d", "Escape", "Down", "Enter"),
        ),
        Fixture("session-empty", "sessions", "SessionPickerApp", EmptySessionHost),
        Fixture(
            "session-loading",
            "sessions",
            "SessionPickerApp",
            EmptySessionHost,
            _session_loading,
            ("set fake local discovery loading",),
        ),
        Fixture(
            "session-error",
            "sessions",
            "SessionPickerApp",
            EmptySessionHost,
            _session_error,
            ("set fake local discovery error",),
        ),
        Fixture(
            "chat-conversation",
            "chat",
            "ChartreuxApp",
            SnapshotTestAppWithConversation,
            _submit_chat_prompt,
            ("type Hello there, who are you?", "Enter", "wait for fake response"),
        ),
        Fixture(
            "completion-slash",
            "composer",
            "CompletionPopup",
            _factory("test_ui_snapshot_completion_popup", "CompletionPopupTestApp"),
            _popup,
            ("show fake slash suggestions",),
        ),
        Fixture(
            "full-shell-path-click",
            "composer",
            "ChartreuxApp path completion click",
            BaseSnapshotTestApp,
            _full_path_click,
            (
                "create fake alpha.txt and alpine.txt in temporary workdir",
                "type look @al",
                "click second completion row at offset (5,0)",
            ),
        ),
        Fixture(
            "mcp-overview",
            "mcp",
            "/mcp",
            _factory("test_ui_snapshot_mcp_command", "SnapshotTestAppWithMcpServers"),
            _mcp,
            ("type /mcp", "Enter", "wait for fake registry"),
        ),
        Fixture(
            "mcp-empty",
            "mcp",
            "/mcp",
            _factory("test_ui_snapshot_mcp_command", "SnapshotTestAppNoMcpServers"),
            _mcp,
            ("type /mcp", "Enter"),
        ),
        Fixture(
            "proxy-empty",
            "mcp",
            "ProxySetupApp",
            _factory("test_ui_snapshot_proxy_setup", "ProxySetupTestApp"),
        ),
        Fixture(
            "proxy-error",
            "mcp",
            "ProxySetupApp",
            _factory("test_ui_snapshot_proxy_setup", "ProxySetupTestApp"),
            _proxy_error,
            ("enter example proxy", "enter invalid HTTPS proxy", "press Save"),
        ),
        Fixture(
            "proxy-discard",
            "mcp",
            "ProxySetupApp",
            _factory("test_ui_snapshot_proxy_setup", "ProxySetupTestApp"),
            _proxy_discard,
            ("type fake proxy URL", "Escape"),
        ),
        Fixture(
            "question-single",
            "approval",
            "QuestionApp",
            _factory("test_ui_snapshot_question_app", "SingleQuestionApp"),
        ),
        Fixture("question-full-host", "approval", "QuestionApp", FullQuestionHost),
        Fixture(
            "question-other-text",
            "approval",
            "QuestionApp Other text",
            _factory("test_ui_snapshot_question_app", "SingleQuestionApp"),
            _question_other_text,
            ("select Other fixture state", "focus Other input", "type SQLite"),
        ),
        Fixture(
            "question-multi",
            "approval",
            "QuestionApp",
            _factory("test_ui_snapshot_question_app", "MultiSelectApp"),
        ),
        Fixture(
            "exit-consequences",
            "approval",
            "ExitConsequencesScreen",
            BaseSnapshotTestApp,
            _exit_consequences,
            ("mark fake main turn pending", "invoke /exit"),
        ),
        Fixture("thinking-picker", "pickers", "ThinkingPickerApp", FullThinkingHost),
        Fixture("theme-picker", "pickers", "ThemePickerApp", FullThemeHost),
        Fixture("loglevel-picker", "pickers", "LogLevelPickerApp", FullLogLevelHost),
        Fixture(
            "oauth-wait",
            "mcp",
            "MCPOAuthApp",
            OAuthHost,
            _oauth_wait,
            ("fake login yields example.invalid URL", "wait for auth"),
        ),
        Fixture(
            "oauth-open-failed",
            "mcp",
            "MCPOAuthApp",
            OAuthHost,
            _oauth_open_failed,
            (
                "fake login yields example.invalid URL",
                "patch browser open -> False",
                "Enter on Open",
            ),
        ),
        Fixture(
            "oauth-show-url",
            "mcp",
            "MCPOAuthApp",
            OAuthHost,
            _oauth_show,
            ("fake login yields example.invalid URL", "Down twice", "Enter Show"),
        ),
        Fixture(
            "oauth-failed",
            "mcp",
            "MCPOAuthApp",
            OAuthFailHost,
            _oauth_failed,
            ("fake login raises authentication denied", "wait for failure"),
        ),
        Fixture(
            "oauth-retried",
            "mcp",
            "MCPOAuthApp",
            OAuthFailHost,
            _oauth_retried,
            ("fake login fails", "press R", "wait for failure"),
        ),
        Fixture(
            "trust-folder",
            "onboarding",
            "TrustFolderApp",
            _factory(
                "test_ui_snapshot_trust_folder_dialog", "TrustFolderDialogSnapshotApp"
            ),
        ),
        Fixture(
            "trust-repo",
            "onboarding",
            "TrustFolderApp",
            _factory(
                "test_ui_snapshot_trust_folder_dialog",
                "TrustFolderDialogWithRepoSnapshotApp",
            ),
        ),
        Fixture(
            "trust-click-second",
            "onboarding",
            "TrustFolderApp",
            _factory(
                "test_ui_snapshot_trust_folder_dialog", "TrustFolderDialogSnapshotApp"
            ),
            _trust_click_second,
            ("click second visible numbered option",),
        ),
        Fixture(
            "trust-down-second",
            "onboarding",
            "TrustFolderApp",
            _factory(
                "test_ui_snapshot_trust_folder_dialog", "TrustFolderDialogSnapshotApp"
            ),
            _trust_down,
            ("Down",),
        ),
        Fixture(
            "agent-transcript",
            "agents",
            "AgentTranscriptViewer",
            _factory(
                "test_ui_snapshot_agent_transcript", "AgentTranscriptViewerSnapshotApp"
            ),
            _agent,
            ("refresh fake transcript",),
        ),
        Fixture(
            "tools-pending",
            "transcript",
            "ParallelToolCallsApp",
            _factory("test_ui_snapshot_parallel_tool_calls", "ParallelToolCallsApp"),
            _tools,
            ("emit three fake tool calls", "freeze spinner"),
        ),
        Fixture(
            "reasoning-collapsed",
            "transcript",
            "ChartreuxApp",
            _factory(
                "test_ui_snapshot_reasoning_content",
                "SnapshotTestAppWithReasoningContent",
            ),
            _reasoning,
            ("ask fake backend a question", "wait for reasoning"),
        ),
        Fixture(
            "reasoning-expanded",
            "transcript",
            "ChartreuxApp",
            _factory(
                "test_ui_snapshot_reasoning_content",
                "SnapshotTestAppWithReasoningContent",
            ),
            _reasoning_expanded,
            ("ask fake backend a question", "click reasoning disclosure"),
        ),
        Fixture(
            "reasoning-body-collapse",
            "transcript",
            "ChartreuxApp",
            _factory(
                "test_ui_snapshot_reasoning_content",
                "SnapshotTestAppWithReasoningContent",
            ),
            _reasoning_body_collapse,
            (
                "ask fake backend a question",
                "open ToolGroup if present",
                "click ReasoningMessage header",
                "click Markdown body at offset (5,0)",
            ),
        ),
        Fixture(
            "reasoning-body-click-after",
            "transcript",
            "ChartreuxApp",
            _factory(
                "test_ui_snapshot_reasoning_content",
                "SnapshotTestAppWithReasoningContent",
            ),
            _reasoning_body_after,
            (
                "ask fake backend a question",
                "open ToolGroup if present",
                "click ReasoningMessage header",
                "click Markdown body at offset (5,0)",
            ),
        ),
        Fixture(
            "resumed-transcript",
            "sessions",
            "ChartreuxApp",
            _factory(
                "test_ui_snapshot_session_resume", "SnapshotTestAppWithResumedSession"
            ),
            None,
            ("load fake prior session",),
        ),
        Fixture(
            "rewind-panel",
            "sessions",
            "RewindApp",
            _factory("test_ui_snapshot_rewind", "RewindSnapshotApp"),
            _rewind,
            ("send three fake turns", "patch rewind preflight", "Escape twice"),
        ),
        Fixture(
            "rewind-mouse-inert",
            "sessions",
            "RewindApp",
            _factory("test_ui_snapshot_rewind", "RewindSnapshotApp"),
            _rewind_mouse_inert,
            (
                "send three fake turns",
                "Escape twice",
                "click third action row offset (5,0)",
            ),
        ),
        Fixture(
            "rewind-double-click-after",
            "sessions",
            "RewindApp double-click guard",
            _factory("test_ui_snapshot_rewind", "RewindSnapshotApp"),
            _rewind_double_click_after,
            (
                "send three fake turns",
                "Escape twice",
                "double-click third action row offset (5,0)",
            ),
        ),
        Fixture(
            "rewind-keyboard-persistence",
            "sessions",
            "RewindApp",
            _factory("test_ui_snapshot_rewind", "RewindSnapshotApp"),
            _rewind_keyboard_persistence,
            ("send three fake turns", "Escape twice", "Down twice", "Enter"),
        ),
        Fixture(
            "debug-console",
            "help-status",
            "DebugConsole",
            _factory("test_ui_snapshot_debug_console", "DebugConsoleSnapshotApp"),
            _debug,
            ("Ctrl+\\",),
        ),
        Fixture(
            "debug-selected-row",
            "help-status",
            "DebugConsole selected row",
            _factory("test_ui_snapshot_debug_console", "DebugConsoleSnapshotApp"),
            _debug_selected,
            ("Ctrl+\\", "click first visible log row"),
        ),
        Fixture(
            "debug-copied-row",
            "help-status",
            "DebugConsole copy selected row",
            _factory("test_ui_snapshot_debug_console", "DebugConsoleSnapshotApp"),
            _debug_copied,
            ("Ctrl+\\", "click first visible log row", "press c"),
        ),
    ]
    for capture in _provider_captures():
        result.append(
            Fixture(
                capture.label,
                "provider",
                "ProviderWorkbenchScreen",
                capture.factory,
                capture.prepare,
                ("open with fixture keyboard sequence",) if capture.prepare else (),
            )
        )
    provider_factory = _provider_captures()[0].factory
    result.extend([
        Fixture(
            "provider-singleclick-route",
            "provider",
            "ProviderWorkbenchScreen",
            provider_factory,
            _provider_single_click,
            ("click #wb-providers offset (15,2) once",),
        ),
        Fixture(
            "provider-doubleclick-route",
            "provider",
            "ProviderWorkbenchScreen",
            provider_factory,
            _provider_double_click,
            ("double-click #wb-providers offset (15,2)",),
        ),
    ])
    result.extend(
        Fixture(name, "provider", route, factory, prepare, actions)
        for name, route, factory, prepare, actions in provider_fixtures()
    )
    result.extend(
        Fixture(name, family, route, factory, prepare, actions)
        for name, family, route, factory, prepare, actions in extra_fixtures()
    )
    return result


def _state_metadata(app: App) -> dict[str, Any]:
    from chartreux.cli.textual_ui.widgets.chat_input import ChatTextArea
    from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import (
        CompletionPopup,
    )
    from chartreux.cli.textual_ui.widgets.messages import ReasoningMessage
    from chartreux.cli.textual_ui.widgets.rewind_app import RewindApp
    from chartreux.setup.trusted_folders.trust_folder_dialog import TrustFolderDialog

    focused = app.screen.focused
    trust_dialogs = list(app.query(TrustFolderDialog))
    rewinds = list(app.query(RewindApp))
    popups = list(app.query(CompletionPopup))
    popup = popups[0] if popups else None
    text_areas = list(app.query(ChatTextArea))
    reasoning = list(app.query(ReasoningMessage))
    debug_logs = list(app.query(_LogView))
    return {
        "focused_id": focused.id if focused is not None else None,
        "selected_option": trust_dialogs[0].selected_option if trust_dialogs else None,
        "rewind_step": str(rewinds[0]._step) if rewinds else None,
        "rewind_selected_option": rewinds[0].selected_option if rewinds else None,
        "popup_visible": bool(popup and popup.display),
        "popup_selected_index": next(
            (
                i
                for i, row in enumerate(popup.children)
                if row.has_class("completion-selected")
            ),
            None,
        )
        if popup
        else None,
        "composer_text": text_areas[0].text if text_areas else None,
        "reasoning_collapsed": reasoning[0].collapsed if reasoning else None,
        "debug_selected_line": debug_logs[0]._selected_line if debug_logs else None,
        "highlighted_options": {
            widget.id or type(widget).__name__: (
                str(widget.highlighted_option.id)
                if widget.highlighted_option is not None
                else None
            )
            for widget in app.screen.query(OptionList)
        },
    }


def _save_review_svg(app: App, svg: Path, *, no_color: bool) -> Any:
    # Rich defaults to a dark SVG palette even for Textual's ANSI light theme.
    terminal_palette = app.ansi_theme_light if no_color else app.ansi_theme
    original_export = Console.export_svg

    def export_with_palette(console: Console, *args: Any, **kwargs: Any) -> str:
        kwargs["theme"] = terminal_palette
        return original_export(console, *args, **kwargs)

    runtime_no_color = os.environ.pop("NO_COLOR", None)
    try:
        with patch.object(Console, "export_svg", export_with_palette):
            app.save_screenshot(filename=svg.name, path=str(svg.parent))
    finally:
        if runtime_no_color is not None:
            os.environ["NO_COLOR"] = runtime_no_color
    return terminal_palette


async def capture(
    fixture: Fixture,
    size: tuple[int, int],
    theme: str,
    output: Path,
    commit: str,
    *,
    ascii_chrome: bool,
    no_color: bool,
) -> dict:
    app = fixture.factory()
    app.theme = theme
    review_app = cast(_ReviewApp, app)
    if ascii_chrome and hasattr(app, "ascii_chrome"):
        review_app.ascii_chrome = True
    stem = f"{fixture.name}-{size[0]}x{size[1]}-{theme}"
    if ascii_chrome:
        stem += "-ascii"
    if no_color:
        stem += "-no-color"
    svg = output / f"{stem}.svg"
    png = output / f"{stem}.png"
    terminal_text_path = output / f"{stem}.txt" if no_color else None
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if hasattr(app, "config"):
            config = cast(_ReviewConfig, review_app.config)
            config.theme = theme.removeprefix("textual-")
            config.ascii_chrome = ascii_chrome
        elif ascii_chrome:
            from types import SimpleNamespace

            review_app.config = SimpleNamespace(ascii_chrome=True)
        app.theme = theme
        if fixture.prepare:
            await fixture.prepare(pilot)
        await pilot.pause()
        state = _state_metadata(app)
        if terminal_text_path is not None:
            frame = app.screen._compositor.render_update(full=True)
            raw = cast(_SegmentFrame, frame).render_segments(app.console)
            terminal_text_path.write_text(re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw))
        terminal_palette = _save_review_svg(app, svg, no_color=no_color)
    subprocess.run(
        ["inkscape", str(svg), "--export-filename", str(png)],
        check=True,
        capture_output=True,
        text=True,
    )
    if not png.is_file():
        raise RuntimeError(f"PNG missing after Inkscape conversion: {png}")
    return {
        "fixture": fixture.name,
        "family": fixture.family,
        "route": fixture.route,
        "source_commit": commit,
        "viewport": f"{size[0]}x{size[1]}",
        "theme": theme,
        "ascii_chrome": ascii_chrome,
        "ascii_glyph_resolver_forced": ascii_chrome,
        "no_color": no_color,
        "effective_NO_COLOR": os.environ.get("NO_COLOR"),
        "terminal_palette_background": tuple(terminal_palette.background_color),
        "terminal_palette_foreground": tuple(terminal_palette.foreground_color),
        "terminal_palette_mode": "light-terminal-for-monochrome"
        if no_color
        else "app-ansi-theme",
        "png_visual_valid": not no_color,
        "visual_limitation": (
            "Rich SVG export renders NO_COLOR widget cells black-on-black; use terminal_text for content and a real PTY for visual contrast"
            if no_color
            else None
        ),
        "terminal_text": str(terminal_text_path.relative_to(ROOT))
        if terminal_text_path
        else None,
        "actions": list(fixture.actions),
        **state,
        "svg": str(svg.relative_to(ROOT)),
        "png": str(png.relative_to(ROOT)),
        "source_fixture": f"{fixture.factory.__module__}.{fixture.factory.__name__}",
        "host_context": "full ChartreuxApp shell"
        if isinstance(app, BaseSnapshotTestApp)
        else "standalone widget host; geometry may differ from production shell",
    }


async def _capture_case(
    fixture: Fixture,
    size: tuple[int, int],
    theme: str,
    output: Path,
    commit: str,
    manifest: Path,
    args: argparse.Namespace,
) -> bool:
    width, height = size
    ascii_context = (
        patch("chartreux.ui.chrome_glyphs.ascii_chrome_enabled", return_value=True)
        if args.ascii
        else nullcontext()
    )
    try:
        # Some snapshot hosts expose config only after compose. Hold the
        # equivalent ASCII resolver state through initial mount and capture.
        with ascii_context:
            record = await capture(
                fixture,
                size,
                theme,
                output,
                commit,
                ascii_chrome=args.ascii,
                no_color=args.no_color,
            )
        with manifest.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"PASS {fixture.name} {width}x{height} {theme}")
        return True
    except Exception as exc:
        print(f"FAIL {fixture.name} {width}x{height} {theme}: {exc}", file=sys.stderr)
        return False


async def _capture_matrix(
    selected: list[Fixture],
    args: argparse.Namespace,
    output: Path,
    commit: str,
    manifest: Path,
) -> int:
    failures = 0
    for fixture in selected:
        for size in args.sizes:
            for theme in args.themes:
                if not await _capture_case(
                    fixture, size, theme, output, commit, manifest, args
                ):
                    failures += 1
    return failures


async def run(args: argparse.Namespace) -> int:
    if not shutil.which("inkscape"):
        raise RuntimeError("Inkscape is required for PNG evidence")
    output = (ROOT / args.output).resolve()
    output.relative_to(ROOT / "docs/reviews/tui-affordances")
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    requested = set(args.fixture.split(","))
    selected = [
        f
        for f in fixtures()
        if ("all" in requested or f.name in requested)
        and args.family in {"all", f.family}
    ]
    if not selected:
        raise ValueError("no matching fixture")
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    # These are synthetic sentinel values. All backends and registries are in-process fakes.
    for key in ("MISTRAL_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        os.environ[key] = "chartreux-review-fake-key"
    manifest = output / "captures.jsonl"
    try:
        with tempfile.TemporaryDirectory(prefix="chartreux-review-") as workdir:
            previous = Path.cwd()
            os.chdir(workdir)
            try:
                failures = await _capture_matrix(
                    selected, args, output, commit, manifest
                )
            finally:
                os.chdir(previous)
    finally:
        reset_harness_files_manager()
    # Repeated runs refresh the same evidence path; keep one current provenance
    # record per PNG so earlier environmental mistakes cannot remain canonical.
    records: dict[str, dict] = {}
    for line in manifest.read_text().splitlines():
        record = json.loads(line)
        records[record["png"]] = record
    manifest.write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n" for record in records.values()
        )
    )
    print(f"{len(selected)} fixtures, {failures} failed; manifest={manifest}")
    return int(failures != 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="all")
    parser.add_argument("--fixture", default="all")
    parser.add_argument("--size", dest="sizes", action="append", default=None)
    parser.add_argument("--theme", dest="themes", action="append", default=None)
    parser.add_argument("--ascii", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--output", default="docs/reviews/tui-affordances/evidence/live"
    )
    args = parser.parse_args()
    args.sizes = [
        tuple(map(int, value.split("x")))
        for value in (args.sizes or ["80x24", "80x48"])
    ]
    # These are the themes Chartreux actually selects for explicit dark/light.
    args.themes = args.themes or ["ansi-dark", "ansi-light"]
    if args.no_color:
        os.environ["NO_COLOR"] = "1"
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
