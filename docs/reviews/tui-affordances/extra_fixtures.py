"""Finite offline fixtures for previously uncaptured TUI states.

These reuse the application's test hosts and fake services. The edit fixture is
an effect-result diff, not an execution approval prompt: this fork has no
per-tool approval dialog.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import patch

from textual.app import App
from textual.pilot import Pilot

from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import (
    CompletionPopup,
    _CompletionRow,
)
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.loading import LoadingWidget
from tests.cli.textual_ui.test_agent_widgets import _agent, _BrowserApp
from tests.snapshots.base_snapshot_test_app import BaseSnapshotTestApp
from tests.snapshots.test_ui_snapshot_completion_popup import (
    FILE_MENTION_SUGGESTIONS,
    CompletionPopupTestApp,
)
from tests.snapshots.test_ui_snapshot_edit_diff import (
    EditApprovalAnsiApp,
    _await_settled_diff,
)
from tests.snapshots.test_ui_snapshot_image_attachments import _ImageAttachmentApp
from tests.snapshots.test_ui_snapshot_mcp_command import SnapshotTestAppWithMcpServers
from tests.snapshots.test_ui_snapshot_question_app import single_question_args
from tests.snapshots.test_ui_snapshot_queued_messages import (
    QueuedMessagesSnapshotApp,
    _enqueue_while_busy,
)
from tests.snapshots.test_ui_snapshot_retrying import SnapshotTestAppRetrying

Prepare = Callable[[Pilot], Awaitable[None]]
ExtraFixture = tuple[str, str, str, Callable[[], App], Prepare | None, tuple[str, ...]]


async def _agent_browser(pilot: Pilot) -> None:
    bar = pilot.app.query_one(AgentBar)
    bar.update_agents((
        _agent("review-agent", availability="running"),
        _agent("build-agent", availability="evicted"),
    ))
    bar.open_browser()
    await pilot.pause(0.1)


async def _agent_details(pilot: Pilot) -> None:
    await _agent_browser(pilot)
    await pilot.press("down", "d")
    await pilot.pause(0.1)
    bar = pilot.app.query_one(AgentBar)
    assert bar.selected_agent_id == "review-agent"
    assert bar._show_full_details
    assert "Identity: review-agent" in str(
        bar.query_one("#agent-bar-full-content").render()
    )


async def _queued(pilot: Pilot) -> None:
    await _enqueue_while_busy(
        pilot, ["first follow-up", "second follow-up", "third follow-up"]
    )
    await pilot.pause(0.1)


async def _mention_popup(pilot: Pilot) -> None:
    pilot.app.query_one(CompletionPopup).update_suggestions(
        FILE_MENTION_SUGGESTIONS, selected=0
    )
    await pilot.pause(0.1)


async def _full_slash_popup(pilot: Pilot) -> None:
    await pilot.press(*"/mo")
    await pilot.pause(0.15)
    assert pilot.app.query_one(CompletionPopup).display


async def _full_slash_click(pilot: Pilot) -> None:
    await _full_slash_popup(pilot)
    popup = pilot.app.query_one(CompletionPopup)
    chosen = popup._suggestions[1].label
    await pilot.click(list(popup.query(_CompletionRow))[1], offset=(5, 0))
    await pilot.pause(0.1)
    assert not popup.display and not popup._suggestions
    composer = pilot.app.query_one(ChatInputContainer)
    assert composer.value.strip() == chosen
    assert composer.input_widget is pilot.app.focused


async def _image_path(pilot: Pilot) -> None:
    with patch(
        "chartreux.cli.textual_ui.widgets.chat_input.paste_path._is_image_file",
        return_value=True,
    ):
        await pilot.press(*"look /snap/shot.png")
        await pilot.pause(0.15)


async def _retrying(pilot: Pilot) -> None:
    app = cast(SnapshotTestAppRetrying, pilot.app)
    app.retry_backend.on_retry = app.loop_under_test.notice_retry
    await pilot.press(*"Hello", "enter")
    await asyncio.wait_for(app.retry_backend.parked.wait(), timeout=5)
    await pilot.pause(0.1)


async def _action_required(pilot: Pilot) -> None:
    app = cast(BaseSnapshotTestApp, pilot.app)
    await app._ensure_loading_widget()
    loading = app.query_one(LoadingWidget)
    loading.begin_action_required("Input required")
    await app._switch_to_question_app(single_question_args())
    await pilot.pause(0.1)


async def _help(pilot: Pilot) -> None:
    await pilot.press("f1")
    await pilot.pause(0.1)


async def _edit_result(pilot: Pilot) -> None:
    await _await_settled_diff(pilot)


async def _mcp_tab(pilot: Pilot) -> None:
    await pilot.press(*"/mcp", "enter")
    await pilot.pause(0.2)
    await pilot.press("tab")
    await pilot.pause(0.1)


def extra_fixtures() -> list[ExtraFixture]:
    return [
        (
            "agent-bar-browser",
            "agents",
            "AgentBar browser",
            _BrowserApp,
            _agent_browser,
            ("populate two fake agent states", "open agent browser"),
        ),
        (
            "agent-bar-details",
            "agents",
            "AgentBar details",
            _BrowserApp,
            _agent_details,
            (
                "populate two fake agent states",
                "open agent browser",
                "Down to review-agent",
                "press d for details",
            ),
        ),
        (
            "queued-composer",
            "composer",
            "ChartreuxApp queued input",
            QueuedMessagesSnapshotApp,
            _queued,
            ("start blocked fake turn", "queue three follow-up prompts"),
        ),
        (
            "file-mention-popup",
            "composer",
            "CompletionPopup file mentions",
            CompletionPopupTestApp,
            _mention_popup,
            ("show three fake path suggestions",),
        ),
        (
            "full-shell-slash-popup",
            "composer",
            "ChartreuxApp slash completion",
            BaseSnapshotTestApp,
            _full_slash_popup,
            ("type /mo",),
        ),
        (
            "full-shell-slash-click",
            "composer",
            "ChartreuxApp slash completion click",
            BaseSnapshotTestApp,
            _full_slash_click,
            ("type /mo", "click second completion row at offset (5,0)"),
        ),
        (
            "image-path-composer",
            "composer",
            "ChartreuxApp image path input",
            _ImageAttachmentApp,
            _image_path,
            ("type synthetic /snap/shot.png image path",),
        ),
        (
            "loading-retrying",
            "help-status",
            "LoadingWidget retry status",
            SnapshotTestAppRetrying,
            _retrying,
            ("submit Hello to gated fake backend", "wait for retry status"),
        ),
        (
            "action-required-question",
            "approval",
            "QuestionApp with loading status",
            BaseSnapshotTestApp,
            _action_required,
            ("show action-required status", "open one-question panel"),
        ),
        (
            "help-command",
            "help-status",
            "ChartreuxApp F1 help",
            BaseSnapshotTestApp,
            _help,
            ("press F1",),
        ),
        (
            "edit-result-diff",
            "transcript",
            "EditResultWidget diff",
            EditApprovalAnsiApp,
            _edit_result,
            ("render fake edit effect diff", "wait for diff layout"),
        ),
        (
            "mcp-tab-search",
            "mcp",
            "MCP browser Tab search",
            SnapshotTestAppWithMcpServers,
            _mcp_tab,
            ("type /mcp", "Enter", "press Tab from MCP browser"),
        ),
    ]
