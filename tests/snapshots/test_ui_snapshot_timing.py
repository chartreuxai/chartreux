from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from textual.app import App, ComposeResult
from textual.pilot import Pilot

from chartreux.app_server.models import (
    CancelledEffectState,
    EffectResultDisplay,
    FailedEffectState,
    PublicError,
)
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from tests.cli.textual_ui.test_tool_stream_message import _effect
from tests.snapshots.snap_compare import SnapCompare
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


class TimingShowcaseApp(App[None]):
    CSS_PATH = Path(__file__).parents[2] / "chartreux/cli/textual_ui/app.tcss"

    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(ascii_chrome=False)

    def compose(self) -> ComposeResult:
        posted = datetime(2026, 6, 15, 12, 34, tzinfo=UTC)
        yield UserMessage("Timestamp-only user", posted_at=posted)
        yield AssistantMessage("Intermediate prose", posted_at=posted)
        yield AssistantMessage(
            "Whole-turn total", posted_at=posted, turn_duration_ms=372000
        )
        group = ToolGroup()
        running = _effect(completed=False).model_copy(update={"id": "running"})
        group.add_content_child(ToolCallMessage(running))
        for index, state in enumerate((
            _effect(completed=True).state.model_copy(update={"duration_ms": 3200}),
            FailedEffectState(
                error=PublicError(message="failed"),
                duration_ms=252000,
                display=EffectResultDisplay(success=False, message="failed"),
            ),
            CancelledEffectState(
                reason="cancelled",
                duration_ms=7389000,
                display=EffectResultDisplay(success=False, message="grep cancelled"),
            ),
            _effect(completed=True).state,
        )):
            entry = _effect(completed=True).model_copy(
                update={"id": f"settled-{index}", "state": state}
            )
            call = ToolCallMessage(entry)
            group.add_content_child(call)
            group.add_content_child(ToolResultMessage(entry, call))
        yield group
        yield UserMessage(
            "Disabled timing: user inset",
            posted_at=posted,
            show_message_timestamps=False,
        )
        yield AssistantMessage(
            "Disabled timing: flush left", show_message_timestamps=False
        )
        yield UserMessage("Legacy unstamped user")

    async def on_load(self) -> None:
        install_snapshot_wake()


@pytest.mark.parametrize("ascii_chrome,width", [(False, 80), (True, 32), (False, 12)])
def test_compact_timing_showcase(
    snap_compare: SnapCompare, ascii_chrome: bool, width: int
) -> None:
    async def before(pilot: Pilot) -> None:
        app = cast(TimingShowcaseApp, pilot.app)
        app.config.ascii_chrome = ascii_chrome
        for message in app.query(AssistantMessage):
            message.header._refresh_time()
            await message.write_initial_content()
            await message.stop_stream()
        await pilot.pause()
        group = app.query_one(ToolGroup)
        assert not group.header.display and group.content_container.display
        assert not app.query_one(ToolCallMessage).query_one(".tool-duration").display

    assert snap_compare(
        "test_ui_snapshot_timing.py:TimingShowcaseApp",
        terminal_size=(width, 30),
        run_before=before,
    )
