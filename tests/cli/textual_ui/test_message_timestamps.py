from __future__ import annotations

from datetime import UTC, datetime
import os
import time
from typing import Literal
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult

from chartreux.app_server.models import (
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    TextContentBlock,
)
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.message_header import (
    MessageHeader,
    format_message_timestamp,
)
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from chartreux.cli.textual_ui.windowing.history import build_history_widgets


@pytest.mark.parametrize(
    ("posted", "expected"),
    [
        ("2026-07-12T14:32:00+00:00", "14:32"),
        ("2026-07-11T14:32:00+00:00", "2026-07-11 14:32"),
        (None, ""),
    ],
)
def test_format(
    posted: str | None, expected: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TZ", "UTC")
    time.tzset()
    try:
        assert (
            format_message_timestamp(
                datetime.fromisoformat(posted) if posted else None,
                now=datetime(2026, 7, 12, 23, tzinfo=UTC),
            )
            == expected
        )
    finally:
        monkeypatch.undo()
        time.tzset()


def test_local_date_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    original = os.environ.get("TZ")
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    try:
        posted = datetime(2026, 7, 12, 2, 32, tzinfo=UTC)
        assert (
            format_message_timestamp(posted, now=datetime(2026, 7, 12, 3, tzinfo=UTC))
            == "22:32"
        )
        assert (
            format_message_timestamp(posted, now=datetime(2026, 7, 12, 5, tzinfo=UTC))
            == "2026-07-11 22:32"
        )
    finally:
        if original is None:
            monkeypatch.delenv("TZ")
        else:
            monkeypatch.setenv("TZ", original)
        time.tzset()


def test_missing_disabled_and_cell_aware_degradation() -> None:
    posted = datetime(2026, 7, 12, 14, 32, tzinfo=UTC)
    header = MessageHeader("Assistant", posted_at=posted)
    stamp = format_message_timestamp(posted, now=posted)
    assert header.metadata_for_width(len(stamp), now=posted) == stamp
    assert header.metadata_for_width(len(stamp) - 1, now=posted) == ""
    header.set_turn_duration(372000)
    assert header.metadata_for_width(100, now=posted) == f"{stamp} · 6m12s"
    assert header.metadata_for_width(5, now=posted) == "6m12s"
    assert header.metadata_for_width(4, now=posted) == ""
    header.set_turn_duration(None)
    header.set_timestamp(posted, show_message_timestamps=False)
    assert header.timestamp_for_width(100, now=posted) == ""
    header.set_timestamp(None, show_message_timestamps=True)
    assert header.timestamp_for_width(100, now=posted) == ""


def test_ascii_metadata_and_user_all_or_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chartreux.ui.chrome_glyphs.ascii_chrome_enabled", lambda: True)
    posted = datetime(2026, 7, 12, tzinfo=UTC)
    assistant = MessageHeader("Assistant", posted_at=posted, turn_duration_ms=3200)
    stamp = format_message_timestamp(posted, now=posted)
    assert assistant.metadata_for_width(80, now=posted) == f"{stamp} . 3.2s"
    user = UserMessage("hello", posted_at=posted)
    user.set_follows_previous(True)
    assert user.header.display
    assert user.header.metadata_for_width(5, now=posted) == stamp
    assert user.header.metadata_for_width(4, now=posted) == ""


@pytest.mark.asyncio
async def test_one_header_per_stream_and_narrow_resize() -> None:
    class HeaderApp(App):
        def compose(self) -> ComposeResult:
            yield AssistantMessage("", posted_at=posted)

    posted = datetime(2026, 7, 12, 14, 32, tzinfo=UTC)
    app = HeaderApp()
    async with app.run_test(size=(60, 20)) as pilot:
        message = app.query_one(AssistantMessage)
        for chunk in ["one", " two", " three"]:
            await message.append_content(chunk)
        assert len(message.query(MessageHeader)) == 1
        assert message.header.posted_at == posted
        await pilot.resize_terminal(10, 20)
        await pilot.pause()
        assert not message.header.display
        assert not list(message.header.query(".message-header-role"))
        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert message.header.query_one(".message-header-time").display
        await message.stop_stream()


@pytest.mark.asyncio
async def test_live_handler_uses_canonical_time_and_preserves_retry() -> None:
    posted = datetime(2026, 7, 12, 14, 32, tzinfo=UTC)
    app = App()
    async with app.run_test():

        async def mount(widget) -> None:
            await app.mount(widget)

        handler = EventHandler(
            mount, lambda: True, get_show_message_timestamps=lambda: False
        )
        await handler._handle_assistant_delta("first", None, posted)
        message = app.query_one(AssistantMessage)
        await handler._handle_assistant_delta(" next", None, posted)
        handler.offer_retry()
        await handler.finalize_streaming()
        assert handler.begin_retry()
        await handler._handle_assistant_delta(
            " retry", None, datetime(2026, 7, 13, tzinfo=UTC)
        )
        assert list(app.query(AssistantMessage)) == [message]
        assert message.header.posted_at == posted
        assert not message.header.show_message_timestamps
        await handler.finalize_streaming()


@pytest.mark.parametrize("show", [True, False])
def test_history_preserves_timestamp_without_created_at_fallback(show: bool) -> None:
    posted = datetime(2026, 7, 12, 14, 32, tzinfo=UTC)
    roles_and_times: list[tuple[Literal["user", "assistant"], datetime | None]] = [
        ("user", posted),
        ("assistant", posted),
        ("user", None),
    ]
    entries = [
        PublicMessageEntry(
            id=str(index),
            session_id="session",
            role=role,
            created_at=42,
            updated_at=42,
            generation_status=PublicEntryGenerationStatus.COMPLETED,
            content=[TextContentBlock(text="message")],
            posted_at=stamp,
        )
        for index, (role, stamp) in enumerate(roles_and_times)
    ]
    widgets = build_history_widgets(
        entries,
        start_index=0,
        history_widget_indices=WeakKeyDictionary(),
        tools_collapsed=True,
        show_message_timestamps=show,
    )
    assert len(widgets) == 3
    for widget, entry in zip(widgets, entries, strict=True):
        assert isinstance(widget, UserMessage | AssistantMessage)
        assert widget.header.posted_at == entry.posted_at
        assert widget.header.show_message_timestamps == show
