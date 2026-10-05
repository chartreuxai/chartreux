from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch
from weakref import WeakKeyDictionary

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widget import Widget
from textual.widgets import Static

from chartreux.app_server.events import HistoryEntryUpdated
from chartreux.app_server.models import (
    ImageAttachment,
    InlineImageSource,
    JsonPatchOperation,
    PublicEntryGenerationStatus,
    PublicReasoningEntry,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.handlers.event_handler import EventHandler
from chartreux.cli.textual_ui.widgets.collapsible import (
    DisclosureHeader,
    HeaderCollapsibleSection,
)
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.cli.textual_ui.windowing.transcript import TranscriptWindow
from chartreux.ui.duration_display import format_duration
from tests.cli.textual_ui.test_history_grouping import (
    _effect as _history_effect,
    _file_edit_effect,
    _message,
)
from tests.cli.textual_ui.test_tool_stream_message import _effect


@pytest.mark.parametrize(
    ("duration", "expected"),
    [
        (None, ""),
        (-1, ""),
        (float("nan"), ""),
        (float("inf"), ""),
        (0, "<0.1s"),
        (99, "<0.1s"),
        (100, "0.1s"),
        (3200, "3.2s"),
        (59940, "59.9s"),
        (59950, "1m00s"),
        (252000, "4m12s"),
        (3599500, "1h00m00s"),
        (7389000, "2h03m09s"),
    ],
)
def test_duration_boundaries(duration: float | None, expected: str) -> None:
    assert format_duration(duration) == expected


class TimingApp(App[None]):
    CSS_PATH = Path(__file__).parents[3] / "chartreux/cli/textual_ui/app.tcss"
    config: SimpleNamespace = SimpleNamespace(ascii_chrome=False)

    def compose(self) -> ComposeResult:
        yield Static("Timing")


@pytest.mark.asyncio
@pytest.mark.parametrize("show,stamped", [(False, True), (True, False)])
async def test_user_inset_includes_attachments_without_double_padding(
    show: bool, stamped: bool
) -> None:
    posted = datetime(2026, 7, 12, tzinfo=UTC) if stamped else None
    image = ImageAttachment(
        source=InlineImageSource(data=""), alias="image.png", mime_type="image/png"
    )
    user = UserMessage(
        "user", posted_at=posted, images=[image], show_message_timestamps=show
    )
    assistant = AssistantMessage(
        "assistant", posted_at=posted, show_message_timestamps=show
    )
    app = TimingApp()
    async with app.run_test() as pilot:
        await app.mount(user, assistant)
        await pilot.pause()
        assert not user.header.display and not assistant.header.display
        content = user.query_one(".user-message-content")
        attachments = user.query_one(".user-message-attachments")
        prompt = user.query_one(".user-message-prompt")
        assert prompt.region.x == assistant.region.x + 2
        assert content.region.x == prompt.region.x + 2
        assert attachments.region.x == prompt.region.x
        assert attachments.styles.padding.left == 0
        await assistant.stop_stream()


@pytest.mark.asyncio
@pytest.mark.parametrize("ascii_mode", [False, True])
@pytest.mark.parametrize("show_timing", [False, True])
async def test_inert_result_disclosure_stays_blank_through_group_folds(
    ascii_mode: bool, show_timing: bool
) -> None:
    app = TimingApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    entry = _effect(completed=True)
    entry = entry.model_copy(
        update={
            "state": entry.state.model_copy(
                update={"output": None, "output_text": "", "duration_ms": 3200}
            )
        }
    )
    group = ToolGroup(show_message_timestamps=show_timing)
    result = ToolResultMessage(entry)
    group.add_content_child(result)
    async with app.run_test() as pilot:
        await app.mount(group)
        await pilot.pause()
        section = result.query_one(HeaderCollapsibleSection)
        assert not section._collapsible
        for folded in (True, False, True, False):
            section.set_group_folded(folded)
            assert str(section._triangle.render()) == " "
        assert not section._toggle_row.can_focus


@pytest.mark.asyncio
@pytest.mark.parametrize("ascii_mode,success", [(False, "✓"), (True, "v")])
async def test_success_is_not_a_disclosure(ascii_mode: bool, success: str) -> None:
    from chartreux.ui.chrome_glyphs import chrome_glyph

    app = TimingApp()
    app.config = SimpleNamespace(ascii_chrome=ascii_mode)
    async with app.run_test():
        assert chrome_glyph("success") == success
        assert chrome_glyph("disclosure_closed") == "+"
        assert chrome_glyph("disclosure_open") == "-"


@pytest.mark.asyncio
async def test_duration_slot_both_paths_and_static_preference_updates() -> None:
    app = TimingApp()
    running = ToolCallMessage(_effect(completed=False))
    settled = _effect(completed=True)
    settled.state = settled.state.model_copy(update={"duration_ms": 3200})
    call = ToolCallMessage(settled)
    result = ToolResultMessage(settled, call)
    async with app.run_test() as pilot:
        await app.mount(running, call, result)
        await pilot.pause()
        assert not running.query_one(DisclosureHeader).duration_slot.display
        assert not call.display
        header = result.query_one(DisclosureHeader)
        assert header.duration_slot.display
        assert str(header.duration_slot.render()) == "3.2s"
        assert header.children[-1] is header.duration_slot
        assert header.duration_slot.region.right == header.content_region.right
        section = result.query_one(HeaderCollapsibleSection)
        section.toggle()
        await pilot.pause()
        assert header.duration_slot.display
        result.set_show_message_timestamps(False)
        assert not header.duration_slot.display
        result.set_show_message_timestamps(True)
        assert header.duration_slot.display
        with patch.object(
            header.duration_slot, "update", wraps=header.duration_slot.update
        ) as update:
            await pilot.pause(0.25)
            assert update.call_count == 0
        # Noncollapsible settled calls use their own header slot.
        await app.mount(ToolCallMessage(settled))
        await pilot.pause()
        visible_call = list(app.query(ToolCallMessage))[-1]
        assert visible_call.query_one(DisclosureHeader).duration_slot.display


@pytest.mark.asyncio
async def test_group_mode_transition_preserves_expansion_and_focus() -> None:
    section = HeaderCollapsibleSection(Static("raw"), header_text="Read file")
    group = ToolGroup(show_message_timestamps=False)
    group.add_content_child(section)
    app = TimingApp()
    async with app.run_test() as pilot:
        await app.mount(group)
        group.set_collapsed(False)
        section.set_collapsed(False)
        group.set_collapsed(True)
        group.header.focus()
        await pilot.pause()
        group.set_show_message_timestamps(True)
        await pilot.pause()
        assert group.content_container.display and not group.header.display
        assert app.focused is section._toggle_row
        assert not section.is_collapsed  # Masking does not change the saved choice.
        assert section._body is not None and not section._body.display
        group.set_collapsed(False)
        await pilot.pause()
        assert section._body is not None and section._body.display
        group.set_collapsed(True)
        group.set_show_message_timestamps(False)
        assert app.focused is group.header
        assert not group.content_container.display
        group.set_collapsed(False)
        assert section._body is not None and section._body.display


@pytest.mark.asyncio
async def test_settled_group_tool_timing_patch_updates_result_header() -> None:
    entry = _history_effect(0)
    window = TranscriptWindow()
    window.admit([entry], start_index=0)
    roots = window.build_unit(entry.id, WeakKeyDictionary())
    app = TimingApp()
    async with app.run_test() as pilot:
        await app.mount(*roots)
        window.register_mounted(entry.id, roots)
        result = app.query_one(ToolResultMessage)
        header = result.query_one(DisclosureHeader)
        # Mount completion does not imply layout/Resize completion. A pause can
        # itself post Resize messages after its queue barrier; drain those too.
        await pilot.pause()
        await pilot.pause()
        assert header.content_size.width > 0
        updated = entry.model_copy(
            update={"state": entry.state.model_copy(update={"duration_ms": 3200})}
        )
        handler = EventHandler(
            app.mount,
            lambda: True,
            mounted_entry_widgets=window.mounted_entry_widgets,
            update_retained_entry=window.update_entry,
        )
        await handler._handle_entry_updated(
            HistoryEntryUpdated(
                entry,
                updated,
                [
                    JsonPatchOperation(
                        op="replace", path="/state/durationMs", value=3200
                    )
                ],
            ),
            None,
        )
        await pilot.pause()
        assert app.query_one(ToolResultMessage) is result
        assert header.duration_slot.display
        assert str(header.duration_slot.render()) == "3.2s"
        assert not app.query_one(ToolCallMessage).display
        group = app.query_one(ToolGroup)
        group.set_show_message_timestamps(False)
        assert not header.duration_slot.display and group.header.display
        group.set_show_message_timestamps(True)
        assert header.duration_slot.display and not group.header.display


@pytest.mark.asyncio
async def test_settled_timing_patch_registry_skips_removed_and_windowed_rows() -> None:
    entry = _message(0)
    window = TranscriptWindow()
    window.admit([entry], start_index=0)
    widget = window.build_unit(entry.id, WeakKeyDictionary())[0]
    assert isinstance(widget, AssistantMessage)
    app = TimingApp()
    async with app.run_test() as pilot:
        await app.mount(widget)
        window.register_mounted(entry.id, [widget])
        handler = EventHandler(
            app.mount,
            lambda: True,
            mounted_entry_widgets=window.mounted_entry_widgets,
            update_retained_entry=window.update_entry,
        )
        updated = entry.model_copy(update={"turn_duration_ms": 372000})
        event = HistoryEntryUpdated(
            entry,
            updated,
            [JsonPatchOperation(op="replace", path="/turnDurationMs", value=372000)],
        )
        await handler._handle_entry_updated(event, None)
        await pilot.pause()
        assert widget.header.turn_duration_ms == 372000
        assert handler.current_streaming_message is None
        parent = widget.parent
        assert isinstance(parent, Widget)
        await window.evict_unit(entry.id)
        assert window.mounted_entry_widgets(entry.id) == []
        await handler._handle_entry_updated(event, None)
        assert await window.restore_unit(
            entry.id, parent, WeakKeyDictionary(), follow_bottom=True
        )
        rebuilt = window.units[entry.id].mounted_roots[0]
        assert isinstance(rebuilt, AssistantMessage)
        await pilot.pause()
        assert rebuilt.header.turn_duration_ms == 372000
        assert window.mounted_entry_widgets("removed-empty") == []


@pytest.mark.asyncio
async def test_open_reasoning_tail_admits_preceding_prose_timing() -> None:
    app = TimingApp()
    area = Vertical()
    entry = _message(0)
    reasoning = PublicReasoningEntry(
        id="reasoning",
        session_id=entry.session_id,
        created_at=1,
        updated_at=1,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        text="thinking",
    )
    window = TranscriptWindow()
    owner: Any = SimpleNamespace(
        _transcript=window,
        _messages_area=area,
        app_server=SimpleNamespace(history=[entry, reasoning]),
        _request_transcript_reconcile=lambda: None,
    )

    def mounted(entry_id: str):
        ChartreuxApp._admit_live_history(owner)
        return window.mounted_entry_widgets(entry_id)

    async def mount(widget: Widget, *, container: Widget | None = None, after=None):
        target = container or (after.parent if after is not None else area)
        await target.mount(widget, after=after)

    handler = EventHandler(
        mount,
        lambda: True,
        mounted_entry_widgets=mounted,
        update_retained_entry=window.update_entry,
    )
    owner.event_handler = handler
    async with app.run_test() as pilot:
        await app.mount(area)
        await handler._handle_entry_added(entry, None)
        await handler._handle_entry_added(reasoning, None)
        assert handler.current_tool_group is not None
        updated = entry.model_copy(update={"turn_duration_ms": 372000})
        await handler._handle_entry_updated(
            HistoryEntryUpdated(
                entry,
                updated,
                [
                    JsonPatchOperation(
                        op="replace", path="/turnDurationMs", value=372000
                    )
                ],
            ),
            None,
        )
        await pilot.pause()
        assert window.mounted_entry_widgets(entry.id)
        assert area.query_one(AssistantMessage).header.turn_duration_ms == 372000
        await handler.finalize_streaming()


@pytest.mark.asyncio
async def test_retry_reused_assistant_registered_for_new_entry_timing() -> None:
    app = TimingApp()
    handler = EventHandler(app.mount, lambda: True)
    first, second = _message(0), _message(1)
    async with app.run_test() as pilot:
        await handler._handle_entry_added(first, None)
        widget = app.query_one(AssistantMessage)
        handler.offer_retry()
        assert handler.begin_retry()
        await handler._handle_entry_added(second, None)
        assert handler.assistant_entry_widgets[second.id] is widget
        updated = second.model_copy(update={"turn_duration_ms": 3200})
        await handler._handle_entry_updated(
            HistoryEntryUpdated(
                second,
                updated,
                [JsonPatchOperation(op="replace", path="/turnDurationMs", value=3200)],
            ),
            None,
        )
        await pilot.pause()
        assert len(app.query(AssistantMessage)) == 1
        assert widget.header.turn_duration_ms == 3200


@pytest.mark.asyncio
async def test_later_result_mount_preserves_individual_expansion_and_focus() -> None:
    app = TimingApp()
    group = ToolGroup(show_message_timestamps=True)
    first = _effect(completed=True)
    call = ToolCallMessage(first)
    result = ToolResultMessage(first, call)
    group.add_content_child(call)
    group.add_content_child(result)
    async with app.run_test() as pilot:
        await app.mount(group)
        await pilot.pause()
        section = result.query_one(HeaderCollapsibleSection)
        section.toggle()
        await pilot.pause()
        section._toggle_row.focus()
        await pilot.pause()
        focused = app.focused
        assert focused is section._toggle_row
        second = first.model_copy(update={"id": "second"})
        call2 = ToolCallMessage(second)
        await group.content_container.mount(call2, ToolResultMessage(second, call2))
        await pilot.pause()
        assert not section.is_collapsed
        assert section._body is not None and section._body.display
        assert app.focused is focused


@pytest.mark.asyncio
async def test_unchanged_timing_preference_keeps_edit_review_visible() -> None:
    entry = _file_edit_effect(0)
    call = ToolCallMessage(entry)
    result = ToolResultMessage(entry, call)
    group = ToolGroup(show_message_timestamps=True)
    group.add_content_child(call)
    group.add_content_child(result)
    app = TimingApp()
    async with app.run_test() as pilot:
        await app.mount(group)
        await pilot.pause()
        assert not result._is_collapsible
        result.display = True  # A mounted review opened individually.
        group.set_show_message_timestamps(True)
        await pilot.pause()
        assert result.display
        assert not group.header.display


@pytest.mark.asyncio
async def test_retry_shared_root_does_not_block_subsequent_admission() -> None:
    app = TimingApp()
    area = Vertical()
    window = TranscriptWindow()
    history = []
    owner: Any = SimpleNamespace(
        _transcript=window,
        _messages_area=area,
        app_server=SimpleNamespace(history=history),
        _request_transcript_reconcile=lambda: None,
    )
    handler = EventHandler(area.mount, lambda: True)
    owner.event_handler = handler
    async with app.run_test() as pilot:
        await app.mount(area)
        first, retry, following = _message(0), _message(1), _message(2)
        history.append(first)
        await handler._handle_entry_added(first, None)
        ChartreuxApp._admit_live_history(owner)
        assert window.admitted_end_index == 1
        handler.offer_retry()
        assert handler.begin_retry()
        history.append(retry)
        await handler._handle_entry_added(retry, None)
        ChartreuxApp._admit_live_history(owner)
        assert window.admitted_end_index == 2
        history.append(following)
        await handler._handle_entry_added(following, None)
        ChartreuxApp._admit_live_history(owner)
        await pilot.pause()
        assert window.admitted_end_index == 3
        assert len(area.query(AssistantMessage)) == 2
        assert window.mounted_entry_widgets(first.id) == window.mounted_entry_widgets(
            retry.id
        )
        rebuilt = window.build_unit(first.id, WeakKeyDictionary())
        assert len(rebuilt) == 1
        assert isinstance(rebuilt[0], AssistantMessage)


@pytest.mark.asyncio
async def test_expanded_timing_header_wraps_full_command() -> None:
    text = "\n".join(f"command {index} " + "long argument " * 8 for index in range(5))
    section = HeaderCollapsibleSection(Static("raw"), header_text=text)
    app = TimingApp()
    async with app.run_test(size=(60, 30)) as pilot:
        await app.mount(section)
        header = section._toggle_row
        assert isinstance(header, DisclosureHeader)
        header.set_duration(3200, show=True)
        section.toggle()
        await pilot.pause()
        assert section._text_widget.region.height >= 5
        assert section._text_widget.styles.text_wrap == "wrap"
        assert str(section._text_widget.render()) == text
        assert header.duration_slot.region.right == header.content_region.right
