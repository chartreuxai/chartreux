from __future__ import annotations

import asyncio
import hashlib
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectResultDisplay,
    FailedEffectState,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    PublicMessageEntry,
    RunningEffectState,
)
from chartreux.app_server.protocol import (
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptSource,
    AgentTranscriptState,
    AgentTranscriptToolStatus,
    AgentTranscriptTruncation,
)
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.messages import (
    AssistantMessage,
    ReasoningMessage,
    UserMessage,
)
from chartreux.cli.textual_ui.widgets.tool_grouping import GroupIndicator
from chartreux.cli.textual_ui.widgets.tools import (
    ToolCallMessage,
    ToolGroup,
    ToolResultMessage,
)
from chartreux.utils.tool_presentation import EffectCallDisplay


def _entry(
    entry_id: str,
    text: str,
    *,
    kind: AgentTranscriptEntryKind = AgentTranscriptEntryKind.USER_TEXT,
    created_at: int = 0,
    revision: int = 0,
    attachment_names: list[str] | None = None,
    attachment_count: int = 0,
    tool_call_id: str | None = None,
) -> AgentTranscriptEntry:
    values: dict[str, object] = {
        "entry_id": entry_id,
        "kind": kind,
        "display_text": text,
        "digest": hashlib.sha256(f"{entry_id}:{revision}".encode()).hexdigest(),
        "created_at": created_at,
        "updated_at": created_at,
        "generation_status": PublicEntryGenerationStatus.COMPLETED,
        "title": "User message",
        "attachment_names": attachment_names or [],
        "attachment_count": attachment_count,
    }
    if kind in {
        AgentTranscriptEntryKind.TOOL_CALL,
        AgentTranscriptEntryKind.TOOL_RESULT,
    }:
        tool_name = "read_file"
        call_display = EffectCallDisplay(
            summary="read_file(src/example.py)",
            verb="Reading",
            message="src/example.py",
            settled_verb="Read",
            settled_message="src/example.py",
            status_text="Reading src/example.py",
        )
        values.update({
            "title": tool_name,
            "tool_name": tool_name,
            "tool_call_id": tool_call_id or f"call-{entry_id}",
            "arguments": {"file_path": "src/example.py"},
            "result": {"content": "Example output"},
            "output_text": "Example output",
            "status": AgentTranscriptToolStatus.COMPLETED,
            "detail": GenericEffectDetail(
                tool_name=tool_name,
                input={"file_path": "src/example.py"},
                display=call_display,
            ),
            "state": CompletedEffectState(
                output={"content": "Example output"},
                output_text="Example output",
                display=EffectResultDisplay(
                    success=True, verb="Read", message="src/example.py"
                ),
            ),
        })
    return AgentTranscriptEntry.model_validate(values)


def _available(
    *entries: AgentTranscriptEntry,
    cursor: str | None = "cursor",
    has_more: bool = False,
) -> AgentTranscriptGetResponse:
    return AgentTranscriptGetResponse(
        state=AgentTranscriptState.AVAILABLE,
        entries=list(entries),
        oldest_cursor=cursor,
        has_more=has_more,
    )


def test_adapter_preserves_truncation_notice_for_text_and_effect_entries() -> None:
    viewer = AgentTranscriptViewer(_FakeSource(), "agent-1")
    clipped_user = _entry("clipped-user", "preview", created_at=0).model_copy(
        update={
            "truncated": True,
            "truncation": AgentTranscriptTruncation.DISPLAY_TEXT_LIMIT,
        }
    )
    user_history = viewer._history_entry(clipped_user)
    assert isinstance(user_history, PublicMessageEntry)
    assert "preview [truncated]" in user_history.text

    clipped_call = _entry(
        "clipped-call", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=1
    ).model_copy(
        update={
            "truncated": True,
            "truncation": AgentTranscriptTruncation.DISPLAY_TEXT_LIMIT,
        }
    )
    effect_history = viewer._history_entry(clipped_call)
    assert isinstance(effect_history, PublicEffectEntry)
    assert effect_history.detail.display.message is not None
    assert effect_history.detail.display.message.endswith("[truncated]")
    assert isinstance(effect_history.state, CompletedEffectState)
    assert effect_history.state.display.message.endswith("[truncated]")


def test_saved_unpaired_pending_call_is_interrupted_but_live_call_keeps_running() -> (
    None
):
    pending_call = _entry(
        "pending-call", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL
    ).model_copy(
        update={
            "status": AgentTranscriptToolStatus.PENDING,
            "state": RunningEffectState(),
        }
    )
    saved = AgentTranscriptViewer(_FakeSource(), "agent-1")._history_entry(pending_call)
    live = AgentTranscriptViewer(_FakeSource(), "agent-1", live=True)._history_entry(
        pending_call
    )

    assert isinstance(saved, PublicEffectEntry)
    assert isinstance(saved.state, FailedEffectState)
    assert saved.state.display.verb == "Interrupted"
    assert "interrupted" in saved.state.error.message.lower()
    assert isinstance(live, PublicEffectEntry)
    assert isinstance(live.state, RunningEffectState)


class _FakeSource(AgentTranscriptSource):
    def __init__(self) -> None:
        self.requests: list[tuple[str, str | None, int]] = []
        self.responses: list[asyncio.Future[AgentTranscriptGetResponse]] = []

    async def read_agent_transcript(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> AgentTranscriptGetResponse:
        self.requests.append((agent_id, before, limit))
        future: asyncio.Future[AgentTranscriptGetResponse] = asyncio.Future()
        self.responses.append(future)
        return await future

    def respond(self, index: int, response: AgentTranscriptGetResponse) -> None:
        self.responses[index].set_result(response)


class _ViewerApp(App[None]):
    def __init__(
        self,
        source: AgentTranscriptSource,
        *,
        live: bool = False,
        page_size: int = 2,
        tools_collapsed: bool = True,
    ) -> None:
        super().__init__()
        self._tools_collapsed = tools_collapsed
        self.viewer = AgentTranscriptViewer(
            source, "agent-1", page_size=page_size, live=live
        )
        self.closed: list[AgentTranscriptViewer.Closed] = []

    def compose(self) -> ComposeResult:
        yield self.viewer

    def on_agent_transcript_viewer_closed(
        self, message: AgentTranscriptViewer.Closed
    ) -> None:
        self.closed.append(message)
        self.call_after_refresh(message.viewer.remove)


@pytest.mark.asyncio
async def test_viewer_mounts_shared_widgets_for_each_entry_kind_and_attachment_placeholders() -> (
    None
):
    source = _FakeSource()
    app = _ViewerApp(source, tools_collapsed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                _entry(
                    "user",
                    "Please inspect this image.",
                    created_at=0,
                    attachment_names=["diagram.png"],
                    attachment_count=2,
                ),
                _entry(
                    "assistant",
                    "I will inspect it.",
                    kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    created_at=1,
                ),
                _entry(
                    "reasoning",
                    "First I will read the file.",
                    kind=AgentTranscriptEntryKind.REASONING,
                    created_at=2,
                ),
                _entry(
                    "call",
                    "{}",
                    kind=AgentTranscriptEntryKind.TOOL_CALL,
                    created_at=3,
                    tool_call_id="operation-1",
                ),
                _entry(
                    "result",
                    "Example output",
                    kind=AgentTranscriptEntryKind.TOOL_RESULT,
                    created_at=4,
                    tool_call_id="operation-1",
                ),
            ),
        )
        await pilot.pause()

        user = app.viewer.query_one(UserMessage)
        assert [image.alias for image in user._images] == [
            "diagram.png",
            "1 more image",
        ]
        assert len(app.viewer.query(AssistantMessage)) == 1
        assert len(app.viewer.query(ReasoningMessage)) == 1
        assert len(app.viewer.query(ToolGroup)) == 1
        assert len(app.viewer.query(ToolCallMessage)) == 1
        assert len(app.viewer.query(ToolResultMessage)) == 1
        assert app.viewer.query_one(ReasoningMessage).collapsed is False
        assistant = app.viewer.query_one(AssistantMessage)
        assert assistant.get_content() == "I will inspect it."


@pytest.mark.asyncio
async def test_poll_keeps_unchanged_widgets_and_replaces_changed_stable_id() -> None:
    source = _FakeSource()
    app = _ViewerApp(source, live=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        user_entry = _entry("user", "Question", created_at=0)
        assistant_entry = _entry(
            "assistant-stable",
            "The answer is",
            kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
            created_at=1,
        )
        source.respond(0, _available(user_entry, assistant_entry))
        await pilot.pause()
        original_user = app.viewer.query_one(UserMessage)
        original_assistant = app.viewer.query_one(AssistantMessage)

        app.viewer._append_live_output()
        await pilot.pause()
        source.respond(
            1,
            _available(
                user_entry,
                _entry(
                    "assistant-stable",
                    "The answer is 42.",
                    kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    created_at=1,
                    revision=1,
                ),
            ),
        )
        await pilot.pause()

        assert app.viewer.query_one(UserMessage) is original_user
        replacement = app.viewer.query_one(AssistantMessage)
        assert replacement is not original_assistant
        assert not original_assistant.is_attached
        assert replacement.get_content() == "The answer is 42."
        assert len(source.requests) == 2


@pytest.mark.asyncio
async def test_empty_assistant_entry_is_mounted_when_streaming_content_arrives() -> (
    None
):
    source = _FakeSource()
    app = _ViewerApp(source, live=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                _entry(
                    "assistant-growing",
                    "",
                    kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    created_at=0,
                )
            ),
        )
        await pilot.pause()
        assert len(app.viewer.query(AssistantMessage)) == 0

        app.viewer._append_live_output()
        await pilot.pause()
        source.respond(
            1,
            _available(
                _entry(
                    "assistant-growing",
                    "The streamed answer arrived.",
                    kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    created_at=0,
                    revision=1,
                )
            ),
        )
        await pilot.pause()

        assert len(app.viewer.query(AssistantMessage)) == 1
        assert app.viewer.query_one(AssistantMessage).get_content() == (
            "The streamed answer arrived."
        )


@pytest.mark.asyncio
async def test_growing_empty_assistant_is_inserted_at_its_timeline_position() -> None:
    source = _FakeSource()
    app = _ViewerApp(source, live=True)
    async with app.run_test() as pilot:
        await pilot.pause()
        empty_assistant = _entry(
            "assistant-growing",
            "",
            kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
            created_at=0,
        )
        first_call = _entry(
            "tool-first",
            "{}",
            kind=AgentTranscriptEntryKind.TOOL_CALL,
            created_at=1,
            tool_call_id="call-first",
        )
        second_call = _entry(
            "tool-second",
            "{}",
            kind=AgentTranscriptEntryKind.TOOL_CALL,
            created_at=2,
            tool_call_id="call-second",
        )
        source.respond(0, _available(empty_assistant, first_call, second_call))
        await pilot.pause()
        assert [unit.entry_ids for unit in app.viewer._units] == [
            ["tool-first", "tool-second"]
        ]

        app.viewer._append_live_output()
        await pilot.pause()
        source.respond(
            1,
            _available(
                _entry(
                    "assistant-growing",
                    "The answer arrived.",
                    kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
                    created_at=0,
                    revision=1,
                ),
                first_call,
                second_call,
            ),
        )
        await pilot.pause()

        assert [unit.entry_ids for unit in app.viewer._units] == [
            ["assistant-growing"],
            ["tool-first", "tool-second"],
        ]
        assert isinstance(app.viewer._units[1].group, ToolGroup)


@pytest.mark.asyncio
async def test_live_batch_continues_the_existing_tool_group() -> None:
    source = _FakeSource()
    app = _ViewerApp(source, live=True, tools_collapsed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        first = _entry(
            "call-first", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=0
        )
        source.respond(0, _available(first))
        await pilot.pause()
        original_group = app.viewer.query_one(ToolGroup)

        app.viewer._append_live_output()
        await pilot.pause()
        continuation = _entry(
            "result-second",
            "Example output",
            kind=AgentTranscriptEntryKind.TOOL_RESULT,
            created_at=1,
        )
        source.respond(1, _available(first, continuation))
        await pilot.pause()

        assert app.viewer.query_one(ToolGroup) is original_group
        assert len(app.viewer.query(ToolGroup)) == 1
        assert len(original_group.content_container.children) == 4
        assert [unit.entry_ids for unit in app.viewer._units] == [
            ["call-first", "result-second"]
        ]


@pytest.mark.asyncio
async def test_live_poll_gap_does_not_merge_separate_tool_groups() -> None:
    source = _FakeSource()
    app = _ViewerApp(source, live=True, tools_collapsed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        first = _entry(
            "call-first", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=0
        )
        source.respond(0, _available(first))
        await pilot.pause()

        app.viewer._append_live_output()
        await pilot.pause()
        second = _entry(
            "call-second", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=2
        )
        source.respond(1, _available(first, second))
        await pilot.pause()

        assert len(app.viewer.query(ToolGroup)) == 2
        assert [unit.entry_ids for unit in app.viewer._units] == [
            ["call-first"],
            ["call-second"],
        ]


@pytest.mark.asyncio
async def test_page_up_prepends_and_continues_a_cross_page_tool_group() -> None:
    source = _FakeSource()
    app = _ViewerApp(source, tools_collapsed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        newest = _entry(
            "newest-tool", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=1
        )
        source.respond(0, _available(newest, cursor="before-newest", has_more=True))
        await pilot.pause()
        group = app.viewer.query_one(ToolGroup)

        app.viewer.action_older_page()
        await pilot.pause()
        assert source.requests[1] == ("agent-1", "before-newest", 2)
        oldest = _entry(
            "oldest-tool", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=0
        )
        source.respond(1, _available(oldest, cursor="before-oldest"))
        await pilot.pause()

        assert app.viewer.query_one(ToolGroup) is group
        assert app.viewer._units[0].entry_ids == ["oldest-tool", "newest-tool"]
        assert len(group.content_container.children) == 4


@pytest.mark.asyncio
async def test_mount_cap_prunes_oldest_units(monkeypatch: pytest.MonkeyPatch) -> None:
    from chartreux.cli.textual_ui.widgets import agent_transcript

    monkeypatch.setattr(agent_transcript, "MAX_MOUNTED_TRANSCRIPT_ENTRIES", 4)
    source = _FakeSource()
    app = _ViewerApp(source, page_size=10)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                *(
                    _entry(f"entry-{index}", f"text {index}", created_at=index)
                    for index in range(7)
                )
            ),
        )
        await pilot.pause()

        assert len(app.viewer._units) == 4
        assert [unit.entry_ids[0] for unit in app.viewer._units] == [
            "entry-3",
            "entry-4",
            "entry-5",
            "entry-6",
        ]
        assert [
            message.history_entry_id for message in app.viewer.query(UserMessage)
        ] == ["entry-3", "entry-4", "entry-5", "entry-6"]
        assert set(app.viewer._known_entries) == {
            "entry-3",
            "entry-4",
            "entry-5",
            "entry-6",
        }


@pytest.mark.asyncio
async def test_page_up_at_mount_cap_keeps_older_entries_mounted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.cli.textual_ui.widgets import agent_transcript

    monkeypatch.setattr(agent_transcript, "MAX_MOUNTED_TRANSCRIPT_ENTRIES", 4)
    source = _FakeSource()
    app = _ViewerApp(source, page_size=2)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                *(
                    _entry(f"entry-{index}", f"text {index}", created_at=index)
                    for index in range(2, 6)
                ),
                cursor="before-entry-2",
                has_more=True,
            ),
        )
        await pilot.pause()
        assert len(app.viewer._units) == 4

        app.viewer.action_older_page()
        await pilot.pause()
        source.respond(
            1,
            _available(
                _entry("entry-0", "text 0", created_at=0),
                _entry("entry-1", "text 1", created_at=1),
                cursor="before-entry-0",
            ),
        )
        await pilot.pause()

        mounted_ids = [unit.entry_ids[0] for unit in app.viewer._units]
        assert mounted_ids == ["entry-0", "entry-1", "entry-2", "entry-3"]
        assert len(app.viewer._known_entries) == 4
        assert [
            message.history_entry_id for message in app.viewer.query(UserMessage)
        ] == (mounted_ids)


@pytest.mark.asyncio
async def test_live_poll_at_mount_cap_keeps_new_assistant_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.cli.textual_ui.widgets import agent_transcript

    monkeypatch.setattr(agent_transcript, "MAX_MOUNTED_TRANSCRIPT_ENTRIES", 2)
    source = _FakeSource()
    app = _ViewerApp(source, live=True, page_size=10)
    async with app.run_test() as pilot:
        await pilot.pause()
        first = _entry("entry-0", "text 0", created_at=0)
        second = _entry("entry-1", "text 1", created_at=1)
        source.respond(0, _available(first, second))
        await pilot.pause()

        app.viewer._append_live_output()
        await pilot.pause()
        assistant = _entry(
            "assistant-final",
            "The final answer.",
            kind=AgentTranscriptEntryKind.ASSISTANT_TEXT,
            created_at=2,
        )
        source.respond(1, _available(first, second, assistant))
        await pilot.pause()

        assert [unit.entry_ids[0] for unit in app.viewer._units] == [
            "entry-1",
            "assistant-final",
        ]
        assert app.viewer.query_one(AssistantMessage).get_content() == (
            "The final answer."
        )
        assert "assistant-final" in app.viewer._known_entries


@pytest.mark.asyncio
async def test_page_up_cap_eviction_recomputes_group_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.cli.textual_ui.widgets import agent_transcript

    monkeypatch.setattr(agent_transcript, "MAX_MOUNTED_TRANSCRIPT_ENTRIES", 3)
    source = _FakeSource()
    app = _ViewerApp(source, page_size=3, tools_collapsed=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        first = _entry(
            "effect-1", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=1
        )
        second = _entry(
            "effect-2", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=2
        )
        failed_newest = _entry(
            "effect-3", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=3
        ).model_copy(
            update={
                "status": AgentTranscriptToolStatus.FAILED,
                "state": FailedEffectState(
                    error={"message": "failed"},  # type: ignore[arg-type]
                    display=EffectResultDisplay(
                        success=False, verb="Read", message="failed"
                    ),
                ),
            }
        )
        source.respond(
            0,
            _available(
                first, second, failed_newest, cursor="before-effect-1", has_more=True
            ),
        )
        await pilot.pause()
        group = app.viewer.query_one(ToolGroup)
        assert group.header._last_state.value == GroupIndicator.ERROR.value

        app.viewer.action_older_page()
        await pilot.pause()
        oldest = _entry(
            "effect-0", "{}", kind=AgentTranscriptEntryKind.TOOL_CALL, created_at=0
        )
        source.respond(1, _available(oldest, cursor="before-effect-0"))
        await pilot.pause()

        assert app.viewer.query_one(ToolGroup) is group
        assert app.viewer._units[0].entry_ids == ["effect-0", "effect-1", "effect-2"]
        assert group._timeline_status.indicator is GroupIndicator.SUCCESS
        assert group.header._last_state.value == GroupIndicator.SUCCESS.value


@pytest.mark.asyncio
async def test_saved_view_is_not_polled_and_refresh_and_close_remain_available() -> (
    None
):
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(0, _available(_entry("saved", "saved text")))
        await pilot.pause()
        app.viewer._append_live_output()
        await pilot.pause()
        assert len(source.requests) == 1
        assert "Saved transcript — may lag the running agent." in str(
            app.viewer.query_one("#agent-transcript-notice", Static).render()
        )

        app.viewer.action_refresh()
        await pilot.pause()
        assert source.requests[1] == ("agent-1", None, 2)
        source.respond(1, _available(_entry("fresh", "fresh text")))
        await pilot.pause()
        assert len(app.viewer.query(UserMessage)) == 1
        assert app.viewer.query_one(UserMessage).get_content() == "fresh text"

        await pilot.press("escape")
        await pilot.pause()
        assert len(app.closed) == 1


@pytest.mark.asyncio
async def test_response_states_and_request_failures_are_visible() -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            AgentTranscriptGetResponse(state=AgentTranscriptState.NO_SAVED_TRANSCRIPT),
        )
        await pilot.pause()
        assert "No saved transcript available." in str(
            app.viewer._status_widget.render()  # pyright: ignore[reportOptionalMemberAccess]
        )

        app.viewer.action_refresh()
        await pilot.pause()
        source.responses[1].set_exception(RuntimeError("offline"))
        await pilot.pause()
        assert "Unable to load saved transcript." in str(
            app.viewer._status_widget.render()  # pyright: ignore[reportOptionalMemberAccess]
        )


@pytest.mark.asyncio
async def test_close_and_unmount_invalidate_late_completions() -> None:
    for invalidate in ("action_close", "on_unmount"):
        source = _FakeSource()
        viewer = AgentTranscriptViewer(source, "agent-1")
        viewer._is_current = lambda epoch, viewer=viewer: (
            not viewer._viewer_closed and epoch == 0
        )  # type: ignore[method-assign]
        task = asyncio.create_task(viewer._load_page(0, before=None, replace=True))
        await asyncio.sleep(0)

        with patch.object(viewer, "post_message"):
            getattr(viewer, invalidate)()
        source.respond(0, _available(_entry("late", "late response")))
        await task
        assert viewer._units == []


def test_escape_key_stops_event_and_posts_close_message() -> None:
    viewer = AgentTranscriptViewer(_FakeSource(), "agent-1")
    event = MagicMock()
    event.key = "escape"
    with patch.object(viewer, "post_message") as post_message:
        viewer.on_key(event)

    assert isinstance(post_message.call_args.args[0], AgentTranscriptViewer.Closed)
    event.stop.assert_called_once()
    event.prevent_default.assert_called_once()
