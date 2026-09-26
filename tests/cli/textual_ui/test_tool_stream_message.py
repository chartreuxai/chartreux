from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.content import Content

from chartreux.app_server.models import (
    CompletedEffectState,
    EffectCallDisplay,
    EffectDetail,
    EffectResultDisplay,
    FileEditEffectDetail,
    FileWriteEffectDetail,
    GenericEffectDetail,
    PublicEffectEntry,
    PublicEntryGenerationStatus,
    RunningEffectState,
    UserQuestionEffectDetail,
)
from chartreux.cli.textual_ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from chartreux.core.events import ToolCallEvent
from chartreux.core.tools.builtins.bash import Bash, BashArgs
from tests.stubs.app_server import CoreEventProjection


class _ToolStreamApp(App[None]):
    CSS_PATH = Path(__file__).parents[3] / "chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        yield Vertical(id="root")


def _effect(
    *,
    completed: bool,
    suffix: str = "",
    verb: str = "",
    message: str = "Found 9 matches",
    detail: EffectDetail | None = None,
) -> PublicEffectEntry:
    generation_status = (
        PublicEntryGenerationStatus.COMPLETED
        if completed
        else PublicEntryGenerationStatus.IN_PROGRESS
    )
    state = (
        CompletedEffectState(
            output_text=message,
            display=EffectResultDisplay(
                success=True, verb=verb, message=message, suffix=suffix
            ),
        )
        if completed
        else RunningEffectState(output_text=message)
    )
    if detail is None:
        detail = GenericEffectDetail(
            tool_name="grep",
            display=EffectCallDisplay(
                summary="Searching", status_text="Searching files"
            ),
        )
    return PublicEffectEntry(
        id="grep-1",
        session_id="session-1",
        turn_id="turn-1",
        created_at=1,
        updated_at=2,
        generation_status=generation_status,
        title="grep",
        detail=detail,
        state=state,
    )


@pytest.mark.asyncio
async def test_restored_terminal_effect_renders_result_immediately() -> None:
    call = ToolCallMessage(_effect(completed=True))

    class _App(App[None]):
        def compose(self) -> ComposeResult:
            yield call

    async with _App().run_test() as pilot:
        await pilot.pause()

        assert call.get_content() == "Found 9 matches"
        assert call._text_widget is not None
        assert "Found 9 matches" in str(call._text_widget.render())
        assert not call._is_spinning


@pytest.mark.asyncio
async def test_terminal_effect_hides_transient_stream_message() -> None:
    app = _ToolStreamApp()

    async with app.run_test() as pilot:
        root = app.query_one("#root", Vertical)
        call = ToolCallMessage(_effect(completed=False))
        await root.mount(call)
        call.set_stream_message("grep: Found 9 matches")
        await pilot.pause(0.06)

        stream = call._stream_widget
        assert stream is not None
        assert stream.display

        await root.mount(ToolResultMessage(_effect(completed=True), call))
        await pilot.pause()

        assert not call._is_spinning
        assert not stream.display


@pytest.mark.asyncio
async def test_tool_stream_messages_coalesce_and_flush_on_finalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _ToolStreamApp()

    async with app.run_test() as pilot:
        call = ToolCallMessage(_effect(completed=False))
        await app.query_one("#root", Vertical).mount(call)
        await pilot.pause()
        stream = call._stream_widget
        assert stream is not None

        call.set_stream_message("first")
        call.set_stream_message("second")
        call.set_stream_message("latest")
        assert call._stream_message_buffer == "latest"
        assert call._stream_write_timer is not None
        assert not stream.display

        await pilot.pause(0.06)
        rendered = stream.render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "→ latest"
        assert call._stream_write_timer is None

        call.set_stream_message("before result")
        call.set_result_text("completed")
        rendered = stream.render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "→ before result"
        assert call._stream_message_buffer is None
        assert call._stream_write_timer is None

        rendered_at_flush: list[str] = []
        flush_stream_message = call._flush_stream_message

        def capture_flush() -> None:
            flush_stream_message()
            rendered = stream.render()
            assert isinstance(rendered, Content)
            rendered_at_flush.append(rendered.plain)

        monkeypatch.setattr(call, "_flush_stream_message", capture_flush)
        call.set_stream_message("before settle")
        call.stop_spinning()
        assert rendered_at_flush == ["→ before settle"]
        assert not stream.display
        assert call._stream_message_buffer is None
        assert call._stream_write_timer is None


@pytest.mark.asyncio
async def test_tool_stream_pending_message_timer_stops_on_unmount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _ToolStreamApp()

    async with app.run_test() as pilot:
        call = ToolCallMessage(_effect(completed=False))
        await app.query_one("#root", Vertical).mount(call)
        await pilot.pause()

        flushes_after_unmount: list[None] = []
        flush_stream_message = call._flush_stream_message

        def record_flush() -> None:
            if not call.is_mounted:
                flushes_after_unmount.append(None)
            flush_stream_message()

        monkeypatch.setattr(call, "_flush_stream_message", record_flush)
        call.set_stream_message("pending at unmount")
        timer = call._stream_write_timer
        assert timer is not None
        assert timer._task is not None
        assert call._stream_message_buffer == "pending at unmount"

        await call.remove()

        assert call._stream_write_timer is None
        assert timer._task is None
        assert call._stream_message_buffer is None
        await pilot.pause(0.06)
        assert flushes_after_unmount == []


@pytest.mark.asyncio
async def test_update_entry_skips_unchanged_header_and_updates_changed_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _ToolStreamApp()
    entry = _effect(completed=False)

    async with app.run_test() as pilot:
        call = ToolCallMessage(entry)
        await app.query_one("#root", Vertical).mount(call)
        await pilot.pause()

        set_text_calls: list[tuple[str, str, str]] = []
        set_text = call._set_text

        def record_set_text(
            text: str, suffix: str, *, verb: str = "", linkify: bool = False
        ) -> None:
            set_text_calls.append((verb, text, suffix))
            set_text(text, suffix, verb=verb, linkify=linkify)

        monkeypatch.setattr(call, "_set_text", record_set_text)

        for updated_at in (3, 4, 5):
            call.update_entry(entry.model_copy(update={"updated_at": updated_at}))

        assert call._entry.updated_at == 5
        assert set_text_calls == []

        changed_entry = _effect(
            completed=False,
            detail=GenericEffectDetail(
                tool_name="grep",
                display=EffectCallDisplay(
                    summary="Searching elsewhere", status_text="Searching files"
                ),
            ),
        )
        call.update_entry(changed_entry)

        assert set_text_calls == [("", "Searching elsewhere", "")]
        assert call._text_widget is not None
        rendered = call._text_widget.render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "Searching elsewhere"


@pytest.mark.asyncio
async def test_running_bash_uses_progressive_verb_and_message() -> None:
    projection = CoreEventProjection()
    projection.project(
        ToolCallEvent(
            tool_call_id="bash-1",
            tool_name="bash",
            tool_class=Bash,
            args=BashArgs(command="sleep 4"),
        )
    )
    entry = projection.history[-1]
    assert isinstance(entry, PublicEffectEntry)

    app = _ToolStreamApp()
    async with app.run_test() as pilot:
        call = ToolCallMessage(entry)
        await app.query_one("#root", Vertical).mount(call)
        await pilot.pause()

        assert call._verb_widget is not None
        assert call._text_widget is not None
        verb = call._verb_widget.render()
        message = call._text_widget.render()
        assert isinstance(verb, Content)
        assert isinstance(message, Content)
        assert verb.plain == "Running"
        assert message.plain == "sleep 4"
        assert call._header_row is not None
        assert call._header_row.has_class("running")
        assert call._header_row.has_class("collapsible-result")
        assert call._verb_widget.styles.text_opacity == 0.55
        assert call._text_widget.styles.text_opacity == 0.55

        call.stop_spinning()
        await pilot.pause()

        assert not call._header_row.has_class("running")


@pytest.mark.asyncio
async def test_collapsible_result_preserves_suffix_in_header() -> None:
    app = _ToolStreamApp()

    async with app.run_test() as pilot:
        root = app.query_one("#root", Vertical)
        entry = _effect(completed=True, suffix="(truncated)")
        call = ToolCallMessage(entry)
        await root.mount(call)
        result = ToolResultMessage(entry, call)
        await root.mount(result)
        await pilot.pause()

        suffix = result.query_one(".status-indicator-suffix", NoMarkupStatic)
        rendered = suffix.render()
        assert isinstance(rendered, Content)
        assert rendered.plain == "(truncated)"
        assert call.display is False


@pytest.mark.parametrize(
    "detail",
    [
        FileWriteEffectDetail(
            tool_name="write_file",
            display=EffectCallDisplay(
                summary="Writing dummy.txt",
                verb="Creating",
                message="dummy.txt",
                status_text="Writing file",
            ),
        ),
        FileEditEffectDetail(
            tool_name="edit",
            display=EffectCallDisplay(
                summary="Editing dummy.txt",
                verb="Editing",
                message="dummy.txt",
                status_text="Editing file",
            ),
        ),
        UserQuestionEffectDetail(
            tool_name="ask_user_question",
            display=EffectCallDisplay(
                summary="Asking a question",
                verb="Asking",
                message="Continue?",
                status_text="Waiting for user input",
            ),
        ),
    ],
    ids=["write_file", "edit", "ask_user_question"],
)
@pytest.mark.asyncio
async def test_running_always_expanded_result_stays_full_contrast(
    detail: EffectDetail,
) -> None:
    app = _ToolStreamApp()

    async with app.run_test() as pilot:
        call = ToolCallMessage(_effect(completed=False, detail=detail))
        await app.query_one("#root", Vertical).mount(call)
        await pilot.pause()

        assert call._header_row is not None
        assert call._verb_widget is not None
        assert call._text_widget is not None
        assert call._header_row.has_class("running")
        assert not call._header_row.has_class("collapsible-result")
        assert call._verb_widget.styles.text_opacity == 1.0
        assert call._text_widget.styles.text_opacity == 1.0
