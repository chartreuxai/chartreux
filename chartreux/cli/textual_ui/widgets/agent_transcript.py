from __future__ import annotations

from typing import ClassVar

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Static

from chartreux.app_server.protocol import (
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptSource,
    AgentTranscriptState,
)

DEFAULT_TRANSCRIPT_PAGE_SIZE = 50
MAX_ENTRY_DISPLAY_LENGTH = 4_096
_FIRST_PRINTABLE_CODE_POINT = 32
_SAVED_SNAPSHOT_NOTICE = "Saved transcript — may lag the running agent."


class AgentTranscriptViewer(Vertical):
    """A read-only, paginated view of an agent transcript."""

    can_focus = True

    DEFAULT_CSS = """
    AgentTranscriptViewer {
        width: 100%;
        height: 1fr;
        border: solid $foreground-muted;
        background: $surface;
    }

    #agent-transcript-header {
        height: auto;
        padding: 0 1;
        background: $boost;
    }

    #agent-transcript-notice {
        height: auto;
        padding: 0 1;
        color: $text-muted;
    }

    #agent-transcript-content-scroll {
        height: 1fr;
        padding: 0 1;
        overflow-y: auto;
    }

    #agent-transcript-content {
        width: 100%;
        height: auto;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("pageup", "older_page", "Older"),
        Binding("r", "refresh", "Refresh"),
        Binding("escape", "close", "Close", priority=True),
    ]

    class Closed(Message):
        """Posted when the viewer should be removed by its owner."""

        def __init__(self, viewer: AgentTranscriptViewer) -> None:
            self.viewer = viewer
            super().__init__()

    def __init__(
        self,
        source: AgentTranscriptSource,
        agent_id: str,
        *,
        profile: str = "unknown",
        live: bool = False,
        page_size: int = DEFAULT_TRANSCRIPT_PAGE_SIZE,
    ) -> None:
        super().__init__(id="agent-transcript-viewer")
        self._source = source
        self._agent_id = agent_id
        self._profile = profile
        self._live = live
        self._live_timer: Timer | None = None
        self._on_newest_page = True
        self._page_size = page_size
        self._content: Static | None = None
        self._content_scroll: VerticalScroll | None = None
        self._cursor: str | None = None
        self._has_more = False
        self._entries: list[AgentTranscriptEntry] = []
        self._seen_entry_ids: set[str] = set()
        self._request_epoch = 0
        self._reading_page = False
        self._viewer_closed = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def compose(self) -> ComposeResult:
        yield Static(
            f"Subagent: {self._agent_id} · {self._profile}  [PageUp: older · r: refresh · Esc: close]",
            id="agent-transcript-header",
        )
        yield Static(self._notice_text(), id="agent-transcript-notice")
        self._content = Static(self._loading_text(), id="agent-transcript-content")
        with VerticalScroll(id="agent-transcript-content-scroll") as content_scroll:
            self._content_scroll = content_scroll
            yield self._content

    def on_mount(self) -> None:
        self.focus()
        self._start_refresh()
        if self._live:
            self._live_timer = self.set_interval(1.0, self._append_live_output)

    def on_unmount(self) -> None:
        self._viewer_closed = True
        self._request_epoch += 1
        if self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None

    def set_live(self, live: bool) -> None:
        """Update live state without replacing the converged transcript view."""
        was_live = self._live
        self._live = live
        if self.is_mounted and live and self._live_timer is None:
            self._live_timer = self.set_interval(1.0, self._append_live_output)
        elif not live and self._live_timer is not None:
            self._live_timer.stop()
            self._live_timer = None
            if was_live:
                self._start_request(before=None, replace=False)
        self._update_notice()

    def _loading_text(self) -> str:
        return "Loading live output…" if self._live else "Loading saved transcript…"

    def _notice_text(self) -> str:
        return (
            "Live output — refreshes about every second."
            if self._live
            else _SAVED_SNAPSHOT_NOTICE
        )

    def _update_notice(self, extra: str = "") -> None:
        notice = self.query_one("#agent-transcript-notice", Static)
        notice.update(f"{self._notice_text()}{extra}")

    def _append_live_output(self) -> None:
        if self._live and self._on_newest_page and not self._reading_page:
            self._start_request(before=None, replace=False)
        elif self._live and not self._on_newest_page:
            self._update_notice(" New output available.")

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.action_close()

    def action_older_page(self) -> None:
        if self._reading_page or not self._has_more or self._cursor is None:
            return
        self._on_newest_page = False
        self._start_request(before=self._cursor, replace=False)

    def action_refresh(self) -> None:
        self._start_refresh()

    def action_close(self) -> None:
        if self._viewer_closed:
            return
        self._viewer_closed = True
        self._request_epoch += 1
        self.post_message(self.Closed(self))

    def _start_refresh(self) -> None:
        if self._viewer_closed:
            return
        self._request_epoch += 1
        self._on_newest_page = True
        self._cursor = None
        self._has_more = False
        self._entries.clear()
        self._seen_entry_ids.clear()
        self._set_content(self._loading_text())
        self._start_request(before=None, replace=True, advance_epoch=False)

    def _start_request(
        self, *, before: str | None, replace: bool, advance_epoch: bool = True
    ) -> None:
        if self._viewer_closed:
            return
        if advance_epoch:
            self._request_epoch += 1
        epoch = self._request_epoch
        self._reading_page = True
        self.run_worker(self._load_page(epoch, before=before, replace=replace))

    async def _load_page(
        self, epoch: int, *, before: str | None, replace: bool
    ) -> None:
        try:
            response = await self._source.read_agent_transcript(
                self._agent_id, before=before, limit=self._page_size
            )
        except Exception:
            if self._is_current(epoch):
                self._set_content("Unable to load saved transcript.")
            return
        finally:
            # A newer request remains in flight independently; it owns this flag.
            if epoch == self._request_epoch:
                self._reading_page = False

        if not self._is_current(epoch):
            return
        self._apply_response(response, replace=replace, before=before)

    def _is_current(self, epoch: int) -> bool:
        return (
            not self._viewer_closed and self.is_mounted and epoch == self._request_epoch
        )

    def _apply_response(
        self, response: AgentTranscriptGetResponse, *, replace: bool, before: str | None
    ) -> None:
        if response.state is AgentTranscriptState.NO_SAVED_TRANSCRIPT:
            self._set_content("No saved transcript available.")
            return
        if response.state is AgentTranscriptState.CHANGED:
            self._set_content("Transcript changed; refresh")
            return
        if response.state is AgentTranscriptState.EXCEEDS_VIEWER_LIMIT:
            self._set_content("Transcript exceeds viewer limit")
            return
        if response.state is not AgentTranscriptState.AVAILABLE:
            self._set_content("Unable to load saved transcript.")
            return

        entries = response.entries or []
        if replace:
            self._entries = []
            self._seen_entry_ids.clear()
        new_entries = [
            entry for entry in entries if entry.entry_id not in self._seen_entry_ids
        ]
        self._seen_entry_ids.update(entry.entry_id for entry in new_entries)
        if replace:
            self._entries = new_entries
        elif before is None:
            self._entries.extend(new_entries)
        else:
            self._entries[0:0] = new_entries
        self._cursor = response.oldest_cursor
        self._has_more = response.has_more or False
        if not self._entries:
            self._set_content("Saved transcript is empty.")
            return
        self._set_content(self._render_entries())

    def _render_entries(self) -> Text:
        rendered = Text()
        for index, entry in enumerate(self._entries):
            if index:
                rendered.append("\n\n")
            rendered.append(self._entry_heading(entry), style="bold")
            rendered.append("\n")
            content, viewer_truncated = _display_text(entry.display_text)
            rendered.append(content)
            if entry.truncated:
                rendered.append("\n[Entry truncated in saved transcript]")
            if viewer_truncated:
                rendered.append("\n[Entry truncated for viewer display]")
        return rendered

    @staticmethod
    def _entry_heading(entry: AgentTranscriptEntry) -> str:
        match entry.kind:
            case AgentTranscriptEntryKind.USER_TEXT:
                return "User"
            case AgentTranscriptEntryKind.ASSISTANT_TEXT:
                return "Assistant"
            case AgentTranscriptEntryKind.TOOL_CALL:
                return f"Tool call: {entry.tool_name}"
            case AgentTranscriptEntryKind.TOOL_RESULT:
                return f"Tool result: {entry.tool_call_id}"

    def _set_content(self, content: str | Text) -> None:
        if self._content is not None:
            self._content.update(content)
        if self._on_newest_page and self._content_scroll is not None:
            self.call_after_refresh(
                self._content_scroll.scroll_end,
                animate=False,
                immediate=True,
                x_axis=False,
            )


def _display_text(value: str) -> tuple[str, bool]:
    """Make terminal controls visible while retaining normal transcript text."""
    literal = "".join(
        character
        if character in {"\n", "\t"} or ord(character) >= _FIRST_PRINTABLE_CODE_POINT
        else repr(character)[1:-1]
        for character in value
    )
    if len(literal) <= MAX_ENTRY_DISPLAY_LENGTH:
        return literal, False
    return literal[:MAX_ENTRY_DISPLAY_LENGTH] + "…", True
