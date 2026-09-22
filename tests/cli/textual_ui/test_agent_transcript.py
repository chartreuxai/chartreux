from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chartreux.app_server.protocol import (
    AgentTranscriptEntry,
    AgentTranscriptEntryKind,
    AgentTranscriptGetResponse,
    AgentTranscriptSource,
    AgentTranscriptState,
    AgentTranscriptTruncation,
)
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer


def _entry(
    entry_id: str,
    text: str,
    *,
    kind: AgentTranscriptEntryKind = AgentTranscriptEntryKind.USER_TEXT,
    truncated: bool = False,
) -> AgentTranscriptEntry:
    return AgentTranscriptEntry(
        entry_id=entry_id,
        kind=kind,
        display_text=text,
        tool_name="tool" if kind is AgentTranscriptEntryKind.TOOL_CALL else None,
        tool_call_id="call-1" if kind is AgentTranscriptEntryKind.TOOL_RESULT else None,
        truncated=truncated,
        truncation=AgentTranscriptTruncation.DISPLAY_TEXT_LIMIT if truncated else None,
    )


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
    def __init__(self, source: AgentTranscriptSource) -> None:
        super().__init__()
        self.viewer = AgentTranscriptViewer(source, "agent-1", page_size=2)
        self.closed: list[AgentTranscriptViewer.Closed] = []

    def compose(self) -> ComposeResult:
        yield self.viewer

    def on_agent_transcript_viewer_closed(
        self, message: AgentTranscriptViewer.Closed
    ) -> None:
        self.closed.append(message)
        self.call_after_refresh(message.viewer.remove)


def _content(app: _ViewerApp) -> str:
    return str(app.viewer._content.render())  # pyright: ignore[reportOptionalMemberAccess]


@pytest.mark.asyncio
async def test_open_fetches_once_and_status_waiting_do_not_fetch_again() -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert source.requests == [("agent-1", None, 2)]

        # The viewer has no status-notification handler or timer to trigger reads.
        await pilot.pause()
        assert len(source.requests) == 1

        source.respond(0, _available(_entry("new", "latest")))
        await pilot.pause()
        assert "latest" in _content(app)


@pytest.mark.asyncio
async def test_older_page_prepends_chronologically_without_boundary_duplicate() -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                _entry("second", "second"),
                _entry("third", "third"),
                cursor="second-cursor",
                has_more=True,
            ),
        )
        await pilot.pause()

        app.viewer.action_older_page()
        await pilot.pause()
        assert source.requests[1] == ("agent-1", "second-cursor", 2)
        source.respond(
            1,
            _available(
                _entry("first", "first"),
                _entry("second", "second"),
                cursor="first-cursor",
            ),
        )
        await pilot.pause()

        content = _content(app)
        assert content.index("first") < content.index("second") < content.index("third")
        assert content.count("second") == 1


@pytest.mark.asyncio
async def test_refresh_resets_pagination_and_discards_delayed_older_response() -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(0, _available(_entry("new", "old latest"), has_more=True))
        await pilot.pause()
        app.viewer.action_older_page()
        await pilot.pause()

        app.viewer.action_refresh()
        await pilot.pause()
        assert source.requests[2] == ("agent-1", None, 2)
        source.respond(2, _available(_entry("fresh", "fresh latest")))
        await pilot.pause()
        source.respond(1, _available(_entry("stale", "stale older")))
        await pilot.pause()

        content = _content(app)
        assert "fresh latest" in content
        assert "stale older" not in content
        assert app.viewer._cursor == "cursor"


@pytest.mark.asyncio
async def test_close_and_unmount_invalidate_late_completions() -> None:
    for invalidate in ("action_close", "on_unmount"):
        source = _FakeSource()
        viewer = AgentTranscriptViewer(source, "agent-1")
        content = MagicMock()
        viewer._content = content
        viewer._is_current = lambda epoch, viewer=viewer: (
            not viewer._viewer_closed and epoch == 0
        )  # type: ignore[method-assign]
        task = asyncio.create_task(viewer._load_page(0, before=None, replace=True))
        await asyncio.sleep(0)

        with patch.object(viewer, "post_message"):
            getattr(viewer, invalidate)()
        source.respond(0, _available(_entry("late", "late response")))
        await task

        content.update.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (_available(cursor=None), "Saved transcript is empty."),
        (
            AgentTranscriptGetResponse(state=AgentTranscriptState.NO_SAVED_TRANSCRIPT),
            "No saved transcript available.",
        ),
        (
            AgentTranscriptGetResponse(state=AgentTranscriptState.CHANGED),
            "Transcript changed; refresh",
        ),
        (
            AgentTranscriptGetResponse(state=AgentTranscriptState.EXCEEDS_VIEWER_LIMIT),
            "Transcript exceeds viewer limit",
        ),
    ],
)
async def test_response_states_are_visible_and_distinct(
    response: AgentTranscriptGetResponse, expected: str
) -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(0, response)
        await pilot.pause()
        assert expected in _content(app)
        assert "Saved transcript — may lag the running agent." in str(
            app.viewer.query_one("#agent-transcript-notice", Static).render()
        )


@pytest.mark.asyncio
async def test_error_and_truncation_markers_are_visible() -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(0, _available(_entry("short", "saved cut", truncated=True)))
        await pilot.pause()
        assert "[Entry truncated in saved transcript]" in _content(app)

        app.viewer.action_refresh()
        await pilot.pause()
        source.responses[1].set_exception(RuntimeError("offline"))
        await pilot.pause()
        assert "Unable to load saved transcript." in _content(app)


@pytest.mark.asyncio
async def test_escape_closes_locally_and_content_is_literal_and_bounded() -> None:
    source = _FakeSource()
    app = _ViewerApp(source)
    async with app.run_test() as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                _entry("markup", "[bold]not markup[/bold] " + chr(27) + "[31mred"),
                _entry("huge", "x" * 5_000),
            ),
        )
        await pilot.pause()
        content = _content(app)
        assert "[bold]not markup[/bold]" in content
        assert "\\x1b[31mred" in content
        assert "[Entry truncated for viewer display]" in content

    viewer = AgentTranscriptViewer(_FakeSource(), "agent-1")
    event = MagicMock()
    event.key = "escape"
    with patch.object(viewer, "post_message") as post_message:
        viewer.on_key(event)

    assert isinstance(post_message.call_args.args[0], AgentTranscriptViewer.Closed)
    event.stop.assert_called_once()
    event.prevent_default.assert_called_once()


@pytest.mark.asyncio
async def test_live_refresh_after_older_page_resumes_appends_and_scrolls_to_newest() -> (
    None
):
    source = _FakeSource()
    app = _ViewerApp(source)
    app.viewer._live = True
    async with app.run_test(size=(80, 12)) as pilot:
        await pilot.pause()
        source.respond(
            0,
            _available(
                *(_entry(str(index), f"entry {index}") for index in range(60)),
                has_more=True,
            ),
        )
        await pilot.pause()
        scroll = app.viewer._content_scroll
        assert scroll is not None and scroll.max_scroll_y > 0
        assert scroll.scroll_y == scroll.max_scroll_y
        scroll_y_before_older_page = scroll.scroll_y

        app.viewer.action_older_page()
        await pilot.pause()
        assert not app.viewer._on_newest_page
        source.respond(1, _available(_entry("older", "older")))
        await pilot.pause()
        assert scroll.scroll_y == scroll_y_before_older_page

        app.viewer.action_refresh()
        await pilot.pause()
        assert app.viewer._on_newest_page
        source.respond(2, _available(_entry("latest", "latest")))
        await pilot.pause()
        app.viewer._append_live_output()
        await pilot.pause()
        assert source.requests[-1] == ("agent-1", None, 2)
        assert scroll.scroll_y == scroll.max_scroll_y
