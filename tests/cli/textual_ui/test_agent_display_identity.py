"""Display-only shortening of a process-topology agent handle (R2).

A process-backed child's handle is ``{identity}-attempt-<uuid4 hex>``: it is
the address every action takes, and it is unreadable in a row, a prompt or a
heading. This module pins the split the RCA asked for:

* the DISPLAY surfaces show ``attempt-<first 8 of the uuid>`` -- the browser
  row's secondary handle, the release confirmation's parenthetical, the
  transcript viewer's header parenthetical and the ``/status`` background
  results handle;
* every surface that ACTS on the handle keeps it whole -- ``full_metadata``,
  the launch ack's structured field and its ``agent_id=`` instruction
  (pinned in ``tests/app_server/test_launch_ack_identity.py``),
  ``check_agents`` rows, wake text and tool arguments;
* the stop prompt keeps the RAW handle by user ruling, so an operator who is
  about to kill a run reads exactly which attempt it is.

The short form is display-only, never a lookup key: two attempts can share
their first eight characters, and an id that is not exactly the
``attempt-<32 hex>`` shape is returned unchanged rather than mangled.

Behavioral discrimination: on the pre-H2 base the row, the release prompt, the
transcript header and the ``/status`` handle render the full wire handle, so
the display pins fail there; the full-handle pins hold on both trees.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from chartreux.app_server.protocol import AgentSummaryModel
from chartreux.app_server.protocol.transcript import (
    AgentTranscriptGetResponse,
    AgentTranscriptState,
)
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer


def _attempt_handle(identity: str = "researcher") -> tuple[str, str]:
    """A realistic process-topology handle and the short form it must display."""
    attempt = uuid4().hex
    return f"{identity}-attempt-{attempt}", f"attempt-{attempt[:8]}"


def _agent(agent_id: str, **kwargs: Any) -> AgentSummaryModel:
    values: dict[str, Any] = {"identity": "researcher", "availability": "idle"}
    values.update(kwargs)
    return AgentSummaryModel(agent_id=agent_id, **values)


class _BarApp(App[None]):
    """The smallest host an :class:`AgentBar` renders in."""

    def on_agent_bar_selection_requested(self, event: AgentBar.SelectionRequested):
        return None

    def on_agent_bar_stop_requested(self, event: AgentBar.StopRequested):
        self.stops.append((event.agent_id, event.run_id))

    def on_agent_bar_release_requested(self, event: AgentBar.ReleaseRequested):
        self.releases.append((event.agent_id, event.expected_run_id))

    def compose(self) -> ComposeResult:
        other = Static("Above the browser", id="other-focus")
        other.can_focus = True
        yield other
        self.stops: list[tuple[str, str]] = []
        self.releases: list[tuple[str, str | None]] = []
        self.bar = AgentBar()
        yield self.bar


def _content(bar: AgentBar) -> str:
    return str(bar.query_one("#agent-bar-content", Static).render())


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


#: The display surfaces are pinned through their rendered widgets, so the
#: rule's own helper is imported lazily where a pin exercises it directly:
#: that keeps every surface pin IMPORTABLE (and therefore behaviorally
#: discriminating) on a tree that predates the helper.


@pytest.mark.parametrize("identity", ["researcher", "coder", "reviewer", "architect"])
def test_a_standard_attempt_handle_shortens_to_its_first_eight(identity: str) -> None:
    from chartreux.cli.textual_ui.widgets.agent_bar import short_agent_display_id

    full, short = _attempt_handle(identity)
    assert short_agent_display_id(full) == short
    assert len(short) == len("attempt-") + 8


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param("coder-3", id="in-process-handle"),
        pytest.param("attempt-1a2b3c4d", id="already-short"),
        pytest.param("coder-attempt-", id="marker-without-uuid"),
        pytest.param("coder-attempt-1a2b3c4d", id="truncated-uuid"),
        pytest.param("coder-attempt-" + "A" * 32, id="uppercase-uuid"),
        pytest.param("coder-attempt-" + "g" * 32, id="non-hex-uuid"),
        pytest.param("coder", id="no-marker"),
        pytest.param("", id="empty"),
    ],
)
def test_an_unexpected_handle_shape_is_returned_unchanged(raw: str) -> None:
    from chartreux.cli.textual_ui.widgets.agent_bar import short_agent_display_id

    assert short_agent_display_id(raw) == raw


def test_an_identity_containing_the_marker_is_not_mistaken_for_it() -> None:
    from chartreux.cli.textual_ui.widgets.agent_bar import short_agent_display_id

    attempt = uuid4().hex
    # The LAST ``attempt-<32 hex>`` wins, so an identity that itself ends in
    # "attempt" still shortens the real handle rather than the identity.
    assert (
        short_agent_display_id(f"attempt-reader-attempt-{attempt}")
        == f"attempt-{attempt[:8]}"
    )
    assert short_agent_display_id("attempt-reader") == "attempt-reader"


def test_the_two_layer_twins_answer_identically() -> None:
    """The ack's sentence and the browser's row must not drift apart."""
    from chartreux.app_server.root_agent import _short_display_handle
    from chartreux.cli.textual_ui.widgets.agent_bar import short_agent_display_id

    attempt = uuid4().hex
    for raw in (
        f"coder-attempt-{attempt}",
        f"attempt-{attempt}",
        "coder-3",
        "attempt-1a2b3c4d",
        "coder-attempt-",
        "",
    ):
        assert short_agent_display_id(raw) == _short_display_handle(raw)


# ---------------------------------------------------------------------------
# The display surfaces
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_browser_row_shows_the_short_handle() -> None:
    full, short = _attempt_handle()
    app = _BarApp()
    async with app.run_test(size=(120, 20)) as pilot:
        app.bar.update_agents((_agent(full, name="researcher-1"),))
        app.bar.open_browser()
        await pilot.pause()
        rendered = _content(app.bar)
    assert short in rendered
    assert full not in rendered
    assert "researcher-1" in rendered


@pytest.mark.asyncio
async def test_release_confirmation_shows_the_short_handle() -> None:
    full, short = _attempt_handle()
    app = _BarApp()
    async with app.run_test(size=(120, 20)) as pilot:
        app.bar.update_agents((_agent(full, name="researcher-1", latest_run_id="r1"),))
        app.bar.open_browser()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("r")
        await pilot.pause()
        assert app.bar._release_confirmation == (full, "r1")
        prompt = str(app.bar.query_one("#agent-release-prompt", Static).render())
        # The confirmation still RELEASES the full handle ...
        assert app.releases == []
        assert short in prompt
        assert full not in prompt
        await pilot.press("left", "enter")
        # ... and the request it posts is the full one.
        assert app.releases == [(full, "r1")]


@pytest.mark.asyncio
async def test_stop_prompt_keeps_the_raw_handle_by_user_ruling() -> None:
    full, short = _attempt_handle()
    app = _BarApp()
    async with app.run_test(size=(120, 20)) as pilot:
        app.bar.update_agents((
            _agent(
                full, name="researcher-1", availability="running", current_run_id="r1"
            ),
        ))
        app.bar.open_browser()
        await pilot.pause()
        await pilot.press("down")
        await pilot.press("c")
        await pilot.pause()
        prompt = str(app.bar.query_one("#agent-stop-prompt", Static).render())
    # An operator about to kill a run reads exactly which attempt it is.
    assert f"Stop {full}, run r1?" in prompt
    # The short form is a prefix of the full handle, so the discriminator is
    # the prompt's own wording: it names the WHOLE handle, not a parenthetical
    # short form.
    assert f"({short})" not in prompt


def test_full_metadata_keeps_the_full_handle() -> None:
    full, short = _attempt_handle()
    bar = AgentBar()
    rendered = bar.full_metadata(_agent(full, name="researcher-1", latest_run_id="r1"))
    assert f"Agent ID: {full}" in rendered
    assert f"({short})" not in rendered


@pytest.mark.asyncio
async def test_transcript_header_shows_the_short_handle() -> None:
    full, short = _attempt_handle()
    source = AsyncMock()
    source.read_agent_transcript = AsyncMock(
        return_value=AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
        )
    )

    class _ViewerApp(App[None]):
        def compose(self) -> ComposeResult:
            self.viewer = AgentTranscriptViewer(
                source, full, identity="researcher", name="researcher-1"
            )
            yield self.viewer

    app = _ViewerApp()
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        header = str(app.viewer.query_one("#agent-transcript-header").render())
    assert header == f"Subagent: researcher-1 ({short}) · researcher"
    assert full not in header
    # The viewer's own requests still address the full handle.
    assert app.viewer.agent_id == full


@pytest.mark.asyncio
async def test_transcript_header_without_a_name_keeps_the_raw_handle() -> None:
    """With no display name the handle is the only identifier: it stays raw."""
    full, short = _attempt_handle()
    source = AsyncMock()
    source.read_agent_transcript = AsyncMock(
        return_value=AgentTranscriptGetResponse(
            state=AgentTranscriptState.NO_SAVED_TRANSCRIPT
        )
    )

    class _ViewerApp(App[None]):
        def compose(self) -> ComposeResult:
            self.viewer = AgentTranscriptViewer(source, full, identity="researcher")
            yield self.viewer

    app = _ViewerApp()
    async with app.run_test(size=(120, 20)) as pilot:
        await pilot.pause()
        header = str(app.viewer.query_one("#agent-transcript-header").render())
    assert header == f"Subagent: {full} · researcher"
    assert f"({full})" not in header
