"""Agent browser controller for the TUI app (W4b C4, stage 8)."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from textual.events import DescendantFocus
from textual.widget import Widget

from chartreux.app_server.events import AgentsUpdate
from chartreux.app_server.protocol import (
    AgentEvictionModel,
    AgentSummaryModel,
    CancelOutcome,
    ReleaseOutcome,
)
from chartreux.cli.textual_ui.bottom_panel import BottomApp
from chartreux.cli.textual_ui.widgets.agent_bar import AgentBar
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.messages import UserCommandMessage
from chartreux.cli.textual_ui.windowing.transcript import TranscriptAnchor
from chartreux.observability.logging import get_effective_log_level, logger

if TYPE_CHECKING:
    from collections.abc import Callable

    from chartreux.cli.textual_ui.app import ChartreuxApp

_MARKDOWN_INLINE_MARKUP = "\\`*_[]<&~"
"""Characters that can open inline markup, entity or autolink sequences."""

_MARKDOWN_ESCAPE_TABLE = str.maketrans({
    char: f"\\{char}" for char in _MARKDOWN_INLINE_MARKUP
})


def _markdown_literal(text: str) -> str:
    """Escape inline-Markdown-active characters so ``text`` renders literally.

    ``UserCommandMessage`` renders its content through a Markdown widget, so
    an interpolated derived agent name must not carry markup-active
    characters (unlike toast notices, which render with markup disabled).
    Only characters that can open emphasis, code, link, autolink, entity or
    strikethrough sequences mid-text are escaped; plain names pass through
    unchanged.
    """
    return text.translate(_MARKDOWN_ESCAPE_TABLE)


class _AgentTranscriptSource:
    """Bind transcript reads to the viewer's parent-session attachment.

    Parameterized by late-bound getters (``get_epoch``/``get_parent_id``/
    ``get_session_id``) instead of an app reference: the attachment validation
    after the await reads the current epoch, parent, and session at call time,
    so supersession is detected exactly as before. app.py re-exports this
    name, so the module-global seam (import from app) is preserved.
    """

    def __init__(
        self,
        *,
        resources: Any,
        epoch: int,
        parent_id: str | None,
        get_epoch: Callable[[], int],
        get_parent_id: Callable[[], str | None],
        get_session_id: Callable[[], str | None],
    ) -> None:
        self._resources = resources
        self._epoch = epoch
        self._parent_id = parent_id
        self._get_epoch = get_epoch
        self._get_parent_id = get_parent_id
        self._get_session_id = get_session_id

    async def read_agent_transcript(
        self, agent_id: str, *, before: str | None = None, limit: int = 50
    ) -> Any:
        response = await self._resources.read_agent_transcript(
            agent_id, before=before, limit=limit
        )
        if (
            self._get_epoch() != self._epoch
            or self._get_parent_id() != self._parent_id
            or self._get_session_id() != self._parent_id
        ):
            raise RuntimeError("Transcript viewer attachment changed")
        return response


class AgentBrowserController:
    """Owns the agent browser/agent-transcript-viewer lifecycle for one app.

    Plain object (no mixins): the app binds itself late via ``_bind_app``
    (never constructor-captured), and collaborators are invoked as
    ``self._app._method(...)`` at call time so instance patches keep
    intercepting. The app exposes thin same-name delegates (plus forwarding
    properties for moved state) so direct callers see no change. The
    converge loop keeps its generation-checked continue-based supersession,
    and the close path joins the transition task exactly as before.
    """

    # Late-bound host reference, assigned via _bind_app after construction.
    _app: ChartreuxApp

    def __init__(self) -> None:
        self._agent_browser_opener: Widget | None = None
        self._last_conversation_focus: Widget | None = None
        self._agent_browser_anchor: TranscriptAnchor | None = None
        self.agent_summaries: list[AgentSummaryModel] = []
        self.agent_transcript_viewer: AgentTranscriptViewer | None = None
        self.agent_transcript_focus_target: Widget | None = None
        self._agent_transcript_epoch = 0
        self.selection_target: str | None = None
        self.selection_generation = 0
        self.transition_task: asyncio.Task[None] | None = None
        self._agent_close_restore_focus = True
        self._agent_transcript_parent_id: str | None = None
        self._agent_evictions: dict[str, AgentEvictionModel] = {}

    def _bind_app(self, app: ChartreuxApp) -> None:
        """Bind the host app after construction."""
        self._app = app

    async def handle_agents_update(
        self, event: AgentsUpdate, received_at: float
    ) -> None:
        """Consume the authoritative inventory: browser, viewer, chrome."""
        viewer = self.agent_transcript_viewer
        selected_id = self.selection_target or (
            viewer.agent_id if viewer is not None else None
        )
        if selected_id is not None and not any(
            agent.agent_id == selected_id for agent in event.agents
        ):
            # Releases and metadata expiry remove the agent from published
            # summaries; evictions and closures remain tombstones in that
            # list and stay browsable. Close the viewer, return to Main.
            label = selected_id
            for agent in self.agent_summaries:
                if agent.agent_id == selected_id:
                    label = agent.name or agent.agent_id
                    break
            await self._app._request_agent_transcript_close(restore_focus=True)
            await self._app._mount_and_scroll(
                UserCommandMessage(
                    f"Background agent {_markdown_literal(label)} is no longer "
                    "available; returned to the main conversation."
                )
            )
        self.agent_summaries = event.agents
        if self._app._turn_outcome_notice.display:
            self._app._show_no_output_outcome()
        self._agent_evictions.update({
            eviction.agent_id: eviction for eviction in event.evictions
        })
        evicted_agent_ids = {
            agent.agent_id
            for agent in event.agents
            if agent.availability.lower() == "evicted"
        }
        self._agent_evictions = {
            agent_id: eviction
            for agent_id, eviction in self._agent_evictions.items()
            if agent_id in evicted_agent_ids
        }
        if self._app._agent_bar is not None:
            self._app._agent_bar.update_agents(
                event.agents,
                tuple(self._agent_evictions.values()),
                received_at=received_at,
            )
        viewer = self.agent_transcript_viewer
        if viewer is None:
            return
        selected = next(
            (agent for agent in event.agents if agent.agent_id == viewer.agent_id), None
        )
        viewer.set_live(selected is not None and self._app._agent_is_active(selected))
        if selected is not None:
            # The viewer is reused per agent: refresh its header in place
            # when the selected handle re-tasks.
            viewer.set_header(selected.name, selected.profile)
            if self._app._agent_bar is not None:
                viewer.set_metadata(self._app._agent_bar.full_metadata(selected))

    def on_descendant_focus(self, event: DescendantFocus) -> None:
        if self.agent_transcript_viewer is not None:
            return
        if event.widget is self._app._agent_bar:
            if self._agent_browser_opener is None:
                self.capture_agent_browser_origin(self._last_conversation_focus)
        elif self._app._agent_bar is None or not self._app._agent_bar.expanded:
            self._last_conversation_focus = event.widget

    def capture_agent_browser_origin(self, opener: Widget | None) -> None:
        self._agent_browser_opener = opener
        self._agent_browser_anchor = self._app._transcript.capture_anchor(
            self._app._messages_area, following=self._app._transcript_following
        )

    async def on_agent_bar_closed(self, message: AgentBar.Closed) -> None:
        message.stop()
        anchor = self._agent_browser_anchor
        self._agent_browser_anchor = None
        if anchor is not None:
            await self._app._transcript.restore_anchor(self._app._messages_area, anchor)
        opener = self._agent_browser_opener
        self._agent_browser_opener = None
        if opener is not None and opener.is_mounted:
            self._app.call_after_refresh(lambda: opener.focus(scroll_visible=False))

    async def toggle_agent_browser(self, **kwargs: Any) -> None:
        if self._app._agent_bar is None:
            return
        if self._app._current_bottom_app == BottomApp.Question:
            await self._app._mount_and_scroll(
                UserCommandMessage("Finish or cancel this question first.")
            )
            return
        # The expanded browser is independent of the collapsed segment: it is
        # always reachable, even while the inventory is empty (design §8).
        if not self._app._agent_bar.expanded:
            self.capture_agent_browser_origin(self._app.screen.focused)
            self._app._refresh_main_agent_details()
            self._app._agent_bar.open_browser()
        else:
            self._app._agent_bar.close_browser()
            await self._app.on_agent_bar_closed(AgentBar.Closed())

    def on_agent_bar_selection_requested(
        self, message: AgentBar.SelectionRequested
    ) -> None:
        self._app._submit_agent_selection(message.agent_id)

    def submit_agent_selection(self, agent_id: str | None) -> None:
        if agent_id is not None and self._app._current_bottom_app == BottomApp.Question:
            self._app.notify("Finish or cancel this question first", severity="warning")
            return
        self.selection_generation += 1
        self.selection_target = agent_id
        logger.debug(
            "Agent selection phase=submitted generation=%d target=%s",
            self.selection_generation,
            agent_id,
        )
        self.ensure_agent_transition()

    def ensure_agent_transition(self) -> None:
        if self.transition_task is None or self.transition_task.done():
            self.transition_task = asyncio.create_task(self.converge_agent_selection())

    async def converge_agent_selection(self) -> None:  # noqa: PLR0915
        while True:
            generation = self.selection_generation
            target = self.selection_target
            logger.debug(
                "Agent selection phase=transition-start generation=%d target=%s",
                generation,
                target,
            )
            try:
                if target is None:
                    await self._app._close_agent_transcript_viewer(
                        restore_focus=self._agent_close_restore_focus, show_chat=True
                    )
                    if generation != self.selection_generation:
                        continue
                    logger.debug(
                        "Agent selection phase=closed generation=%d", generation
                    )
                    return
                if self._app._app_server is None or not self._app.is_running:
                    return
                current = self.agent_transcript_viewer
                if (
                    current is not None
                    and current.agent_id == target
                    and current.is_mounted
                ):
                    break
                agent = next(
                    (item for item in self.agent_summaries if item.agent_id == target),
                    None,
                )
                if agent is None:
                    return
                debug_timings = get_effective_log_level() == "DEBUG"
                await self._app._close_agent_transcript_viewer(
                    restore_focus=False, show_chat=False
                )
                if generation != self.selection_generation or not self._app.is_running:
                    continue
                self.agent_transcript_focus_target = self._app.screen.focused
                self._agent_transcript_epoch += 1
                self._agent_transcript_parent_id = self._app.app_server.session_id
                viewer = AgentTranscriptViewer(
                    _AgentTranscriptSource(
                        resources=self._app.app_server.resources.sessions,
                        epoch=self._agent_transcript_epoch,
                        parent_id=self._agent_transcript_parent_id,
                        get_epoch=lambda: self._agent_transcript_epoch,
                        get_parent_id=lambda: self._agent_transcript_parent_id,
                        get_session_id=lambda: (
                            self._app.app_server.session_id
                            if self._app._app_server is not None
                            else None
                        ),
                    ),
                    agent.agent_id,
                    profile=agent.profile,
                    name=agent.name,
                    metadata=(
                        self._app._agent_bar.full_metadata(agent)
                        if self._app._agent_bar is not None
                        else ""
                    ),
                    live=self._app._agent_is_active(agent),
                    show_message_timestamps=self._app.config.show_message_timestamps,
                )
                self.agent_transcript_viewer = viewer
                self._app._chat_widget.display = False
                self.hide_conversation_chrome()
                self._app._indicate_pending_action()
                try:
                    logger.debug(
                        "Agent selection phase=mount-start generation=%d target=%s",
                        generation,
                        target,
                    )
                    mount_start = time.perf_counter() if debug_timings else 0.0
                    await self._app.mount(viewer, before="#chat")
                    self._app._indicate_pending_action()
                    if debug_timings:
                        logger.debug(
                            "Agent transcript mount took %.3fs generation=%d target=%s",
                            time.perf_counter() - mount_start,
                            generation,
                            target,
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to mount agent transcript viewer", exc_info=exc
                    )
                    self.agent_transcript_viewer = None
                    self._agent_transcript_epoch += 1
                    self._agent_transcript_parent_id = None
                    self.recover_agent_mount_failure()
                    if generation != self.selection_generation:
                        continue
                    return
                await asyncio.sleep(0)
                if generation != self.selection_generation:
                    continue
                viewer.focus()
                logger.debug(
                    "Agent selection phase=ready generation=%d target=%s",
                    generation,
                    target,
                )
                return
            except Exception as exc:
                logger.warning("Agent selection transition failed", exc_info=exc)
                if generation != self.selection_generation:
                    continue
                return

    async def request_agent_transcript_close(
        self, *, restore_focus: bool = True
    ) -> None:
        self.selection_generation += 1
        self.selection_target = None
        logger.debug(
            "Agent selection phase=close-requested generation=%d",
            self.selection_generation,
        )
        self._agent_close_restore_focus = restore_focus
        self.ensure_agent_transition()
        task = self.transition_task
        if task is not None and task is not asyncio.current_task():
            await task

    async def on_agent_transcript_viewer_closed(
        self, message: AgentTranscriptViewer.Closed
    ) -> None:
        if message.viewer is self.agent_transcript_viewer:
            await self._app._request_agent_transcript_close(restore_focus=True)

    async def close_agent_transcript_viewer(
        self, *, restore_focus: bool, show_chat: bool = True
    ) -> None:
        viewer = self.agent_transcript_viewer
        self._agent_transcript_epoch += 1
        self.agent_transcript_viewer = None
        self._agent_transcript_parent_id = None
        if viewer is not None and viewer.is_mounted:
            debug = get_effective_log_level() == "DEBUG"
            start = time.perf_counter() if debug else 0.0
            if debug:
                logger.debug(
                    "Agent transcript teardown phase=start generation=%d agent=%s",
                    self.selection_generation,
                    viewer.agent_id,
                )
            try:
                await viewer.dispose()
                if viewer.is_mounted:
                    await viewer.remove()
            finally:
                if debug:
                    logger.debug(
                        "Agent transcript teardown took %.3fs generation=%d agent=%s",
                        time.perf_counter() - start,
                        self.selection_generation,
                        viewer.agent_id,
                    )
        if show_chat:
            if self._app._cached_chat is not None:
                self._app._cached_chat.display = True
            self.restore_conversation_chrome()
            if self._agent_browser_anchor is not None:
                await self._app._transcript.restore_anchor(
                    self._app._messages_area, self._agent_browser_anchor
                )
        if not restore_focus:
            self.agent_transcript_focus_target = None
            return
        target = self.agent_transcript_focus_target
        self.agent_transcript_focus_target = None
        if self._app._agent_bar is not None and self._app._agent_bar.expanded:
            self._app.call_after_refresh(self._app._agent_bar.focus)
        elif target is not None and target.is_mounted:
            self._app.call_after_refresh(lambda: target.focus(scroll_visible=False))
        elif self._app._chat_input_container is not None:
            if input_widget := self._app._chat_input_container.input_widget:
                input_widget.set_app_focus(True)
            self._app.call_after_refresh(self._app._chat_input_container.focus_input)
        if show_chat and self.selection_target is None:
            await self._app._present_pending_callback()

    def recover_agent_mount_failure(self) -> None:
        if self._app._cached_chat is not None:
            self._app._cached_chat.display = True
        self.restore_conversation_chrome()

    def hide_conversation_chrome(self) -> None:
        for chrome_id in (
            "loading-area",
            "agent-bar",
            "bottom-app-container",
            "bottom-bar",
        ):
            self._app.query_one(f"#{chrome_id}").display = False

    def restore_conversation_chrome(self) -> None:
        for chrome_id in ("loading-area", "bottom-app-container", "bottom-bar"):
            self._app.query_one(f"#{chrome_id}").display = True
        if self._app._agent_bar is not None:
            # The expanded browser stays visible even with an empty inventory.
            self._app._agent_bar.display = (
                bool(self._app._agent_bar.agents) or self._app._agent_bar.expanded
            )

    def on_agent_bar_stop_requested(self, message: AgentBar.StopRequested) -> None:
        message.stop()
        # Do not block agents/update processing while the RPC is outstanding.
        self._app.run_worker(
            self.cancel_agent_run(message.agent_id, message.run_id), exclusive=False
        )

    def on_agent_bar_release_requested(
        self, message: AgentBar.ReleaseRequested
    ) -> None:
        message.stop()
        # Do not block agents/update processing while the RPC is outstanding.
        self._app.run_worker(
            self.release_agent(message.agent_id, message.expected_run_id),
            exclusive=False,
        )

    async def release_agent(self, agent_id: str, expected_run_id: str | None) -> None:
        try:
            response = await self._app.app_server.release_agent(
                agent_id, expected_run_id
            )
        except Exception as exc:
            # The registry purges records and results before awaiting runtime
            # teardown, so a lost RPC response does not prove the release did
            # not commit: keep the outcome uncertain instead of claiming the
            # agent is unchanged.
            self._app.notify(
                f"Failed to release {agent_id}: {exc}. "
                "Release may or may not have completed; "
                "check the agents browser or check_agents before retrying.",
                severity="error",
                markup=False,
            )
            return
        # The outcome never encodes the tombstone reason; conflict and unknown
        # mean nothing was stopped or purged, so a stale reply cannot affect a
        # replacement run's state.
        if response.outcome is ReleaseOutcome.CONFLICT:
            self._app.notify(
                f"Release conflict for {agent_id}: "
                f"{response.reason or 'the agent changed state'}. "
                "Nothing was released; the agent may have started a new run.",
                severity="warning",
                markup=False,
            )
        elif response.outcome is ReleaseOutcome.UNKNOWN:
            self._app.notify(
                f"{agent_id} is no longer known to the server; nothing was released.",
                severity="warning",
                markup=False,
            )

    async def cancel_agent_run(self, agent_id: str, run_id: str) -> None:
        bar = self._app._agent_bar
        if bar is None or not bar.stop_is_pending(agent_id, run_id):
            return
        try:
            response = await self._app.app_server.cancel_agent(agent_id, run_id)
        except Exception as exc:
            bar.settle_stop(agent_id, run_id, None)
            self._app.notify(
                f"Failed to stop {agent_id}, run {run_id}: {exc}. "
                "Retained output remains inspectable; retry or close the browser.",
                severity="error",
                markup=False,
            )
            return
        bar.settle_stop(agent_id, run_id, response)
        if response.outcome is CancelOutcome.FORBIDDEN:
            self._app.notify(
                f"Stop forbidden for {agent_id}, run {run_id}. "
                "Retained output remains inspectable; close the browser.",
                severity="error",
                markup=False,
            )
