from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, ClassVar

from rich.cells import cell_len, chop_cells
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer

from chartreux.app_server.protocol import AgentEvictionModel, AgentSummaryModel
from chartreux.cli.textual_ui.widgets.spinner import create_spinner
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import logger
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def agent_state(agent: AgentSummaryModel) -> str:
    """Return the presentation state for an agent summary."""
    run_status = (agent.current_run_status or agent.last_run_status or "").lower()
    availability = agent.availability.lower()
    if availability == "evicted":
        return "evicted"
    if run_status in {"failed", "error"} or availability in {"failed", "error"}:
        return "failed"
    if run_status in {"running", "in_progress", "in-progress"} or availability in {
        "running",
        "finalizing",
    }:
        return "running"
    if run_status in {"cancelled", "canceled"}:
        return "cancelled"
    if availability == "idle":
        return "idle"
    return "unknown"


def agent_model_display_name(agent: AgentSummaryModel) -> str:
    """Return the agent's resolved provider/wire-model display identity."""
    model = agent.effective_model or agent.base_model
    if (
        agent.effective_model
        and agent.base_model
        and agent.effective_model.endswith(f"/{agent.base_model}")
    ):
        model = agent.base_model
    return format_model_display_name(agent.active_provider, model)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:g}s"


class AgentBar(Vertical):
    """Clickable status line with an in-place, keyboard-navigable agent browser."""

    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Previous agent", show=False),
        Binding("down", "cursor_down", "Next agent", show=False),
        Binding("enter", "select", "Select agent", show=False),
        Binding("d", "details", "Full agent details", show=False),
        Binding("escape", "collapse", "Close agents", show=False, priority=True),
    ]
    agents: reactive[tuple[AgentSummaryModel, ...]] = reactive(())

    class SelectionRequested(Message):
        """Request opening an agent transcript, or returning to the main pane."""

        def __init__(self, agent_id: str | None) -> None:
            self.agent_id = agent_id
            super().__init__()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="agent-bar", **kwargs)
        self._spinner = create_spinner()
        self._spinner_timer: Timer | None = None
        self._expanded = False
        self._show_full_details = False
        self._details_scroll: VerticalScroll | None = None
        self._details_content: NoMarkupStatic | None = None
        self._selected_agent_id: str | None = None
        self._evictions: dict[str, AgentEvictionModel] = {}
        self._previous_agent_ids: tuple[str, ...] = ()
        self._content: NoMarkupStatic | None = None
        self._rows_scroll: VerticalScroll | None = None
        self._detail: NoMarkupStatic | None = None
        self._footer: NoMarkupStatic | None = None
        self._rendered = ""
        self._last_rendered_content: str | Content | None = None
        self._last_rendered_focused: bool | None = None

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic("Background agents", id="agent-bar-title")
        self._rows_scroll = VerticalScroll(id="agent-bar-rows")
        with self._rows_scroll:
            self._content = NoMarkupStatic(id="agent-bar-content")
            yield self._content
        self._detail = NoMarkupStatic(id="agent-bar-detail")
        yield self._detail
        self._details_scroll = VerticalScroll(id="agent-bar-full-details")
        with self._details_scroll:
            self._details_content = NoMarkupStatic(id="agent-bar-full-content")
            yield self._details_content
        self._footer = NoMarkupStatic(id="agent-bar-footer")
        yield self._footer

    def render(self) -> str:
        return self._rendered

    @property
    def expanded(self) -> bool:
        return self._expanded

    @property
    def selected_agent_id(self) -> str | None:
        return self._selected_agent_id

    def on_mount(self) -> None:
        self._render_agents()
        self._update_spinner_timer()

    def on_focus(self) -> None:
        self.call_after_refresh(self._render_agents)

    def on_blur(self) -> None:
        self.call_after_refresh(self._render_agents)

    def on_resize(self) -> None:
        self._render_agents()

    def on_unmount(self) -> None:
        if self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def watch_agents(self) -> None:
        self._render_agents()
        if self.is_mounted:
            self._update_spinner_timer()

    def update_agents(
        self,
        agents: Sequence[AgentSummaryModel],
        evictions: Sequence[AgentEvictionModel] = (),
    ) -> None:
        selectable = tuple(
            agent for agent in agents if agent.availability.lower() != "released"
        )
        ids = tuple(agent.agent_id for agent in selectable)
        self._evictions.update({eviction.agent_id: eviction for eviction in evictions})
        evicted_ids = {
            agent.agent_id for agent in selectable if agent_state(agent) == "evicted"
        }
        self._evictions = {
            agent_id: eviction
            for agent_id, eviction in self._evictions.items()
            if agent_id in evicted_ids
        }
        self._update_selection(ids)
        self._previous_agent_ids = ids
        self.agents = selectable

    def toggle(self) -> None:
        self.close_browser() if self._expanded else self.open_browser()

    def open_browser(self) -> None:
        if not self.agents:
            return
        self._expanded = True
        self.set_class(True, "-expanded")
        self._render_agents()
        if self.is_mounted:
            self.focus()

    def close_browser(self) -> None:
        self._expanded = False
        self._show_full_details = False
        self.set_class(False, "-expanded")
        self._render_agents()

    def action_cursor_up(self) -> None:
        if self._show_full_details and self._details_scroll is not None:
            self._details_scroll.scroll_up(animate=False)
            return
        self._move_selection(-1)

    def action_cursor_down(self) -> None:
        if self._show_full_details and self._details_scroll is not None:
            self._details_scroll.scroll_down(animate=False)
            return
        self._move_selection(1)

    def action_select(self) -> None:
        self.post_message(self.SelectionRequested(self._selected_agent_id))

    def action_details(self) -> None:
        if self._expanded and self._selected_agent_id is not None:
            self._show_full_details = True
            self._render_agents()

    def action_collapse(self) -> None:
        if self._show_full_details:
            self._show_full_details = False
            self._render_agents()
            return
        self.close_browser()
        self.post_message(self.SelectionRequested(None))

    def on_click(self, event: events.Click) -> None:
        if not self._expanded:
            logger.debug("Agent bar click phase=expand-start row=%d", event.y)
            self.open_browser()
            logger.debug("Agent bar click phase=expand-done row=%d", event.y)
            return
        row = event.screen_y - self.region.y
        if row <= 0 or self._rows_scroll is None or row > self._rows_scroll.size.height:
            logger.debug("Agent bar click phase=chrome row=%d", row)
            return
        index = row - 1 + int(self._rows_scroll.scroll_y)
        ids = (None, *(agent.agent_id for agent in self.agents))
        if index < len(ids):
            logger.debug(
                "Agent bar click phase=row-select-start row=%d target=%s",
                row,
                ids[row - 1],
            )
            self._selected_agent_id = ids[index]
            self._render_agents()
            self.action_select()
            logger.debug(
                "Agent bar click phase=row-select-posted row=%d target=%s",
                row,
                self._selected_agent_id,
            )

    def _move_selection(self, offset: int) -> None:
        ids = (None, *(agent.agent_id for agent in self.agents))
        index = (
            ids.index(self._selected_agent_id) if self._selected_agent_id in ids else 0
        )
        self._selected_agent_id = ids[max(0, min(index + offset, len(ids) - 1))]
        self._render_agents()
        self._scroll_selected_row_into_view()

    def _scroll_selected_row_into_view(self) -> None:
        ids = (None, *(agent.agent_id for agent in self.agents))
        if self._selected_agent_id not in ids:
            return
        row = ids.index(self._selected_agent_id)
        scroll = self._rows_scroll
        if scroll is None:
            return
        top = int(scroll.scroll_y)
        height = scroll.size.height
        if row < top:
            scroll.scroll_to(y=row, animate=False, force=True, immediate=True)
        elif height and row >= top + height:
            scroll.scroll_to(
                y=row - height + 1, animate=False, force=True, immediate=True
            )

    def _update_selection(self, ids: tuple[str, ...]) -> None:
        if self._selected_agent_id is None or self._selected_agent_id in ids:
            return
        index = (
            self._previous_agent_ids.index(self._selected_agent_id)
            if self._selected_agent_id in self._previous_agent_ids
            else 0
        )
        self._selected_agent_id = ids[min(index, len(ids) - 1)] if ids else None

    def _render_agents(self) -> None:
        self.display = (
            bool(self.agents)
            and getattr(self.app, "_agent_transcript_viewer", None) is None
        )
        if self._rows_scroll is not None:
            self._rows_scroll.display = self._expanded and not self._show_full_details
        if self._details_scroll is not None:
            self._details_scroll.display = self._expanded and self._show_full_details
        if self._detail is not None:
            self._detail.display = self._expanded and not self._show_full_details
        if self._footer is not None:
            self._footer.display = self._expanded
        if not self.agents:
            self._expanded = False
            self.set_class(False, "-expanded")
            self._update_content("")
            return
        if not self._expanded:
            counts = Counter(agent_state(agent) for agent in self.agents)
            parts = [f"{count} {state}" for state, count in counts.items()]
            spinner = f"{self._spinner.current_frame()} " if counts["running"] else ""
            summary = f"{spinner}{len(self.agents)} agents: {' · '.join(parts)}"
            if self.is_mounted:
                self.query_one("#agent-bar-title", NoMarkupStatic).update(summary)
            self._update_content(summary)
            return
        if self.is_mounted:
            self.query_one("#agent-bar-title", NoMarkupStatic).update(
                "Background agents"
            )
        lines = [self._row(None, "Main agent")]
        lines.extend(
            self._row(
                agent.agent_id,
                f"{agent.agent_id} · {agent.profile}",
                state=agent_state(agent),
            )
            for agent in self.agents
        )
        selected = next(
            (
                agent
                for agent in self.agents
                if agent.agent_id == self._selected_agent_id
            ),
            None,
        )
        detail = self._agent_details(selected) if selected else "Return to conversation"
        if self._detail is not None:
            self._detail.update(detail)
        if self._details_content is not None:
            self._details_content.update(
                self.full_metadata(selected) if selected else ""
            )
        if self._footer is not None:
            self._footer.update(
                "Up/Down Scroll  Esc Back"
                if self._show_full_details
                else f"{chrome_glyph('vertical')} Move  Enter Open  D Details  Esc Close"
            )
        content = Content("\n".join(lines))
        if self.has_focus:
            index = next(
                (
                    i + 1
                    for i, agent in enumerate(self.agents)
                    if agent.agent_id == self._selected_agent_id
                ),
                0,
            )
            start = sum(len(line) + 1 for line in lines[:index])
            content = content.stylize("bold reverse", start, start + len(lines[index]))
        self._update_content(content)

    def _update_content(self, content: str | Content) -> None:
        self._rendered = str(content)
        if self._content is not None and (
            content != self._last_rendered_content
            or self.has_focus != self._last_rendered_focused
        ):
            self._content.update(content)
            self._last_rendered_content = content
            self._last_rendered_focused = self.has_focus

    def _row(self, agent_id: str | None, details: str, *, state: str = "") -> str:
        prefix = (
            f"{chrome_glyph('cursor')} "
            if agent_id == self._selected_agent_id
            else "  "
        )
        width = max(
            1, (self._content.size.width if self._content else 0) or self.size.width
        )
        suffix = f" · {state}" if state else ""
        truncation = chrome_glyph("truncation")
        available = width - cell_len(prefix) - cell_len(suffix)
        if cell_len(details) > available:
            if available > cell_len(truncation):
                details = (
                    chop_cells(details, available - cell_len(truncation))[0]
                    + truncation
                )
            else:
                details = ""
        row = f"{prefix}{details}{suffix}"
        if cell_len(row) > width:
            row = chop_cells(row, width)[0]
        return row + " " * max(0, width - cell_len(row))

    def _agent_details(self, agent: AgentSummaryModel) -> str:
        state = agent_state(agent)
        glyph = {
            "running": chrome_glyph("running"),
            "idle": chrome_glyph("success"),
            "failed": chrome_glyph("error"),
            "cancelled": chrome_glyph("warning"),
            "evicted": chrome_glyph("warning"),
            "unknown": chrome_glyph("information"),
        }[state]
        model = agent_model_display_name(agent) or "unknown"
        turns = "—" if agent.turns_used is None else str(agent.turns_used)
        details = f"{agent.agent_id} · {agent.profile} · {glyph} {state}"
        if state == "evicted":
            eviction = self._evictions.get(agent.agent_id)
            details += (
                " · result expired" if agent.result_expired else " · result preserved"
            )
            if eviction is not None:
                details += f" · evicted: {eviction.reason} ({_format_seconds(eviction.idle_duration_seconds)} idle)"
        if state == "failed":
            details += f" · run status {agent.current_run_status or agent.last_run_status or 'failed'}"
        details += f" · {model} · turns {turns} · run {agent.current_run_id or '—'}"
        if agent.idle_seconds is not None:
            details += f" · idle {_format_seconds(agent.idle_seconds)}"
        if agent.ttl_remaining_seconds is not None:
            details += f" · TTL {_format_seconds(agent.ttl_remaining_seconds)}"
        return details

    def full_metadata(self, agent: AgentSummaryModel) -> str:
        """Unclipped agent summary for the scrollable transcript inspection."""
        fields = [
            ("Identity", agent.agent_id),
            ("Profile", agent.profile),
            ("Availability", agent.availability),
            ("Current run status", agent.current_run_status),
            ("Last run status", agent.last_run_status),
            ("Result expired", str(agent.result_expired)),
            ("Initial task", agent.initial_task_summary),
            ("Current task", agent.current_task_summary),
            ("Provider", agent.active_provider),
            ("Model", agent.effective_model),
            ("Base model", agent.base_model),
            ("Thinking", agent.effective_thinking),
            ("Turns", str(agent.turns_used) if agent.turns_used is not None else None),
            ("Run ID", agent.current_run_id),
            (
                "Idle seconds",
                str(agent.idle_seconds) if agent.idle_seconds is not None else None,
            ),
            (
                "TTL remaining seconds",
                str(agent.ttl_remaining_seconds)
                if agent.ttl_remaining_seconds is not None
                else None,
            ),
        ]
        eviction = self._evictions.get(agent.agent_id)
        if eviction is not None:
            fields.extend([
                ("Eviction reason", eviction.reason),
                ("Eviction idle seconds", str(eviction.idle_duration_seconds)),
                ("Eviction run ID", eviction.run_id),
                ("Eviction root generation", str(eviction.root_generation)),
            ])
        return "\n".join(
            f"{name}: {value}" for name, value in fields if value is not None
        )

    def _update_spinner_timer(self) -> None:
        running = any(agent_state(agent) == "running" for agent in self.agents)
        if running and self._spinner_timer is None:
            self._spinner_timer = self.set_interval(0.1, self._advance_spinner)
        elif not running and self._spinner_timer is not None:
            self._spinner_timer.stop()
            self._spinner_timer = None

    def _advance_spinner(self) -> None:
        if any(agent_state(agent) == "running" for agent in self.agents):
            self._spinner.next_frame()
            self._render_agents()
