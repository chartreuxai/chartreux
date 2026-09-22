from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any, ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.timer import Timer

from chartreux.app_server.protocol import AgentEvictionModel, AgentSummaryModel
from chartreux.cli.textual_ui.widgets.spinner import create_spinner
from chartreux.model_display import format_model_display_name
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


class AgentBar(VerticalScroll):
    """Clickable status line with an in-place, keyboard-navigable agent browser."""

    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Previous agent", show=False),
        Binding("down", "cursor_down", "Next agent", show=False),
        Binding("enter", "select", "Select agent", show=False),
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
        self._selected_agent_id: str | None = None
        self._evictions: dict[str, AgentEvictionModel] = {}
        self._previous_agent_ids: tuple[str, ...] = ()
        self._content: NoMarkupStatic | None = None
        self._rendered = ""

    def compose(self) -> ComposeResult:
        self._content = NoMarkupStatic(id="agent-bar-content")
        yield self._content

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
        self.set_class(False, "-expanded")
        self._render_agents()

    def action_cursor_up(self) -> None:
        self._move_selection(-1)

    def action_cursor_down(self) -> None:
        self._move_selection(1)

    def action_select(self) -> None:
        self.post_message(self.SelectionRequested(self._selected_agent_id))

    def action_collapse(self) -> None:
        self.close_browser()
        self.post_message(self.SelectionRequested(None))

    def on_click(self, event: events.Click) -> None:
        if not self._expanded:
            self.open_browser()
            return
        row = event.y
        if row <= 0:
            return
        ids = (None, *(agent.agent_id for agent in self.agents))
        if row - 1 < len(ids):
            self._selected_agent_id = ids[row - 1]
            self._render_agents()
            self.action_select()

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
        row = 1 + ids.index(self._selected_agent_id)
        top = int(self.scroll_y)
        height = self.size.height
        if row < top:
            self.scroll_to(y=row, animate=False, force=True, immediate=True)
        elif height and row >= top + height:
            self.scroll_to(
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
        self.display = bool(self.agents)
        if not self.agents:
            self._expanded = False
            self.set_class(False, "-expanded")
            self._update_content("")
            return
        if not self._expanded:
            counts = Counter(agent_state(agent) for agent in self.agents)
            parts = [f"{count} {state}" for state, count in counts.items()]
            spinner = f"{self._spinner.current_frame()} " if counts["running"] else ""
            self._update_content(
                f"{spinner}{len(self.agents)} agents: {' · '.join(parts)}"
            )
            return
        lines = ["Background Agents  [↑↓ select · Enter open · Esc close]"]
        lines.append(self._row(None, "Main agent — return to conversation"))
        lines.extend(
            self._row(agent.agent_id, self._agent_details(agent))
            for agent in self.agents
        )
        self._update_content("\n".join(lines))

    def _update_content(self, content: str) -> None:
        self._rendered = content
        if self._content is not None:
            self._content.update(content)

    def _row(self, agent_id: str | None, details: str) -> str:
        return f"{'›' if agent_id == self._selected_agent_id else ' '} {details}"

    def _agent_details(self, agent: AgentSummaryModel) -> str:
        state = agent_state(agent)
        glyph = {
            "running": self._spinner.current_frame(),
            "idle": "✓",
            "failed": "✗",
            "cancelled": "⊘",
            "evicted": "⊘",
            "unknown": "?",
        }[state]
        model = agent_model_display_name(agent) or "unknown"
        turns = "—" if agent.turns_used is None else str(agent.turns_used)
        details = f"{agent.agent_id} · {agent.profile} · {glyph} {state} · {model} · turns {turns} · run {agent.current_run_id or '—'}"
        if agent.idle_seconds is not None:
            details += f" · idle {_format_seconds(agent.idle_seconds)}"
        if agent.ttl_remaining_seconds is not None:
            details += f" · TTL {_format_seconds(agent.ttl_remaining_seconds)}"
        if state == "evicted":
            eviction = self._evictions.get(agent.agent_id)
            details += (
                " · result expired" if agent.result_expired else " · result preserved"
            )
            if eviction is not None:
                details += f" · evicted: {eviction.reason} ({_format_seconds(eviction.idle_duration_seconds)} idle)"
        return details

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
