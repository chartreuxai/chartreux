from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from textual.binding import Binding, BindingType
from textual.message import Message
from textual.reactive import reactive

from chartreux.app_server.protocol import AgentEvictionModel, AgentSummaryModel
from chartreux.cli.textual_ui.widgets.agent_bar import (
    agent_model_display_name,
    agent_state,
)
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def _format_seconds(seconds: float) -> str:
    return f"{seconds:g}s"


def _compact(value: str, limit: int = 48) -> str:
    return value if len(value) <= limit else f"{value[: limit - 1]}…"


class AgentSidebar(NoMarkupStatic):
    """Docked, selectable detail view for background-agent summaries."""

    DEFAULT_CSS = """
    AgentSidebar:focus {
        border-left: solid $accent;
    }
    """

    can_focus = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Previous agent", show=False),
        Binding("down", "cursor_down", "Next agent", show=False),
        Binding("enter", "open_transcript", "Open transcript", show=False),
    ]

    agents: reactive[tuple[AgentSummaryModel, ...]] = reactive(())

    class TranscriptOpen(Message):
        """Request that the parent open an agent transcript by its stable ID."""

        def __init__(self, agent_id: str) -> None:
            self.agent_id = agent_id
            super().__init__()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="agent-sidebar", **kwargs)
        self._evictions: dict[str, AgentEvictionModel] = {}
        self._selected_agent_id: str | None = None
        self._previous_agent_ids: tuple[str, ...] = ()
        self._render_agents()

    @property
    def selected_agent_id(self) -> str | None:
        """The selected agent's stable ID, or ``None`` when the list is empty."""
        return self._selected_agent_id

    def on_mount(self) -> None:
        self._render_agents()
        self.focus_selection()

    def watch_agents(self) -> None:
        self._render_agents()

    def focus_selection(self) -> None:
        """Focus the sidebar's row-selection control without submitting input."""
        self.focus()

    def update_agents(
        self,
        agents: Sequence[AgentSummaryModel],
        evictions: Sequence[AgentEvictionModel] = (),
    ) -> None:
        selectable_agents = tuple(
            agent for agent in agents if agent.availability.lower() != "released"
        )
        agent_ids = tuple(agent.agent_id for agent in selectable_agents)
        self._evictions.update({eviction.agent_id: eviction for eviction in evictions})
        evicted_agent_ids = {
            agent.agent_id
            for agent in selectable_agents
            if agent.availability.lower() == "evicted"
        }
        self._evictions = {
            agent_id: eviction
            for agent_id, eviction in self._evictions.items()
            if agent_id in evicted_agent_ids
        }
        self._update_selection(agent_ids)
        self._previous_agent_ids = agent_ids
        self.agents = selectable_agents

    def action_cursor_up(self) -> None:
        self._move_selection(-1)

    def action_cursor_down(self) -> None:
        self._move_selection(1)

    def action_open_transcript(self) -> None:
        if self._selected_agent_id is not None:
            self.post_message(self.TranscriptOpen(self._selected_agent_id))

    def _move_selection(self, offset: int) -> None:
        agent_ids = tuple(agent.agent_id for agent in self.agents)
        if not agent_ids:
            return
        try:
            index = agent_ids.index(self._selected_agent_id)
        except ValueError:
            index = 0
        self._selected_agent_id = agent_ids[
            max(0, min(index + offset, len(agent_ids) - 1))
        ]
        self._render_agents()

    def _update_selection(self, agent_ids: tuple[str, ...]) -> None:
        if self._selected_agent_id in agent_ids:
            return
        if not agent_ids:
            self._selected_agent_id = None
            return
        if self._selected_agent_id is None:
            self._selected_agent_id = agent_ids[0]
            return
        try:
            previous_index = self._previous_agent_ids.index(self._selected_agent_id)
        except ValueError:
            previous_index = 0
        self._selected_agent_id = agent_ids[min(previous_index, len(agent_ids) - 1)]

    def _render_agents(self) -> None:
        lines = ["Background Agents", ""]
        if not self.agents:
            lines.append("No background agents")
        else:
            for index, agent in enumerate(self.agents):
                if index:
                    lines.append("")
                marker = "› " if agent.agent_id == self._selected_agent_id else "  "
                lines.extend((
                    f"{marker}{agent.agent_id}",
                    f"Profile: {agent.profile}",
                    f"Availability: {agent.availability}",
                    f"Run: {agent.current_run_id or '—'}",
                    f"Status: {agent.current_run_status or agent_state(agent)}",
                ))
                if agent.initial_task_summary is not None:
                    lines.append(f"Initial task: {agent.initial_task_summary}")
                if agent.current_task_summary is not None:
                    lines.append(f"Current task: {agent.current_task_summary}")
                model = agent_model_display_name(agent)
                if model == "unknown":
                    model = agent.effective_model
                if model is not None:
                    lines.append(f"Model: {_compact(model)}")
                if agent.effective_thinking is not None:
                    lines.append(f"Thinking: {agent.effective_thinking}")
                if agent.idle_seconds is not None:
                    lines.append(f"Idle: {_format_seconds(agent.idle_seconds)}")
                if agent.ttl_remaining_seconds is not None:
                    lines.append(
                        "TTL remaining (advisory): "
                        f"{_format_seconds(agent.ttl_remaining_seconds)}"
                    )
                eviction = self._evictions.get(agent.agent_id)
                if agent_state(agent) == "evicted":
                    lines.append(
                        "Evicted: result expired"
                        if agent.result_expired
                        else "Evicted: result preserved"
                    )
                    if eviction is not None:
                        lines.append(
                            f"Eviction reason: {eviction.reason} "
                            f"({_format_seconds(eviction.idle_duration_seconds)} idle)"
                        )
        self.update("\n".join(lines))
