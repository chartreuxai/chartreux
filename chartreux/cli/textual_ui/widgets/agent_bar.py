from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from textual import events
from textual.reactive import reactive
from textual.timer import Timer

from chartreux.app_server.protocol import AgentSummaryModel
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


def _truncate_chip(chip: str, width: int) -> str:
    if width <= 0 or len(chip) <= width:
        return chip
    if width == 1:
        return "…"
    return f"{chip[: width - 1]}…"


def pack_agent_chips(
    chips: Sequence[str], width: int, max_lines: int | None = None
) -> list[str]:
    """Pack whole agent chips without hiding retained-agent identities."""
    if not chips:
        return []
    if width <= 0:
        return ["  ".join(chips)]
    chips = tuple(_truncate_chip(chip, width) for chip in chips)

    def pack(items: Sequence[str]) -> list[str]:
        lines: list[str] = []
        for chip in items:
            if not lines or len(lines[-1]) + 2 + len(chip) > width:
                lines.append(chip)
            else:
                lines[-1] = f"{lines[-1]}  {chip}"
        return lines

    lines = pack(chips)
    if max_lines is None or len(lines) <= max_lines:
        return lines

    for visible_count in range(len(chips) - 1, -1, -1):
        hidden_count = len(chips) - visible_count
        lines = pack((*chips[:visible_count], f"+{hidden_count} more"))
        if len(lines) <= max_lines:
            return lines

    raise AssertionError("the overflow chip must fit on one line")


class AgentBar(NoMarkupStatic):
    """Wrapped summary of retained background agents."""

    agents: reactive[tuple[AgentSummaryModel, ...]] = reactive(())

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="agent-bar", **kwargs)
        self._spinner = create_spinner()
        self._spinner_timer: Timer | None = None

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

    def update_agents(self, agents: Sequence[AgentSummaryModel]) -> None:
        self.agents = tuple(agents)

    def on_resize(self, event: events.Resize) -> None:
        """Repack chips when the available terminal width changes."""
        self._render_agents()

    def _render_agents(self) -> None:
        self.display = bool(self.agents)
        if not self.agents:
            self.update("")
            return
        glyphs = {
            "running": self._spinner.current_frame(),
            "idle": "✓",
            "failed": "✗",
            "cancelled": "⊘",
            "evicted": "⊘",
            "unknown": "?",
        }
        chips = []
        for agent in self.agents:
            state = agent_state(agent)
            suffix = " evicted" if state == "evicted" else ""
            model = agent_model_display_name(agent)
            if model == "unknown":
                model = agent.effective_model
            details = (
                f" [{agent.effective_thinking}; {model}]"
                if agent.effective_thinking and model
                else (
                    f" [{agent.effective_thinking}]"
                    if agent.effective_thinking
                    else f" [{model}]"
                    if model
                    else ""
                )
            )
            chips.append(
                f"{glyphs[state]} {agent.agent_id}·{agent.profile}{details}{suffix}"
            )
        self.update("\n".join(pack_agent_chips(chips, self.size.width, max_lines=3)))

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
