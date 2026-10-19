from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import time
from typing import Any, ClassVar, Literal

from rich.cells import cell_len, chop_cells
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content, Span
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button

from chartreux.app_server.protocol import (
    AgentEvictionModel,
    AgentsCancelResponse,
    AgentSummaryModel,
    CancelOutcome,
)
from chartreux.model_display import format_model_display_name
from chartreux.observability.logging import logger
from chartreux.ui.chrome_glyphs import ascii_chrome_enabled, chrome_glyph
from chartreux.ui.context_display import format_context
from chartreux.ui.duration_display import format_duration
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


def _run_outcome(agent: AgentSummaryModel) -> str | None:
    status = (agent.current_run_status or agent.last_run_status or "").lower()
    if agent.stop_reason in {"budget_exceeded", "budget_unverifiable"}:
        return "budget-stopped"
    if agent.stop_reason in {"user_cancelled", "orchestrator_cancelled", "retasked"}:
        return "cancelled"
    if agent.stop_reason == "error" or status in {"failed", "error"}:
        return "failed"
    if status in {"cancelled", "canceled"}:
        return "cancelled"
    if agent.availability.lower() in {"failed", "error"}:
        return "failed"
    return None


def agent_state(agent: AgentSummaryModel) -> str:  # noqa: PLR0911 - state matrix
    """Presentation state; finalization is not active generation."""
    availability = agent.availability.lower()
    if availability == "evicted":
        return "evicted"
    if availability == "closed":
        return _run_outcome(agent) or "closed"
    if availability == "finalizing":
        return "finishing"
    if availability in {"suspended", "restoring"}:
        # WP2: a suspended agent is held warm after a recoverable failure
        # (its failed run is NOT the presentation state); a restoring agent
        # has an admitted successor starting. Neither is an idle default.
        return availability
    outcome = _run_outcome(agent)
    if outcome is not None:
        return outcome
    status = (agent.current_run_status or agent.last_run_status or "").lower()
    if availability == "running" or status in {"running", "in_progress", "in-progress"}:
        return "compacting" if agent.compacting else "running"
    return "idle"


def agent_is_active(agent: AgentSummaryModel) -> bool:
    """Shared UI predicate for output polling and active-agent chrome."""
    return agent_state(agent) in {"running", "compacting", "restoring"}


# Lifecycle groups (design §8). Grouping and counting key on availability,
# never on agent_state, which mixes availability with historical outcomes.
GROUP_ACTIVE = "active"
GROUP_IDLE = "idle"
GROUP_RECENT = "recent"

_FilterKey = Literal["resident", "active", "idle", "recent", "all"]
_FILTER_ORDER: tuple[_FilterKey, ...] = ("resident", "active", "idle", "recent", "all")
_FILTER_LABELS: dict[_FilterKey, str] = {
    "resident": "Resident",
    "active": "Active",
    "idle": "Idle",
    "recent": "Recent",
    "all": "All",
}
_GROUP_LABELS = {
    GROUP_ACTIVE: "Active",
    GROUP_IDLE: "Retained Idle",
    GROUP_RECENT: "Recent",
}
_FILTER_EMPTY_NOTES = {
    "resident": "No resident agents",
    "active": "No active agents",
    "idle": "No retained idle agents",
    "recent": "No recent agents",
    "all": "No agents",
}


def agent_group(agent: AgentSummaryModel) -> str:
    """The lifecycle group of an agent: active work, retained idle, or recent."""
    availability = agent.availability.lower()
    if availability in {"running", "finalizing", "suspended", "restoring"}:
        # WP2: suspended/restoring agents are resident with recovery work
        # pending or admitted — never the retained-idle default.
        return GROUP_ACTIVE
    if availability in {"closed", "evicted"}:
        return GROUP_RECENT
    return GROUP_IDLE


@dataclass(frozen=True, slots=True)
class _Row:
    """One rendered browser row in the shared visible-row mapping."""

    kind: str  # "main" | "note" | "header" | "agent"
    agent_id: str | None
    text: str
    dim: tuple[int, int] | None = None


def _separator() -> str:
    return " | " if ascii_chrome_enabled() else " · "


def _context_text(
    tokens: int | None,
    threshold: int | None,
    *,
    compacting: bool = False,
    last_recorded: bool = False,
) -> str:
    text = format_context(
        tokens, threshold, compacting=compacting, last_recorded=last_recorded
    )
    return text.replace("—", "-") if ascii_chrome_enabled() else text


def _state_label(state: str, outcome: str | None = None) -> str:
    labels = {
        "running": "Running",
        "stopping": "Stopping",
        "compacting": f"{chrome_glyph('running')} Compacting",
        "finishing": f"{chrome_glyph('running')} Finishing",
        "idle": "Idle",
        "suspended": f"{chrome_glyph('warning')} Suspended",
        "restoring": f"{chrome_glyph('running')} Restoring",
        "failed": f"{chrome_glyph('error')} Failed",
        "cancelled": "! Cancelled",
        "budget-stopped": "! Budget stopped",
        "evicted": "Evicted",
        "closed": "Closed",
    }
    text = labels[state]
    if state == "finishing" and outcome:
        text += _separator() + labels[outcome].split(" ", 1)[1]
    return text


def _clip(text: str, width: int) -> str:
    """Ellipsize by terminal cells, never splitting a wide glyph."""
    text = " ".join(text.splitlines())
    if cell_len(text) <= width:
        return text
    marker = chrome_glyph("truncation")
    if width <= cell_len(marker):
        return (
            marker
            if width == cell_len(marker)
            else (chop_cells(text, width)[0] if width > 0 else "")
        )
    return chop_cells(text, width - cell_len(marker))[0] + marker


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


def _format_seconds(seconds: float | None) -> str:
    return format_duration(seconds * 1000 if seconds is not None else None) or "unknown"


class AgentBar(Vertical):  # noqa: PLR0904 -- Textual actions and event handlers
    """Clickable status line with an in-place, keyboard-navigable agent browser."""

    DEFAULT_CSS = """
    AgentBar {
        width: 100%; height: 1; max-height: 11; color: $text-muted;
    }
    AgentBar:focus { color: $foreground; }
    AgentBar.-expanded {
        height: 11; border: solid $primary; border-title-color: $foreground;
        border-title-style: bold;
    }
    AgentBar .agent-sheet-title {
        width: 100%; height: 1; text-wrap: nowrap;
    }
    AgentBar .agent-sheet-rows {
        width: 100%; height: 5; overflow-y: auto;
    }
    AgentBar .agent-sheet-content { width: 100%; height: auto; text-wrap: nowrap; }
    AgentBar .agent-sheet-detail {
        width: 100%; height: 2; text-wrap: nowrap;
    }
    AgentBar .agent-stop-confirmation { width: 100%; height: 2; }
    AgentBar .agent-stop-prompt-scroll { width: 100%; height: 1; }
    AgentBar .agent-stop-prompt { width: 100%; height: auto; text-wrap: wrap; }
    AgentBar .agent-stop-actions { width: 100%; height: 1; }
    AgentBar .agent-stop-actions Button {
        /* ANSI and hover Button borders must not consume the single label row. */
        width: auto; min-width: 0; height: 1; border: none !important;
        background: $surface; color: $foreground; padding: 0 1; margin-right: 1;
    }
    AgentBar .agent-stop-actions Button:focus { text-style: bold reverse; }
    AgentBar #agent-stop-submit { color: $error; }
    AgentBar .agent-release-confirmation { width: 100%; height: 2; }
    AgentBar .agent-release-prompt-scroll { width: 100%; height: 1; }
    AgentBar .agent-release-prompt { width: 100%; height: auto; text-wrap: wrap; }
    AgentBar .agent-release-actions { width: 100%; height: 1; }
    AgentBar .agent-release-actions Button {
        width: auto; min-width: 0; height: 1; border: none !important;
        background: $surface; color: $foreground; padding: 0 1; margin-right: 1;
    }
    AgentBar .agent-release-actions Button:focus { text-style: bold reverse; }
    AgentBar #agent-release-submit { color: $error; }
    AgentBar .agent-sheet-footer {
        width: 100%; height: 1; text-wrap: nowrap;
    }
    AgentBar .agent-sheet-full { width: 100%; height: 7; overflow-y: auto; }
    AgentBar .agent-sheet-full-content { width: 100%; height: auto; text-wrap: wrap; }
    """
    can_focus = True
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Previous agent", show=False),
        Binding("down", "cursor_down", "Next agent", show=False),
        Binding("enter", "select", "Select agent", show=False),
        Binding("c", "stop_run", "Stop run", show=False),
        Binding("r", "release", "Release agent", show=False),
        Binding(
            "left,right,tab,shift+tab",
            "switch_action",
            "Switch action",
            show=False,
            priority=True,
        ),
        Binding("left", "filter_prev", "Previous filter", show=False),
        Binding("right", "filter_next", "Next filter", show=False),
        Binding("pageup", "stop_scroll(-1)", "Scroll confirmation", show=False),
        Binding("pagedown", "stop_scroll(1)", "Scroll confirmation", show=False),
        Binding("d", "details", "Full agent details", show=False),
        Binding("f1", "help", "Agent browser help", show=False, priority=True),
        Binding("escape", "collapse", "Close agents", show=False, priority=True),
    ]
    agents: reactive[tuple[AgentSummaryModel, ...]] = reactive(())

    class SelectionRequested(Message):
        """Request opening an agent transcript, or returning to the main pane."""

        def __init__(self, agent_id: str | None) -> None:
            self.agent_id = agent_id
            super().__init__()

    class StopRequested(Message):
        """A user-confirmed, run-pinned stop (never root interruption)."""

        def __init__(self, agent_id: str, run_id: str) -> None:
            self.agent_id = agent_id
            self.run_id = run_id
            super().__init__()

    class ReleaseRequested(Message):
        """A user-confirmed, run-pinned release of an idle or recent agent."""

        def __init__(self, agent_id: str, expected_run_id: str | None) -> None:
            self.agent_id = agent_id
            self.expected_run_id = expected_run_id
            super().__init__()

    class Closed(Message):
        """Browser closed without requesting output (restore its opener)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(id="agent-bar", **kwargs)
        self.border_title = "Background agents"
        self._show_help = False
        self._main_metadata: dict[str, object] = {}
        self._main_context_tokens: int | None = None
        self._main_context_threshold: int | None = None
        self._main_compacting = False
        self._expanded = False
        self._show_full_details = False
        self._details_scroll: VerticalScroll | None = None
        self._details_content: NoMarkupStatic | None = None
        self._selected_agent_id: str | None = None
        self._stop_confirmation: tuple[str, str] | None = None
        self._release_confirmation: tuple[str, str | None] | None = None
        self._filter: _FilterKey = "resident"
        self._pending_stops: set[tuple[str, str]] = set()
        self._evictions: dict[str, AgentEvictionModel] = {}
        # Samples and their receipt anchors are installed before reactive rendering.
        self._duration_samples: dict[str, tuple[AgentSummaryModel, float]] = {}
        self._previous_agent_ids: tuple[str | None, ...] = ()
        self._content: NoMarkupStatic | None = None
        self._rows_scroll: VerticalScroll | None = None
        self._detail: NoMarkupStatic | None = None
        self._footer: NoMarkupStatic | None = None
        self._rendered = ""
        self._last_rendered_content: str | Content | None = None
        self._last_rendered_focused: bool | None = None

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic(id="agent-bar-title", classes="agent-sheet-title")
        self._rows_scroll = VerticalScroll(
            id="agent-bar-rows", classes="agent-sheet-rows"
        )
        self._rows_scroll.can_focus = False
        with self._rows_scroll:
            self._content = NoMarkupStatic(
                id="agent-bar-content", classes="agent-sheet-content"
            )
            yield self._content
        self._detail = NoMarkupStatic(
            id="agent-bar-detail", classes="agent-sheet-detail"
        )
        yield self._detail
        with Vertical(classes="agent-stop-confirmation", id="agent-stop-confirmation"):
            with VerticalScroll(
                classes="agent-stop-prompt-scroll", id="agent-stop-scroll"
            ) as scroll:
                scroll.can_focus = False
                yield NoMarkupStatic(
                    id="agent-stop-prompt", classes="agent-stop-prompt"
                )
            with Horizontal(classes="agent-stop-actions"):
                yield Button(Content("[Stop run]"), id="agent-stop-submit")
                yield Button(Content("[Cancel]"), id="agent-stop-cancel")
        with Vertical(
            classes="agent-release-confirmation", id="agent-release-confirmation"
        ):
            with VerticalScroll(
                classes="agent-release-prompt-scroll", id="agent-release-scroll"
            ) as scroll:
                scroll.can_focus = False
                yield NoMarkupStatic(
                    id="agent-release-prompt", classes="agent-release-prompt"
                )
            with Horizontal(classes="agent-release-actions"):
                yield Button(Content("[Release]"), id="agent-release-submit")
                yield Button(Content("[Cancel]"), id="agent-release-cancel")
        self._details_scroll = VerticalScroll(
            id="agent-bar-full-details", classes="agent-sheet-full"
        )
        with self._details_scroll:
            self._details_content = NoMarkupStatic(
                id="agent-bar-full-content", classes="agent-sheet-full-content"
            )
            yield self._details_content
        self._footer = NoMarkupStatic(
            id="agent-bar-footer", classes="agent-sheet-footer"
        )
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

    def on_focus(self) -> None:
        self.call_after_refresh(self._render_agents)

    def on_blur(self) -> None:
        self.call_after_refresh(self._render_agents)

    def on_resize(self) -> None:
        self._render_agents()

    def watch_agents(self) -> None:
        self._render_agents()

    def update_main_details(
        self,
        metadata: Mapping[str, object],
        *,
        context_tokens: int | None = None,
        auto_compact_threshold: int | None = None,
        compacting: bool = False,
    ) -> None:
        """Accept public root-session metadata, not a synthetic registry entry.

        The caller supplies display names mapped to public session/config/run
        values. Root identity remains None for output activation.
        """
        self._main_metadata = dict(metadata)
        self._main_context_tokens = context_tokens
        self._main_context_threshold = auto_compact_threshold
        self._main_compacting = compacting
        self._render_agents()

    def update_agents(
        self,
        agents: Sequence[AgentSummaryModel],
        evictions: Sequence[AgentEvictionModel] = (),
        *,
        received_at: float | None = None,
    ) -> None:
        receipt = time.monotonic() if received_at is None else received_at
        # The published inventory is authoritative: ordering is consumed from
        # the server and released handles are simply absent (design §8).
        incoming = tuple(agents)
        self._evictions.update({eviction.agent_id: eviction for eviction in evictions})
        evicted_ids = {
            agent.agent_id
            for agent in incoming
            if agent.availability.lower() == "evicted"
        }
        self._evictions = {
            agent_id: eviction
            for agent_id, eviction in self._evictions.items()
            if agent_id in evicted_ids
        }
        self._refresh_selection(incoming)
        self._duration_samples = {
            agent.agent_id: (agent, receipt) for agent in incoming
        }
        self.agents = incoming
        # Updates are authoritative. Clearing a target also fences late RPC replies.
        self._pending_stops.intersection_update(
            (agent.agent_id, run_id)
            for agent in incoming
            if agent_is_active(agent) or agent.availability.lower() == "finalizing"
            if (run_id := agent.current_run_id or agent.latest_run_id) is not None
        )
        if self._stop_confirmation is not None:
            target = self._stop_confirmation
            if not any(
                agent.agent_id == target[0]
                and agent.current_run_id == target[1]
                and agent_is_active(agent)
                for agent in incoming
            ):
                self.dismiss_stop_confirmation()
        if self._release_confirmation is not None:
            target = self._release_confirmation
            if not any(
                agent.agent_id == target[0]
                and (agent.current_run_id or agent.latest_run_id) == target[1]
                and agent_group(agent) in {GROUP_IDLE, GROUP_RECENT}
                for agent in incoming
            ):
                self.dismiss_release_confirmation()
        self._render_agents()
        if self._expanded and self.is_mounted:
            self.call_after_refresh(self._scroll_selected_row_into_view)

    def toggle(self) -> None:
        self.close_browser() if self._expanded else self.open_browser()

    def open_browser(self) -> None:
        # The expanded browser is independent of the collapsed segment: it is
        # reachable even while the inventory is empty (design §8).
        self._expanded = True
        self.set_class(True, "-expanded")
        self._render_agents()
        if self.is_mounted:
            self.focus()

    def close_browser(self) -> None:
        self._stop_confirmation = None
        self._release_confirmation = None
        self._expanded = False
        self._show_full_details = False
        self._show_help = False
        self.set_class(False, "-expanded")
        self._render_agents()

    def _confirmation_open(self) -> bool:
        return (
            self._stop_confirmation is not None
            or self._release_confirmation is not None
        )

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"switch_action", "stop_scroll"}:
            return self._confirmation_open()
        if action == "stop_run":
            return self._stop_target() is not None and not self._confirmation_open()
        if action == "release":
            return self._release_target() is not None and not self._confirmation_open()
        if action in {"filter_prev", "filter_next"}:
            return (
                self._expanded
                and not self._show_full_details
                and not self._show_help
                and not self._confirmation_open()
                and bool(self.agents)
            )
        if self._confirmation_open() and action in {"select", "details", "help"}:
            return False
        return True

    def _stop_target(self) -> tuple[str, str] | None:
        if not self._expanded or self._show_full_details or self._show_help:
            return None
        for agent in self.agents:
            if (
                agent.agent_id == self._selected_agent_id
                and agent.current_run_id
                and agent_is_active(agent)
            ):
                target = (agent.agent_id, agent.current_run_id)
                return target if target not in self._pending_stops else None
        return None

    def _release_target(self) -> tuple[str, str | None] | None:
        """The selected idle/recent agent pinned to its latest run identity."""
        if not self._expanded or self._show_full_details or self._show_help:
            return None
        for agent in self.agents:
            if agent.agent_id == self._selected_agent_id and agent_group(agent) in {
                GROUP_IDLE,
                GROUP_RECENT,
            }:
                return (agent.agent_id, agent.current_run_id or agent.latest_run_id)
        return None

    def action_stop_run(self) -> None:
        if self._confirmation_open() or (target := self._stop_target()) is None:
            return
        self._stop_confirmation = target
        self._render_agents()
        self.query_one("#agent-stop-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#agent-stop-cancel", Button).focus(scroll_visible=False)

    def action_release(self) -> None:
        if self._confirmation_open() or (target := self._release_target()) is None:
            return
        self._release_confirmation = target
        self._render_agents()
        self.query_one("#agent-release-scroll", VerticalScroll).scroll_home(
            animate=False
        )
        self.query_one("#agent-release-cancel", Button).focus(scroll_visible=False)

    def action_switch_action(self) -> None:
        if self._stop_confirmation is not None:
            submit = self.query_one("#agent-stop-submit", Button)
            cancel = self.query_one("#agent-stop-cancel", Button)
        elif self._release_confirmation is not None:
            submit = self.query_one("#agent-release-submit", Button)
            cancel = self.query_one("#agent-release-cancel", Button)
        else:
            return
        (cancel if submit.has_focus else submit).focus(scroll_visible=False)

    def stop_is_pending(self, agent_id: str, run_id: str) -> bool:
        return (agent_id, run_id) in self._pending_stops

    def action_stop_scroll(self, direction: int) -> None:
        if self._stop_confirmation is not None:
            scroll = self.query_one("#agent-stop-scroll", VerticalScroll)
        elif self._release_confirmation is not None:
            scroll = self.query_one("#agent-release-scroll", VerticalScroll)
        else:
            return
        if direction < 0:
            scroll.scroll_up(animate=False)
        else:
            scroll.scroll_down(animate=False)

    def dismiss_stop_confirmation(self) -> None:
        self._stop_confirmation = None
        self._render_agents()
        if self.is_mounted and self._expanded:
            self.focus(scroll_visible=False)

    def dismiss_release_confirmation(self) -> None:
        self._release_confirmation = None
        self._render_agents()
        if self.is_mounted and self._expanded:
            self.focus(scroll_visible=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id not in {
            "agent-stop-submit",
            "agent-stop-cancel",
            "agent-release-submit",
            "agent-release-cancel",
        }:
            return
        event.stop()
        if event.button.id in {"agent-stop-submit", "agent-stop-cancel"}:
            target = self._stop_confirmation
            if event.button.id == "agent-stop-submit" and target is not None:
                # Recheck the captured identity, never substitute the current run.
                if self._stop_target() == target:
                    self._pending_stops.add(target)
                    self.post_message(self.StopRequested(*target))
            self.dismiss_stop_confirmation()
            return
        target = self._release_confirmation
        if event.button.id == "agent-release-submit" and target is not None:
            # Recheck the captured identity, never substitute the current run.
            if self._release_target() == target:
                self.post_message(self.ReleaseRequested(*target))
        self.dismiss_release_confirmation()

    def settle_stop(
        self, agent_id: str, run_id: str, response: AgentsCancelResponse | None
    ) -> None:
        target = (agent_id, run_id)
        if target not in self._pending_stops:
            return
        if (
            response is None
            or response.outcome
            not in {CancelOutcome.STOP_REQUESTED, CancelOutcome.ALREADY_STOPPING}
            or response.run_id != run_id
        ):
            self._pending_stops.discard(target)
        self._render_agents()

    def _display_state(self, agent: AgentSummaryModel) -> str:
        if (
            agent.agent_id,
            agent.current_run_id or agent.latest_run_id,
        ) in self._pending_stops:
            return "stopping"
        return agent_state(agent)

    def action_cursor_up(self) -> None:
        if self._confirmation_open():
            self.action_switch_action()
            return
        if (
            self._show_full_details or self._show_help
        ) and self._details_scroll is not None:
            self._render_agents()
            self._details_scroll.scroll_up(animate=False)
            return
        self._move_selection(-1)

    def action_cursor_down(self) -> None:
        if self._confirmation_open():
            self.action_switch_action()
            return
        if (
            self._show_full_details or self._show_help
        ) and self._details_scroll is not None:
            self._render_agents()
            self._details_scroll.scroll_down(animate=False)
            return
        self._move_selection(1)

    def action_select(self) -> None:
        if (
            self._expanded
            and not self._show_help
            and not self._show_full_details
            and not self._confirmation_open()
        ):
            self.post_message(self.SelectionRequested(self._selected_agent_id))

    def action_details(self) -> None:
        if self._expanded and not self._show_help:
            self._show_full_details = True
            self._render_agents()

    def action_help(self) -> None:
        if self._expanded:
            self._show_help = True
            self._render_agents()

    def action_collapse(self) -> None:
        if self._stop_confirmation is not None:
            self.dismiss_stop_confirmation()
            return
        if self._release_confirmation is not None:
            self.dismiss_release_confirmation()
            return
        if self._show_help:
            self._show_help = False
            self._render_agents()
            return
        if self._show_full_details:
            self._show_full_details = False
            self._render_agents()
            return
        self.close_browser()
        self.post_message(self.Closed())

    def action_filter_prev(self) -> None:
        self._cycle_filter(-1)

    def action_filter_next(self) -> None:
        self._cycle_filter(1)

    def _cycle_filter(self, offset: int) -> None:
        keys = list(_FILTER_ORDER)
        self._filter = keys[(keys.index(self._filter) + offset) % len(keys)]
        self._refresh_selection()
        self._render_agents()
        if self.is_mounted and self._expanded:
            self.call_after_refresh(self._scroll_selected_row_into_view)

    def on_click(self, event: events.Click) -> None:
        if self._confirmation_open():
            event.stop()
            return
        if not self._expanded:
            logger.debug("Agent bar click phase=expand-start row=%d", event.y)
            self.open_browser()
            logger.debug("Agent bar click phase=expand-done row=%d", event.y)
            return
        if self._show_help or self._show_full_details or self._rows_scroll is None:
            return
        row = event.screen_y - self._rows_scroll.region.y
        if (
            row < 0
            or row >= self._rows_scroll.size.height
            or self._content is None
            or not self._content.region.contains(event.screen_x, event.screen_y)
        ):
            logger.debug("Agent bar click phase=chrome row=%d", row)
            return
        index = row + int(self._rows_scroll.scroll_y)
        rows = self._visible_rows()
        if index < 0 or index >= len(rows):
            return
        target = rows[index]
        if target.kind not in {"main", "agent"}:
            # Group headers and notes are labels, never selections.
            logger.debug(
                "Agent bar click phase=row-skip row=%d kind=%s", row, target.kind
            )
            return
        logger.debug(
            "Agent bar click phase=row-select-start row=%d target=%s",
            row,
            target.agent_id,
        )
        self._selected_agent_id = target.agent_id
        self.focus(scroll_visible=False)
        self._render_agents()
        self.action_select()
        logger.debug(
            "Agent bar click phase=row-select-posted row=%d target=%s",
            row,
            self._selected_agent_id,
        )

    def _visible_groups(self) -> tuple[str, ...]:
        if self._filter == "active":
            return (GROUP_ACTIVE,)
        if self._filter == "idle":
            return (GROUP_IDLE,)
        if self._filter == "recent":
            return (GROUP_RECENT,)
        if self._filter == "all":
            return (GROUP_ACTIVE, GROUP_IDLE, GROUP_RECENT)
        return (GROUP_ACTIVE, GROUP_IDLE)

    def _group_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter(agent_group(agent) for agent in self.agents)
        counts["resident"] = counts[GROUP_ACTIVE] + counts[GROUP_IDLE]
        counts["all"] = len(self.agents)
        return counts

    def _visible_rows(
        self, agents: Sequence[AgentSummaryModel] | None = None
    ) -> list[_Row]:
        """The shared visible-row mapping: Main, notes, group headers, agents.

        One mapping drives keyboard movement, mouse hit-testing, scrolling and
        highlighting. Group headers and notes are never selectable.
        """
        inventory = tuple(self.agents if agents is None else agents)
        rows = [_Row("main", None, self._row(None, "Main agent"))]
        if not inventory:
            rows.append(
                _Row(
                    "note",
                    None,
                    "No background agents yet"
                    + _separator()
                    + "launch a background task to inspect it here.",
                )
            )
            return rows
        counts = Counter(agent_group(agent) for agent in inventory)
        groups = self._visible_groups()
        if not any(counts[group] for group in groups):
            if self._filter == "resident" and counts[GROUP_RECENT]:
                # Never an empty screen: fall back to the Recent group.
                rows.append(
                    _Row(
                        "note",
                        None,
                        "No resident agents" + _separator() + "showing recent",
                    )
                )
                groups = (GROUP_RECENT,)
            else:
                rows.append(_Row("note", None, _FILTER_EMPTY_NOTES[self._filter]))
                return rows
        for group in groups:
            if not counts[group]:
                continue
            rows.append(_Row("header", None, _GROUP_LABELS[group]))
            for agent in inventory:
                if agent_group(agent) == group:
                    text, dim = self._agent_row(agent)
                    rows.append(_Row("agent", agent.agent_id, text, dim))
        return rows

    def _visible_ids(
        self, agents: Sequence[AgentSummaryModel] | None = None
    ) -> tuple[str | None, ...]:
        """Selectable targets (Main first) in the shared visible-row order."""
        return tuple(
            row.agent_id
            for row in self._visible_rows(agents)
            if row.kind in {"main", "agent"}
        )

    def _refresh_selection(
        self, agents: Sequence[AgentSummaryModel] | None = None
    ) -> None:
        ids = self._visible_ids(agents)
        self._update_selection(ids)
        self._previous_agent_ids = ids

    def _move_selection(self, offset: int) -> None:
        ids = self._visible_ids()
        index = (
            ids.index(self._selected_agent_id) if self._selected_agent_id in ids else 0
        )
        self._selected_agent_id = ids[max(0, min(index + offset, len(ids) - 1))]
        self._render_agents()
        self._scroll_selected_row_into_view()

    def _scroll_selected_row_into_view(self) -> None:
        rows = self._visible_rows()
        index = next(
            (
                position
                for position, row in enumerate(rows)
                if row.kind in {"main", "agent"}
                and row.agent_id == self._selected_agent_id
            ),
            None,
        )
        if index is None:
            return
        scroll = self._rows_scroll
        if scroll is None:
            return
        top = int(scroll.scroll_y)
        height = scroll.size.height
        if index < top:
            scroll.scroll_to(y=index, animate=False, force=True, immediate=True)
        elif height and index >= top + height:
            scroll.scroll_to(
                y=index - height + 1, animate=False, force=True, immediate=True
            )

    def _update_selection(self, ids: tuple[str | None, ...]) -> None:
        if self._selected_agent_id is None or self._selected_agent_id in ids:
            return
        index = (
            self._previous_agent_ids.index(self._selected_agent_id)
            if self._selected_agent_id in self._previous_agent_ids
            else 0
        )
        self._selected_agent_id = ids[min(index, len(ids) - 1)] if ids else None

    def _render_agents(self) -> None:
        self.display = (bool(self.agents) or self._expanded) and getattr(
            self.app, "_agent_transcript_viewer", None
        ) is None
        full = self._show_full_details or self._show_help
        confirmation = self._confirmation_open()
        if self.is_mounted:
            self._render_chrome()
        if self._rows_scroll is not None:
            self._rows_scroll.display = self._expanded and not full
        if self._details_scroll is not None:
            self._details_scroll.display = self._expanded and full
        if self._detail is not None:
            self._detail.display = self._expanded and not full and not confirmation
        if self._footer is not None:
            self._footer.display = self._expanded
        if not self._expanded:
            summary = self._collapsed_content(self.content_size.width)
            if self.is_mounted:
                self.query_one("#agent-bar-title", NoMarkupStatic).update(summary)
            self._update_content(summary)
            return
        rows = self._visible_rows()
        selected = next(
            (
                agent
                for agent in self.agents
                if agent.agent_id == self._selected_agent_id
            ),
            None,
        )
        if self._detail is not None:
            self._detail.update(
                self._agent_details(selected) if selected else self._main_details()
            )
        if self._details_content is not None:
            self._details_content.update(
                self._help_body() if self._show_help else self.full_metadata(selected)
            )
        if self._footer is not None:
            self._footer.update(self._footer_hint(full))
        self._update_content(self._rows_content(rows))

    def _render_chrome(self) -> None:
        """Title (tabs or collapsed summary) and the confirmation prompts."""
        title = self.query_one("#agent-bar-title", NoMarkupStatic)
        title.display = bool(self.agents)
        if self._expanded:
            title.update(self._tabs_content(self.content_size.width))
        self.query_one("#agent-stop-confirmation").display = (
            self._expanded and self._stop_confirmation is not None
        )
        self.query_one("#agent-release-confirmation").display = (
            self._expanded and self._release_confirmation is not None
        )
        if self._stop_confirmation is not None:
            agent_id, run_id = self._stop_confirmation
            self.query_one("#agent-stop-prompt", NoMarkupStatic).update(
                f"Stop {agent_id}, run {run_id}? Retained agent, transcript and partial "
                "output preserved subject to retention/release policy."
            )
        if self._release_confirmation is not None:
            agent_id = self._release_confirmation[0]
            agent = next(
                (item for item in self.agents if item.agent_id == agent_id), None
            )
            label = (agent.name or agent.agent_id) if agent is not None else agent_id
            self.query_one("#agent-release-prompt", NoMarkupStatic).update(
                f"Release {label} ({agent_id})? Retained results and browser "
                "access are discarded."
            )

    def _rows_content(self, rows: list[_Row]) -> Content:
        """Assemble the visible rows with dim handles/notes and the highlight."""
        lines: list[str] = []
        spans: list[Span] = []
        for row in rows:
            base = sum(len(line) + 1 for line in lines)
            lines.append(row.text)
            if row.dim is not None:
                spans.append(Span(base + row.dim[0], base + row.dim[1], "dim"))
            elif row.kind in {"note", "header"}:
                spans.append(Span(base, base + len(row.text), "dim"))
        content = Content("\n".join(lines), spans=spans)
        if not self.has_focus:
            return content
        index = next(
            (
                position
                for position, row in enumerate(rows)
                if row.kind in {"main", "agent"}
                and row.agent_id == self._selected_agent_id
            ),
            None,
        )
        if index is None:
            return content
        start = sum(len(line) + 1 for line in lines[:index])
        return content.stylize("bold reverse", start, start + len(lines[index]))

    def _active_count(self) -> int:
        return sum(
            1
            for agent in self.agents
            if agent.availability.lower() in {"running", "finalizing"}
        )

    def _collapsed_content(self, width: int) -> str | Content:
        """The collapsed segment: active counts, or a dim inventory note."""
        text = self.collapsed_summary(width)
        if not text or self._active_count():
            return text
        return Content(text, spans=[Span(0, len(text), "dim")])

    def _tabs_content(self, width: int) -> str | Content:
        """Filter tabs with truthful counts; the current tab is highlighted."""
        counts = self._group_counts()
        separator = _separator()
        text = ""
        with_counts = True
        for candidate in (True, False):
            parts = [
                (
                    f"{_FILTER_LABELS[key]} {counts[key]}"
                    if candidate
                    else _FILTER_LABELS[key]
                )
                for key in _FILTER_ORDER
            ]
            text = separator.join(parts)
            with_counts = candidate
            if width <= 0 or cell_len(text) <= width:
                break
        else:
            text = _clip(text, width)
        active = _FILTER_LABELS[self._filter]
        active_text = f"{active} {counts[self._filter]}" if with_counts else active
        start = text.find(active_text)
        if start < 0:
            return text
        return Content(
            text, spans=[Span(start, start + len(active_text), "bold reverse")]
        )

    def collapsed_summary(self, width: int) -> str:
        """Active-only counts; a dim note keeps the bar non-empty.

        Only running and finalizing instances are counted, each once, keyed on
        lifecycle availability. Failed, cancelled and budget outcomes surface
        through completion and eviction notifications, not this status line.
        """
        running = sum(
            1 for agent in self.agents if agent.availability.lower() == "running"
        )
        finishing = sum(
            1 for agent in self.agents if agent.availability.lower() == "finalizing"
        )
        if running or finishing:
            parts = []
            if running:
                parts.append(f"{running} running")
            if finishing:
                parts.append(f"{finishing} finishing")
            summary = _separator().join(parts)
            if width <= 0 or cell_len(summary) <= width:
                return summary
            if running and finishing:
                # The finishing count is the optional segment.
                summary = f"{running} running"
                if cell_len(summary) <= width:
                    return summary
            return _clip(summary, width)
        counts = self._group_counts()
        if counts[GROUP_RECENT]:
            note = f"{counts[GROUP_RECENT]} recent"
        elif counts[GROUP_IDLE]:
            note = f"{counts[GROUP_IDLE]} idle"
        else:
            return ""
        if width <= 0 or cell_len(note) <= width:
            return note
        return _clip(note, width)

    def _footer_hint(self, full: bool) -> Content:
        if self._confirmation_open():
            variants = [
                [
                    ("↑↓/Tab", "Action"),
                    ("PgUp/PgDn", "Read"),
                    ("Enter", "Choose"),
                    ("Esc", "Cancel"),
                ]
            ]
        elif full:
            variants = [[("↑↓", "Scroll"), ("F1", "Help"), ("Esc", "Back")]]
        else:
            base = [
                ("↑↓", "Move"),
                ("Enter", "Open"),
                *([("C", "Stop run")] if self._stop_target() is not None else []),
                *([("R", "Release")] if self._release_target() is not None else []),
                ("D", "Details"),
                ("F1", "Help"),
                ("Esc", "Close"),
            ]
            # The filter hint is the first entry to yield its slot: ascii chrome
            # renders the arrow keys ("Up/Down", "Left/Right") far wider than
            # the glyphs, and the remaining labels must keep fitting at 80 cols.
            variants = [base[:1] + [("←→", "Filter")] + base[1:], base]
        width = self._footer.size.width if self._footer else self.content_size.width
        for descriptions in (True, False):
            for entries in variants:
                hint = shortcut_hint(
                    "  ".join(
                        shortcut(key) + (f" {label}" if descriptions else "")
                        for key, label in entries
                    )
                )
                if hint.cell_length <= width:
                    return hint
        # Keep Close reachable when there is no room for the complete key list.
        close = shortcut_hint(shortcut("Esc") + (" Back" if full else " Close"))
        return (
            close
            if close.cell_length <= width
            else shortcut_hint(shortcut(_clip("Esc", width)))
        )

    @staticmethod
    def _help_body() -> str:
        return (
            "Agent browser help\n\n"
            "Up/Down: move between agents without changing their order.\n"
            "Enter or single-click a row: open that agent's output.\n"
            "Main agent: return to the root conversation.\n"
            "Left/Right: cycle the Resident, Active, Idle, Recent and All filters.\n"
            "Rows are grouped by lifecycle: Active, Retained Idle, Recent.\n"
            "Group headers are labels, never selections.\n"
            "D: full metadata, context usage and compaction state.\n"
            "F1: this local help. Up/Down scroll details and help.\n"
            "C: confirm stopping the highlighted Running/Compacting run.\n"
            "R: confirm releasing the highlighted Idle or Recent agent.\n"
            "Confirmation: Up/Down or Tab choose actions; PgUp/PgDn read the scope.\n"
            "Cancel preserves the running task; Stop preserves retained output subject to retention policy.\n"
            "Release discards retained results and browser access for that agent.\n"
            "Esc: dismiss confirmation, back from help, then details, then close the browser.\n"
            "Running / Compacting are active; Finishing is finalization.\n"
            "Failed, Cancelled and Budget stopped retain the run outcome.\n"
            "Closed and Evicted rows are recent terminal metadata.\n"
            "The collapsed bar counts active work only; outcomes surface as notifications.\n"
            "Narrow rows clip by cells; the recent and idle notes render dimmed."
        )

    def _update_content(self, content: str | Content) -> None:
        self._rendered = str(content)
        if self._content is not None and (
            content != self._last_rendered_content
            or self.has_focus != self._last_rendered_focused
        ):
            self._content.update(content)
            self._last_rendered_content = content
            self._last_rendered_focused = self.has_focus

    def _row(
        self,
        agent_id: str | None,
        identity: str,
        *,
        secondary: str | None = None,
        state: str = "",
    ) -> str:
        return self._row_spans(agent_id, identity, secondary=secondary, state=state)[0]

    def _row_spans(
        self,
        agent_id: str | None,
        identity: str,
        *,
        secondary: str | None = None,
        state: str = "",
    ) -> tuple[str, tuple[int, int] | None]:
        """Render one row, returning its text and the dim span of the handle."""
        prefix = (
            f"{chrome_glyph('cursor')} "
            if agent_id == self._selected_agent_id
            else "  "
        )
        width = max(
            1,
            (self._content.size.width if self._content else 0)
            or self.content_size.width,
        )
        separator = _separator()
        # At very narrow widths use explicit state abbreviations rather than
        # clipping away a failure/budget outcome at the right edge.
        if state and cell_len(prefix + separator + state) >= width:
            if "Finishing" in state:
                outcome = (
                    "B"
                    if "Budget" in state
                    else "F"
                    if "Failed" in state
                    else "C"
                    if "Cancelled" in state
                    else ""
                )
                state = "Fin" + (":" + outcome if outcome else "")
            else:
                state = next(
                    (
                        short
                        for word, short in (
                            ("Compacting", "Comp"),
                            ("Running", "Run"),
                            ("Suspended", "Susp"),
                            ("Restoring", "Rest"),
                            ("Failed", "F"),
                            ("Budget", "B"),
                            ("Cancelled", "C"),
                            ("Idle", "Idle"),
                            ("Evicted", "Evict"),
                            ("Closed", "Cld"),
                        )
                        if word in state
                    ),
                    state,
                )
            if cell_len(prefix + separator + state) >= width:
                marker = state.rsplit(":", 1)[-1]
                state = marker if marker in {"F", "B", "C"} else chrome_glyph("running")
                separator = " " if width > 1 else ""
                prefix = ""
        if cell_len(prefix) > width:
            prefix = ""
        suffix = separator + state if state else ""
        available = max(0, width - cell_len(prefix) - cell_len(suffix))
        details = _clip(identity, available)
        dim: tuple[int, int] | None = None
        if secondary:
            remaining = available - cell_len(details) - cell_len(separator)
            if remaining > 0:
                handle = _clip(secondary, remaining)
                # Spans index characters, not cells: a wide-glyph name must
                # not shift the dim handle span (cell_len stays above, where
                # it belongs, for the clipping width math).
                start = len(prefix) + len(details) + len(separator)
                dim = (start, start + len(handle))
                details += separator + handle
        row = prefix + details + suffix
        return row + " " * max(0, width - cell_len(row)), dim

    def _agent_row(
        self, agent: AgentSummaryModel
    ) -> tuple[str, tuple[int, int] | None]:
        """The agent row: derived name primary, handle dim secondary."""
        state = _state_label(self._display_state(agent), _run_outcome(agent))
        return self._row_spans(
            agent.agent_id,
            agent.name or agent.agent_id,
            secondary=agent.agent_id,
            state=state,
        )

    def _bounded_details(self, first: str, second: str) -> str:
        width = (
            self._detail.size.width if self._detail else 0
        ) or self.content_size.width
        # Unknown width (before the first layout pass): keep the text intact,
        # matching collapsed_summary and _tabs_content; the next render clips.
        if width <= 0:
            return first + "\n" + second
        return _clip(first, width) + "\n" + _clip(second, width)

    def _main_details(self) -> str:
        state = (
            _state_label("compacting")
            if self._main_compacting
            else str(self._main_metadata.get("State", "Return to conversation"))
        )
        context = _context_text(
            self._main_context_tokens,
            self._main_context_threshold,
            compacting=self._main_compacting,
        )
        return self._bounded_details("Main agent" + _separator() + state, context)

    def _duration_details(self, agent: AgentSummaryModel) -> str:
        sample, receipt = self._duration_samples.get(
            agent.agent_id, (agent, time.monotonic())
        )
        # Textual may retain an equal-valued snapshot; use its latest receipt anchor.
        # Unequal snapshots (including older runs) must not borrow that anchor.
        elapsed = max(0.0, time.monotonic() - receipt) if sample == agent else 0.0
        run = agent.run_elapsed_seconds
        active = agent_is_active(agent)
        if active and run is not None:
            run += elapsed
        label = (
            "Run"
            if active or agent.availability.lower() == "finalizing"
            else "Last run"
        )
        parts = [f"{label} {_format_seconds(run)}"]
        availability = agent.availability.lower()
        if availability == "idle":
            idle = agent.idle_seconds
            if idle is not None:
                idle += elapsed
            parts.append(f"Idle {_format_seconds(idle)}")
        elif availability == "evicted":
            parts.append(f"Idle at eviction {_format_seconds(agent.idle_seconds)}")
        # A closed tombstone is terminal: it keeps no idle segment.
        return _separator().join(parts)

    def _agent_details(self, agent: AgentSummaryModel) -> str:
        state = self._display_state(agent)
        details = _state_label(state, _run_outcome(agent))
        if state == "evicted" and (outcome := _run_outcome(agent)):
            details += _separator() + _state_label(outcome)
        details += _separator() + self._duration_details(agent)
        availability = agent.availability.lower()
        if availability in {"closed", "evicted"}:
            details += _separator() + (
                "result expired" if agent.result_expired else "result preserved"
            )
            if availability == "evicted":
                eviction = self._evictions.get(agent.agent_id)
                if eviction is not None:
                    details += (
                        _separator()
                        + f"evicted: {eviction.reason} ({_format_seconds(eviction.idle_duration_seconds)} idle)"
                    )
        unknown = "-" if ascii_chrome_enabled() else "—"
        turns = unknown if agent.turns_used is None else str(agent.turns_used)
        model = agent_model_display_name(agent) or "unknown"
        context = _context_text(
            agent.context_tokens,
            agent.context_window,
            compacting=agent.compacting,
            last_recorded=availability in {"closed", "evicted"},
        )
        return self._bounded_details(
            details, _separator().join((context, model, f"turns {turns}"))
        )

    def full_metadata(self, agent: AgentSummaryModel | None = None) -> str:
        """Unclipped public metadata for Main or a child, in a scrollable view."""
        if agent is None:
            lines = ["Identity: Main agent"]
            lines.extend(
                f"{name}: {value}"
                for name, value in self._main_metadata.items()
                if value is not None
            )
            lines.append(
                _context_text(
                    self._main_context_tokens,
                    self._main_context_threshold,
                    compacting=self._main_compacting,
                )
            )
            lines.append(f"Compacting: {self._main_compacting}")
            return "\n".join(lines)
        fields = [
            ("Name", agent.name or None),
            ("Agent ID", agent.agent_id),
            ("Profile", agent.profile),
            ("Availability", agent.availability),
            ("Current run status", agent.current_run_status),
            ("Last run status", agent.last_run_status),
            ("State", _state_label(agent_state(agent), _run_outcome(agent))),
            ("Stop reason", agent.stop_reason),
            ("Compacting", str(agent.compacting)),
            (
                "Context tokens",
                str(agent.context_tokens) if agent.context_tokens is not None else None,
            ),
            (
                "Auto-compact threshold",
                str(agent.context_window) if agent.context_window is not None else None,
            ),
            ("Result expired", str(agent.result_expired)),
            ("Initial task", agent.initial_task_summary),
            ("Current task", agent.current_task_summary),
            ("Provider", agent.active_provider),
            ("Model", agent.effective_model),
            ("Base model", agent.base_model),
            ("Thinking", agent.effective_thinking),
            ("Turns", str(agent.turns_used) if agent.turns_used is not None else None),
            ("Run ID", agent.current_run_id or agent.latest_run_id),
            ("Duration", self._duration_details(agent)),
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
        return (
            "\n".join(f"{name}: {value}" for name, value in fields if value is not None)
            + "\n"
            + _context_text(
                agent.context_tokens,
                agent.context_window,
                compacting=agent.compacting,
                last_recorded=agent.availability.lower() in {"closed", "evicted"},
            )
        )
