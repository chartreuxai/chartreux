from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import time
from typing import Any, ClassVar

from rich.cells import cell_len, chop_cells
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
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


def agent_state(agent: AgentSummaryModel) -> str:
    """Presentation state; finalization is not active generation."""
    availability = agent.availability.lower()
    if availability == "evicted":
        return "evicted"
    if availability == "finalizing":
        return "finishing"
    outcome = _run_outcome(agent)
    if outcome is not None:
        return outcome
    status = (agent.current_run_status or agent.last_run_status or "").lower()
    if availability == "running" or status in {"running", "in_progress", "in-progress"}:
        return "compacting" if agent.compacting else "running"
    return "idle"


def agent_is_active(agent: AgentSummaryModel) -> bool:
    """Shared UI predicate for output polling and active-agent chrome."""
    return agent_state(agent) in {"running", "compacting"}


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
        "failed": f"{chrome_glyph('error')} Failed",
        "cancelled": "! Cancelled",
        "budget-stopped": "! Budget stopped",
        "evicted": "Evicted",
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
        width: 100%; height: 1; max-height: 10; color: $text-muted;
    }
    AgentBar:focus { color: $foreground; }
    AgentBar.-expanded {
        height: 10; border: solid $primary; border-title-color: $foreground;
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
        Binding(
            "left,right,tab,shift+tab",
            "stop_focus",
            "Switch action",
            show=False,
            priority=True,
        ),
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
        self._pending_stops: set[tuple[str, str]] = set()
        self._evictions: dict[str, AgentEvictionModel] = {}
        # Samples and their receipt anchors are installed before reactive rendering.
        self._duration_samples: dict[str, tuple[AgentSummaryModel, float]] = {}
        self._previous_agent_ids: tuple[str, ...] = ()
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
        by_id = {
            agent.agent_id: agent
            for agent in agents
            if agent.availability.lower() != "released"
        }
        ids = tuple(
            agent_id for agent_id in self._previous_agent_ids if agent_id in by_id
        )
        ids += tuple(agent_id for agent_id in by_id if agent_id not in ids)
        selectable = tuple(by_id[agent_id] for agent_id in ids)
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
        self._duration_samples = {
            agent.agent_id: (agent, receipt) for agent in selectable
        }
        self.agents = selectable
        # Updates are authoritative. Clearing a target also fences late RPC replies.
        self._pending_stops.intersection_update(
            (agent.agent_id, run_id)
            for agent in selectable
            if agent_is_active(agent) or agent.availability.lower() == "finalizing"
            if (run_id := agent.current_run_id or agent.latest_run_id) is not None
        )
        if self._stop_confirmation is not None:
            target = self._stop_confirmation
            if not any(
                agent.agent_id == target[0]
                and agent.current_run_id == target[1]
                and agent_is_active(agent)
                for agent in selectable
            ):
                self.dismiss_stop_confirmation()
        self._render_agents()
        if self._expanded and self.is_mounted:
            self.call_after_refresh(self._scroll_selected_row_into_view)

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
        self._stop_confirmation = None
        self._expanded = False
        self._show_full_details = False
        self._show_help = False
        self.set_class(False, "-expanded")
        self._render_agents()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"stop_focus", "stop_scroll"}:
            return self._stop_confirmation is not None
        if action == "stop_run":
            return self._stop_target() is not None and self._stop_confirmation is None
        if self._stop_confirmation is not None and action in {
            "select",
            "details",
            "help",
        }:
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

    def action_stop_run(self) -> None:
        if (
            self._stop_confirmation is not None
            or (target := self._stop_target()) is None
        ):
            return
        self._stop_confirmation = target
        self._render_agents()
        self.query_one("#agent-stop-scroll", VerticalScroll).scroll_home(animate=False)
        self.query_one("#agent-stop-cancel", Button).focus(scroll_visible=False)

    def action_stop_focus(self) -> None:
        if self._stop_confirmation is None:
            return
        submit = self.query_one("#agent-stop-submit", Button)
        cancel = self.query_one("#agent-stop-cancel", Button)
        (cancel if submit.has_focus else submit).focus(scroll_visible=False)

    def stop_is_pending(self, agent_id: str, run_id: str) -> bool:
        return (agent_id, run_id) in self._pending_stops

    def action_stop_scroll(self, direction: int) -> None:
        scroll = self.query_one("#agent-stop-scroll", VerticalScroll)
        if direction < 0:
            scroll.scroll_up(animate=False)
        else:
            scroll.scroll_down(animate=False)

    def dismiss_stop_confirmation(self) -> None:
        self._stop_confirmation = None
        self._render_agents()
        if self.is_mounted and self._expanded:
            self.focus(scroll_visible=False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id not in {"agent-stop-submit", "agent-stop-cancel"}:
            return
        event.stop()
        target = self._stop_confirmation
        if event.button.id == "agent-stop-submit" and target is not None:
            # Recheck the captured identity, never substitute the current run.
            if self._stop_target() == target:
                self._pending_stops.add(target)
                self.post_message(self.StopRequested(*target))
        self.dismiss_stop_confirmation()

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
        if self._stop_confirmation is not None:
            self.action_stop_focus()
            return
        if (
            self._show_full_details or self._show_help
        ) and self._details_scroll is not None:
            self._render_agents()
            self._details_scroll.scroll_up(animate=False)
            return
        self._move_selection(-1)

    def action_cursor_down(self) -> None:
        if self._stop_confirmation is not None:
            self.action_stop_focus()
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
            and self._stop_confirmation is None
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

    def on_click(self, event: events.Click) -> None:
        if self._stop_confirmation is not None:
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
        ids = (None, *(agent.agent_id for agent in self.agents))
        if index < len(ids):
            logger.debug(
                "Agent bar click phase=row-select-start row=%d target=%s",
                row,
                ids[index],
            )
            self._selected_agent_id = ids[index]
            self.focus(scroll_visible=False)
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
        full = self._show_full_details or self._show_help
        if self.is_mounted:
            self.query_one("#agent-bar-title").display = not self._expanded
            confirmation = self.query_one("#agent-stop-confirmation")
            confirmation.display = (
                self._expanded and self._stop_confirmation is not None
            )
            if self._stop_confirmation is not None:
                agent_id, run_id = self._stop_confirmation
                self.query_one("#agent-stop-prompt", NoMarkupStatic).update(
                    f"Stop {agent_id}, run {run_id}? Retained agent, transcript and partial "
                    "output preserved subject to retention/release policy."
                )
        if self._rows_scroll is not None:
            self._rows_scroll.display = self._expanded and not full
        if self._details_scroll is not None:
            self._details_scroll.display = self._expanded and full
        if self._detail is not None:
            self._detail.display = (
                self._expanded and not full and self._stop_confirmation is None
            )
        if self._footer is not None:
            self._footer.display = self._expanded
        if not self.agents:
            if self._expanded:
                self.close_browser()
                self.post_message(self.Closed())
            else:
                self._update_content("")
            return
        if not self._expanded:
            summary = self.collapsed_summary(self.content_size.width)
            if self.is_mounted:
                self.query_one("#agent-bar-title", NoMarkupStatic).update(summary)
            self._update_content(summary)
            return
        lines = [self._row(None, "Main agent")]
        lines.extend(
            self._row(
                agent.agent_id,
                agent.agent_id,
                profile=agent.profile,
                state=_state_label(self._display_state(agent), _run_outcome(agent)),
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
        detail = self._agent_details(selected) if selected else self._main_details()
        if self._detail is not None:
            self._detail.update(detail)
        if self._details_content is not None:
            self._details_content.update(
                self._help_body() if self._show_help else self.full_metadata(selected)
            )
        if self._footer is not None:
            self._footer.update(self._footer_hint(full))
        content = Content("\n".join(lines))
        if self.has_focus:
            ids = (None, *(agent.agent_id for agent in self.agents))
            index = (
                ids.index(self._selected_agent_id)
                if self._selected_agent_id in ids
                else 0
            )
            start = sum(len(line) + 1 for line in lines[:index])
            content = content.stylize("bold reverse", start, start + len(lines[index]))
        self._update_content(content)

    def collapsed_summary(self, width: int) -> str:
        """Drop optional counts before shortening protected outcome counts.

        Compacting contributes to running; finishing outcomes are counted as
        well as finishing so a terminal failure is never hidden by finalization.
        If even the protected counts exceed width, retain them intact rather
        than silently clipping an outcome (the physical minimum is unavoidable).
        """
        order = (
            "running",
            "stopping",
            "finishing",
            "failed",
            "budget-stopped",
            "cancelled",
            "idle",
            "evicted",
        )
        counts: Counter[str] = Counter()
        for agent in self.agents:
            state = self._display_state(agent)
            counts["running" if state == "compacting" else state] += 1
            if state == "finishing" and (outcome := _run_outcome(agent)):
                counts[outcome] += 1
        parts = {state: f"{counts[state]} {state}" for state in order if counts[state]}
        prefix = f"{len(self.agents)} agents: "
        separator = _separator()
        summary = prefix + separator.join(parts.values())
        if width <= 0 or cell_len(summary) <= width:
            return summary
        summary = separator.join(parts.values())
        if cell_len(summary) <= width:
            return summary
        for state in ("evicted", "idle", "cancelled", "finishing", "running"):
            if len(parts) == 1:
                break
            parts.pop(state, None)
            summary = separator.join(parts.values())
            if cell_len(summary) <= width:
                return summary
        protected = " ".join(
            f"{counts[state]}{label}"
            for state, label in (("failed", "F"), ("budget-stopped", "B"))
            if counts[state]
        )
        return protected or _clip(summary, width)

    def _footer_hint(self, full: bool) -> Content:
        if self._stop_confirmation is not None:
            entries = [
                ("↑↓/Tab", "Action"),
                ("PgUp/PgDn", "Read"),
                ("Enter", "Choose"),
                ("Esc", "Cancel"),
            ]
        else:
            entries = (
                [("↑↓", "Scroll"), ("F1", "Help"), ("Esc", "Back")]
                if full
                else [
                    ("↑↓", "Move"),
                    ("Enter", "Open output"),
                    *([("C", "Stop run")] if self._stop_target() is not None else []),
                    ("D", "Details"),
                    ("F1", "Help"),
                    ("Esc", "Close"),
                ]
            )
        width = self._footer.size.width if self._footer else self.content_size.width
        for descriptions in (True, False):
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
            "D: full metadata, context usage and compaction state.\n"
            "F1: this local help. Up/Down scroll details and help.\n"
            "C: confirm stopping the highlighted Running/Compacting run.\n"
            "Confirmation: Up/Down or Tab choose actions; PgUp/PgDn read the scope.\n"
            "Cancel preserves the running task; Stop preserves retained output subject to retention policy.\n"
            "Esc: dismiss confirmation, back from help, then details, then close the browser.\n"
            "Running / Compacting are active; Finishing is finalization.\n"
            "Failed, Cancelled and Budget stopped retain the run outcome.\n"
            "Evicted context is the last recorded measurement.\n"
            "Narrow summaries: F = failed, B = budget-stopped."
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
        self, agent_id: str | None, identity: str, *, profile: str = "", state: str = ""
    ) -> str:
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
                            ("Failed", "F"),
                            ("Budget", "B"),
                            ("Cancelled", "C"),
                            ("Idle", "Idle"),
                            ("Evicted", "Evict"),
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
        profile_width = available - cell_len(details) - cell_len(separator)
        if profile and profile_width > 0:
            details += separator + _clip(profile, profile_width)
        row = prefix + details + suffix
        return row + " " * max(0, width - cell_len(row))

    def _bounded_details(self, first: str, second: str) -> str:
        width = (
            self._detail.size.width if self._detail else 0
        ) or self.content_size.width
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
        if agent.availability.lower() in {"idle", "evicted"}:
            idle = agent.idle_seconds
            evicted = agent.availability.lower() == "evicted"
            if idle is not None and not evicted:
                idle += elapsed
            parts.append(
                f"{'Idle at eviction' if evicted else 'Idle'} {_format_seconds(idle)}"
            )
        return _separator().join(parts)

    def _agent_details(self, agent: AgentSummaryModel) -> str:
        state = self._display_state(agent)
        details = _state_label(state, _run_outcome(agent))
        if state == "evicted" and (outcome := _run_outcome(agent)):
            details += _separator() + _state_label(outcome)
        details += _separator() + self._duration_details(agent)
        if state == "evicted":
            eviction = self._evictions.get(agent.agent_id)
            details += _separator() + (
                "result expired" if agent.result_expired else "result preserved"
            )
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
            last_recorded=state == "evicted",
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
            ("Identity", agent.agent_id),
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
                last_recorded=agent_state(agent) == "evicted",
            )
        )
