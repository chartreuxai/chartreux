"""Cell-aware session chrome; all session data is supplied by the caller."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.cells import cell_len
from rich.text import Text
from textual.widget import Widget

from chartreux.app_server.config import StatusLineConfigView
from chartreux.app_server.models import UsageWindowSummary
from chartreux.ui.context_display import format_context
from chartreux.ui.usage_display import format_usage_cost

_REQUIRED_SEGMENTS = frozenset({"directory", "context"})
_MIN_USAGE_WIDTH = 5
_ASCII_SPACE = 32
_ASCII_DELETE = 127


@dataclass(frozen=True)
class SessionStatusState:
    """Presentation snapshot, not a resource reader or usage accumulator.

    model_identity is the caller's active provider/model display identity.
    home_directory is captured at construction, never queried during rendering.
    """

    cwd: Path | str
    pid: int | None = None
    model_identity: str | None = None
    context_tokens: int | None = None
    auto_compact_threshold: int | None = None
    compacting: bool = False
    last_recorded: bool = False
    branch: str | None = None
    branch_status: Literal["branch", "detached", "not_repository", "unknown"] = (
        "unknown"
    )
    home_directory: Path | None = field(default_factory=Path.home)
    ascii_chrome: bool = False
    # Global recorded usage across all projects, supplied by the caller.
    usage_day: UsageWindowSummary | None = None
    usage_week: UsageWindowSummary | None = None
    usage_month: UsageWindowSummary | None = None


def _single_line(value: str) -> str:
    # Paths and externally supplied identities may contain terminal controls.
    return "".join(
        " " if ord(char) < _ASCII_SPACE or ord(char) == _ASCII_DELETE else char
        for char in value
    )


def format_segment(
    segment: str, state: SessionStatusState, config: StatusLineConfigView
) -> str:
    text = _format_segment(segment, state, config)
    return text.replace("—", "-") if state.ascii_chrome else text


def _format_segment(  # noqa: PLR0911 -- one renderer per configured segment
    segment: str, state: SessionStatusState, config: StatusLineConfigView
) -> str:
    """Format one configured segment without reading resources."""
    match segment:
        case "directory":
            path = Path(state.cwd)
            if config.directory_style == "name":
                return _single_line(path.name or str(path))
            # Abbreviate the supplied home directory without reading resources.
            home = state.home_directory
            if home is not None and path.is_relative_to(home):
                path = Path("~") / path.relative_to(home)
            return _single_line(str(path))
        case "pid":
            return f"pid {state.pid}" if state.pid is not None else "pid —"
        case "model":
            return _single_line(state.model_identity or "Model —")
        case "context":
            return format_context(
                state.context_tokens,
                state.auto_compact_threshold,
                style=config.context_style,
                compacting=state.compacting,
                last_recorded=state.last_recorded,
            )
        case "git-branch":
            if state.branch_status == "detached":
                return "Git detached"
            if state.branch_status == "not_repository":
                return "Git not a repository"
            if state.branch_status == "branch" and state.branch:
                return _single_line(state.branch)
            return "Git —"
        case "spend-today":
            return f"Today {format_usage_cost(state.usage_day)}"
        case "spend-week":
            return f"Week {format_usage_cost(state.usage_week)}"
        case "spend-month":
            return f"Month {format_usage_cost(state.usage_month)}"
        case _:
            raise ValueError(f"Unknown status line segment: {segment}")


def _fit(value: str, width: int, *, ascii_chrome: bool) -> str:
    if width <= 0:
        return ""
    if cell_len(value) <= width:
        return value
    marker = "..." if ascii_chrome else "…"
    if width <= cell_len(marker):
        return marker[:width]
    text = Text(value, no_wrap=True)
    text.truncate(width - cell_len(marker), overflow="crop")
    return text.plain.rstrip() + marker


def format_status_line(
    state: SessionStatusState, config: StatusLineConfigView, width: int
) -> str:
    """Fit ordered segments, dropping pid first and then the optional tail.

    Once only required segments remain, shorten directory before context. At
    widths below five cells, show only a truncation marker (or an empty row at
    zero cells), never a misleading cropped usage value.
    """
    width = max(0, width)
    if width < _MIN_USAGE_WIDTH:
        return _fit(
            "…" if not state.ascii_chrome else "...",
            width,
            ascii_chrome=state.ascii_chrome,
        )
    parts = [(name, format_segment(name, state, config)) for name in config.segments]
    separator = " | " if config.separator == "pipe" else " "

    def joined() -> str:
        return separator.join(value for _, value in parts)

    if cell_len(joined()) > width:
        parts = [(name, value) for name, value in parts if name != "pid"]
    while cell_len(joined()) > width:
        optional = next(
            (
                i
                for i in range(len(parts) - 1, -1, -1)
                if parts[i][0] not in _REQUIRED_SEGMENTS
            ),
            None,
        )
        if optional is None:
            break
        parts.pop(optional)
    if cell_len(joined()) <= width:
        return joined()
    available = width - cell_len(separator)
    values = dict(parts)
    context_width = min(cell_len(values["context"]), available - 1)
    directory_width = available - context_width
    fitted = {
        "directory": _fit(
            values["directory"], directory_width, ascii_chrome=state.ascii_chrome
        ),
        "context": _fit(
            values["context"], context_width, ascii_chrome=state.ascii_chrome
        ),
    }
    return separator.join(fitted[name] for name, _ in parts)


class SessionStatusLine(Widget):
    """One neutral row with feedback taking precedence over session refreshes."""

    DEFAULT_CSS = """
    SessionStatusLine {
        width: 1fr;
        height: 1;
        min-height: 1;
        max-height: 1;
        padding: 0;
        margin: 0;
        overflow: hidden hidden;
        text-wrap: nowrap;
        color: $foreground;
        background: $background;
    }
    """

    def __init__(
        self,
        state: SessionStatusState,
        config: StatusLineConfigView | None = None,
        *,
        id: str | None = None,
        classes: str | None = None,
    ) -> None:
        super().__init__(id=id, classes=classes)
        self.state = state
        self.config = config or StatusLineConfigView()
        self._feedback: str | None = None

    def set_state(self, state: SessionStatusState) -> None:
        self.state = state
        self.refresh()

    def set_config(self, config: StatusLineConfigView) -> None:
        self.config = config
        self.refresh()

    def set_feedback(self, message: str) -> None:
        """Override status until explicitly cleared (no widget-owned timer)."""
        self._feedback = _single_line(message)
        self.refresh()

    def clear_feedback(self) -> None:
        """Restore the latest supplied state and configuration."""
        self._feedback = None
        self.refresh()

    def render(self) -> Text:
        width = self.content_size.width
        value = (
            _fit(self._feedback, width, ascii_chrome=self.state.ascii_chrome)
            if self._feedback is not None
            else format_status_line(self.state, self.config, width)
        )
        return Text(value, no_wrap=True, overflow="crop")

    def on_resize(self) -> None:
        self.refresh()
