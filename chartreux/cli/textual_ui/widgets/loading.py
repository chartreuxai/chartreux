from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import time

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Static

from chartreux.cli.textual_ui.widgets.spinner import SpinnerMixin, SpinnerType
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

DEFAULT_LOADING_STATUS = "Generating"
THINKING_LOADING_STATUS = "Thinking"
RETRYING_LOADING_STATUS = "Retrying"
INTERRUPTING_LOADING_STATUS = "Interrupting"
INITIALIZING_LOADING_STATUS = "Initializing"


def _format_elapsed(seconds: int) -> str:
    if seconds < 60:  # noqa: PLR2004
        return f"{seconds}s"

    minutes, secs = divmod(seconds, 60)
    if minutes < 60:  # noqa: PLR2004
        return f"{minutes}m{secs}s"

    hours, mins = divmod(minutes, 60)
    return f"{hours}h{mins}m{secs}s"


class LoadingWidget(SpinnerMixin, Static):
    SPINNER_TYPE = SpinnerType.SNAKE

    def __init__(self, status: str | None = None, *, show_hint: bool = True) -> None:
        super().__init__(classes="loading-widget")
        self.init_spinner()
        self._base_status = status or DEFAULT_LOADING_STATUS
        self.status = self._base_status
        self._indicator_widget: Static | None = None
        self._status_widget: Static | None = None
        self.hint_widget: Static | None = None
        self._show_hint = show_hint
        self._hint_suppressed = False
        self.debounce_widget: Static | None = None
        self.start_time: float | None = None
        self._last_elapsed: int = -1
        self._last_hint_width: int = -1
        self._paused_total: float = 0.0
        self._pause_start: float | None = None
        self._action_required_status: str | None = None
        self._status_before_action_required: str | None = None
        self._queued_count: int = 0
        self._interrupting = False

    @property
    def base_status(self) -> str:
        """The semantic status label."""
        return self._base_status

    def show_debounce_hint(self) -> None:
        if self.debounce_widget:
            self.debounce_widget.update(
                f"typing detected, waiting{chrome_glyph('truncation')}"
            )
            self.debounce_widget.display = True

    def hide_debounce_hint(self) -> None:
        if self.debounce_widget:
            self.debounce_widget.display = False

    def pause_timer(self) -> None:
        if self._pause_start is None:
            self._pause_start = time()

    def resume_timer(self) -> None:
        if self._pause_start is not None:
            self._paused_total += time() - self._pause_start
            self._pause_start = None

    def begin_action_required(self, status: str) -> None:
        """Show a blocking user-action status without losing live progress state."""
        if self._interrupting:
            return
        if self._action_required_status is None:
            self._status_before_action_required = self._base_status
            self.pause_timer()
        self._action_required_status = status
        self._set_status(status)

    def end_action_required(self) -> None:
        """Resume progress after the final open callback has been answered."""
        if self._action_required_status is None:
            return
        status = self._status_before_action_required or DEFAULT_LOADING_STATUS
        self._action_required_status = None
        self._status_before_action_required = None
        self.resume_timer()
        self.set_status(status)

    def set_status(self, status: str) -> None:
        # Once interrupting, ignore late status updates: the turn keeps streaming
        # until the cancel propagates and the event handler drives set_status on
        # those events, which would otherwise clobber the "Interrupting" label
        # and make the interrupt look ignored. The widget is torn down right
        # after, so this latch never needs releasing.
        if self._interrupting and status != INTERRUPTING_LOADING_STATUS:
            return
        if status == INTERRUPTING_LOADING_STATUS:
            self._interrupting = True
        elif self._action_required_status is not None:
            if status != self._action_required_status:
                self._status_before_action_required = status
            return
        self._set_status(status)

    def _set_status(self, status: str) -> None:
        # Repeated semantic status updates must not flicker the label.
        if status == self._base_status:
            return
        self._base_status = status
        self.status = status
        if self._status_widget:
            self._status_widget.update(self._build_status_text())

    def set_retrying(self, retrying: bool) -> None:
        if retrying:
            self.set_status(RETRYING_LOADING_STATUS)
        elif self._base_status == RETRYING_LOADING_STATUS:
            self.set_status(DEFAULT_LOADING_STATUS)

    def set_queue_count(self, count: int) -> None:
        if count == self._queued_count:
            return
        self._queued_count = count
        self._update_hint(max(self._last_elapsed, 0))

    def set_hint_suppressed(self, suppressed: bool) -> None:
        """Hide keyboard shortcuts while another control owns the keys."""
        self._hint_suppressed = suppressed
        if self.hint_widget is not None:
            self.hint_widget.display = not suppressed

    def _update_hint(self, elapsed: int) -> None:
        if self.hint_widget is None:
            return
        hint = shortcut_hint(self._format_hint(elapsed))
        # Only relayout when the width changes (e.g. 9s -> 10s). This assumes
        # that the line never wraps; equal width does not imply equal rendered
        # size for wrapped text.
        layout = hint.cell_length != self._last_hint_width
        self._last_hint_width = hint.cell_length
        self.hint_widget.update(hint, layout=layout)

    def _format_hint(self, elapsed: int) -> str:
        elapsed_str = _format_elapsed(elapsed)
        if self._queued_count > 0:
            return (
                f"({elapsed_str} {shortcut('Esc')} to interrupt · "
                f"{shortcut('Enter')} queues next turn · "
                f"{shortcut('Ctrl+C')} to cancel last queued message)"
            )
        return (
            f"({elapsed_str} {shortcut('Esc/Ctrl+C')} to interrupt · "
            f"{shortcut('Enter')} queues next turn)"
        )

    def compose(self) -> ComposeResult:
        with Horizontal(classes="loading-container"):
            self._indicator_widget = Static(
                self._spinner.current_frame(), classes="loading-indicator"
            )
            yield self._indicator_widget

            self._status_widget = Static(
                self._build_status_text(), classes="loading-status"
            )
            yield self._status_widget

            if self._show_hint:
                initial_hint = shortcut_hint(self._format_hint(0))
                self._last_hint_width = initial_hint.cell_length
                self.hint_widget = NoMarkupStatic(initial_hint, classes="loading-hint")
                self.hint_widget.display = not self._hint_suppressed
                yield self.hint_widget

            self.debounce_widget = Static("", classes="loading-debounce")
            self.debounce_widget.display = False
            yield self.debounce_widget

    def on_mount(self) -> None:
        self.start_time = time()
        self._update_animation()
        self.start_spinner_timer()

    def on_resize(self) -> None:
        self.refresh_spinner()

    def _update_spinner_frame(self) -> None:
        if not self._is_spinning:
            return
        self._update_animation()

    def _build_status_text(self) -> str:
        from rich.markup import escape

        return f"Running: {escape(self.status)}{chrome_glyph('truncation')}"

    def _update_animation(self) -> None:
        # The spinner and text share the active theme's interactive role.
        if self._indicator_widget:
            spinner_char = self._spinner.next_frame()
            self._indicator_widget.update(spinner_char, layout=False)

        if self.hint_widget and self.start_time is not None:
            paused = self._paused_total + (
                time() - self._pause_start if self._pause_start else 0
            )
            elapsed = int(time() - self.start_time - paused)
            if elapsed != self._last_elapsed:
                self._last_elapsed = elapsed
                self._update_hint(elapsed)


@contextmanager
def paused_timer(loading_widget: LoadingWidget | None) -> Iterator[None]:
    if loading_widget:
        loading_widget.pause_timer()
    try:
        yield
    finally:
        if loading_widget:
            loading_widget.resume_timer()
