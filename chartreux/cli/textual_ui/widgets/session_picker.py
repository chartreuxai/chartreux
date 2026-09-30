from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical
from textual.content import Content
from textual.message import Message
from textual.timer import Timer
from textual.widgets import Button, OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.models import PublicSession, SavedSessionSummary
from chartreux.cli.textual_ui.widgets.spinner_text import SpinnerText
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import rich_theme_style, shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

_SECONDS_PER_MINUTE = 60
_SECONDS_PER_HOUR = 3600
_SECONDS_PER_DAY = 86400
_SECONDS_PER_WEEK = 604800
_PREVIEW_DEBOUNCE_SECONDS = 0.1
_EMPTY_OPTION_ID = "state:empty"
_DeleteStateKind = Literal["confirmation", "feedback", "pending"]
type _PickerSession = PublicSession | SavedSessionSummary


@dataclass(frozen=True)
class _DeleteState:
    kind: _DeleteStateKind
    option_id: str


def _session_datetime(timestamp: int | str | None) -> datetime | None:
    try:
        if isinstance(timestamp, int):
            return datetime.fromtimestamp(timestamp / 1000, UTC)
        if isinstance(timestamp, str):
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return (
                parsed.replace(tzinfo=UTC)
                if parsed.tzinfo is None
                else parsed.astimezone(UTC)
            )
    except (OverflowError, OSError, ValueError):
        return None
    return None


def _format_relative_time(timestamp: int | str | None) -> str:
    if (dt := _session_datetime(timestamp)) is None:
        return "unknown"

    seconds = int((datetime.now(UTC) - dt).total_seconds())
    if seconds < _SECONDS_PER_MINUTE:
        return "just now"
    for threshold, divisor, unit in [
        (_SECONDS_PER_HOUR, _SECONDS_PER_MINUTE, "m"),
        (_SECONDS_PER_DAY, _SECONDS_PER_HOUR, "h"),
        (_SECONDS_PER_WEEK, _SECONDS_PER_DAY, "d"),
        (float("inf"), _SECONDS_PER_WEEK, "w"),
    ]:
        if seconds < threshold:
            return f"{seconds // divisor}{unit} ago"
    return "unknown"


def _session_id(session: _PickerSession) -> str:
    if isinstance(session, PublicSession):
        return session.id
    return session.session_id


def _session_updated_at(session: _PickerSession) -> int | str | None:
    if isinstance(session, PublicSession):
        return session.updated_at
    return session.end_time


def _session_sort_key(session: _PickerSession) -> float:
    if (updated_at := _session_datetime(_session_updated_at(session))) is None:
        return 0
    return updated_at.timestamp()


def _build_header_text(cwd: str | None, muted_style: str = "") -> Text:
    text = Text(no_wrap=True)
    text.append("local ", style=muted_style)
    text.append(cwd or "this folder", style=muted_style)
    return text


def _session_harness_tag(session: _PickerSession) -> str | None:
    """Return the harness provenance tag for a session, or ``None``.

    Only ``PublicSession`` carries a ``harness`` field; legacy
    ``SavedSessionSummary`` rows never have one.
    """
    return None


def _build_option_text(
    session: _PickerSession,
    message: str,
    muted_style: str = "",
    *,
    active: bool = False,
) -> Content:
    time_str = _format_relative_time(_session_updated_at(session))
    session_id = _session_id(session)[:8]
    parts: list[tuple[str, str] | str] = [
        (f"{time_str:10}", muted_style),
        "  ",
        (f"{session_id}  ", muted_style),
    ]
    harness = _session_harness_tag(session)
    if harness is not None:
        parts.append((f"[{harness}]  ", muted_style))
    if active:
        parts.append(("Active  ", muted_style))
    parts.append(message)
    return Content.assemble(*parts)


class SessionPickerApp(Container):
    """Session picker for /resume command."""

    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("d", "request_delete", "Delete", show=False),
        Binding("i", "inspect_session_id", "Inspect session ID", show=False),
        Binding("c", "copy_session_id", "Copy session ID", show=False),
    ]

    class SessionSelected(Message):
        option_id: str
        session_id: str

        def __init__(self, option_id: str, session_id: str) -> None:
            self.option_id = option_id
            self.session_id = session_id
            super().__init__()

    class Cancelled(Message):
        pass

    class SessionHighlighted(Message):
        session_id: str | None

        def __init__(self, session_id: str | None) -> None:
            self.session_id = session_id
            super().__init__()

    class SessionDeleteRequested(Message):
        option_id: str
        session_id: str

        def __init__(self, option_id: str, session_id: str) -> None:
            self.option_id = option_id
            self.session_id = session_id
            super().__init__()

    def __init__(
        self,
        sessions: Sequence[_PickerSession],
        latest_messages: dict[str, str],
        current_session_id: str | None = None,
        cwd: str | None = None,
        loading: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(id="sessionpicker-app", **kwargs)
        self._sessions = list(sessions)
        self._latest_messages = latest_messages
        self._current_session_id = current_session_id
        self._cwd = cwd
        self._loading = loading
        self._load_error: str | None = None
        self._preview_timer: Timer | None = None
        self._pending_preview_session_id: str | None = None
        self._preview_generation = 0
        self._selection_pending = False
        self._delete_state: _DeleteState | None = None
        self._id_detail_widget: NoMarkupStatic | None = None
        self._id_detail_open = False
        self._initial_highlighted: int | None = next(
            (i for i, s in enumerate(sessions) if _session_id(s) == current_session_id),
            None,
        )

    @property
    def has_sessions(self) -> bool:
        return bool(self._sessions)

    @property
    def is_loading(self) -> bool:
        return self._loading

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def set_loading(self, loading: bool, *, error: str | None = None) -> None:
        """Update the discovery state without touching an unmounted picker."""
        self._loading = loading
        self._load_error = error
        if error is not None:
            self._cancel_pending_preview()
        if not self.is_mounted:
            return

        status = self.query_one("#sessionpicker-loading", Horizontal)
        spinner = status.query_one(SpinnerText)
        if loading:
            spinner.set_pending(True)
            status.query_one(".sessionpicker-loading-status", NoMarkupStatic).update(
                f"{chrome_glyph('running')} Running: Loading sessions"
                if self._sessions
                else ""
            )
            status.display = bool(self._sessions)
            self._update_empty_option()
            return

        spinner.set_pending(False)
        message = (
            f"{chrome_glyph('error')} Failed: Loading sessions: {error}"
            if error is not None
            else ""
        )
        status.query_one(".sessionpicker-loading-status", NoMarkupStatic).update(
            message
        )
        status.set_class(error is not None, "-failed")
        status.display = bool(message and self._sessions)
        self._update_empty_option()

    def _option_list(self) -> OptionList:
        return self.query_one(OptionList)

    def _empty_option_text(self) -> Content:
        if self._loading:
            return Content.styled(
                f"{chrome_glyph('running')} Running: Loading sessions",
                self._running_style(),
            )
        if self._load_error is not None:
            return Content.styled(
                f"{chrome_glyph('error')} Failed: Loading sessions: {self._load_error}",
                self._error_style(),
            )
        muted = (
            rich_theme_style(self.app.theme_variables["text-muted"])
            if self.is_attached
            else ""
        )
        return Content.styled(
            "No saved sessions. Start a conversation to create one.", muted
        )

    def _update_empty_option(self) -> None:
        if not self.is_mounted:
            return
        option_list = self._option_list()
        if self._sessions:
            if any(option.id == _EMPTY_OPTION_ID for option in option_list.options):
                option_list.remove_option(_EMPTY_OPTION_ID)
            return
        message = self._empty_option_text()
        if any(option.id == _EMPTY_OPTION_ID for option in option_list.options):
            option_list.replace_option_prompt(_EMPTY_OPTION_ID, message)
        else:
            option_list.add_option(Option(message, id=_EMPTY_OPTION_ID))
        option_list.highlighted = 0

    def _session_by_option_id(self, option_id: str | None) -> _PickerSession | None:
        if option_id is None:
            return None

        return next(
            (
                session
                for session in self._sessions
                if _session_id(session) == option_id
            ),
            None,
        )

    def _highlighted_option_id(self) -> str | None:
        option = self._option_list().highlighted_option
        if option is None or option.id is None:
            return None

        return str(option.id)

    def _highlighted_session(self) -> _PickerSession | None:
        return self._session_by_option_id(self._highlighted_option_id())

    def _session_message(self, session: _PickerSession) -> str:
        return self._latest_messages.get(_session_id(session), "(empty session)")

    def _normal_option_text(self, session: _PickerSession) -> Content:
        return _build_option_text(
            session,
            self._session_message(session),
            rich_theme_style(self.app.theme_variables["text-muted"])
            if self.is_attached
            else "",
            active=_session_id(session) == self._current_session_id,
        )

    def _option_text(self, session: _PickerSession) -> Content:
        state = self._delete_state
        if state is None or state.option_id != _session_id(session):
            return self._normal_option_text(session)
        match state.kind:
            case "confirmation":
                return self._delete_confirmation_option_text(session)
            case "feedback":
                return self._delete_feedback_option_text(session)
            case "pending":
                return self._delete_pending_option_text(session)

    def _delete_confirmation_option_text(self, session: _PickerSession) -> Content:
        return _build_option_text(
            session, f"Delete session {_session_id(session)[:8]}?"
        )

    def _update_delete_confirmation(self) -> None:
        if not self.is_mounted:
            return
        confirmation = self.query_one("#sessionpicker-delete", Vertical)
        state = self._delete_state
        confirmation.display = state is not None and state.kind == "confirmation"
        help_widget = self.query_one("#sessionpicker-help", NoMarkupStatic)
        if self._delete_is_pending() or self._selection_pending:
            help_widget.update("Session operation in progress; Escape unavailable")
        elif state is not None and state.kind == "confirmation":
            help_widget.update(
                shortcut_hint(
                    f"{shortcut('Tab')} Choose  {shortcut('Enter')} Select  "
                    f"{shortcut('Esc')} Back"
                )
            )
        else:
            if self._sessions and not self._loading:
                hint = (
                    f"{shortcut('↑↓/jk')} Navigate  {shortcut('Enter')} Select  "
                    f"{shortcut('d')} Delete  {shortcut('i')} ID  "
                    f"{shortcut('c')} Copy ID  {shortcut('Esc')} Cancel"
                )
            else:
                hint = f"{shortcut('Esc')} Cancel"
            help_widget.update(shortcut_hint(hint))
        if state is not None and state.kind == "confirmation":
            session = self._session_by_option_id(state.option_id)
            label = self._session_message(session) if session else state.option_id
            confirmation.query_one(NoMarkupStatic).update(
                f"Delete session {state.option_id} ({label})? "
                "Cancel preserves this session and its saved history."
            )
            confirmation.query_one("#sessionpicker-cancel-delete", Button).focus()

    def _set_help_text(self, text: str) -> None:
        self.query_one("#sessionpicker-help", NoMarkupStatic).update(text)

    def _delete_feedback_option_text(self, session: _PickerSession) -> Content:
        return _build_option_text(
            session, "", active=_session_id(session) == self._current_session_id
        ) + Content.styled(self._delete_feedback_message(session), self._error_style())

    def _error_style(self) -> str:
        return (
            rich_theme_style(self.app.theme_variables["error"])
            if self.is_attached
            else ""
        )

    def _delete_feedback_message(self, session: _PickerSession) -> str:
        if _session_id(session) == self._current_session_id:
            return f"{chrome_glyph('error')} Failed: Can't delete current session"

        return f"{chrome_glyph('error')} Failed: Can't delete session"

    def _delete_pending_option_text(self, session: _PickerSession) -> Content:
        return _build_option_text(session, "") + Content.styled(
            f"{chrome_glyph('running')} Running: Deleting session",
            self._running_style(),
        )

    def _running_style(self) -> str:
        return (
            rich_theme_style(self.app.theme_variables["primary"])
            if self.is_attached
            else ""
        )

    def _restore_option_text(self, session: _PickerSession) -> None:
        self._option_list().replace_option_prompt(
            _session_id(session), self._normal_option_text(session)
        )

    def _delete_state_matches(
        self, option_id: str, kind: _DeleteStateKind | None = None
    ) -> bool:
        if self._delete_state is None or self._delete_state.option_id != option_id:
            return False
        if kind is not None and self._delete_state.kind != kind:
            return False
        return True

    def _delete_is_pending(self) -> bool:
        return self._delete_state is not None and self._delete_state.kind == "pending"

    def _cancel_pending_preview(self) -> None:
        self._preview_generation += 1
        if self._preview_timer is not None:
            self._preview_timer.stop()
            self._preview_timer = None
        self._pending_preview_session_id = None

    def _schedule_preview(self, session_id: str | None) -> None:
        self._cancel_pending_preview()
        if session_id is None:
            self.post_message(self.SessionHighlighted(session_id=None))
            return
        if not self.is_mounted:
            return

        self._pending_preview_session_id = session_id
        generation = self._preview_generation
        self._preview_timer = self.set_timer(
            _PREVIEW_DEBOUNCE_SECONDS,
            lambda: self._flush_preview(session_id, generation),
        )

    def _flush_preview(self, session_id: str, generation: int) -> None:
        if (
            generation != self._preview_generation
            or session_id != self._pending_preview_session_id
        ):
            return

        self._preview_timer = None
        self._pending_preview_session_id = None
        self.post_message(self.SessionHighlighted(session_id=session_id))

    def _clear_delete_state(self) -> None:
        state = self._delete_state
        if state is None:
            return

        self._delete_state = None
        self._update_delete_confirmation()
        if session := self._session_by_option_id(state.option_id):
            self._restore_option_text(session)

    def _show_delete_state(
        self, session: _PickerSession, kind: _DeleteStateKind, prompt: Content
    ) -> None:
        self._clear_delete_state()
        session_id = _session_id(session)
        self._delete_state = _DeleteState(kind=kind, option_id=session_id)
        self._option_list().replace_option_prompt(session_id, prompt)
        self._update_delete_confirmation()

    def remove_session(self, option_id: str) -> bool:
        session = self._session_by_option_id(option_id)
        if session is None:
            return False

        self._sessions = [
            session for session in self._sessions if _session_id(session) != option_id
        ]
        self._latest_messages.pop(option_id, None)
        if self._delete_state_matches(option_id):
            self._delete_state = None
        option_list = self._option_list()
        option_list.remove_option(option_id)
        self._update_empty_option()
        # Textual doesn't fire OptionHighlighted when the highlight moves due to
        # removal, so notify the app manually.
        option = option_list.highlighted_option
        new_id = (
            str(option.id) if option is not None and option.id is not None else None
        )
        self._schedule_preview(new_id)
        return True

    def add_sessions(
        self, sessions: list[PublicSession], latest_messages: dict[str, str]
    ) -> None:
        existing = {_session_id(session) for session in self._sessions}
        new_sessions = [
            session for session in sessions if _session_id(session) not in existing
        ]
        if not new_sessions:
            return

        self._sessions = sorted(
            [*self._sessions, *new_sessions], key=_session_sort_key, reverse=True
        )
        self._latest_messages.update(latest_messages)

        option_list = self._option_list()
        highlighted = self._highlighted_option_id()
        option_list.clear_options()
        option_list.add_options([
            Option(self._option_text(session), id=_session_id(session))
            for session in self._sessions
        ])
        self._update_empty_option()
        self._refresh_header()
        if highlighted is None:
            return
        for index, session in enumerate(self._sessions):
            if _session_id(session) == highlighted:
                option_list.highlighted = index
                return

    def load_sessions(
        self, sessions: list[PublicSession], latest_messages: dict[str, str]
    ) -> None:
        """Populate the picker after initial mount. Highlights current session if present."""
        if not self.is_mounted:
            return
        self.add_sessions(sessions, latest_messages)
        self.set_loading(False)
        option_list = self._option_list()
        if option_list.highlighted is not None:
            return
        target_id = self._current_session_id
        for index, session in enumerate(self._sessions):
            if target_id is not None and _session_id(session) == target_id:
                option_list.highlighted = index
                return
        if self._sessions:
            option_list.highlighted = 0

    def _refresh_header(self) -> None:
        header = self.query_one(".sessionpicker-header", NoMarkupStatic)
        header.update(
            _build_header_text(
                self._cwd, rich_theme_style(self.app.theme_variables["text-muted"])
            )
        )

    def clear_pending_delete(self, option_id: str) -> bool:
        if not self._delete_state_matches(option_id, "pending"):
            return False

        self._clear_delete_state()
        return True

    def compose(self) -> ComposeResult:
        options = [
            Option(self._normal_option_text(session), id=_session_id(session))
            for session in self._sessions
        ] or [Option(self._empty_option_text(), id=_EMPTY_OPTION_ID)]
        with Vertical(id="sessionpicker-content"):
            yield NoMarkupStatic(
                _build_header_text(
                    self._cwd, rich_theme_style(self.app.theme_variables["text-muted"])
                ),
                classes="sessionpicker-header",
            )
            with Horizontal(id="sessionpicker-loading") as status:
                status.display = bool(
                    self._sessions and (self._loading or self._load_error)
                )
                yield SpinnerText(classes="sessionpicker-loading-indicator")
                yield NoMarkupStatic(
                    f"{chrome_glyph('running')} Running: Loading sessions"
                    if self._loading and self._sessions
                    else f"{chrome_glyph('error')} Failed: Loading sessions: {self._load_error}"
                    if self._load_error is not None and self._sessions
                    else "",
                    classes="sessionpicker-loading-status",
                )
            option_list = NavigableOptionList(*options, id="sessionpicker-options")
            if self._initial_highlighted is not None:
                option_list.highlighted = self._initial_highlighted
            yield option_list
            with Vertical(id="sessionpicker-delete") as confirmation:
                confirmation.display = False
                yield NoMarkupStatic("")
                yield Button("Cancel", id="sessionpicker-cancel-delete")
                yield Button("Delete session", id="sessionpicker-confirm-delete")
            id_detail = NoMarkupStatic("", id="sessionpicker-id-detail")
            self._id_detail_widget = id_detail
            id_detail.display = False
            yield id_detail
            yield NoMarkupStatic(
                shortcut_hint(
                    (
                        f"{shortcut('↑↓/jk')} Navigate  {shortcut('Enter')} Select  "
                        f"{shortcut('d')} Delete  {shortcut('i')} ID  "
                        f"{shortcut('c')} Copy ID  "
                        if self._sessions and not self._loading
                        else ""
                    )
                    + f"{shortcut('Esc')} Cancel"
                ),
                id="sessionpicker-help",
                classes="sessionpicker-help",
            )

    def on_mount(self) -> None:
        self.call_after_refresh(
            lambda: self.set_loading(self._loading, error=self._load_error)
        )
        option_list = self.query_one(OptionList)
        option_list.focus()
        option = option_list.highlighted_option
        initial_id = (
            str(option.id) if option is not None and option.id is not None else None
        )
        self._schedule_preview(None if initial_id == _EMPTY_OPTION_ID else initial_id)

    def on_unmount(self) -> None:
        # SpinnerText stops its timer on unmount; clear the state so a late
        # discovery result cannot revive a picker that the user dismissed.
        self._loading = False
        self._cancel_pending_preview()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if self._delete_is_pending():
            return

        option_id = str(event.option.id) if event.option.id is not None else None
        self._hide_id_detail()
        if self._delete_state is not None and self._delete_state.option_id != option_id:
            self._clear_delete_state()
        self._schedule_preview(None if option_id == _EMPTY_OPTION_ID else option_id)
        if self.is_mounted:
            self._update_delete_confirmation()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if self._delete_is_pending() or self._selection_pending:
            return

        if event.option.id == _EMPTY_OPTION_ID:
            if self._loading:
                self._set_help_text("Sessions are still loading. Esc Cancel")
            elif self._load_error is not None:
                self._set_help_text("Sessions could not be loaded. Esc Cancel")
            else:
                self._set_help_text("No session available. Esc Cancel")
            return

        if event.option.id:
            option_id = str(event.option.id)
            if self._delete_state_matches(option_id, "confirmation"):
                self._clear_delete_state()
                return

            self._selection_pending = True
            self._update_delete_confirmation()
            self._cancel_pending_preview()
            self.post_message(
                self.SessionSelected(option_id=option_id, session_id=option_id)
            )

    def action_cancel(self) -> None:
        if self._delete_is_pending() or self._selection_pending:
            return

        self._cancel_pending_preview()
        if self._delete_state is not None:
            self._clear_delete_state()
            if self.is_mounted:
                self._option_list().focus()
            return
        if self._id_detail_open:
            self._hide_id_detail()
            if self.is_mounted:
                self._option_list().focus()
            return

        self.post_message(self.Cancelled())

    def action_request_delete(self) -> None:
        if self._delete_is_pending():
            return

        session = self._highlighted_session()
        if session is None:
            return

        session_id = _session_id(session)
        if session_id == self._current_session_id:
            self._show_delete_state(
                session, "feedback", self._delete_feedback_option_text(session)
            )
            return

        if self._delete_state_matches(session_id, "confirmation"):
            return

        self._show_delete_state(
            session, "confirmation", self._delete_confirmation_option_text(session)
        )

    def action_inspect_session_id(self) -> None:
        if self._delete_state is not None or self._selection_pending:
            return
        session = self._highlighted_session()
        id_detail = self._id_detail_widget
        if session is None or id_detail is None or not id_detail.is_mounted:
            return
        if self._id_detail_open:
            self._hide_id_detail()
            return
        id_detail.update(f"Session ID: {_session_id(session)}")
        id_detail.display = True
        self._id_detail_open = True

    def _hide_id_detail(self) -> None:
        if not self._id_detail_open:
            return
        self._id_detail_open = False
        if self._id_detail_widget is not None and self._id_detail_widget.is_mounted:
            self._id_detail_widget.display = False

    def action_copy_session_id(self) -> None:
        if self._delete_state is not None or self._selection_pending:
            return
        session = self._highlighted_session()
        if session is None:
            return
        self.app.copy_to_clipboard(_session_id(session))
        self.app.notify("Session ID copied", timeout=2.0)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "sessionpicker-cancel-delete":
            self._clear_delete_state()
            self._option_list().focus()
        elif event.button.id == "sessionpicker-confirm-delete":
            state = self._delete_state
            if state is None or state.kind != "confirmation":
                return
            session = self._session_by_option_id(state.option_id)
            if session is not None:
                self._confirm_delete(session)

    def _confirm_delete(self, session: _PickerSession) -> None:
        session_id = _session_id(session)
        if not self._delete_state_matches(session_id, "confirmation"):
            return
        self._show_delete_state(
            session, "pending", self._delete_pending_option_text(session)
        )
        self._cancel_pending_preview()
        self.post_message(
            self.SessionDeleteRequested(option_id=session_id, session_id=session_id)
        )
