from __future__ import annotations

import asyncio
from datetime import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

from chartreux.app_server.models import (
    FileImageSource,
    HookSeverity,
    ImageAttachment,
    InlineImageSource,
)
from chartreux.observability.logging import logger
from chartreux.utils.io import read_safe_async

if TYPE_CHECKING:
    from textual.timer import Timer
    from textual.widgets import Markdown
    from textual.widgets._markdown import MarkdownStream

    from chartreux.cli.textual_ui.app import ChatScroll


from textual import events
from textual.app import ComposeResult
from textual.color import Color
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.geometry import Size
from textual.reactive import reactive
from textual.strip import Strip
from textual.visual import Visual
from textual.widget import Widget
from textual.widgets import Link, Static

from chartreux.cli.textual_ui.widgets.collapsible import (
    ClickWithoutDragMixin,
    DisclosureHeader,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.message_header import MessageHeader
from chartreux.cli.textual_ui.widgets.spinner import SpinnerMixin, SpinnerType
from chartreux.cli.textual_ui.widgets.tool_widgets import clean_output
from chartreux.ui.chrome_glyphs import ascii_chrome_enabled, chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic, NonSelectableStatic

# Full-content MarkdownStream reparsing per delta dominated CPU; 50ms frames cut it ~40%.
STREAM_WRITE_FRAME_SECONDS = 0.05


class ExpandingBorder(NonSelectableStatic):
    def __init__(self, *, classes: str | None = None) -> None:
        super().__init__(classes=classes)
        self._row_colors: dict[int, str] = {}

    def set_row_colors(self, colors: dict[int, str]) -> None:
        self._row_colors = colors
        self.refresh()

    # The border is always a single glyph column. Returning a constant avoids the
    # default measurement, which renders the widget (reading self.size) during
    # arrange and forces an O(N) compositor map rebuild on every layout pass.
    def get_content_width(self, container: Size, viewport: Size) -> int:
        return 1

    def render_line(self, y: int) -> Strip:
        if not self.size.width or y < 0 or y >= self.size.height:
            return Strip.blank(self.size.width, self.visual_style.rich_style)
        char = "⎣" if y == self.size.height - 1 else "⎢"
        color = self._row_colors.get(y)
        content = Content.styled(char, color) if color else Content(char)
        strip = Visual.to_strips(self, content, 1, 1, self.visual_style)[0]
        return strip.apply_offsets(0, y)


# Mimic a border bottom with this component in order to have dimmed colors in ANSI themes
# Move back to border when Textual supports dimmed borders or foreground-muted in ANSI themes
class ExpandingSeparator(NonSelectableStatic):
    def render(self) -> str:
        return "─" * max(self.size.width, 1)


def _attachment_label(attachment: ImageAttachment) -> str:
    alias_path = Path(attachment.alias).expanduser()
    if not alias_path.is_absolute():
        return attachment.alias
    return _format_display_path(alias_path)


def _format_display_path(path: Path) -> str:
    home = Path.home()
    try:
        relative = path.relative_to(home)
    except ValueError:
        return str(path)
    if str(relative) == ".":
        return "~"
    return str(Path("~") / relative)


class UserMessageAttachment(Horizontal):
    def __init__(self, attachment: ImageAttachment) -> None:
        super().__init__(classes="user-message-attachment-line")
        self._attachment = attachment

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic(
            "└ attached image: ", classes="user-message-attachment-label"
        )
        match self._attachment.source:
            case FileImageSource(path=path):
                image_path = Path(path)
                yield Link(
                    _attachment_label(self._attachment),
                    url=image_path.as_uri(),
                    classes="user-message-attachment-link",
                )
            case InlineImageSource():
                # Inline images have no file on disk, so there's nothing to link.
                yield NoMarkupStatic(
                    _attachment_label(self._attachment),
                    classes="user-message-attachment-link",
                )


class UserMessage(Static):
    PROMPT_CHAR: ClassVar[str] = "You"
    SHOW_SEPARATOR: ClassVar[bool] = False

    def __init__(
        self,
        content: str,
        pending: bool = False,
        history_entry_id: str | None = None,
        images: list[ImageAttachment] | None = None,
        *,
        posted_at: datetime | None = None,
        show_message_timestamps: bool = True,
    ) -> None:
        super().__init__()
        self.header = MessageHeader(
            self.PROMPT_CHAR, show_message_timestamps=show_message_timestamps
        )
        self.posted_at = posted_at
        self.add_class("user-message")
        self._content = content
        self._pending = pending
        self.header.set_timestamp(None if pending else posted_at)
        self._images = images or []
        self.history_entry_id = history_entry_id

    def get_content(self) -> str:
        return self._content

    def update_content(self, content: str) -> None:
        self._content = content
        try:
            content_widget = self.query_one(".user-message-content", NoMarkupStatic)
            content_widget.update(content)
        except Exception:
            pass

    @property
    def pending(self) -> bool:
        return self._pending

    def compose(self) -> ComposeResult:
        self.set_class(
            ascii_chrome_enabled()
            or os.environ.get("NO_COLOR") is not None
            or bool(self.app.console.no_color),
            "plain-user-accent",
        )
        with Vertical(classes="user-message-wrapper"):
            self.header.set_timestamp(None if self._pending else self.posted_at)
            yield self.header
            with Horizontal(classes="user-message-container"):
                yield NonSelectableStatic(">", classes="user-message-prompt")
                yield NoMarkupStatic(self._content, classes="user-message-content")
            if self._images:
                with Vertical(classes="user-message-attachments"):
                    for image in self._images:
                        yield UserMessageAttachment(image)
            if self.SHOW_SEPARATOR:
                yield ExpandingSeparator(classes="user-message-separator")
            if self._pending:
                self.add_class("pending")

    def on_mount(self) -> None:
        self._apply_accent_tint()
        self.watch(self.app, "theme", self._refresh_accent_tint, init=False)

    def _refresh_accent_tint(self) -> None:
        # App updates its CSS variables after notifying theme watchers.
        self.call_after_refresh(self._apply_accent_tint)

    def _apply_accent_tint(self) -> None:
        wrapper = self.query_one(".user-message-wrapper")
        if self.has_class("plain-user-accent"):
            wrapper.styles.background = None
            return
        accent = Color.parse(self.app.theme_variables["accent"])
        # Textual treats ANSI destinations as opaque even with TCSS alpha.
        # Resolve their theme-provided RGB channels before applying the tint.
        if accent.ansi is not None:
            base = Color.parse(
                "ansi_black" if self.app.current_theme.dark else "ansi_white"
            )
            wrapper.styles.background = Color(base.r, base.g, base.b).blend(
                Color(accent.r, accent.g, accent.b), 0.08
            )
        else:
            wrapper.styles.background = None

    @staticmethod
    def _attachment_label(attachment: ImageAttachment) -> str:
        return _attachment_label(attachment)

    @staticmethod
    def _format_display_path(path: Path) -> str:
        return _format_display_path(path)

    async def set_pending(self, pending: bool) -> None:
        if pending == self._pending:
            return

        self._pending = pending
        self.header.set_timestamp(None if pending else self.posted_at)

        if pending:
            self.add_class("pending")
            return

        self.remove_class("pending")

    def reconcile_timestamp(self, posted_at: datetime | None) -> None:
        """Apply history's authority without stamping optimistic or queued input."""
        self.posted_at = posted_at
        self.header.set_timestamp(None if self._pending else posted_at)

    def set_show_message_timestamps(self, show: bool) -> None:
        self.header.set_timestamp(
            None if self._pending else self.posted_at, show_message_timestamps=show
        )

    def set_show_separator(self, show: bool) -> None:
        self.set_class(not show, "no-separator")

    def on_resize(self) -> None:
        self.header._refresh_time()

    def set_follows_previous(self, follows: bool) -> None:
        self.set_class(follows, "follows-user")


class QueueHeaderMessage(Static):
    DEFAULT_LABEL = "» Queued"
    PAUSED_LABEL = f"» Queued — press {shortcut('Enter')} to send, type to add"

    def __init__(self, *, paused: bool = False) -> None:
        super().__init__()
        self.add_class("queue-header-message")
        self._paused = paused
        self._label_widget: NoMarkupStatic | None = None

    def compose(self) -> ComposeResult:
        with Vertical(classes="queue-header-container"):
            self._label_widget = NoMarkupStatic(
                shortcut_hint(self._current_label()), classes="queue-header-content"
            )
            yield self._label_widget
            yield ExpandingSeparator(classes="queue-header-separator")

    def set_paused(self, paused: bool) -> None:
        if paused == self._paused:
            return
        self._paused = paused
        if self._label_widget is not None:
            self._label_widget.update(shortcut_hint(self._current_label()))

    def _current_label(self) -> str:
        return self.PAUSED_LABEL if self._paused else self.DEFAULT_LABEL


class SlashCommandMessage(UserMessage):
    PROMPT_CHAR = "You"
    SHOW_SEPARATOR = False

    def __init__(self, content: str, pending: bool = False) -> None:
        super().__init__(
            content if content.startswith("/") else f"/{content}", pending=pending
        )
        self.add_class("slash-command-message")


class StreamingMessageBase(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self._content = content
        self._markdown: Markdown | None = None
        self._stream: MarkdownStream | None = None
        self._write_timer: Timer | None = None
        self._content_initialized = False
        self._to_write_buffer = ""

    def _get_markdown(self) -> Markdown:
        if self._markdown is None:
            raise RuntimeError(
                "Markdown widget not initialized. compose() must be called first."
            )
        return self._markdown

    def _ensure_stream(self) -> MarkdownStream:
        if self._stream is None:
            from textual.widgets import Markdown

            self._stream = Markdown.get_stream(self._get_markdown())
        return self._stream

    def _is_chat_at_bottom(self) -> bool:
        try:
            chat = cast("ChatScroll", self.app.query_one("#chat"))
            return chat.is_at_bottom
        except Exception:
            return True

    async def append_content(self, content: str) -> None:
        if not content:
            return

        self._content += content

        if not self._should_write_content():
            return

        self._to_write_buffer += content
        if self._is_chat_at_bottom() and self._write_timer is None:
            self._write_timer = self.set_timer(
                STREAM_WRITE_FRAME_SECONDS, self._flush_write_buffer
            )

    async def _flush_write_buffer(self) -> None:
        self._write_timer = None
        if (
            not self._to_write_buffer
            or not self._should_write_content()
            or not self._is_chat_at_bottom()
        ):
            return

        to_write = self._to_write_buffer
        self._to_write_buffer = ""
        stream = self._ensure_stream()
        await stream.write(to_write)

    def _cancel_write_timer(self) -> None:
        if self._write_timer is not None:
            self._write_timer.stop()
            self._write_timer = None

    async def write_initial_content(self) -> None:
        if self._content_initialized:
            return
        self._content_initialized = True
        if self._content and self._should_write_content():
            self._cancel_write_timer()
            stream = self._ensure_stream()
            await stream.write(self._content)
            self._to_write_buffer = ""

    async def stop_stream(self) -> None:
        self._cancel_write_timer()
        if self._to_write_buffer and self._should_write_content():
            stream = self._ensure_stream()
            await stream.write(self._to_write_buffer)
        self._to_write_buffer = ""

        if self._stream is None:
            return

        await self._stream.stop()
        self._stream = None

    def _should_write_content(self) -> bool:
        return True

    def get_content(self) -> str:
        return self._content

    def is_stripped_content_empty(self) -> bool:
        return self._content.strip() == ""


class AssistantMessage(StreamingMessageBase):
    def __init__(
        self,
        content: str,
        *,
        posted_at: datetime | None = None,
        turn_duration_ms: float | None = None,
        show_message_timestamps: bool = True,
    ) -> None:
        super().__init__(content)
        self.header = MessageHeader(
            "Assistant",
            posted_at=posted_at,
            turn_duration_ms=turn_duration_ms,
            show_message_timestamps=show_message_timestamps,
        )
        self.add_class("assistant-message")

    def reconcile_timing(self, duration_ms: float | None) -> None:
        self.header.set_turn_duration(duration_ms)

    def on_resize(self) -> None:
        self.header._refresh_time()

    def reconcile_timestamp(self, posted_at: datetime | None) -> None:
        self.header.set_timestamp(posted_at)

    def set_show_message_timestamps(self, show: bool) -> None:
        self.header.set_timestamp(self.header.posted_at, show_message_timestamps=show)

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        markdown = Markdown("")
        self._markdown = markdown
        yield self.header
        yield markdown


class ReasoningMessage(ClickWithoutDragMixin, SpinnerMixin, StreamingMessageBase):
    SPINNER_TYPE = SpinnerType.PULSE
    SPINNING_TEXT = "Thinking"
    COMPLETED_TEXT = "Thought"

    def __init__(
        self,
        content: str,
        collapsed: bool = True,
        *,
        completed: bool = False,
        entry_id: str | None = None,
        expansion_state: EntryExpansionState | None = None,
    ) -> None:
        super().__init__(content)
        self.add_class("reasoning-message")
        self.entry_id = entry_id
        self._expansion_state = expansion_state
        self.collapsed = (
            expansion_state.register(entry_id)
            if entry_id is not None and expansion_state is not None
            else collapsed
        )
        self._indicator_widget: Static | None = None
        self._header_widget: Horizontal | None = None
        self.init_spinner()
        self._is_spinning = not completed

    def compose(self) -> ComposeResult:
        with Vertical(classes="reasoning-message-wrapper"):
            self._header_widget = DisclosureHeader(
                activate=self._toggle_collapsed, classes="reasoning-message-header"
            )
            with self._header_widget:
                self._indicator_widget = NonSelectableStatic(
                    self._spinner.current_frame()
                    if self._is_spinning
                    else chrome_glyph("checked"),
                    classes="reasoning-indicator",
                )
                yield self._indicator_widget
                self._status_text_widget = NoMarkupStatic(
                    self.SPINNING_TEXT if self._is_spinning else self.COMPLETED_TEXT,
                    classes="reasoning-collapsed-text",
                )
                yield self._status_text_widget
            from textual.widgets import Markdown

            markdown = Markdown("", classes="reasoning-message-content")
            markdown.display = not self.collapsed
            self._markdown = markdown
            yield markdown

    def on_mount(self) -> None:
        if self._is_spinning:
            self.start_spinner_timer()

    def on_resize(self) -> None:
        self.refresh_spinner()

    def stop_spinning(self, success: bool = True) -> None:
        super().stop_spinning(success)
        if self._indicator_widget:
            self._indicator_widget.remove_class("success", "error")
            self._indicator_widget.update(
                chrome_glyph(
                    "disclosure_closed" if self.collapsed else "disclosure_open"
                )
            )

    def _is_click_on_toggle(self, event: events.Click) -> bool:
        return self._is_click_within(event, self._header_widget)

    async def on_click(self, event: events.Click) -> None:
        if not self._is_click_on_toggle(event) or self._click_is_passive(event):
            return
        await self._toggle_collapsed()

    async def _toggle_collapsed(self) -> None:
        await self.set_collapsed(not self.collapsed)

    def _should_write_content(self) -> bool:
        return not self.collapsed

    async def set_collapsed(self, collapsed: bool) -> None:
        if self.collapsed == collapsed:
            return

        focused = self.screen.focused
        if (
            collapsed
            and self._markdown is not None
            and self._markdown.is_mounted
            and focused is not None
            and self._markdown in focused.ancestors_with_self
        ):
            assert self._header_widget is not None
            self._header_widget.focus()
        self.collapsed = collapsed
        if self.entry_id is not None and self._expansion_state is not None:
            self._expansion_state.set_collapsed(self.entry_id, collapsed)
        if self._indicator_widget and not self._is_spinning:
            self._indicator_widget.update(
                chrome_glyph("disclosure_closed" if collapsed else "disclosure_open")
            )
        if self._markdown:
            self._markdown.display = not collapsed
            if not collapsed and self._content:
                self._cancel_write_timer()
                if self._stream is not None:
                    await self._stream.stop()
                    self._stream = None
                await self._markdown.update("")
                stream = self._ensure_stream()
                await stream.write(self._content)
                self._to_write_buffer = ""


class UserCommandMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.add_class("user-command-message")
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        with Horizontal(classes="user-command-container"):
            yield ExpandingBorder(classes="user-command-border")
            with Vertical(classes="user-command-content"):
                yield Markdown(self._content)


class WhatsNewMessage(Static):
    def __init__(self, content: str) -> None:
        super().__init__()
        self.add_class("whats-new-message")
        self._content = content

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        yield Markdown(self._content)


class GreetingMessage(Static):
    def __init__(self, username: str) -> None:
        super().__init__()
        self.add_class("greeting-message")
        self._username = username

    def compose(self) -> ComposeResult:
        yield NoMarkupStatic(f"Hello {self._username}, how can I help you?")


class CustomToolsDeprecationMessage(Static):
    def __init__(self, tool_names: list[str]) -> None:
        super().__init__()
        self.add_class("custom-tools-deprecation-message")
        names = ", ".join(f"`{name}`" for name in sorted(tool_names))
        replacement = "a skill" if len(tool_names) == 1 else "skills"
        self._content = (
            "**Support for custom tools will be deprecated soon.** "
            f"Ask Chartreux to help replace yours ({names}) with {replacement}."
        )

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        yield Markdown(self._content)


class InterruptMessage(Static):
    def __init__(self) -> None:
        super().__init__()
        self.add_class("interrupt-message")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="interrupt-container"):
            yield ExpandingBorder(classes="interrupt-border")
            yield NoMarkupStatic(
                "Interrupted — completed tool results are kept; in-progress effects may be partial. What next?",
                classes="interrupt-content",
            )


class ErrorMessage(Static):
    def __init__(
        self, error: str | Content, collapsed: bool = False, show_border: bool = True
    ) -> None:
        super().__init__()
        self.add_class("error-message")
        self._error = error
        self.collapsed = collapsed
        self._show_border = show_border
        self._content_widget: Static | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="error-container"):
            if self._show_border:
                yield ExpandingBorder(classes="error-border")
            error = (
                self._error
                if isinstance(self._error, Content)
                else Content(clean_output(self._error))
            )
            text = Content("Error: ") + error if self._show_border else error
            self._content_widget = NoMarkupStatic(text, classes="error-content")
            yield self._content_widget

    def set_collapsed(self, collapsed: bool) -> None:
        pass


class HookRunContainer(Vertical):
    def __init__(self) -> None:
        super().__init__(classes="hook-run-container")
        self.display = False

    async def add_message(self, widget: HookSystemMessageLine) -> None:
        await self.mount(widget)
        self.display = True


_HOOK_SEVERITY_ICONS: dict[HookSeverity, str] = {
    HookSeverity.OK: "success",
    HookSeverity.WARNING: "warning",
    HookSeverity.ERROR: "error",
}


class HookSystemMessageLine(Static):
    def __init__(
        self,
        hook_name: str,
        content: str,
        severity: HookSeverity = HookSeverity.WARNING,
    ) -> None:
        super().__init__()
        self.add_class("hook-system-message")
        self.add_class(f"hook-severity-{severity}")
        self._hook_name = hook_name
        self._content = content
        self._severity = severity

    def compose(self) -> ComposeResult:
        icon = chrome_glyph(
            _HOOK_SEVERITY_ICONS.get(
                self._severity, _HOOK_SEVERITY_ICONS[HookSeverity.WARNING]
            )
        )
        wording = {
            HookSeverity.OK: "Saved:",
            HookSeverity.WARNING: "Warning:",
            HookSeverity.ERROR: "Failed:",
        }.get(self._severity, "Warning:")
        with Horizontal(classes="hook-system-container"):
            yield NonSelectableStatic(icon, classes="hook-system-icon")
            yield NoMarkupStatic(
                f"[{self._hook_name}] {wording} {self._content}",
                classes="hook-system-content",
            )


class WarningMessage(Static):
    def __init__(self, message: str, show_border: bool = True) -> None:
        super().__init__()
        self.add_class("warning-message")
        self._message = message
        self._show_border = show_border

    def compose(self) -> ComposeResult:
        with Horizontal(classes="warning-container"):
            if self._show_border:
                yield ExpandingBorder(classes="warning-border")
            yield NoMarkupStatic(
                f"! Warning: {self._message}", classes="warning-content"
            )


class PlanFileMessage(Widget):
    content: reactive[str] = reactive("")

    def __init__(self, file_path: Path) -> None:
        super().__init__()
        self.add_class("plan-file-message")
        self._file_path = file_path
        self._watch_task: asyncio.Task | None = None

    def compose(self) -> ComposeResult:
        from textual.widgets import Markdown

        with Vertical(classes="plan-file-wrapper"):
            yield Markdown(self.content, classes="plan-file-content")

    def watch_content(self, new_content: str) -> None:
        from textual.widgets import Markdown

        try:
            self.query_one(Markdown).update(new_content)
        except NoMatches:
            pass

    async def on_mount(self) -> None:
        self.content = (await read_safe_async(self._file_path)).text
        self._watch_task = asyncio.create_task(self._watch_file())

    async def _watch_file(self) -> None:
        from watchfiles import awatch

        try:
            async for _ in awatch(self._file_path):
                self.content = (await read_safe_async(self._file_path)).text
        except (asyncio.CancelledError, FileNotFoundError):
            pass

    def open_in_editor(self) -> None:
        from chartreux.cli.textual_ui.external_editor import ExternalEditor

        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            with self.app.suspend():
                ExternalEditor.edit_file(self._file_path)
        except OSError:
            logger.warning(
                "Failed to open plan file in editor: %s", self._file_path, exc_info=True
            )
            self.app.notify(
                f"Could not open plan in editor: {self._file_path}",
                severity="error",
                timeout=6,
            )

    def stop_watching(self) -> None:
        if self._watch_task is None:
            return

        if not self._watch_task.done():
            self._watch_task.cancel()

        self._watch_task = None
