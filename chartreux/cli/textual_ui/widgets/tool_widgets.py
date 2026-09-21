from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, ClassVar

from pydantic import BaseModel, JsonValue
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Static
from textual.worker import Worker

from chartreux.app_server.models import (
    EffectDetail,
    FileEditEffectOutput as FileEditOutput,
    FileReadEffectOutput as FileReadOutput,
    FileSearchEffectOutput as FileSearchOutput,
    FileWriteEffectOutput as FileWriteOutput,
    ShellEffectOutput as ShellOutput,
    TodoEffectOutput as TodoOutput,
    UserQuestionResult as AskUserQuestionResult,
    WebFetchEffectOutput as WebFetchOutput,
    WebSearchEffectOutput as WebSearchOutput,
    WebSearchEffectSource as WebSearchSourceView,
)
from chartreux.cli.textual_ui.widgets.diff_rendering import (
    DiffOccurrence,
    DiffView,
    language_for_path,
    render_edit_diff_async,
)
from chartreux.cli.textual_ui.widgets.links import LinkStatic, link_content
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic
from chartreux.utils.tool_presentation import ToolEffectKind

_LINE_NUMBER_PREFIX = re.compile(r"^ *\d+→")
_BACKTICK_RUN = re.compile(r"`+")
_UNSAFE_INFO_STRING = re.compile(r"[^A-Za-z0-9_+\-.]")
_MAX_INFO_STRING_LEN = 32

# ANSI escape sequences (CSI, OSC, and other ESC-prefixed forms).
_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\][^\x07\x1b]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_])"
)
# Remaining control bytes to drop (keep tab \x09; newlines handled per line).
_CONTROL_BYTES = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_output(content: str) -> str:
    """Sanitize captured command output for terminal-safe display.

    Command output (e.g. uv's in-place progress bars) carries ANSI escapes,
    carriage-return redraws, and other control bytes. Textual renders these
    straight to the terminal (it does not strip ESC), corrupting the display,
    and scrolling emits a different slice so the glitches shift. Collapse each
    ``\\r``-redrawn line to its final state and strip escape/control bytes.

    Mirrors the webview's ``collapseCarriageReturns``: a trailing ``\\r`` only
    parks the cursor at column 0, so the line it sits on stands until something
    overwrites it.
    """
    cleaned: list[str] = []
    for line in content.replace("\r\n", "\n").split("\n"):
        written = line.rstrip("\r")
        cleaned.append(
            _CONTROL_BYTES.sub("", _ANSI_ESCAPE.sub("", written.rsplit("\r", 1)[-1]))
        )
    return "\n".join(cleaned)


class GenericToolData(BaseModel):
    data: JsonValue = None


def _strip_line_numbers(content: str) -> str:
    """Remove the model-facing ``   12→`` line-number prefixes for CLI display."""
    return "\n".join(_LINE_NUMBER_PREFIX.sub("", line) for line in content.split("\n"))


def _fenced_code_block(content: str, ext: str) -> str:
    """Wrap content in a code fence long enough to survive embedded backticks.

    Untrusted content (file/command output) may contain ``` runs that would
    otherwise break out of a fixed three-backtick fence and render as live
    Markdown. CommonMark resolves this by requiring the fence to be strictly
    longer than any backtick run it encloses.

    ``ext`` is derived from attacker-controlled paths in some call sites, so
    strip anything that could escape the fence's info string (newlines,
    backticks, whitespace) and cap the length defensively.
    """
    safe_ext = _UNSAFE_INFO_STRING.sub("", ext)[:_MAX_INFO_STRING_LEN]
    longest_run = max(
        (len(m.group(0)) for m in _BACKTICK_RUN.finditer(content)), default=0
    )
    fence = "`" * max(3, longest_run + 1)
    return f"{fence}{safe_ext}\n{content}\n{fence}"


class ToolResultWidget[TResult: BaseModel](Static):
    class BorderColorsChanged(Message):
        def __init__(self, result_widget: ToolResultWidget[Any]) -> None:
            self.result_widget = result_widget
            super().__init__()

        @property
        def control(self) -> ToolResultWidget[Any]:
            return self.result_widget

    # When True the whole result collapses into a one-line header; when False it
    # is always rendered in full (used by diff-style results like edit/write).
    COLLAPSIBLE: ClassVar[bool] = True

    def __init__(
        self,
        result: TResult | None,
        success: bool,
        message: str,
        warnings: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.result = result
        self.success = success
        self.message = message
        self.warnings = warnings or []
        self.border_row_colors: dict[int, str] = {}
        self.add_class("tool-result-widget")

    def _footer(self, extra: str | None = None) -> ComposeResult:
        if extra:
            yield NoMarkupStatic(extra, classes="tool-result-hint")

    def _yield_text(
        self, content: str, *, classes: str = "tool-result-detail"
    ) -> Iterable[Widget]:
        cleaned = clean_output(content.strip("\n"))
        if cleaned:
            yield NoMarkupStatic(cleaned, classes=classes)

    def _yield_markdown(self, content: str, *, ext: str) -> Iterable[Widget]:
        if content:
            from textual.widgets import Markdown

            yield Markdown(_fenced_code_block(content.strip("\n"), ext))

    def compose(self) -> ComposeResult:
        if self.result:
            lines = [
                f"{field_name}: {value}"
                for field_name in type(self.result).model_fields
                if (value := getattr(self.result, field_name)) is not None
                and value not in ("", [])
            ]
            if lines:
                yield from self._yield_text("\n".join(lines))
        yield from self._footer()


class GenericToolResultWidget(ToolResultWidget[GenericToolData]):
    def compose(self) -> ComposeResult:
        if self.result and (text := _format_generic_result(self.result.data)):
            yield from self._yield_text(text)
        yield from self._footer()


def _format_generic_result(value: JsonValue) -> str:
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {_format_generic_value(item)}"
            for key, item in value.items()
            if item is not None and item not in ("", [])
        )
    return _format_generic_value(value)


def _format_generic_value(value: JsonValue) -> str:
    if value is None:
        return ""
    return str(value)


class BashResultWidget(ToolResultWidget[ShellOutput]):
    def _collapsed_output(self) -> str:
        return self.result.transcript.strip("\n") if self.result else ""

    def compose(self) -> ComposeResult:
        if not self.result:
            yield from self._footer()
            return
        output = self._collapsed_output()
        if output:
            yield from self._yield_text(output)
        else:
            yield NoMarkupStatic("(no content)", classes="tool-result-detail")
        yield from self._footer()


class WriteFileResultWidget(ToolResultWidget[FileWriteOutput]):
    COLLAPSIBLE = False

    def compose(self) -> ComposeResult:
        if not self.result:
            yield from self._footer()
            return
        if self.result.content:
            yield from self._yield_markdown(
                self.result.content, ext=language_for_path(self.result.file_path)
            )
        yield from self._footer()


class EditResultWidget(ToolResultWidget[FileEditOutput]):
    COLLAPSIBLE = False

    def __init__(
        self,
        result: FileEditOutput | None,
        success: bool,
        message: str,
        warnings: list[str] | None = None,
    ) -> None:
        super().__init__(result, success, message, warnings)
        self._diff_view = DiffView([], ansi=False, dark=True)
        self._requested_render_theme: tuple[bool, bool] | None = None
        self._render_worker: Worker[None] | None = None
        if result is None:
            self._occurrences = []
        elif result.occurrences:
            self._occurrences = [
                DiffOccurrence(item.start_line, item.old_text, item.new_text)
                for item in result.occurrences
            ]
        elif result.old_string is not None and result.new_string is not None:
            self._occurrences = [
                DiffOccurrence(None, result.old_string, result.new_string)
            ]
        else:
            self._occurrences = []

    def compose(self) -> ComposeResult:
        if not self.result:
            yield from self._footer()
            return
        warnings = [
            NoMarkupStatic(f"⚠ {w}", classes="tool-result-warning")
            for w in self.warnings
        ]
        # Wrap the diff in a horizontal-scroll container so wide lines can be
        # scrolled instead of clipped (overflow-x is `auto`, so the scrollbar
        # only shows when a line overruns the width). For a diff taller than the
        # viewport the bar sits at the bottom -- the same trade-off write_file's
        # code fence makes -- but that beats silently truncating long lines.
        yield Vertical(*warnings, self._diff_view, classes="diff-scroll")
        yield from self._footer()

    def on_mount(self) -> None:
        self.request_diff_render(
            ansi=self.app.native_ansi_color, dark=self.app.current_theme.dark
        )

    def request_diff_render(self, *, ansi: bool, dark: bool) -> Worker[None] | None:
        self._diff_view.set_render_mode(ansi=ansi, dark=dark)
        self._requested_render_theme = (ansi, dark)
        if not self.result:
            return None
        if self._render_worker is None or self._render_worker.is_finished:
            self._render_worker = self.run_worker(
                self._drain_diff_renders(), group="edit-diff-render"
            )
        return self._render_worker

    async def _drain_diff_renders(self) -> None:
        rendered_theme: tuple[bool, bool] | None = None
        while self.is_attached and rendered_theme != self._requested_render_theme:
            rendered_theme = self._requested_render_theme
            if rendered_theme is None or not self.result:
                return
            ansi, dark = rendered_theme
            lines = await render_edit_diff_async(
                self._occurrences,
                language_for_path(self.result.file),
                ansi=ansi,
                dark=dark,
            )
            if rendered_theme != self._requested_render_theme:
                continue
            self._diff_view.set_render_data(lines, ansi=ansi, dark=dark)
            # Border rows sit below the warning lines, so shift the diff's own row
            # colors down by the number of warnings.
            self.border_row_colors = {
                len(self.warnings) + row: color
                for row, color in self._diff_view.border_row_colors.items()
            }
            self.post_message(self.BorderColorsChanged(self))


class TodoResultWidget(ToolResultWidget[TodoOutput]):
    def compose(self) -> ComposeResult:
        if not self.result or not self.result.todos:
            yield NoMarkupStatic("No todos", classes="todo-empty")
            yield from self._footer()
            return

        by_status: dict[str, list] = {
            "in_progress": [],
            "pending": [],
            "completed": [],
            "cancelled": [],
        }
        for todo in self.result.todos:
            status = (
                todo.status.value if hasattr(todo.status, "value") else str(todo.status)
            )
            if status in by_status:
                by_status[status].append(todo)

        for status in ["in_progress", "pending", "completed", "cancelled"]:
            for todo in by_status[status]:
                icon = self._get_status_icon(status)
                yield NoMarkupStatic(f"{icon} {todo.content}", classes=f"todo-{status}")
        yield from self._footer()

    def _get_status_icon(self, status: str) -> str:
        icons = {"pending": "☐", "in_progress": "☐", "completed": "☑", "cancelled": "☒"}
        return icons.get(status, "☐")


class ReadResultWidget(ToolResultWidget[FileReadOutput]):
    def compose(self) -> ComposeResult:
        if not self.result:
            yield from self._footer()
            return
        for warning in self.warnings:
            yield NoMarkupStatic(f"⚠ {warning}", classes="tool-result-warning")
        if self.result.content:
            ext = Path(self.result.file_path).suffix.lstrip(".") or "text"
            yield from self._yield_markdown(
                _strip_line_numbers(self.result.content), ext=ext
            )
        yield from self._footer()


class GrepResultWidget(ToolResultWidget[FileSearchOutput]):
    def compose(self) -> ComposeResult:
        for warning in self.warnings:
            yield NoMarkupStatic(f"⚠ {warning}", classes="tool-result-warning")
        if not self.result or not self.result.matches:
            yield from self._footer()
            return
        yield from self._yield_text(self.result.matches)
        yield from self._footer()


class AskUserQuestionResultWidget(ToolResultWidget[AskUserQuestionResult]):
    # Shown as a single wrapping "Answered <question> → <answer>" line on the
    # call widget (see get_result_display); no folded body.
    COLLAPSIBLE = False

    def compose(self) -> ComposeResult:
        yield from ()


class WebSearchResultWidget(ToolResultWidget[WebSearchOutput]):
    @staticmethod
    def _source_content(source: WebSearchSourceView) -> Content:
        label = source.title or source.url
        return Content("  • ") + link_content(label, source.url)

    def compose(self) -> ComposeResult:
        if not self.result:
            yield from self._footer()
            return
        result = self.result
        yield NoMarkupStatic(f"query: {result.query}", classes="tool-result-detail")
        if result.answer:
            yield from self._yield_text(f"answer: {result.answer}")
        if result.sources:
            yield NoMarkupStatic("")
            if len(result.sources) > 1:
                yield NoMarkupStatic("Sources:", classes="tool-result-detail")
            lines = [self._source_content(s) for s in result.sources]
            yield LinkStatic(Content("\n").join(lines), classes="tool-result-detail")
        yield from self._footer()


class WebFetchResultWidget(ToolResultWidget[WebFetchOutput]):
    def compose(self) -> ComposeResult:
        if not self.result:
            yield from self._footer()
            return
        yield from self._yield_text(self.result.content)
        yield from self._footer()


@dataclass(frozen=True, slots=True)
class EffectWidgets:
    output_model: type[BaseModel] | None = None
    result: type[ToolResultWidget] = GenericToolResultWidget
    linkify_result: bool = False


EFFECT_WIDGETS: dict[ToolEffectKind, EffectWidgets] = {
    ToolEffectKind.SHELL: EffectWidgets(
        output_model=ShellOutput, result=BashResultWidget
    ),
    ToolEffectKind.FILE_READ: EffectWidgets(
        output_model=FileReadOutput, result=ReadResultWidget
    ),
    ToolEffectKind.FILE_WRITE: EffectWidgets(
        output_model=FileWriteOutput, result=WriteFileResultWidget
    ),
    ToolEffectKind.FILE_EDIT: EffectWidgets(
        output_model=FileEditOutput, result=EditResultWidget
    ),
    ToolEffectKind.FILE_SEARCH: EffectWidgets(
        output_model=FileSearchOutput, result=GrepResultWidget
    ),
    ToolEffectKind.TODO: EffectWidgets(
        output_model=TodoOutput, result=TodoResultWidget
    ),
    ToolEffectKind.USER_QUESTION: EffectWidgets(
        output_model=AskUserQuestionResult, result=AskUserQuestionResultWidget
    ),
    ToolEffectKind.WEB_SEARCH: EffectWidgets(
        output_model=WebSearchOutput, result=WebSearchResultWidget
    ),
    ToolEffectKind.WEB_FETCH: EffectWidgets(
        output_model=WebFetchOutput, result=WebFetchResultWidget, linkify_result=True
    ),
}


def get_result_widget(
    detail: EffectDetail,
    result: JsonValue,
    success: bool,
    message: str,
    warnings: list[str] | None = None,
) -> ToolResultWidget:
    widgets = EFFECT_WIDGETS.get(detail.kind, EffectWidgets())
    if result is None:
        parsed = None
    elif widgets.output_model is not None:
        parsed = widgets.output_model.model_validate(result)
    else:
        parsed = GenericToolData(data=result)
    return widgets.result(parsed, success, message, warnings)


def linkify_effect_result(detail: EffectDetail) -> bool:
    return EFFECT_WIDGETS.get(detail.kind, EffectWidgets()).linkify_result


def effect_result_is_collapsible(detail: EffectDetail) -> bool:
    return EFFECT_WIDGETS.get(detail.kind, EffectWidgets()).result.COLLAPSIBLE
