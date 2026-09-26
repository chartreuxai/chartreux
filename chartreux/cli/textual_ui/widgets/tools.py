from __future__ import annotations

from typing import TYPE_CHECKING

from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.visual import VisualType
from textual.widget import Widget
from textual.widgets import Static

if TYPE_CHECKING:
    from textual.timer import Timer

from chartreux.app_server.models import (
    CancelledEffectState,
    CompletedEffectState,
    EffectCallDisplay,
    EffectResultDisplay,
    EffectState,
    FailedEffectState,
    PublicEffectEntry,
    ShellEffectDetail,
    SkippedEffectState,
)
from chartreux.cli.textual_ui.widgets.collapsible import (
    ClickWithoutDragMixin,
    CollapsibleSection,
    HeaderCollapsibleSection,
    OverflowCollapsibleSection,
    lines_label,
)
from chartreux.cli.textual_ui.widgets.entry_expansion import EntryExpansionState
from chartreux.cli.textual_ui.widgets.links import LinkStatic, linkify_urls_in_text
from chartreux.cli.textual_ui.widgets.messages import ExpandingBorder
from chartreux.cli.textual_ui.widgets.status_message import (
    IndicatorState,
    StatusMessage,
)
from chartreux.cli.textual_ui.widgets.tool_grouping import (
    GroupIndicator,
    TimelineStatus,
    ToolGroupExpansionState,
    ToolGroupKey,
    effect_state_to_indicator,
    is_manual_shell_entry,
)
from chartreux.cli.textual_ui.widgets.tool_widgets import (
    ToolResultWidget,
    clean_output,
    effect_result_is_collapsible,
    get_result_widget,
    linkify_effect_result,
    shell_output_body,
    shell_output_is_large,
)
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic, NonSelectableStatic
from chartreux.utils.tool_presentation import ToolEffectKind

TOOL_STREAM_WRITE_FRAME_SECONDS = 0.05

_TOOL_CATEGORY_LABELS: dict[ToolEffectKind, str] = {
    ToolEffectKind.FILE_READ: "read files",
    ToolEffectKind.FILE_EDIT: "edited files",
    ToolEffectKind.FILE_WRITE: "wrote files",
    ToolEffectKind.FILE_SEARCH: "searched files",
    ToolEffectKind.SHELL: "ran commands",
    ToolEffectKind.WEB_SEARCH: "searched the web",
    ToolEffectKind.WEB_FETCH: "fetched pages",
    ToolEffectKind.TODO: "updated todos",
    ToolEffectKind.USER_QUESTION: "asked questions",
    ToolEffectKind.SKILL: "loaded skills",
    ToolEffectKind.SUBAGENT: "ran subagents",
    ToolEffectKind.WORKTREE: "created worktrees",
    ToolEffectKind.TOOL: "called tools",
}

_TOOL_CATEGORY_RUNNING_LABELS: dict[ToolEffectKind, str] = {
    ToolEffectKind.FILE_READ: "reading files",
    ToolEffectKind.FILE_EDIT: "editing files",
    ToolEffectKind.FILE_WRITE: "writing files",
    ToolEffectKind.FILE_SEARCH: "searching files",
    ToolEffectKind.SHELL: "running commands",
    ToolEffectKind.WEB_SEARCH: "searching the web",
    ToolEffectKind.WEB_FETCH: "fetching pages",
    ToolEffectKind.TODO: "updating todos",
    ToolEffectKind.USER_QUESTION: "asking questions",
    ToolEffectKind.SKILL: "loading skills",
    ToolEffectKind.SUBAGENT: "running subagents",
    ToolEffectKind.WORKTREE: "creating worktrees",
    ToolEffectKind.TOOL: "calling tools",
}


def _category_label(kind: ToolEffectKind, *, running: bool = False) -> str:
    labels = _TOOL_CATEGORY_RUNNING_LABELS if running else _TOOL_CATEGORY_LABELS
    return labels.get(kind, "calling tools" if running else "called tools")


def _group_indicator_state(indicator: GroupIndicator) -> IndicatorState:
    return IndicatorState(indicator.value)


def _failed_header_display(
    entry: PublicEffectEntry, state: FailedEffectState
) -> EffectResultDisplay:
    call_display = entry.detail.display
    if call_display.settled_message is None:
        return state.display
    return EffectResultDisplay(
        success=False,
        verb=call_display.settled_verb,
        message=call_display.settled_message,
        suffix=call_display.suffix,
    )


def _result_is_collapsible(entry: PublicEffectEntry) -> bool:
    """Whether an effect result folds into a one-line header.

    The output of a user-issued ``!`` command is the point of the timeline
    entry, so it remains expanded. Agent shell calls keep normal folding.
    """
    return not is_manual_shell_entry(entry) and effect_result_is_collapsible(
        entry.detail
    )


class ToolGroupHeader(ClickWithoutDragMixin, StatusMessage):
    """Collapsible, persisted summary line for a :class:`ToolGroup`."""

    SETTLED_GLYPH = "⏵"

    def __init__(self) -> None:
        super().__init__(initial_text="")
        self._categories: list[ToolEffectKind] = []
        self._has_reasoning = False
        self._last_rendered_text: VisualType | None = None
        self._last_state = IndicatorState.SUCCESS
        self._is_collapsed = True
        self.add_class("tool-group-header")

    def compose(self) -> ComposeResult:
        with Horizontal(classes="tool-call-header"):
            self._indicator_widget = NonSelectableStatic(
                self._spinner.current_frame(), classes="status-indicator-icon"
            )
            yield self._indicator_widget
            self._text_widget = LinkStatic("", classes="status-indicator-text")
            yield self._text_widget

    def add_category(self, kind: ToolEffectKind) -> None:
        if kind not in self._categories:
            self._categories.append(kind)
        self._update_text()

    def mark_reasoning(self) -> None:
        self._has_reasoning = True
        self._update_text()

    def settle(self, state: IndicatorState) -> None:
        """Remember a call outcome while keeping the group visibly running."""
        self._last_state = state

    def stop_spinning(self, success: bool = True) -> None:
        super().settle(self._last_state)
        self._update_disclosure_glyph()

    def resume(self) -> None:
        """Resume the summary spinner when a later call joins this group."""
        self._is_spinning = True
        self.update_display()
        if self.is_mounted and self._spinner_timer is None:
            self.start_spinner_timer()

    def set_collapsed(self, collapsed: bool) -> None:
        self._is_collapsed = collapsed
        if not self._is_spinning:
            self._update_disclosure_glyph()

    def get_content(self) -> str:
        labels = [
            _category_label(kind, running=self._is_spinning)
            for kind in self._categories
        ]
        if self._has_reasoning:
            labels.append("thinking" if self._is_spinning else "thought")
        return ", ".join(labels).capitalize()

    def _update_disclosure_glyph(self) -> None:
        if self._indicator_widget is not None:
            self._indicator_widget.update(
                "⏵" if self._is_collapsed else "⏷", layout=False
            )

    def update_display(self) -> None:
        if self._indicator_widget is None or self._text_widget is None:
            return

        if self._is_spinning:
            self._indicator_widget.update(self._spinner.next_frame(), layout=False)
        else:
            self._indicator_widget.update(
                self.SETTLED_GLYPH or self._state.glyph, layout=False
            )

        for state in IndicatorState:
            self._indicator_widget.set_class(
                not self._is_spinning and state is self._state, state.css_class
            )

        self._update_text()

    def _update_text(self) -> None:
        text = self._format_text(self.get_content())
        if self._text_widget is not None and text != self._last_rendered_text:
            self._text_widget.update(text)
            self._last_rendered_text = text

    async def on_click(self, event: events.Click) -> None:
        if self._click_is_passive(event):
            return
        event.stop()
        if isinstance(self.parent, ToolGroup):
            self.parent.set_collapsed(not self.parent.is_collapsed)


class ToolGroup(Vertical):
    """A grouped tool timeline with a collapsible summary and indented body."""

    def __init__(
        self,
        *,
        key: ToolGroupKey | None = None,
        expansion_state: ToolGroupExpansionState | None = None,
    ) -> None:
        super().__init__(classes="tool-group")
        self._key = key
        self._expansion_state = expansion_state
        self._header = ToolGroupHeader()
        self._content = Vertical(classes="tool-group-content tool-group")
        self._border = ExpandingBorder(classes="tool-result-border")
        self._timeline_status = TimelineStatus()
        self._is_collapsed = (
            expansion_state.register(key)
            if key is not None and expansion_state is not None
            else True
        )

    def compose(self) -> ComposeResult:
        yield self._header
        self._content.display = not self._is_collapsed
        self._border.display = not self._is_collapsed
        with Horizontal(classes="tool-group-body"):
            yield self._border
            yield self._content

    def on_mount(self) -> None:
        self._header.set_collapsed(self._is_collapsed)

    @property
    def content_container(self) -> Vertical:
        return self._content

    @property
    def header(self) -> ToolGroupHeader:
        return self._header

    @property
    def is_collapsed(self) -> bool:
        return self._is_collapsed

    def add_call_kind(self, kind: ToolEffectKind) -> None:
        self._header.add_category(kind)

    def mark_reasoning(self) -> None:
        self._header.mark_reasoning()

    def settle_indicator(self, state: IndicatorState | GroupIndicator) -> None:
        indicator = (
            _group_indicator_state(state)
            if isinstance(state, GroupIndicator)
            else state
        )
        self._header.settle(indicator)

    def record_effect(self, timeline_index: int, state: EffectState) -> None:
        """Record a terminal outcome using D23's timeline-order policy."""
        self._timeline_status.record_effect(timeline_index, state)
        if indicator := self._timeline_status.indicator:
            self.settle_indicator(indicator)

    def forget_effect(self, timeline_index: int) -> None:
        """Forget an effect removed from the retained transcript window."""
        self._timeline_status.forget_effect(timeline_index)
        self.settle_indicator(self._timeline_status.indicator or GroupIndicator.SUCCESS)
        if not self._header._is_spinning:
            self._header.stop_spinning()

    def failure_is_muted(self, timeline_index: int) -> bool:
        """Whether this call's error is resolved by a later successful call."""
        return self._timeline_status.failure_is_muted(timeline_index)

    def settle_effect(self, state: EffectState) -> None:
        """Settle from an effect state through D23's canonical mapping."""
        self.settle_indicator(effect_state_to_indicator(state))

    def finalize(self) -> None:
        self._header.stop_spinning()

    def resume(self) -> None:
        self._header.resume()

    def set_collapsed(self, collapsed: bool) -> None:
        self._is_collapsed = collapsed
        self._content.display = not collapsed
        self._border.display = not collapsed
        self._header.set_collapsed(collapsed)
        if self._key is not None and self._expansion_state is not None:
            self._expansion_state.set_collapsed(self._key, collapsed)

    def add_content_child(self, widget: Widget) -> None:
        """Pre-mount a history child before this group is attached to an app."""
        self._content._add_child(widget)

    def sync_visibility(self) -> None:
        """Hide groups whose body contains only hidden timeline entries."""
        self.display = any(child.display for child in self._content.children)


class ToolCallMessage(StatusMessage):
    SETTLED_GLYPH = "⏵"

    def __init__(self, entry: PublicEffectEntry) -> None:
        self._entry = entry
        self._tool_name = entry.detail.tool_name
        self._stream_widget: NoMarkupStatic | None = None
        self._stream_message_buffer: str | None = None
        self._stream_write_timer: Timer | None = None
        self._suffix_widget: NoMarkupStatic | None = None
        self._verb_widget: NoMarkupStatic | None = None
        self._header_row: Horizontal | None = None

        super().__init__()
        self.add_class("tool-call")

        if isinstance(entry.state, CompletedEffectState):
            self._is_spinning = False
            self._state = (
                IndicatorState.SUCCESS
                if entry.state.display.success
                else IndicatorState.ERROR
            )
        elif isinstance(entry.state, FailedEffectState):
            self._is_spinning = False
            self._state = IndicatorState.ERROR
        elif isinstance(entry.state, SkippedEffectState | CancelledEffectState):
            self._is_spinning = False
            self._state = IndicatorState.MUTED

    def compose(self) -> ComposeResult:
        with Vertical(classes="tool-call-container"):
            self._header_row = Horizontal(classes="tool-call-header")
            with self._header_row:
                self._indicator_widget = NonSelectableStatic(
                    self._spinner.current_frame(), classes="status-indicator-icon"
                )
                yield self._indicator_widget
                self._verb_widget = NoMarkupStatic(
                    "", classes="collapsible-header-verb"
                )
                self._verb_widget.display = False
                yield self._verb_widget
                self._text_widget = LinkStatic("", classes="status-indicator-text")
                yield self._text_widget
                self._suffix_widget = NoMarkupStatic(
                    "", classes="status-indicator-suffix"
                )
                self._suffix_widget.display = False
                yield self._suffix_widget
            self._stream_widget = NoMarkupStatic("", classes="tool-stream-message")
            self._stream_widget.display = False
            yield self._stream_widget

    def on_mount(self) -> None:
        super().on_mount()
        self.recompute_gap()

    def recompute_gap(self) -> None:
        # Outside a group (history / standalone), collapse the gap when the
        # previous visible widget is another tool widget. Inside a group the CSS
        # packs children directly, so there is nothing to toggle.
        # "no-gap" is a ToolResultMessage signal, intentionally not a CSS hook.
        self.set_class(self._follows_visible_tool_widget(), "no-gap")

    def _follows_visible_tool_widget(self) -> bool:
        if self.parent is None or self.parent.has_class("tool-group"):
            return False
        siblings = list(self.parent.children)
        idx = siblings.index(self) if self in siblings else -1
        if idx <= 0:
            return False
        prev_idx = idx - 1
        while prev_idx > 0 and not siblings[prev_idx].display:
            prev_idx -= 1
        prev = siblings[prev_idx]
        return prev.display and isinstance(prev, (ToolCallMessage, ToolResultMessage))

    @property
    def tool_call_id(self) -> str:
        return self._entry.id

    def get_content(self) -> str:
        return self._header_parts()[1]

    def get_content_suffix(self) -> str:
        return self._header_parts()[2]

    def _header_parts(self) -> tuple[str, str, str]:
        if display := self._settled_display():
            return display.verb, display.message, display.suffix
        display = self._call_display()
        message = display.message if display.message is not None else display.summary
        return display.verb, message, display.suffix

    def _settled_display(self) -> EffectResultDisplay | None:
        state = self._entry.state
        if isinstance(state, FailedEffectState):
            return _failed_header_display(self._entry, state)
        if isinstance(state, CompletedEffectState | SkippedEffectState):
            return state.display
        if isinstance(state, CancelledEffectState):
            return state.display
        return None

    def _call_display(self) -> EffectCallDisplay:
        return self._entry.detail.display

    def update_entry(self, entry: PublicEffectEntry) -> None:
        previous_header = self._header_parts()
        self._entry = entry
        self._tool_name = entry.detail.tool_name
        verb, message, suffix = self._header_parts()
        if (verb, message, suffix) != previous_header:
            self._set_text(message, suffix, verb=verb)

    def set_stream_message(self, message: str) -> None:
        """Coalesce stream-message refreshes while retaining the latest delta."""
        if self._stream_widget is None:
            return
        self._stream_message_buffer = message
        if not self.is_mounted:
            self._flush_stream_message()
        elif self._stream_write_timer is None:
            self._stream_write_timer = self.set_timer(
                TOOL_STREAM_WRITE_FRAME_SECONDS, self._flush_stream_message
            )

    def _flush_stream_message(self) -> None:
        self._cancel_stream_write_timer()
        message = self._stream_message_buffer
        self._stream_message_buffer = None
        if message is None or self._stream_widget is None:
            return
        self._stream_widget.update(f"→ {message}")
        self._stream_widget.display = True

    def _cancel_stream_write_timer(self) -> None:
        if self._stream_write_timer is not None:
            self._stream_write_timer.stop()
            self._stream_write_timer = None

    def on_unmount(self) -> None:
        self._cancel_stream_write_timer()
        self._stream_message_buffer = None

    def settle(self, state: IndicatorState) -> None:
        self._flush_stream_message()
        super().settle(state)
        if self._stream_widget is None:
            return
        self._stream_widget.update("")
        self._stream_widget.display = False

    def set_result_text(
        self, text: str, suffix: str = "", *, verb: str = "", linkify: bool = False
    ) -> None:
        self._flush_stream_message()
        self._set_text(text, suffix, verb=verb, linkify=linkify)

    def _set_text(
        self, text: str, suffix: str, *, verb: str = "", linkify: bool = False
    ) -> None:
        if self._verb_widget:
            self._verb_widget.update(verb)
            self._verb_widget.display = bool(verb)
        if self._text_widget:
            content = linkify_urls_in_text(text) if linkify else text
            self._text_widget.update(content)
        self._update_suffix(suffix)

    def _update_suffix(self, suffix: str) -> None:
        if self._suffix_widget:
            self._suffix_widget.update(suffix)
            self._suffix_widget.display = bool(suffix)
        if self._header_row is not None:
            # With a suffix present the title drops to auto width so the suffix
            # sits right after it; otherwise the title takes 1fr and wraps.
            self._header_row.set_class(bool(suffix), "has-suffix")

    def update_display(self) -> None:
        super().update_display()
        verb, _, suffix = self._header_parts()
        if self._header_row is not None:
            self._header_row.set_class(self._is_spinning, "running")
            self._header_row.set_class(
                effect_result_is_collapsible(self._entry.detail), "collapsible-result"
            )
        if self._verb_widget:
            self._verb_widget.update(verb)
            self._verb_widget.display = bool(verb)
        self._update_suffix(suffix)

    def show_muted(self) -> None:
        # Neutral grey square with the call summary -- used for an error whose
        # verdict is still unknown, a user-declined call, and a cancelled call.
        self.settle(IndicatorState.MUTED)

    def escalate_error(self) -> None:
        # No recovery followed: promote the held square to a hard red cross.
        self.settle(IndicatorState.ERROR)


class ToolResultMessage(ClickWithoutDragMixin, Static):
    def __init__(
        self,
        entry: PublicEffectEntry,
        call_widget: ToolCallMessage | None = None,
        *,
        expansion_state: EntryExpansionState | None = None,
    ) -> None:
        self._entry = entry
        self._expansion_state = expansion_state
        if expansion_state is not None and (
            _result_is_collapsible(entry) or isinstance(entry.state, FailedEffectState)
        ):
            expansion_state.register(entry.id)
        self._call_widget = call_widget
        self._tool_name = entry.detail.tool_name
        self._content_container: Vertical | None = None
        self._result_widget: ToolResultWidget | None = None
        self._is_collapsible = self._determine_collapsible()
        self._is_error = False
        # History restoration decides this from the shared group policy before
        # the widget mounts; live rendering escalates directly at group end.
        self._should_escalate = False
        # The collapsed error/skip section whose triangle carries the muted
        # state (and gets recoloured red on escalation).
        self._muted_section: HeaderCollapsibleSection | None = None

        super().__init__()
        self.add_class("tool-result")

    @property
    def tool_name(self) -> str:
        return self._tool_name

    def _determine_collapsible(self) -> bool:
        return _result_is_collapsible(self._entry)

    def compose(self) -> ComposeResult:
        if self._is_collapsible:
            # Collapsible results mount their section directly onto this widget
            # (see `_render_result_collapsible`); no intermediate container.
            return
        with Horizontal(classes="tool-result-container"):
            self._border = ExpandingBorder(classes="tool-result-border")
            yield self._border
            self._content_container = Vertical(classes="tool-result-content")
            yield self._content_container

    async def on_mount(self) -> None:
        if self._call_widget:
            if isinstance(self._state, FailedEffectState):
                # Start muted; the verdict (recoverable vs terminal) lands later.
                self._call_widget.show_muted()
                verb, message, suffix = self._get_result_parts()
                self._call_widget.set_result_text(message, suffix, verb=verb)
            elif isinstance(self._state, SkippedEffectState | CancelledEffectState):
                # A declined/denied call is the user's choice, not a failure.
                self._call_widget.show_muted()
                verb, message, suffix = self._get_result_parts()
                self._call_widget.set_result_text(message, suffix, verb=verb)
            else:
                success = self._determine_success()
                if success:
                    self._call_widget.stop_spinning(success=True)
                else:
                    # CompletedEffectState with success=False (for example, a
                    # non-zero exit code) follows the same muting/escalation
                    # policy as an explicit failed effect.
                    self._call_widget.show_muted()
                    self._is_error = True
                # Collapsible results fold into a header and hide the call
                # widget, so its inline text is only set for expanded results.
                if not self._is_collapsible:
                    verb, message, suffix = self._get_result_parts()
                    linkify = linkify_effect_result(self._entry.detail)
                    self._call_widget.set_result_text(
                        message, suffix, verb=verb, linkify=linkify
                    )
        self.recompute_gap()
        await self._render_result()
        if self._should_escalate:
            self.escalate_error()

    def recompute_gap(self) -> None:
        self.set_class(self._needs_standalone_gap(), "has-gap")

    def _needs_standalone_gap(self) -> bool:
        # Grouped results are packed with no gaps (CSS); a standalone collapsible
        # result keeps a gap unless its call widget was itself gap-collapsed.
        if self.parent is not None and self.parent.has_class("tool-group"):
            return False
        if self._call_widget is None or not self._is_collapsible:
            return False
        return "no-gap" not in self._call_widget.classes

    def _muted_header_text(self) -> str:
        # The call widget reflects the settled display when one is available and
        # otherwise falls back to the summary of what was attempted.
        if self._call_widget is not None:
            summary = self._call_widget.get_content()
            if summary:
                return summary
        return self._get_result_text()

    @staticmethod
    def _bordered(body: Widget) -> Horizontal:
        """Wrap a result body in the expanding-border container used by all
        collapsible result bodies.
        """
        return Horizontal(
            ExpandingBorder(classes="tool-result-border"),
            Vertical(body, classes="tool-result-content"),
            classes="tool-result-container",
        )

    def set_error_escalation(self, *, escalate: bool) -> None:
        """Set pre-mount escalation selected by the shared grouping policy."""
        self._should_escalate = escalate

    def escalate_error(self) -> None:
        # Turn ended without a follow-up tool call: promote the held muted state
        # to a hard error. For collapsible results the arrow turns red; for
        # expanded (call-widget) results the call icon turns into a red cross.
        if not self._is_error:
            return
        if self._muted_section is not None:
            self._muted_section.mark_error()
        elif self._call_widget is not None:
            self._call_widget.escalate_error()
            verb, message, suffix = self._get_result_parts()
            self._call_widget.set_result_text(message, suffix, verb=verb)

    def _determine_success(self) -> bool:
        display = self._result_display()
        return display.success if display is not None else False

    def _manual_shell_output(self) -> str | None:
        """Return terminal-safe output retained by a failed manual ``!`` command."""
        if not is_manual_shell_entry(self._entry):
            return None
        if not isinstance(self._entry.detail, ShellEffectDetail):
            return None
        if not isinstance(self._state, FailedEffectState | CancelledEffectState):
            return None
        cleaned = clean_output(self._state.output_text)
        return (
            cleaned
            if shell_output_is_large(cleaned)
            else cleaned.strip("\n") or "(no output)"
        )

    def _get_result_parts(self) -> tuple[str, str, str]:
        if isinstance(self._state, FailedEffectState):
            display = _failed_header_display(self._entry, self._state)
            return display.verb, display.message, display.suffix
        if isinstance(self._state, SkippedEffectState | CancelledEffectState):
            return "", f"{self._tool_name}: skipped", ""
        if display := self._result_display():
            return display.verb, display.message, display.suffix
        return "", f"{self._tool_name} completed", ""

    def _get_result_text(self) -> str:
        verb, message, _ = self._get_result_parts()
        return f"{verb} {message}".strip() if verb else message

    async def _render_result(self) -> None:
        if self._is_collapsible:
            await self._render_result_collapsible()
        else:
            await self._render_result_expanded()

    async def _mount_section(
        self, section: CollapsibleSection, container: Widget
    ) -> None:
        state = self._expansion_state
        if state is not None:
            collapsed = state.register(self._entry.id)
            section.on_collapse_changed = lambda value: state.set_collapsed(
                self._entry.id, value
            )
        else:
            collapsed = True
        await container.mount(section)
        section.set_collapsed(collapsed)

    async def _render_result_collapsible(self) -> None:
        # Bodies are built lazily (factory closures): a collapsed result keeps
        # only its header, and the heavy content widget (Markdown/Syntax/diff) is
        # created on first expand and torn down on collapse. See
        # `CollapsibleSection`.
        await self.remove_children()

        if isinstance(self._state, FailedEffectState):
            self._is_error = True
            # Fold the whole result into a single muted-arrow header; the error
            # detail is the collapsed, bordered body. No separate square icon or
            # "N lines" row. Escalation recolours red.
            error = clean_output(self._state.error.message)
            output = self._manual_shell_output()
            verb, message, suffix = self._get_result_parts()

            def build_error_body() -> Widget:
                error_widget = Static(
                    Content.from_markup("[$error]Error[/]: ") + Content(error)
                )
                body: Widget = (
                    Vertical(error_widget, shell_output_body(output))
                    if output is not None
                    else error_widget
                )
                return self._bordered(body)

            self._muted_section = HeaderCollapsibleSection(
                build_error_body,
                header_text=message,
                header_verb=verb,
                header_suffix=suffix,
                header_muted=True,
            )
            await self._mount_section(self._muted_section, self)
            if self._call_widget:
                self._call_widget.display = False
            self.display = True
            return

        if isinstance(self._state, SkippedEffectState | CancelledEffectState):
            reason = self._state.reason
            output = self._manual_shell_output()

            def build_skipped_body() -> Widget:
                reason_widget = NoMarkupStatic(f"Skipped: {reason}")
                return self._bordered(
                    Vertical(reason_widget, shell_output_body(output))
                    if output is not None
                    else reason_widget
                )

            self._muted_section = HeaderCollapsibleSection(
                build_skipped_body,
                header_text=self._muted_header_text(),
                header_muted=True,
            )
            await self._mount_section(self._muted_section, self)
            if self._call_widget:
                self._call_widget.display = False
            self.display = True
            return

        self.remove_class("error-text")
        self.remove_class("warning-text")

        display = self._result_display()
        if display is None:
            self.display = False
            return

        output = (
            self._state.output
            if isinstance(self._state, CompletedEffectState)
            else None
        )
        # A hook that replaces a tool result leaves no structured output; the model-facing
        # text lives in output_text. Fall back to it so the replacement (e.g. a deny
        # reason) is still visible when unfolded.
        fallback_text = (
            clean_output(self._state.output_text).strip()
            if output is None and isinstance(self._state, CompletedEffectState)
            else ""
        )

        def build_result_body() -> Widget:
            if output is None and fallback_text:
                widget: Widget = NoMarkupStatic(
                    fallback_text, classes="tool-result-detail"
                )
            else:
                widget = get_result_widget(
                    self._entry.detail,
                    output,
                    success=display.success,
                    message=display.message,
                    warnings=display.warnings,
                )
                self._result_widget = widget
            return Horizontal(
                ExpandingBorder(classes="tool-result-border"),
                Vertical(widget, classes="tool-result-content"),
                classes="tool-result-container",
            )

        # The header is inert only when there is genuinely nothing to unfold: no
        # structured output, no fallback text, and no warnings.
        has_body = not (output is None and not fallback_text and not display.warnings)
        section = HeaderCollapsibleSection(
            build_result_body,
            header_text=display.message,
            header_verb=display.verb,
            header_suffix=display.suffix,
            header_success=display.success,
            header_muted=not display.success,
            collapsible=has_body,
        )
        self._muted_section = section if not display.success else None
        await self._mount_section(section, self)
        if self._call_widget:
            self._call_widget.display = False
        self.display = True

    async def _render_result_expanded(self) -> None:
        if self._content_container is None:
            return

        await self._content_container.remove_children()

        if isinstance(self._state, FailedEffectState):
            self._is_error = True
            # Only the inline "Error" span is ever colored; escalation changes the
            # call icon to a red cross but leaves this folded body untouched.
            message = clean_output(self._state.error.message)
            output = self._manual_shell_output()
            line_count = len(message.strip("\n").split("\n"))

            def build_error_body() -> Widget:
                error_widget = Static(
                    Content.from_markup("[$error]Error[/]: ") + Content(message)
                )
                return (
                    Vertical(error_widget, shell_output_body(output))
                    if output is not None
                    else error_widget
                )

            await self._mount_section(
                OverflowCollapsibleSection(
                    build_error_body, collapsed_label=lines_label(line_count)
                ),
                self._content_container,
            )
            self.display = True
            return

        if isinstance(self._state, SkippedEffectState | CancelledEffectState):
            self.add_class("warning-text")
            reason = self._state.reason
            output = self._manual_shell_output()
            await self._content_container.mount(NoMarkupStatic(f"Skipped: {reason}"))
            if output is not None:
                await self._content_container.mount(shell_output_body(output))
            self.display = True
            return

        self.remove_class("error-text")
        self.remove_class("warning-text")

        display = self._result_display()
        if display is None:
            self.display = False
            return

        widget = get_result_widget(
            self._entry.detail,
            self._state.output
            if isinstance(self._state, CompletedEffectState)
            else None,
            success=display.success,
            message=display.message,
            warnings=display.warnings,
        )
        await self._content_container.mount(widget)
        self._result_widget = widget
        self._apply_border_colors()
        self.display = bool(widget.children)

    @property
    def _state(self) -> EffectState:
        return self._entry.state

    def _result_display(self) -> EffectResultDisplay | None:
        match self._state:
            case (
                CompletedEffectState()
                | FailedEffectState()
                | SkippedEffectState() as state
            ):
                return state.display
            case CancelledEffectState() as state:
                return state.display
            case _:
                return None

    def _apply_border_colors(self) -> None:
        if self._result_widget is None:
            return
        self._border.set_row_colors(self._result_widget.border_row_colors)

    def on_tool_result_widget_border_colors_changed(
        self, message: ToolResultWidget.BorderColorsChanged
    ) -> None:
        if message.control is self._result_widget:
            self._apply_border_colors()

    async def on_click(self, event: events.Click) -> None:
        if self._click_is_passive(event):
            return
        sections = list(self.query(CollapsibleSection))
        if sections:
            sections[0].toggle()
