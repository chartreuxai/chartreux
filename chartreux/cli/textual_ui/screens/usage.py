"""Live, read-only browser of recorded usage wire snapshots."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import ROUND_DOWN, Decimal, localcontext
import json
from typing import ClassVar

from rich.cells import cell_len, set_cell_size
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.models import (
    UsageModelSummary,
    UsageTotals,
    UsageWindow,
    UsageWindowSummary,
)
from chartreux.app_server.protocol import UsageReadResponse, UsageUpdatedParams
from chartreux.app_server.resources import UsageResource
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

MIN_MODAL_WIDTH = 84
MIN_MODAL_HEIGHT = 28
MIN_SELECTOR_WIDTH = 74


type ReadProvider = Callable[[UsageWindow, str | None], Awaitable[UsageReadResponse]]
type UpdateSubscription = Callable[
    [Callable[[UsageUpdatedParams], None]], Callable[[], None]
]


def row_key(row: UsageModelSummary) -> str:
    """Deployment identity, not row order or a potentially ambiguous label."""
    return json.dumps([row.model, row.provider, row.wire_name])


def tokens(value: int, incomplete: bool) -> str:
    return f"{value:,}{'+' if incomplete else ''}"


TOTAL_KEY = "total"
DECIMAL_SUFFIX_BASE = 1000


def bounded_number(
    value: int | float, width: int, *, prefix: str = "", incomplete: bool = False
) -> str:
    """Decimal K/M/B, with one fractional digit (then none if space is tight).

    All reductions truncate down, including cents. Advance to the next suffix
    only when needed to fit. Scientific notation is a last resort after even
    integral B cannot fit; its mantissa also truncates. Affixes are never cropped.
    An impossibly small budget raises rather than silently clipping a cell.
    """
    number = Decimal(str(value))
    suffix = "+" if incomplete else ""
    budget = width - len(prefix) - len(suffix)
    with localcontext() as context:
        context.prec = max(
            28, len(number.as_tuple().digits) + abs(number.adjusted()) + 4
        )

        def fixed(scaled: Decimal, places: int) -> str:
            return format(
                scaled.quantize(Decimal(1).scaleb(-places), rounding=ROUND_DOWN),
                f".{places}f",
            )

        ordinary = fixed(number, 2) if prefix else str(value)
        if number < DECIMAL_SUFFIX_BASE and len(ordinary) <= budget:
            return prefix + ordinary + suffix
        magnitude = min(9, max(3, (number.adjusted() // 3) * 3))
        for exponent, marker in ((3, "K"), (6, "M"), (9, "B")):
            if exponent < magnitude:
                continue
            for places in (1, 0):
                result = fixed(number.scaleb(-exponent), places) + marker
                if len(result) <= budget:
                    return prefix + result + suffix
        exponent = number.adjusted()
        for places in (1, 0):
            result = f"{fixed(number.scaleb(-exponent), places)}e{exponent}"
            if len(result) <= budget:
                return prefix + result + suffix
    raise ValueError(
        f"Numeric column width {width} cannot retain affixes and magnitude"
    )


def usage_cost(summary: UsageTotals, width: int | None = None) -> str:
    """Unscaled wire-decimal precision in Details; bounded lower bounds in cells."""
    if summary.state != "ready" or (
        isinstance(summary, UsageWindowSummary) and summary.degraded
    ):
        return "—"
    if not summary.has_known_cost and (
        summary.requests > 0 or summary.has_unknown_cost
    ):
        return "Unknown"
    if width is not None:
        return bounded_number(
            summary.known_cost_usd,
            width,
            prefix="$",
            incomplete=summary.has_unknown_cost,
        )
    # str(float) is the wire value's decimal representation, not its binary
    # expansion. Preserve all its digits; padding cents never rounds a lower bound.
    amount = format(Decimal(str(summary.known_cost_usd)), "f")
    whole, _, fraction = amount.partition(".")
    return f"${whole}.{fraction.ljust(2, '0')}{'+' if summary.has_unknown_cost else ''}"


def table_values(totals: UsageTotals, narrow: bool) -> list[str]:
    if totals.state != "ready":
        return ["—"] * (2 if narrow else 5)
    values = [bounded_number(totals.requests, 8)]
    if not narrow:
        values.extend(
            bounded_number(value, 9, incomplete=totals.has_unknown_tokens)
            for value in (
                totals.input_tokens,
                totals.output_tokens,
                totals.cached_input_tokens,
            )
        )
    values.append(usage_cost(totals, 12))
    return values


def exact_totals(totals: UsageTotals) -> str:
    if totals.state != "ready":
        return "Requests: —\nInput: — · Output: — · Cached (subset of input): —\nKnown cost: —"
    return (
        f"Requests: {totals.requests:,}\n"
        f"Input: {tokens(totals.input_tokens, totals.has_unknown_tokens)} "
        f"{chrome_glyph('metadata_separator')} Output: {tokens(totals.output_tokens, totals.has_unknown_tokens)} "
        f"{chrome_glyph('metadata_separator')} Cached (subset of input): {tokens(totals.cached_input_tokens, totals.has_unknown_tokens)}\n"
        f"Known cost: {usage_cost(totals)}"
    )


def identity_cell(value: str, width: int) -> str:
    """Pad identities by terminal cells, marking any hidden suffix explicitly."""
    if cell_len(value) > width:
        marker = chrome_glyph("truncation")
        marker = set_cell_size(marker, min(width, cell_len(marker)))
        return set_cell_size(value, width - cell_len(marker)) + marker
    return set_cell_size(value, width)


def model_columns(identity: str, values: list[str], identity_width: int) -> str:
    compact_widths = [8, 12]
    widths = compact_widths if len(values) == len(compact_widths) else [8, 9, 9, 9, 12]
    return (
        "  "
        + identity_cell(identity, identity_width)
        + " "
        + " ".join(
            " " * max(0, width - cell_len(value)) + value
            for value, width in zip(values, widths, strict=True)
        )
    )


class UsageSelector(Button):
    BINDINGS: ClassVar[list[BindingType]] = [Binding("space", "press", show=False)]


class UsageModels(NavigableOptionList):
    def preserve_position(self) -> Callable[[], None]:
        """Capture single-line row identities before rebuilding the option list."""
        old_ids = [option.id for option in self.options]
        selected = self.highlighted_option.id if self.highlighted_option else None
        selected_index = self.highlighted or 0
        anchor_index = min(int(self.scroll_y), max(0, len(old_ids) - 1))
        anchor = old_ids[anchor_index] if old_ids else None
        offset = self.scroll_y - anchor_index

        def restore() -> None:
            new_ids = [option.id for option in self.options]
            # Missing identities fall back to the nearest surviving position.
            self.highlighted = (
                new_ids.index(selected)
                if selected in new_ids
                else min(selected_index, len(new_ids) - 1)
                if new_ids
                else None
            )
            y = (
                new_ids.index(anchor)
                if anchor in new_ids
                else min(anchor_index, max(0, len(new_ids) - 1))
            ) + offset

            def scroll() -> None:
                self.scroll_to(y=y, animate=False, immediate=True, force=True)

            # clear_options resets scroll and highlighting auto-scrolls. Restore
            # immediately and again after layout has recomputed the scroll bounds.
            scroll()
            self.call_after_refresh(scroll)

        return restore

    def action_select(self) -> None:
        if self.option_count:
            super().action_select()
        elif isinstance(self.screen, UsageScreen):
            self.screen.action_details()

    def action_cursor_up(self) -> None:
        if self.highlighted != 0:
            super().action_cursor_up()

    def action_cursor_down(self) -> None:
        if self.highlighted != self.option_count - 1:
            super().action_cursor_down()

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        event.stop()


class UsageHints(NoMarkupStatic):
    """One shortcut row with pointer targets, without extra focus stops."""

    def __init__(self) -> None:
        super().__init__("", id="usage-hint")
        self.targets: list[tuple[int, int, str]] = []

    def show_shortcuts(self, pairs: list[tuple[str, str, str]]) -> None:
        self.targets = []
        parts = []
        position = 0
        separator = f" {chrome_glyph('metadata_separator')} "
        for key, label, action in pairs:
            part = f"{shortcut(key)} {label}"
            length = cell_len(shortcut_hint(part).plain)
            self.targets.append((position, position + length, action))
            parts.append(part)
            position += length + cell_len(separator)
        self.update(shortcut_hint(separator.join(parts)))

    async def on_click(self, event: events.Click) -> None:
        for start, end, action in self.targets:
            if start <= event.x < end and action:
                event.stop()
                await self.screen.run_action(action)
                break


class UsageDetails(VerticalScroll):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up,k", "scroll_up", show=False),
        Binding("down,j", "scroll_down", show=False),
    ]


class UsageScreen(ModalScreen[None]):
    CSS_PATH = "usage.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", show=False, priority=True),
        Binding("r,ctrl+r", "refresh", show=False),
        Binding("d", "details", show=False),
        Binding("f1", "help", show=False),
        Binding("tab", "next_group", show=False, priority=True),
        Binding("shift+tab", "previous_group", show=False, priority=True),
        Binding("left", "selector_move(-1)", show=False),
        Binding("right", "selector_move(1)", show=False),
    ]
    FOCUS_GROUPS: ClassVar[tuple[tuple[str, ...], ...]] = (
        ("usage-day", "usage-week", "usage-month"),
        ("usage-all", "usage-current"),
        ("usage-models",),
        ("usage-details",),
    )

    def __init__(
        self,
        read_provider: ReadProvider,
        subscribe: UpdateSubscription | None = None,
        *,
        project_key: str | None = None,
    ) -> None:
        super().__init__(id="usage-screen")
        self.read_provider = read_provider
        self.subscribe = subscribe
        self.project_key = project_key
        # Activated request, independent of focus and of the displayed snapshot.
        self.window: UsageWindow = "day"
        self.current_project = False
        self._displayed_window: UsageWindow | None = None
        self._snapshot_current_project: bool | None = None
        self._group_members = [group[0] for group in self.FOCUS_GROUPS]
        self.snapshot: UsageReadResponse | None = None
        self._generation = 0
        self._last_revision: int | None = None
        self._pending_revision: int | None = None
        self._dirty = False
        self._reads_in_flight = 0
        self._freeze_epoch = 0
        self._unsubscribe: Callable[[], None] | None = None
        self._browser_closed = False
        self._expanded = False
        self._details_natural_height: int | None = None
        self._help = False
        self._status = f"Loading recorded usage{chrome_glyph('running')}"

    def compose(self) -> ComposeResult:
        with Vertical(id="usage-content"):
            yield NoMarkupStatic("Usage", id="usage-title")
            with Horizontal(id="usage-selectors"):
                with Horizontal(id="usage-windows"):
                    for name in ("day", "week", "month"):
                        yield UsageSelector(name.title(), id=f"usage-{name}")
                with Horizontal(id="usage-scopes"):
                    yield UsageSelector("All projects", id="usage-all")
                    yield UsageSelector("Current project", id="usage-current")
            yield NoMarkupStatic("", id="usage-status")
            yield NoMarkupStatic("", id="usage-heading")
            yield UsageModels(id="usage-models")
            with UsageDetails(id="usage-details"):
                yield NoMarkupStatic("", id="usage-detail-text")
            yield UsageHints()

    def on_mount(self) -> None:
        if self.subscribe is not None:
            if isinstance(getattr(self.subscribe, "__self__", None), UsageResource):
                # The resource suppresses equal revisions within a host epoch.
                # An equal revision published here is therefore post-reconnect,
                # and must invalidate the displayed snapshot too.
                def callback(update: UsageUpdatedParams) -> None:
                    self.usage_updated(
                        update, post_reconnect=update.revision == self._last_revision
                    )

            else:
                callback = self.usage_updated
            self._unsubscribe = self.subscribe(callback)
        self._resize_surface()
        self.query_one("#usage-day").focus()
        self.action_refresh()

    def on_unmount(self) -> None:
        self._browser_closed = True
        self._generation += 1
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self) -> None:
        self._resize_surface()

    def _resize_surface(self) -> None:
        self._details_natural_height = None
        content = self.query_one("#usage-content", Vertical)
        width, height = self.size
        fullscreen = width < MIN_MODAL_WIDTH or height < MIN_MODAL_HEIGHT
        content.set_class(fullscreen, "fullscreen")
        content.styles.width = "100%" if fullscreen else min(92, width - 2)
        content.styles.height = "100%" if fullscreen else height - 2
        content.border_title = "" if fullscreen else "Usage"
        self.call_after_refresh(self._render_snapshot)

    def usage_updated(
        self, update: UsageUpdatedParams, *, post_reconnect: bool = False
    ) -> None:
        """Coalesce revisions, including a new host's reset revision sequence."""
        if self._browser_closed or (
            update.revision == self._last_revision and not post_reconnect
        ):
            return
        self._last_revision = self._pending_revision = update.revision
        self._dirty = True
        self.call_later(self._schedule_automatic_read)

    def _schedule_automatic_read(self) -> None:
        if (
            self._browser_closed
            or self._expanded
            or self._reads_in_flight
            or not self._dirty
        ):
            return
        self._start_read(automatic=True)

    def action_refresh(self) -> None:
        # Explicit refresh/selector activation may commit even in Details.
        self._generation += 1
        changing_window = (
            not self.snapshot
            or self.window != self.snapshot.window
            or self.current_project != self._snapshot_current_project
        )
        self._status = (
            f"Loading recorded usage: {self.window.title()} / "
            f"{'Current project' if self.current_project else 'All projects'}{chrome_glyph('running')}"
            if changing_window
            else ""
        )
        self._render_snapshot_status()
        self._update_details()
        self._start_read(automatic=False)

    def _start_read(self, *, automatic: bool) -> None:
        self._dirty = False
        self._pending_revision = None
        self._reads_in_flight += 1
        self.run_worker(
            self._read(
                self._generation,
                self.window,
                self.project_key if self.current_project else None,
                automatic=automatic,
                freeze_epoch=self._freeze_epoch,
            ),
            group="usage-read",
        )

    async def _read(
        self,
        generation: int,
        window: UsageWindow,
        project: str | None,
        *,
        automatic: bool,
        freeze_epoch: int,
    ) -> None:
        try:
            try:
                response = await self.read_provider(window, project)
            except Exception:
                if generation == self._generation and not self._browser_closed:
                    retained = (
                        "previous snapshot retained" if self.snapshot else "no snapshot"
                    )
                    self._status = f"Failed: Usage read failed · {retained} · r Retry"
                    self._render_snapshot_status()
                    self._update_details()
                return
            if generation != self._generation or self._browser_closed:
                return
            # UsageResource publishes the read response before returning it. That
            # callback is covered by this result, not a reason to read forever.
            if self._pending_revision == response.revision:
                self._dirty = False
                self._pending_revision = None
            if not self._dirty:
                self._last_revision = response.revision
            if automatic and (self._expanded or freeze_epoch != self._freeze_epoch):
                self._dirty = True
                return
            self.snapshot = response.model_copy(deep=True)
            # Even an unavailable summary is an accepted snapshot.
            self._displayed_window = response.window
            self._snapshot_current_project = project is not None
            self.project_key = response.project_key
            self._status = ""
            self._render_snapshot()
        finally:
            self._reads_in_flight -= 1
            self.call_later(self._schedule_automatic_read)

    def _selected(self) -> str | None:
        option = self.query_one(UsageModels).highlighted_option
        return option.id if option else None

    def _render_snapshot_status(self) -> None:
        status = self._status
        snapshot = self.snapshot
        if not status and snapshot:
            summary = getattr(snapshot.summaries, snapshot.window)
            if summary.state != "ready":
                status = (
                    f"Loading recorded usage{chrome_glyph('running')}"
                    if summary.state == "loading"
                    else "Unavailable: Recorded usage could not be read. Refresh to retry."
                )
        if not snapshot:
            options = self.query_one(UsageModels)
            label = (
                "Read failed"
                if status.startswith("Failed:")
                else "Loading" + chrome_glyph("running")
            )
            prompt = Content(f"{chrome_glyph('cursor')} {label}")
            if options.option_count:
                options.replace_option_prompt_at_index(0, prompt)
            else:
                options.add_option(Option(prompt, id="empty"))
                options.highlighted = 0
        widget = self.query_one("#usage-status", NoMarkupStatic)
        widget.display = bool(status)
        widget.update(
            Content.assemble((
                status,
                "$error"
                if status.startswith("Failed:")
                else "$primary"
                if status.startswith("Loading")
                else "$warning"
                if "Incomplete" in status or "Unavailable" in status
                else "$text-muted",
            ))
        )

    def _render_snapshot(self) -> None:
        if not self.is_mounted:
            return
        for name in ("day", "week", "month", "all", "current"):
            # No markers before the first accepted snapshot exists.
            active = name == self._displayed_window or (
                self._snapshot_current_project is not None
                and name == ("current" if self._snapshot_current_project else "all")
            )
            button = self.query_one(f"#usage-{name}", Button)
            label = (
                name.title()
                if name in {"day", "week", "month"}
                else "All projects"
                if name == "all"
                else "Current project"
            )
            button.label = f"{'(*)' if active else '( )'} {label}"
            # Button label changes need an explicit cell budget even before a read
            # completes; otherwise the loading/error selectors retain old widths.
            button.styles.width = cell_len(str(button.label)) + 2
        current = self.query_one("#usage-current", Button)
        current.disabled = self.project_key is None
        current.tooltip = "No attached project" if current.disabled else None
        width = self.query_one("#usage-content").content_size.width
        compact = width < MIN_SELECTOR_WIDTH
        self.query_one("#usage-content").set_class(compact, "compact")
        narrow = width < MIN_MODAL_WIDTH
        # Leave room for the model list's scrollbar even before it is visible.
        identity_width = min(30, max(1, width - 26)) if narrow else 24
        headings = (
            ["Requests", "Cost (USD)"]
            if narrow
            else ["Requests", "Input", "Output", "Cached", "Cost (USD)"]
        )
        self.query_one("#usage-heading", NoMarkupStatic).update(
            model_columns("Model / provider", headings, identity_width)
        )
        options = self.query_one(UsageModels)
        restore_position = options.preserve_position()
        rows = self.snapshot.models if self.snapshot else []
        options.clear_options()
        for row in rows:
            identity = f"{row.model} / {row.provider}"
            if (
                sum(
                    other.model == row.model and other.provider == row.provider
                    for other in rows
                )
                > 1
            ):
                identity = f"{row.model} / {row.wire_name} / {row.provider}"
            values = table_values(row, narrow)
            options.add_option(
                Option(
                    Content(model_columns(identity, values, identity_width)),
                    id=row_key(row),
                )
            )
        if self.snapshot:
            summary = getattr(self.snapshot.summaries, self.snapshot.window)
            options.add_option(
                Option(
                    Content(
                        model_columns(
                            "TOTAL", table_values(summary, narrow), identity_width
                        )
                    ),
                    id=TOTAL_KEY,
                )
            )
        options.can_focus = True
        restore_position()
        self._render_snapshot_status()
        self._update_details()

    def _details_text(self) -> str:
        rows = self.snapshot.models if self.snapshot else []
        row = next((row for row in rows if row_key(row) == self._selected()), None)
        if not self.snapshot:
            text = (
                "No model rows available."
                if self._status.startswith("Failed:")
                else "Model rows will be available when recorded usage finishes loading."
            )
        else:
            text = "Snapshot details"
        if row:
            text = f"Model: {row.model}\nProvider: {row.provider} {chrome_glyph('metadata_separator')} Wire name: {row.wire_name}\n{exact_totals(row)}"
        if self.snapshot:
            summary = getattr(self.snapshot.summaries, self.snapshot.window)
            selected_totals = row or summary
            if self._selected() == TOTAL_KEY:
                text = "TOTAL\n" + exact_totals(summary)
                for label, component in (
                    ("Uncached input", self.snapshot.components.uncached_input),
                    (
                        "Cached input (subset of input)",
                        self.snapshot.components.cached_input,
                    ),
                    ("Output", self.snapshot.components.output),
                ):
                    cost = UsageTotals(
                        state=summary.state,
                        requests=summary.requests,
                        **component.model_dump(
                            exclude={"tokens", "has_unknown_tokens"}
                        ),
                    )
                    text += f"\n{label}: {tokens(component.tokens, component.has_unknown_tokens) if summary.state == 'ready' else '—'} / {usage_cost(cost) if not summary.degraded else '—'}"
            if selected_totals.has_unknown_cost or selected_totals.has_unknown_tokens:
                text += "\nIncomplete accounting: + denotes a known lower bound; unreported cost or tokens are unknown."
            if summary.state != "ready":
                text += f"\nSnapshot state: {summary.state}"
            elif not summary.requests:
                text += "\nNo recorded calls in this window."
            if summary.degraded:
                text += "\nIncomplete coverage: snapshot is degraded; aggregate cost is unavailable."
        if self.snapshot and self.snapshot.warnings:
            text += "\nIncomplete coverage: " + ", ".join(
                w.code for w in self.snapshot.warnings
            )
        if self.project_key is None:
            text += "\nNo attached project: Current project is unavailable."
        if self._status.startswith("Failed:"):
            text += "\n" + self._status
        if self._help:
            text = (
                "Tab/Shift+Tab: move between groups (period, scope, table, open Details).\n"
                "Left/Right: move within selectors; Up/Down: move through model rows and TOTAL, or scroll Details. Arrows stop at group boundaries.\n"
                "Enter/Space: activate a selector to fetch a snapshot. Navigation alone never fetches; (*) marks the displayed snapshot.\n"
                "d: Details; Enter on a row: Details; Esc: Back from Details, then Close.\n"
                "The shortcut row offers pointer actions: r Refresh, d Details, Esc Close/Back. F1: Help.\n"
                "Status-line Today/Week/Month spend is recorded across all projects; the Current project filter here does not change it.\n"
                "Cached tokens are a subset of input. + means incomplete; Unknown means nothing priced.\n"
                "No attached project: Current project is unavailable until the attached project's identity is supplied."
            )
        return text

    def _size_table_and_details(self) -> None:
        """Give spare table rows to Details, retaining its eight-row minimum."""
        # A queued refresh callback may run after dismissal removes the children.
        if self._browser_closed or not self.is_mounted:
            return
        content = self.query_one("#usage-content", Vertical)
        models = self.query_one(UsageModels)
        details = self.query_one(UsageDetails)
        if not self._expanded:
            models.styles.height = "1fr"
            return
        fixed_height = sum(
            self.query_one(selector).outer_size.height
            for selector in (
                "#usage-title",
                "#usage-selectors",
                "#usage-status",
                "#usage-heading",
                "#usage-hint",
            )
            if self.query_one(selector).display
        )
        available = max(0, content.content_size.height - fixed_height)
        minimum = min(8, max(0, available - 1))
        table_height = min(max(1, models.option_count), available - minimum)
        models.styles.height = table_height
        if self._details_natural_height is None:
            # Measure once per inspection mode / viewport, not per selected row.
            detail_text = self.query_one("#usage-detail-text", NoMarkupStatic)
            self._details_natural_height = detail_text.get_content_height(
                content.content_size, self.size, content.content_size.width
            )
        details.styles.height = min(
            available - table_height, max(minimum, self._details_natural_height)
        )

    def _update_details(self) -> None:
        details = self.query_one(UsageDetails)
        scroll_y = details.scroll_y
        details.display = self._expanded
        details.can_focus = self._expanded
        if not self._expanded:
            details.scroll_to(y=0, animate=False, immediate=True)
        self.query_one("#usage-detail-text", NoMarkupStatic).update(
            self._details_text() if self._expanded else ""
        )
        if self._expanded:
            self.call_after_refresh(
                lambda: details.scroll_to(y=scroll_y, animate=False, immediate=True)
            )
        actions = (
            [
                ("Tab", "Groups", "next_group"),
                ("Arrows", "Move", ""),
                ("r", "Refresh", "refresh"),
                ("d", "Details", "details"),
                ("F1", "Help", "help"),
                ("Esc", "Close", "close"),
            ]
            if not self._expanded
            else [
                ("Arrows", "Scroll", ""),
                ("r", "Refresh", "refresh"),
                ("d", "Hide details", "details"),
                ("F1", "Help", "help"),
                ("Esc", "Back", "close"),
            ]
        )
        hint = self.query_one(UsageHints)
        if (
            sum(cell_len(f"{key} {label}") for key, label, _ in actions)
            + 3 * (len(actions) - 1)
            > hint.content_size.width
        ):
            actions = [
                ("r", "Refresh", "refresh"),
                ("d", "Details" if not self._expanded else "Hide details", "details"),
                ("F1", "Help", "help"),
                ("Esc", "Back" if self._expanded else "Close", "close"),
            ]
        hint.show_shortcuts(actions)
        # Selector/status geometry may have changed in this same refresh.
        self.call_after_refresh(self._size_table_and_details)

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        options = self.query_one(UsageModels)
        for index, option in enumerate(options.options):
            prompt = option.prompt
            if isinstance(prompt, Content):
                options.replace_option_prompt_at_index(
                    index,
                    Content(
                        (
                            chrome_glyph("cursor")
                            if index == options.highlighted
                            else " "
                        )
                        + prompt.plain[1:]
                    ),
                )
        self._update_details()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        # The state row opens snapshot-level Details, never a model selection.
        self.action_details()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        name = (event.button.id or "").removeprefix("usage-")
        if name in {"day", "week", "month"}:
            self.window = (
                "day" if name == "day" else "week" if name == "week" else "month"
            )
            self.action_refresh()
            self._render_snapshot()
        elif name in {"all", "current"}:
            self.current_project = name == "current"
            self.action_refresh()
            self._render_snapshot()

    def action_details(self) -> None:
        self._details_natural_height = None
        self._expanded = not self._expanded
        if self._expanded:
            self._freeze_epoch += 1
        else:
            self.call_later(self._schedule_automatic_read)
        self._help = False
        self._update_details()
        (
            self.query_one(UsageDetails)
            if self._expanded
            else self.query_one(UsageModels)
        ).focus()

    def action_help(self) -> None:
        if not self._help:
            self._details_natural_height = None
        if not self._expanded:
            self._freeze_epoch += 1
        self._expanded = True
        self._help = True
        self._update_details()
        self.query_one(UsageDetails).focus()

    def action_close(self) -> None:
        if self._expanded:
            self._expanded = self._help = False
            self._update_details()
            self.query_one(UsageModels).focus()
            self.call_later(self._schedule_automatic_read)
            return
        self._browser_closed = True
        self._generation += 1
        self.workers.cancel_group(self, "usage-read")
        self.dismiss(None)

    def on_descendant_focus(self, event: events.DescendantFocus) -> None:
        for index, group in enumerate(self.FOCUS_GROUPS):
            if event.widget.id in group:
                self._group_members[index] = event.widget.id or group[0]
                break

    def _focused_group(self) -> int | None:
        focused = self.focused
        return next(
            (
                index
                for index, group in enumerate(self.FOCUS_GROUPS)
                if focused is not None and focused.id in group
            ),
            None,
        )

    def _move_group(self, direction: int) -> None:
        current = self._focused_group()
        if current is not None and self.focused and self.focused.id:
            self._group_members[current] = self.focused.id
        start = current if current is not None else -1 if direction > 0 else 0
        for step in range(1, len(self.FOCUS_GROUPS) + 1):
            index = (start + direction * step) % len(self.FOCUS_GROUPS)
            members = [
                self.query_one(f"#{member}") for member in self.FOCUS_GROUPS[index]
            ]
            available = [member for member in members if member.focusable]
            if available:
                target = next(
                    (
                        member
                        for member in available
                        if member.id == self._group_members[index]
                    ),
                    available[0],
                )
                target.focus()
                return

    def action_previous_group(self) -> None:
        self._move_group(-1)

    def action_next_group(self) -> None:
        self._move_group(1)

    def action_selector_move(self, direction: int) -> None:
        group_index = self._focused_group()
        if group_index not in {0, 1} or self.focused is None:
            return
        group = self.FOCUS_GROUPS[group_index]
        position = group.index(self.focused.id or "")
        for index in range(
            position + direction, len(group) if direction > 0 else -1, direction
        ):
            member = self.query_one(f"#{group[index]}")
            if member.focusable:
                member.focus()
                self._group_members[group_index] = group[index]
                return
