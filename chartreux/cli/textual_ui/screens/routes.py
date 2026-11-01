"""Read-only browser of the conversation's LIVE purpose-route table."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from rich.cells import cell_len
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import OptionList
from textual.widgets.option_list import Option

from chartreux.app_server.config import RouteTargetView, RouteView
from chartreux.route_labels import PREVIEW_LABEL, UNKNOWN
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.navigable_option_list import NavigableOptionList
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic

MIN_MODAL_WIDTH = 72
MIN_MODAL_HEIGHT = 20

#: The live table plus the one reason it cannot be inspected at all, if any.
type RoutesRead = Callable[[], "RoutesSnapshot"]
type RoutesUpdate = Callable[[Callable[[], None]], Callable[[], None]]


@dataclass(frozen=True)
class RoutesSnapshot:
    """One read of the conversation's live route table."""

    routes: tuple[RouteView, ...]
    error: str | None = None


def resolved_label(route: RouteView) -> str:
    """The resolved distinct-model count, or UNKNOWN when resolution was partial."""
    return (
        UNKNOWN
        if route.resolved_distinct_models is None
        else str(route.resolved_distinct_models)
    )


def route_row(route: RouteView) -> str:
    return (
        f"{route.purpose}  requires {route.requires_distinct_models} distinct"
        f" {chrome_glyph('metadata_separator')} resolved {resolved_label(route)}"
        f" {chrome_glyph('metadata_separator')} reload {route.reload_status}"
        f" {chrome_glyph('metadata_separator')} launch {route.launch_status}"
    )


def target_line(target: RouteTargetView) -> str:
    """One target with every design field, and the preview label every time."""
    preview = target.preview if target.preview is not None else UNKNOWN
    line = (
        f"{target.seat or '-'}  binding {target.binding} -> {preview}"
        f" ({PREVIEW_LABEL})"
        f"{f' (marker {target.marker})' if target.marker is not None else ''}"
        f"  identity {target.identity}"
        f"  tier {target.tier}  overlay {target.overlay or '-'}"
    )
    if target.error is not None:
        line += f"\n  error: {target.error}"
    return line


def details_text(route: RouteView) -> str:
    """The selected route's full report: both facts, then every target."""
    lines = [
        f"Purpose: {route.purpose}",
        f"Requires {route.requires_distinct_models} distinct canonical models"
        f" {chrome_glyph('metadata_separator')} resolved {resolved_label(route)}"
        f" {chrome_glyph('metadata_separator')} reload validation"
        f" {route.reload_status} {chrome_glyph('metadata_separator')}"
        f" launch diversity {route.launch_status}",
    ]
    if route.reload_detail is not None:
        lines.append(f"Reload: {route.reload_detail}")
    if route.launch_detail is not None:
        lines.append(f"Launch: {route.launch_detail}")
    lines.extend(target_line(target) for target in route.targets)
    return "\n".join(lines)


class RoutesList(NavigableOptionList):
    pass


class RoutesScreen(ModalScreen[None]):
    CSS_PATH = "routes.tcss"
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", show=False, priority=True),
        Binding("r,ctrl+r", "refresh", show=False),
    ]

    def __init__(
        self, read_routes: RoutesRead, subscribe: RoutesUpdate | None = None
    ) -> None:
        super().__init__(id="routes-screen")
        self.read_routes = read_routes
        self.subscribe = subscribe
        self.snapshot: RoutesSnapshot | None = None
        self._status = ""
        self._unsubscribe: Callable[[], None] | None = None
        self._browser_closed = False

    def compose(self) -> ComposeResult:
        with Vertical(id="routes-content"):
            yield NoMarkupStatic("Routes", id="routes-title")
            yield NoMarkupStatic("", id="routes-heading")
            yield NoMarkupStatic("", id="routes-status")
            with RoutesList(id="routes-list"):
                pass
            with VerticalScroll(id="routes-details"):
                yield NoMarkupStatic("", id="routes-detail-text")
            yield NoMarkupStatic("", id="routes-hint")

    def on_mount(self) -> None:
        # The live table can change under an open browser (a reload swaps it
        # atomically); the subscription re-reads instead of caching.
        if self.subscribe is not None:
            self._unsubscribe = self.subscribe(self.action_refresh)
        self._resize_surface()
        self.query_one(RoutesList).focus()
        self.action_refresh()

    def on_unmount(self) -> None:
        self._browser_closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None

    def on_resize(self) -> None:
        self._resize_surface()

    def _resize_surface(self) -> None:
        content = self.query_one("#routes-content", Vertical)
        width, height = self.size
        fullscreen = width < MIN_MODAL_WIDTH or height < MIN_MODAL_HEIGHT
        content.set_class(fullscreen, "fullscreen")
        content.styles.width = "100%" if fullscreen else min(92, width - 2)
        content.styles.height = "100%" if fullscreen else height - 2
        content.border_title = "" if fullscreen else "Routes"

    def action_refresh(self) -> None:
        if self._browser_closed:
            return
        try:
            self.snapshot = self.read_routes()
        except Exception:
            # The read is read-only; a failure keeps the last good rows.
            self._status = (
                "Failed: Route table read failed; previous rows retained. r Retry."
            )
            self._render_status()
            return
        self._status = ""
        self._render_table()

    def _render_table(self) -> None:
        """Rebuild the purpose rows from the snapshot (never from a cache)."""
        options = self.query_one(RoutesList)
        selected = options.highlighted_option
        selected_id = selected.id if selected is not None else None
        options.clear_options()
        for route in self.snapshot.routes if self.snapshot else ():
            options.add_option(Option(Content(route_row(route)), id=route.purpose))
        options.can_focus = True
        if selected_id is not None and options.option_count:
            surviving = [
                index
                for index, option in enumerate(options.options)
                if option.id == selected_id
            ]
            options.highlighted = surviving[0] if surviving else 0
        elif options.option_count:
            options.highlighted = 0
        self._render_status()
        self._render_details()
        self._render_hint()

    def _render_status(self) -> None:
        widget = self.query_one("#routes-status", NoMarkupStatic)
        error = self.snapshot.error if self.snapshot else None
        text = self._status
        color = "$error" if text.startswith("Failed:") else "$text-muted"
        if not text and error:
            text = f"Route table unavailable: {error}"
            color = "$warning"
        if not text and self.snapshot is not None and not self.snapshot.routes:
            text = "No purpose routes are configured."
            color = "$text-muted"
        widget.display = bool(text)
        widget.update(Content.assemble((text, color)))
        heading = self.query_one("#routes-heading", NoMarkupStatic)
        heading.update(
            Content(
                f"Live [orchestration.routes] {chrome_glyph('metadata_separator')} "
                f"every model is a {PREVIEW_LABEL}"
            )
        )

    def _selected_route(self) -> RouteView | None:
        option = self.query_one(RoutesList).highlighted_option
        if option is None or self.snapshot is None:
            return None
        purpose = option.id
        return next(
            (route for route in self.snapshot.routes if route.purpose == purpose), None
        )

    def _render_details(self) -> None:
        route = self._selected_route()
        self.query_one("#routes-detail-text", NoMarkupStatic).update(
            details_text(route) if route is not None else ""
        )

    def _render_hint(self) -> None:
        hint = self.query_one("#routes-hint", NoMarkupStatic)
        pairs = [
            ("Up/Down", "Move", ""),
            ("r", "Refresh", "refresh"),
            ("Esc", "Close", "close"),
        ]
        if (
            sum(cell_len(f"{key} {label}") for key, label, _ in pairs)
            + 3 * (len(pairs) - 1)
            > hint.content_size.width
        ):
            pairs = [("r", "Refresh", "refresh"), ("Esc", "Close", "close")]
        hint.update(
            shortcut_hint(
                f" {chrome_glyph('metadata_separator')} ".join(
                    f"{shortcut(key)} {label}" for key, label, _ in pairs
                )
            )
        )

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        self._render_details()

    def action_close(self) -> None:
        self._browser_closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self.dismiss(None)
