from __future__ import annotations

from textual.geometry import Size
from textual.widget import Widget

from chartreux.cli.textual_ui.widgets.virtual_output import VirtualOutputText


class PlacementPlaceholder(Widget):
    """One measured Stream placement, without retaining its original widget.

    ``geometry_width`` matches Stream's get_content_height width argument:
    parent arrange width minus this placement's horizontal gutter, unlike
    TranscriptUnit.geometry_width (the parent's outer size.width).
    """

    ALLOW_SELECT = False

    def __init__(self, content_height: int, original: Widget) -> None:
        super().__init__()
        self.exempt_body_height = sum(
            widget.line_count
            for widget in (original, *original.walk_children())
            if isinstance(widget, VirtualOutputText)
        )
        self.content_height: int | None = content_height
        self._remainder_height: int | None = max(
            0, content_height - self.exempt_body_height
        )
        self._reserved = False
        # Stream passes arrange width minus the widget's horizontal gutter.
        self.geometry_width = (
            original.region.width
            + original.styles._base_styles.margin.totals[0]
            - original.styles._base_styles.gutter.totals[0]
        )
        styles = original.styles._base_styles
        self._margin = styles.margin
        self._gutter = styles.gutter

    def on_mount(self) -> None:
        # Stream reads base styles directly, not the inline styles on a Widget.
        self.styles._base_styles.margin = self._margin
        self.styles._base_styles.padding = self._gutter

    @classmethod
    def from_widget(cls, original: Widget) -> PlacementPlaceholder:
        gutter_height = original.styles._base_styles.gutter.totals[1]
        return cls(original.region.height - gutter_height, original)

    def reserve(self) -> None:
        self._reserved = True
        # Roots take over this placement immediately; reservation is bookkeeping.
        self.display = False
        self.refresh(layout=True)

    def release(self) -> None:
        self._reserved = False
        self.display = True
        self.refresh(layout=True)

    def invalidate_geometry(self) -> None:
        self._remainder_height = None
        self.content_height = None
        self.refresh(layout=True)

    def get_content_height(self, container: Size, viewport: Size, width: int) -> int:
        if width != self.geometry_width and self._remainder_height is not None:
            self.invalidate_geometry()
        if self._reserved:
            return 0
        return self.exempt_body_height + (self._remainder_height or 0)
