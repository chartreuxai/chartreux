from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Container, Horizontal, Vertical
from textual.events import Resize
from textual.widgets import Markdown, Static

from chartreux.setup.onboarding.base import OnboardingHost, OnboardingScreen
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.theme import resolve_theme
from chartreux.ui.widgets.theme_picker import sorted_theme_names

THEMES = sorted_theme_names()
_TEXTUAL_THEME_MAP = {"auto": None, "light": "ansi-light", "dark": "ansi-dark"}
THEME_EXPLANATIONS = (
    "Auto follows your terminal or system preference.",
    "Light keeps the interface bright.",
    "Dark keeps the interface dim.",
)

VISIBLE_NEIGHBORS = 3
MIN_HORIZONTAL_WIDTH = 62
FADE_CLASSES = ["fade-1", "fade-2", "fade-3"]

PREVIEW_MARKDOWN = """\
### Heading

**Bold**, *italic*, and `inline code`.

- Bullet point
- Another bullet point

1. First item
2. Second item

```python
def greet(name: str = "World") -> str:
    return f"Hello, {name}!"
```

> Blockquote

---

| Column 1 | Column 2 |
|----------|----------|
| Item 1   | Item 2   |
"""


class ThemeSelectionScreen(OnboardingScreen):
    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("enter", "next", "Next", show=False, priority=True),
        Binding("up", "prev_theme", "Previous", show=False),
        Binding("down", "next_theme", "Next Theme", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False),
        Binding("escape", "cancel", "Cancel", show=False),
    ]

    def __init__(self, *, initial_theme: str, host: OnboardingHost) -> None:
        super().__init__(host)
        self._theme_index = THEMES.index(initial_theme)
        self._theme_widgets: list[Static] = []

    def _compose_theme_list(self) -> ComposeResult:
        for _ in range(VISIBLE_NEIGHBORS * 2 + 1):
            widget = Static("", classes="theme-item")
            self._theme_widgets.append(widget)
            yield widget

    def compose(self) -> ComposeResult:
        with Center(id="theme-outer"):
            with Vertical(id="theme-content"):
                yield Static("Select your preferred theme", id="theme-title")
                yield Static("\n".join(THEME_EXPLANATIONS), id="theme-explanations")
                yield Center(
                    Horizontal(
                        Static(
                            shortcut_hint(f"Navigate {shortcut('↑↓')}"), id="nav-hint"
                        ),
                        Vertical(*self._compose_theme_list(), id="theme-list"),
                        Static(
                            shortcut_hint(f"Press {shortcut('Enter')} ↵"),
                            id="enter-hint",
                        ),
                        id="theme-row",
                    )
                )
                with Container(id="preview-center"):
                    preview = Container(id="preview")
                    preview.border_title = "Preview"
                    with preview:
                        yield Container(Markdown(PREVIEW_MARKDOWN), id="preview-inner")

    def on_mount(self) -> None:
        self._update_display()
        self._update_responsive_layout()
        self._update_preview_height()
        self.focus()

    def on_resize(self, _: Resize) -> None:
        self._update_responsive_layout()
        self._update_preview_height()

    def _update_responsive_layout(self) -> None:
        row = self.query_one("#theme-row", Horizontal)
        row.set_class(self.app.size.width < MIN_HORIZONTAL_WIDTH, "narrow")

    def _update_preview_height(self) -> None:
        preview = self.query_one("#preview", Container)
        header_height = 17
        available = self.app.size.height - header_height
        preview.styles.max_height = max(7, available)

    def _get_theme_at_offset(self, offset: int) -> str:
        index = (self._theme_index + offset) % len(THEMES)
        return THEMES[index]

    def _update_display(self) -> None:
        for i, widget in enumerate(self._theme_widgets):
            offset = i - VISIBLE_NEIGHBORS
            theme = self._get_theme_at_offset(offset)

            widget.remove_class("selected", *FADE_CLASSES)

            if offset == 0:
                widget.update(f" {theme} ")
                widget.add_class("selected")
            else:
                distance = min(abs(offset) - 1, len(FADE_CLASSES) - 1)
                widget.update(theme)
                widget.add_class(FADE_CLASSES[distance])

    def _navigate(self, direction: int) -> None:
        self._theme_index = (self._theme_index + direction) % len(THEMES)
        resolved_theme = resolve_theme(self.selected_theme)
        textual_theme = _TEXTUAL_THEME_MAP[resolved_theme]
        if textual_theme is not None:
            self.app.theme = textual_theme
        self._update_display()

    @property
    def selected_theme(self) -> str:
        return THEMES[self._theme_index]

    def action_next_theme(self) -> None:
        self._navigate(1)

    def action_prev_theme(self) -> None:
        self._navigate(-1)

    def action_next(self) -> None:
        self.host.confirm_theme(self.selected_theme)
