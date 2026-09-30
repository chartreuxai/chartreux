from __future__ import annotations

from typing import ClassVar

from textual import events
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.widgets import Button, Input, Static

from chartreux.app_server.config import ProxySettingsView
from chartreux.cli.textual_ui.widgets.vscode_compat import VscodeCompatInput
from chartreux.ui.chrome_glyphs import chrome_glyph
from chartreux.ui.shortcut_hints import shortcut, shortcut_hint
from chartreux.ui.widgets.no_markup_static import NoMarkupStatic


class ProxySetupApp(Container):
    can_focus = True
    can_focus_children = True

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "close", "Cancel", show=False)
    ]

    class ProxySetupClosed(Message):
        def __init__(
            self, saved: bool, changes: dict[str, str | None] | None = None
        ) -> None:
            super().__init__()
            self.saved = saved
            self.changes = changes or {}

    def __init__(self, settings: ProxySettingsView) -> None:
        super().__init__(id="proxysetup-app")
        self._settings = settings
        self.inputs: dict[str, Input] = {}
        self._confirming_discard = False
        self._save_pending = False
        self._body_max_height: int | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="proxysetup-content"):
            yield NoMarkupStatic("Proxy Configuration", classes="settings-title")

            with VerticalScroll(id="proxysetup-form"):
                for key, description in self._settings.descriptions.items():
                    yield Static(key, classes="proxy-label-line")

                    initial_value = self._settings.values.get(key) or ""
                    input_widget = VscodeCompatInput(
                        value=initial_value,
                        placeholder="",
                        id=f"proxy-input-{key}",
                        classes="proxy-input",
                    )
                    self.inputs[key] = input_widget
                    yield input_widget
                    error_widget = NoMarkupStatic(
                        "", id=f"proxy-error-{key}", classes="proxy-field-error"
                    )
                    error_widget.styles.height = "auto"
                    error_widget.styles.text_wrap = "wrap"
                    error_widget.display = False
                    yield error_widget
                    if description:
                        yield NoMarkupStatic(description, classes="settings-help")

            error_widget = NoMarkupStatic("", id="proxysetup-error")
            error_widget.styles.height = "auto"
            error_widget.styles.text_wrap = "wrap"
            error_widget.display = False
            yield error_widget
            with Horizontal(id="proxysetup-actions", classes="proxy-actions"):
                yield Button("Apply changes", id="proxysetup-save")
                yield Button("Cancel", id="proxysetup-cancel")
            with Vertical(id="proxysetup-discard-confirm") as confirmation:
                confirmation.display = False
                yield NoMarkupStatic("", id="proxysetup-discard-message")
                with Horizontal(classes="proxy-actions"):
                    yield Button("Cancel", id="proxysetup-keep")
                    yield Button("Discard edits", id="proxysetup-discard")
            yield NoMarkupStatic(
                shortcut_hint(
                    f"{shortcut('Tab')} next field  {shortcut('Enter')} accept field  "
                    f"{shortcut('Esc')} cancel"
                ),
                id="proxysetup-help",
                classes="settings-help",
            )

    def on_mount(self) -> None:
        self._schedule_body_max_height()

    def on_resize(self, _event: events.Resize) -> None:
        self._schedule_body_max_height()

    def _schedule_body_max_height(self) -> None:
        if self.is_mounted:
            self.call_after_refresh(self._set_body_max_height)

    def _set_body_max_height(self) -> None:
        content = self.query_one("#proxysetup-content", Vertical)
        body = self.query_one("#proxysetup-form", VerticalScroll)
        fixed_chrome = (
            self.styles.gutter.height
            + content.virtual_size.height
            - body.outer_size.height
        )
        max_height = max(1, self.app.size.height // 2 - fixed_chrome)
        if max_height != self._body_max_height:
            self._body_max_height = max_height
            body.styles.max_height = max_height

    def focus(self, scroll_visible: bool = True) -> ProxySetupApp:
        """Override focus to focus the first input widget."""
        if self.inputs:
            first_input = list(self.inputs.values())[0]
            first_input.focus(scroll_visible=scroll_visible)
        else:
            super().focus(scroll_visible=scroll_visible)
        return self

    def action_focus_next(self) -> None:
        inputs = list(self.inputs.values())
        focused = self.screen.focused
        if isinstance(focused, Input) and focused in inputs:
            idx = inputs.index(focused)
            if idx + 1 < len(inputs):
                inputs[idx + 1].focus()
            else:
                self.query_one("#proxysetup-save", Button).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.action_focus_next()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._save_pending and event.button.id != "proxysetup-save":
            return
        match event.button.id:
            case "proxysetup-save":
                self._save_and_close()
            case "proxysetup-discard":
                self.post_message(self.ProxySetupClosed(saved=False))
            case "proxysetup-keep":
                self._hide_discard_confirmation()
            case "proxysetup-cancel":
                self.action_close()

    def on_blur(self, _event: events.Blur) -> None:
        self.call_after_refresh(self._refocus_if_needed)

    def on_input_blurred(self, _event: Input.Blurred) -> None:
        self.call_after_refresh(self._refocus_if_needed)

    def _refocus_if_needed(self) -> None:
        if (
            self.has_focus
            or any(inp.has_focus for inp in self.inputs.values())
            or any(button.has_focus for button in self.query(Button))
        ):
            return
        self.focus()

    def _pending_changes(self) -> dict[str, str | None]:
        return {
            key: value or None
            for key, input_widget in self.inputs.items()
            if (value := input_widget.value.strip())
            != (self._settings.values.get(key) or "")
        }

    def _save_and_close(self) -> None:
        if self._save_pending:
            return
        self._save_pending = True
        self.query_one("#proxysetup-save", Button).disabled = True
        self.query_one("#proxysetup-cancel", Button).disabled = True
        self.query_one("#proxysetup-help", NoMarkupStatic).update(
            "Saving proxy settings…"
        )
        self.post_message(
            self.ProxySetupClosed(saved=True, changes=self._pending_changes())
        )

    def finish_save(self, *, success: bool) -> None:
        if success:
            return
        self._save_pending = False
        self.query_one("#proxysetup-save", Button).disabled = False
        self.query_one("#proxysetup-cancel", Button).disabled = False
        self.query_one("#proxysetup-help", NoMarkupStatic).update(
            shortcut_hint(
                f"{shortcut('Tab')} next field  {shortcut('Enter')} accept field  "
                f"{shortcut('Esc')} cancel"
            )
        )

    def show_error(self, message: str) -> None:
        failed_field = next(
            (key for key in self.inputs if message.startswith(key)), None
        )
        form_error = self.query_one("#proxysetup-error", NoMarkupStatic)
        form_error.update(
            f"{chrome_glyph('error')} Failed: {message}" if failed_field is None else ""
        )
        form_error.display = failed_field is None
        for key, input_widget in self.inputs.items():
            invalid = key == failed_field
            input_widget.set_class(invalid, "-invalid")
            error_widget = self.query_one(f"#proxy-error-{key}", NoMarkupStatic)
            error_widget.update(
                f"{chrome_glyph('error')} Error: {message}" if invalid else ""
            )
            error_widget.display = invalid
        self._schedule_body_max_height()

    def _hide_discard_confirmation(self) -> None:
        self._confirming_discard = False
        self.query_one("#proxysetup-discard-confirm", Vertical).display = False
        self.query_one("#proxysetup-actions", Horizontal).display = True
        self.query_one("#proxysetup-cancel", Button).focus()
        self.query_one("#proxysetup-help", NoMarkupStatic).update(
            shortcut_hint(
                f"{shortcut('Tab')} next field  {shortcut('Enter')} accept field  "
                f"{shortcut('Esc')} cancel"
            )
        )
        self._schedule_body_max_height()

    def action_close(self) -> None:
        if self._save_pending:
            return
        if self._confirming_discard:
            self._hide_discard_confirmation()
        elif changes := self._pending_changes():
            self._confirming_discard = True
            names = ", ".join(changes)
            self.query_one("#proxysetup-discard-message", NoMarkupStatic).update(
                f"Discard edits to {names}? Cancel preserves your edits "
                "and all saved proxy settings."
            )
            self.query_one("#proxysetup-actions", Horizontal).display = False
            self.query_one("#proxysetup-discard-confirm", Vertical).display = True
            self.query_one("#proxysetup-keep", Button).focus()
            self.query_one("#proxysetup-help", NoMarkupStatic).update(
                shortcut_hint(
                    f"{shortcut('Tab')} choose  {shortcut('Enter')} select  "
                    f"{shortcut('Esc')} back"
                )
            )
            self._schedule_body_max_height()
        else:
            self.post_message(self.ProxySetupClosed(saved=False))
