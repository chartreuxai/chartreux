from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.color import Color

from chartreux.app_server.models import ImageAttachment, InlineImageSource
from chartreux.cli.textual_ui.widgets.agent_transcript import AgentTranscriptViewer
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage, UserMessage
from tests.cli.textual_ui.test_agent_transcript import _available, _entry, _FakeSource


class AccentApp(App[None]):
    CSS_PATH = Path(__file__).parents[3] / "chartreux/cli/textual_ui/app.tcss"

    def __init__(self, *, ascii_chrome: bool = False, show: bool = True) -> None:
        super().__init__()
        self.config = SimpleNamespace(ascii_chrome=ascii_chrome)
        self.message = UserMessage(
            "hello\nsecond line",
            pending=True,
            posted_at=datetime(2026, 7, 12, tzinfo=UTC),
            show_message_timestamps=show,
            images=[
                ImageAttachment(
                    source=InlineImageSource(data="aW1n"),
                    alias="pasted.png",
                    mime_type="image/png",
                )
            ],
        )

    def compose(self) -> ComposeResult:
        yield self.message
        yield AssistantMessage("unmarked")


@pytest.mark.parametrize(
    "theme", ["textual-dark", "textual-light", "ansi-dark", "ansi-light"]
)
@pytest.mark.parametrize("show", [True, False])
@pytest.mark.asyncio
async def test_accent_covers_user_block_and_survives_promotion(
    theme: str, show: bool
) -> None:
    app = AccentApp(show=show)
    app.theme = theme
    async with app.run_test() as pilot:
        message = app.message
        prompt = message.query_one(".user-message-prompt")
        assert str(prompt.render()) == ">"
        assert prompt.styles.color == Color.parse(app.theme_variables["accent"])
        wrapper = message.query_one(".user-message-wrapper")
        if prompt.styles.color.ansi is None:
            assert wrapper.styles.background == prompt.styles.color.with_alpha(0.08)
        else:
            base = Color.parse("ansi_black" if app.current_theme.dark else "ansi_white")
            accent = prompt.styles.color
            assert wrapper.styles.background == Color(base.r, base.g, base.b).blend(
                Color(accent.r, accent.g, accent.b), 0.08
            )
            assert wrapper.styles.background.ansi is None
        assert wrapper.styles.padding.left == 2
        for selector in [
            ".user-message-content",
            ".user-message-attachment-line",
            ".user-message-attachment-link",
        ]:
            assert (
                message.query_one(selector).background_colors[1]
                == wrapper.background_colors[1]
            )
        assert not message.header.display
        await message.set_pending(False)
        await pilot.pause()
        assert message.header.display is show
        assert message.header.background_colors[1] == wrapper.background_colors[1]
        assert prompt.styles.color == Color.parse(app.theme_variables["accent"])
        message.set_show_message_timestamps(False)
        assert prompt.display
        assistant = app.query_one(AssistantMessage)
        assert not list(assistant.query(".user-message-prompt"))
        assert assistant.styles.background.a == 0
        app.theme = "textual-light" if theme != "textual-light" else "ansi-dark"
        await pilot.pause()
        assert prompt.styles.color == Color.parse(app.theme_variables["accent"])
        if app.theme == "textual-light":
            assert wrapper.styles.background == prompt.styles.color.with_alpha(0.08)
        else:
            assert wrapper.styles.background.ansi is None
            assert wrapper.styles.background != prompt.styles.color


@pytest.mark.parametrize("fallback", ["ascii", "no-color", "console"])
@pytest.mark.asyncio
async def test_plain_accent_fallback(
    fallback: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    if fallback == "no-color":
        monkeypatch.setenv("NO_COLOR", "1")
    app = AccentApp(ascii_chrome=fallback == "ascii")
    if fallback == "console":
        app.console.no_color = True
    async with app.run_test():
        message = app.message
        assert message.has_class("plain-user-accent")
        prompt = message.query_one(".user-message-prompt")
        assert str(prompt.render()) == ">"
        assert prompt.styles.color == Color.parse(app.theme_variables["foreground"])
        assert message.query_one(".user-message-wrapper").styles.background.a == 0
        assert message.query_one(".user-message-wrapper").styles.padding.left == 2


@pytest.mark.asyncio
async def test_child_transcript_uses_same_accent() -> None:
    app = AccentApp()
    source = _FakeSource()
    viewer = AgentTranscriptViewer(source, "agent-1")
    async with app.run_test() as pilot:
        await app.mount(viewer)
        await pilot.pause()
        source.respond(0, _available(_entry("accent", "child message")))
        await pilot.pause()
        child = viewer.query_one(UserMessage)
        assert (
            child.query_one(".user-message-prompt").styles.color
            == app.message.query_one(".user-message-prompt").styles.color
        )
        assert (
            child.query_one(".user-message-wrapper").styles.background
            == app.message.query_one(".user-message-wrapper").styles.background
        )
        await viewer.dispose()
