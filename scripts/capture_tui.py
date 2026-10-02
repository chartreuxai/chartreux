#!/usr/bin/env python3
"""Capture deterministic Chartreux TUI surfaces for visual review.

This hosts modal widgets in tiny Textual apps and runs the chat surface with
ChartreuxApp's in-process fake app-server. No surface calls an LLM or network.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Protocol, cast

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.pilot import Pilot
from textual.widgets import Static

# Make the repository importable when this file is run as scripts/capture_tui.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.screens.settings import SettingsScreen
from chartreux.cli.textual_ui.widgets.messages import AssistantMessage
from chartreux.cli.textual_ui.widgets.model_picker import ModelOption, ModelPickerApp
from chartreux.cli.textual_ui.widgets.session_picker import SessionPickerApp
from chartreux.core.config.harness_files import (
    init_harness_files_manager,
    reset_harness_files_manager,
)
from chartreux.ui.settings_service import SettingsService
from tests.cli.textual_ui.test_settings_app import FakeService
from tests.snapshots.test_ui_snapshot_basic_conversation import (
    SnapshotTestAppWithConversation,
)
from tests.snapshots.test_ui_snapshot_session_picker import _LATEST_MESSAGES, _SESSIONS
from tests.ui.providers.test_workbench import Host, setup

DEFAULT_SIZES = [(80, 24), (120, 36), (120, 72)]
THEMES = ("textual-dark", "textual-light", "ansi-dark", "ansi-light")
APP_CSS_PATH = str(
    Path(__file__).resolve().parents[1] / "chartreux/cli/textual_ui/app.tcss"
)


class _PreviewHost(Protocol):
    config: SimpleNamespace


class SettingsHost(App[None]):
    def __init__(self) -> None:
        super().__init__()
        service = FakeService()
        self.settings_screen = SettingsScreen(
            cast(SettingsService, service), service.snapshot
        )

    def compose(self) -> ComposeResult:
        yield Static("host")

    def on_mount(self) -> None:
        self.push_screen(self.settings_screen)


class PickerHost(App[None]):
    CSS_PATH = APP_CSS_PATH

    picker: object

    def compose(self) -> ComposeResult:
        transcript = Container(id="preview-transcript")
        transcript.styles.height = "1fr"
        yield transcript
        with Static(id="bottom-app-container"):
            yield self.picker  # type: ignore[misc]


class SessionHost(PickerHost):
    def __init__(self) -> None:
        super().__init__()
        self.picker = SessionPickerApp(
            sessions=_SESSIONS, latest_messages=_LATEST_MESSAGES, cwd="/test/workdir"
        )


class ModelHost(PickerHost):
    def __init__(self) -> None:
        super().__init__()
        self.picker = ModelPickerApp(
            models=[
                ModelOption("mistral-large", "Mistral Large"),
                ModelOption("devstral", "Devstral"),
                ModelOption("codestral", "Codestral"),
            ],
            current_model="devstral",
            is_pinned=True,
            default_display_name="Devstral",
        )


@dataclass(frozen=True)
class Capture:
    surface: str
    label: str
    factory: Callable[[], App]
    prepare: Callable[[Pilot], Awaitable[None]] | None = None


def _parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x", 1)
        result = (int(width), int(height))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("size must be WIDTHxHEIGHT") from exc
    if result[0] < 1 or result[1] < 1:
        raise argparse.ArgumentTypeError("size dimensions must be positive")
    return result


def _provider_captures() -> list[Capture]:
    def browser() -> App:
        screen, _services = setup(active="a")
        return Host(screen)

    def actions() -> App:
        screen, _services = setup(active="a")
        app = Host(screen)
        app._capture_screen = screen  # type: ignore[attr-defined]
        return app

    async def to_actions(pilot: Pilot) -> None:
        await pilot.press("enter")

    def models() -> App:
        screen, _services = setup(active="a")
        app = Host(screen)
        app._capture_screen = screen  # type: ignore[attr-defined]
        return app

    async def to_models(pilot: Pilot) -> None:
        screen = pilot.app._capture_screen  # type: ignore[attr-defined]
        await pilot.press("enter")  # provider browser -> actions
        actions = screen.query_one("#wb-actions")
        actions.highlighted = next(
            i for i, option in enumerate(actions.options) if option.id == "models"
        )
        await pilot.press("enter")

    def detail() -> App:
        screen, _services = setup(active="a")
        app = Host(screen)
        app._capture_screen = screen  # type: ignore[attr-defined]
        return app

    async def to_detail(pilot: Pilot) -> None:
        screen = pilot.app._capture_screen  # type: ignore[attr-defined]
        await pilot.press("enter")
        actions = screen.query_one("#wb-actions")
        actions.highlighted = next(
            i for i, option in enumerate(actions.options) if option.id == "models"
        )
        await pilot.press("enter")
        await pilot.pause()
        screen._open_detail("a")

    return [
        Capture("provider", "provider-browser", browser),
        Capture("provider", "provider-actions", actions, to_actions),
        Capture("provider", "provider-models", models, to_models),
        Capture("provider", "provider-model-detail", detail, to_detail),
    ]


async def _submit_chat_prompt(pilot: Pilot) -> None:
    expected = "I'm the Vibe agent and I'm ready to help."
    await pilot.press(*"Hello there, who are you?")
    await pilot.press("enter")
    for _ in range(100):
        await pilot.pause(0.05)  # poll the rendered response, bounded at 5 seconds
        app = pilot.app
        assistants = list(app.query(AssistantMessage))
        if (
            assistants
            and assistants[-1].get_content().strip() == expected
            and not getattr(app, "_pending_turn", True)
            and getattr(app, "_loading_widget", None) is None
        ):
            return
    raise RuntimeError(
        "Timed out after 5 seconds waiting for the deterministic assistant response"
    )


def _chat_capture() -> Capture:
    return Capture(
        "chat",
        "chat-conversation",
        SnapshotTestAppWithConversation,
        _submit_chat_prompt,
    )


def _captures(surface: str) -> Iterable[Capture]:
    captures = _provider_captures()
    captures.extend([
        Capture("settings", "settings-browser", SettingsHost),
        Capture("session", "session-picker", SessionHost),
        Capture("model", "model-picker", ModelHost),
        _chat_capture(),
    ])
    return (capture for capture in captures if surface in {"all", capture.surface})


def _configure_app(app: App, theme: str, ascii_chrome: bool) -> None:
    app.theme = theme
    if ascii_chrome and not hasattr(type(app), "config"):
        cast(_PreviewHost, app).config = SimpleNamespace(ascii_chrome=True)


async def _capture_one(
    capture: Capture,
    size: tuple[int, int],
    theme: str,
    output: Path,
    ascii_chrome: bool,
) -> list[str]:
    app = capture.factory()
    _configure_app(app, theme, ascii_chrome)
    stem = f"{capture.label}-{size[0]}x{size[1]}-{theme}"
    if ascii_chrome:
        stem += "-ascii"
    svg_path = output / f"{stem}.svg"
    async with app.run_test(size=size) as pilot:
        await pilot.pause()
        if isinstance(app, ChartreuxApp):
            app.config.theme = theme
            app.config.ascii_chrome = ascii_chrome
            app.theme = theme
        if capture.prepare is not None:
            await capture.prepare(pilot)
            await pilot.pause()
        app.save_screenshot(filename=svg_path.name, path=str(output))
    paths = [str(svg_path)]
    inkscape = shutil.which("inkscape")
    if inkscape:
        png_path = output / f"{stem}.png"
        result = subprocess.run(
            [inkscape, str(svg_path), "--export-filename", str(png_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and png_path.exists():
            paths.append(str(png_path))
        else:
            print(
                f"warning: inkscape failed for {svg_path}: {result.stderr.strip()}",
                file=sys.stderr,
            )
    return paths


async def _run(args: argparse.Namespace) -> int:
    os.environ["MISTRAL_API_KEY"] = "chartreux-ui-fake-key"
    os.environ["ANTHROPIC_API_KEY"] = "chartreux-ui-fake-key"
    os.environ["OPENAI_API_KEY"] = "chartreux-ui-fake-key"
    reset_harness_files_manager()
    init_harness_files_manager("user", "project")
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    captures = list(_captures(args.surface))
    generated: list[str] = []
    try:
        with tempfile.TemporaryDirectory(prefix="chartreux-ui-capture-") as workdir:
            previous_cwd = os.getcwd()
            os.chdir(workdir)
            try:
                for size in args.size:
                    for capture in captures:
                        generated.extend(
                            await _capture_one(
                                capture, size, args.theme, output, args.ascii
                            )
                        )
            finally:
                os.chdir(previous_cwd)
    finally:
        reset_harness_files_manager()
    for path in generated:
        print(path)
    if not shutil.which("inkscape"):
        print(
            "note: inkscape not found; SVG files were generated, PNG conversion skipped",
            file=sys.stderr,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--surface",
        choices=("all", "provider", "settings", "session", "model", "chat"),
        default="all",
    )
    parser.add_argument(
        "--size",
        type=_parse_size,
        action="append",
        default=None,
        metavar="WIDTHxHEIGHT",
    )
    parser.add_argument("--theme", choices=THEMES, default="textual-dark")
    parser.add_argument("--output", default="/tmp/chartreux-ui/screenshots")
    parser.add_argument(
        "--ascii", action="store_true", help="Use ASCII chrome glyphs in captures"
    )
    args = parser.parse_args()
    args.size = args.size or DEFAULT_SIZES
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
