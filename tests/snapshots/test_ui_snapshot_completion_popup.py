from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.pilot import Pilot

from chartreux.cli.autocompletion.base import CompletionEntry
from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import CompletionPopup
from tests.snapshots.snap_compare import SnapCompare

FILE_MENTION_SUGGESTIONS: list[CompletionEntry] = [
    CompletionEntry(
        "@chartreux/cli/textual_ui/widgets/chat_input/completion_popup.py", ""
    ),
    CompletionEntry(
        "@chartreux/core/tools/builtins/very_long_deeply_nested_module_name.py", ""
    ),
    CompletionEntry(
        "@tests/snapshots/test_ui_snapshot_completion_popup_fixtures.py", ""
    ),
]

SLASH_COMMAND_SUGGESTIONS: list[CompletionEntry] = [
    CompletionEntry(
        "/model",
        "Pick the model used for the conversation from every configured provider",
    ),
    CompletionEntry(
        "/compact",
        "Summarize the conversation so far to reclaim context window headroom",
    ),
    CompletionEntry(
        "/resume",
        "Reopen a previous local session and continue exactly where you left off",
    ),
]

LONG_SLASH_COMMAND_SUGGESTIONS: list[CompletionEntry] = [
    CompletionEntry(
        "/main-test-generator",
        "Generate unit and integration tests for a single file/module, or bounded "
        "characterization tests at an explicit subsystem boundary, with framework "
        "detection and self-review. Repo-wide generation is out of scope.",
    ),
    CompletionEntry(
        "/main-debugging",
        "Systematic, language-agnostic debugging assistant that helps reproduce, "
        "isolate, diagnose, fix, and prevent bugs using a structured methodology "
        "with AI-powered root cause analysis and regression test generation.",
    ),
    CompletionEntry(
        "/main-plan",
        "Turn an approved design into an executable, bounded plan with work "
        "packages, dependencies, safe parallelism, acceptance checks, and "
        "verification. Delegates deep planning analysis to the advisor profile.",
    ),
    CompletionEntry(
        "/mcp",
        "Display available MCP servers. Pass a name to list tools; subcommands: "
        "add <url> [--transport streamable-http], status, login <alias>, logout "
        "<alias>",
    ),
]


class CompletionPopupTestApp(App):
    CSS_PATH = "../../chartreux/cli/textual_ui/app.tcss"

    def compose(self) -> ComposeResult:
        with Container():
            yield CompletionPopup()


async def _show(pilot: Pilot, suggestions: list[CompletionEntry]) -> None:
    pilot.app.query_one(CompletionPopup).update_suggestions(suggestions, selected=0)
    await pilot.pause(0.1)


@pytest.mark.asyncio
async def test_completion_popup_stays_within_narrow_viewport() -> None:
    app = CompletionPopupTestApp()
    async with app.run_test(size=(30, 20)) as pilot:
        await _show(pilot, FILE_MENTION_SUGGESTIONS)
        popup = app.query_one(CompletionPopup)
        assert popup.region.right <= app.size.width
        assert popup.size.width <= 30


@pytest.mark.asyncio
async def test_completion_popup_selected_description_wraps_with_room() -> None:
    app = CompletionPopupTestApp()
    async with app.run_test(size=(80, 24)) as pilot:
        await _show(pilot, LONG_SLASH_COMMAND_SUGGESTIONS)
        popup = app.query_one(CompletionPopup)
        assert popup.region.width == 78
        assert popup.size.height > len(LONG_SLASH_COMMAND_SUGGESTIONS) + 2


@pytest.mark.asyncio
@pytest.mark.parametrize("width, expected_width", [(80, 78), (120, 92)])
async def test_completion_popup_width_uses_viewport_cap(
    width: int, expected_width: int
) -> None:
    app = CompletionPopupTestApp()
    async with app.run_test(size=(width, 36)) as pilot:
        await _show(pilot, LONG_SLASH_COMMAND_SUGGESTIONS)
        assert app.query_one(CompletionPopup).region.width == expected_width


def test_snapshot_completion_popup_file_mentions_stretch_full_width(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await _show(pilot, FILE_MENTION_SUGGESTIONS)

    assert snap_compare(
        "test_ui_snapshot_completion_popup.py:CompletionPopupTestApp",
        terminal_size=(80, 20),
        run_before=run_before,
    )


def test_snapshot_completion_popup_slash_commands_two_columns(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await _show(pilot, SLASH_COMMAND_SUGGESTIONS)

    assert snap_compare(
        "test_ui_snapshot_completion_popup.py:CompletionPopupTestApp",
        terminal_size=(80, 20),
        run_before=run_before,
    )


def test_snapshot_completion_popup_long_selected_description_80x24(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await _show(pilot, LONG_SLASH_COMMAND_SUGGESTIONS)

    assert snap_compare(
        "test_ui_snapshot_completion_popup.py:CompletionPopupTestApp",
        terminal_size=(80, 24),
        run_before=run_before,
    )


def test_snapshot_completion_popup_long_selected_description_120x36(
    snap_compare: SnapCompare,
) -> None:
    async def run_before(pilot: Pilot) -> None:
        await _show(pilot, LONG_SLASH_COMMAND_SUGGESTIONS)

    assert snap_compare(
        "test_ui_snapshot_completion_popup.py:CompletionPopupTestApp",
        terminal_size=(120, 36),
        run_before=run_before,
    )
