from __future__ import annotations

import pytest
import pytest_asyncio

from chartreux.cli.autocompletion.base import CompletionEntry
from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import (
    SELECTED_CLASS,
    CompletionPopup,
)
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from tests.conftest import build_test_chartreux_app
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


@pytest.mark.asyncio
async def test_popup_width_tracks_container_resize_without_rebuilding_rows() -> None:
    app = build_test_chartreux_app()
    suggestions = [
        CompletionEntry("/model", "Choose the model used for this conversation."),
        CompletionEntry(
            "/long-command",
            "A long command description that needs a wide popup to remain readable "
            "after the terminal grows beyond its initial narrow viewport.",
        ),
        CompletionEntry("/resume", "Reopen a saved conversation."),
    ]

    async with app.run_test(size=(80, 24)) as pilot:
        container = app.query_one(ChatInputContainer)
        container.focus_input()
        input_widget = container.input_widget
        assert input_widget is not None
        popup = app.query_one(CompletionPopup)
        container.render_completion_suggestions(suggestions, selected_index=1)
        await pilot.pause()

        rows = tuple(popup.children)
        assert popup.region.width == 78
        assert rows[1].has_class(SELECTED_CLASS)
        assert app.focused is input_widget

        await pilot.resize_terminal(120, 36)
        await pilot.pause()
        assert popup.region.width == 92
        assert tuple(popup.children) == rows
        assert rows[1].has_class(SELECTED_CLASS)
        assert app.focused is input_widget

        await pilot.resize_terminal(60, 20)
        await pilot.pause()
        assert popup.region.width == 58
        assert tuple(popup.children) == rows
        assert rows[1].has_class(SELECTED_CLASS)
        assert app.focused is input_widget
