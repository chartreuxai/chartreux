from __future__ import annotations

import time

import pytest
import pytest_asyncio

from chartreux.cli.textual_ui.widgets.chat_input import ChatInputContainer, ChatTextArea
from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import (
    CompletionPopup,
    _CompletionRow,
)
from tests.conftest import build_test_chartreux_app
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


async def _wait_for_rows(pilot, popup: CompletionPopup, count: int) -> None:
    deadline = time.monotonic() + 2
    while len(popup.query(_CompletionRow)) < count:
        if time.monotonic() >= deadline:
            raise AssertionError(f"Expected {count} completion rows")
        await pilot.pause(0.01)


@pytest.mark.asyncio
async def test_click_slash_suggestion_accepts_without_submitting() -> None:
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        input_widget = app.query_one(ChatTextArea)
        popup = app.query_one(CompletionPopup)
        input_widget.focus()
        await pilot.press("/")
        await _wait_for_rows(pilot, popup, 2)

        label = popup._suggestions[1].label
        row = list(popup.query(_CompletionRow))[1]
        await pilot.click(row, offset=(5, 0))
        await pilot.pause()

        assert app.query_one(ChatInputContainer).value == label
        assert not popup._suggestions
        assert app.focused is input_widget
        assert not app._agent_job_active()


@pytest.mark.asyncio
async def test_click_path_suggestion_accepts_without_submitting(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "alpha.txt").write_text("a")
    (tmp_path / "alpine.txt").write_text("b")
    monkeypatch.chdir(tmp_path)
    app = build_test_chartreux_app()
    async with app.run_test(size=(80, 24)) as pilot:
        input_widget = app.query_one(ChatTextArea)
        popup = app.query_one(CompletionPopup)
        input_widget.focus()
        await pilot.press(*"look @al")
        await _wait_for_rows(pilot, popup, 2)

        label = popup._suggestions[1].label
        row = list(popup.query(_CompletionRow))[1]
        await pilot.click(row, offset=(5, 0))
        await pilot.pause()

        assert app.query_one(ChatInputContainer).value == f"look {label} "
        assert not popup._suggestions
        assert app.focused is input_widget
        assert not app._agent_job_active()
