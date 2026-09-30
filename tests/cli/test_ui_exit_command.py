from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import pytest_asyncio

from chartreux.cli.textual_ui.widgets.chat_input.completion_popup import CompletionPopup
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import SlashCommandMessage, UserMessage
from tests.conftest import build_test_chartreux_app
from tests.snapshots.snapshot_event_loop import install_snapshot_wake


@pytest_asyncio.fixture(autouse=True)
async def _snapshot_event_loop_wake() -> None:
    install_snapshot_wake()


@pytest.mark.parametrize("alias", ["/exit", "exit", "quit", ":q", ":quit"])
@pytest.mark.asyncio
async def test_exit_synonym_runs_exit_handler_and_is_not_sent_as_prompt(
    alias: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    chartreux_app = build_test_chartreux_app()
    async with chartreux_app.run_test() as pilot:
        await pilot.pause(0.1)

        calls: list[str] = []

        async def _record_exit(**_kwargs: Any) -> None:
            calls.append(alias)

        monkeypatch.setattr(chartreux_app, "_exit_app", _record_exit)

        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted(alias))
        await pilot.pause(0.2)

        assert calls == [alias]
        prompts = [
            m
            for m in chartreux_app.query(UserMessage)
            if not isinstance(m, SlashCommandMessage)
        ]
        assert prompts == []


@pytest.mark.asyncio
async def test_typed_exit_with_completion_exits_on_first_enter() -> None:
    chartreux_app = build_test_chartreux_app()
    async with chartreux_app.run_test(size=(80, 24)) as pilot:
        await pilot.pause(0.1)
        assert chartreux_app._app_server is not None

        await pilot.press("/", "e", "x", "i", "t")
        await pilot.pause(0.1)
        assert chartreux_app.query_one(CompletionPopup).styles.display == "block"

        with (
            patch.object(chartreux_app, "_force_quit") as force_quit,
            patch.object(
                chartreux_app._quit_manager, "request_confirmation"
            ) as request_confirmation,
        ):
            await pilot.press("enter")
            await pilot.pause(0.2)

        force_quit.assert_called_once()
        request_confirmation.assert_not_called()
