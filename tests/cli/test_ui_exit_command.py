from __future__ import annotations

from typing import Any

import pytest

from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import SlashCommandMessage, UserMessage
from tests.conftest import build_test_chartreux_app


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
