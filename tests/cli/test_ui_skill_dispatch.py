from __future__ import annotations

import asyncio
from pathlib import Path
import time
from weakref import WeakKeyDictionary

import pytest

from chartreux.app_server.models import CompletedEffectState, PublicEffectEntry
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage, UserMessage
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from tests.conftest import (
    build_test_agent_loop,
    build_test_chartreux_app,
    build_test_vibe_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.skills.conftest import create_skill
from tests.stubs.fake_backend import FakeBackend

SKILL_BODY = "## Instructions\n\nDo the thing."


class _BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([[mock_llm_chunk(content="done")]] * 4)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await self.release.wait()
        return await super().complete(**kwargs)


_blocking_backends: WeakKeyDictionary[ChartreuxApp, _BlockingBackend] = (
    WeakKeyDictionary()
)


async def _block_agent_job(app: ChartreuxApp, pilot) -> _BlockingBackend:
    backend = _blocking_backends[app]
    chat_input = app.query_one(ChatInputContainer)
    chat_input.post_message(ChatInputContainer.Submitted("block queue"))
    assert await _wait_until(pilot, backend.started.is_set)
    # Wait until the client has handled TurnStarted, not just until the session
    # projection reports the turn active (that flag flips earlier, often before
    # backend.started returns). Only once TurnStarted is processed is the blocking
    # turn cleared from the optimistic len(app._queue), so a later follow-up count
    # is accurate instead of being inflated by the still-pending running turn.
    assert await _wait_until(
        pilot, lambda: not app._pending_turn and len(app._queue) == 0
    )
    return backend


async def _release_agent_job(
    app: ChartreuxApp, pilot, backend: _BlockingBackend
) -> None:
    backend.release.set()
    assert await _wait_until(pilot, lambda: not app._agent_job_active(), timeout=5.0)


@pytest.fixture
def chartreux_app_with_skills(tmp_path: Path) -> ChartreuxApp:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    create_skill(skills_dir, "my-skill", body=SKILL_BODY)
    config = build_test_vibe_config(skill_paths=[skills_dir])
    backend = _BlockingBackend()
    app = build_test_chartreux_app(
        config=config, agent_loop=build_test_agent_loop(config=config, backend=backend)
    )
    _blocking_backends[app] = backend
    return app


async def _wait_for_user_message_containing(
    chartreux_app: ChartreuxApp, pilot, text: str, timeout: float = 1.0
) -> UserMessage:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for message in chartreux_app.query(UserMessage):
            if text in message._content:
                return message
        await pilot.pause(0.05)
    raise TimeoutError(
        f"UserMessage containing {text!r} did not appear within {timeout}s"
    )


async def _wait_for_error_message_containing(
    chartreux_app: ChartreuxApp, pilot, text: str, timeout: float = 1.0
) -> ErrorMessage:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for error in chartreux_app.query(ErrorMessage):
            if text in str(error._error):
                return error
        await pilot.pause(0.05)
    raise TimeoutError(
        f"ErrorMessage containing {text!r} did not appear within {timeout}s"
    )


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return False


def _skill_effect_loaded(app: ChartreuxApp, name: str) -> bool:
    marker = f'<skill_content name="{name}">'
    for entry in app.app_server.history:
        if not isinstance(entry, PublicEffectEntry):
            continue
        if entry.detail.tool_name != "skill":
            continue
        if not isinstance(entry.state, CompletedEffectState):
            continue
        output = entry.state.output
        if not isinstance(output, dict):
            continue
        content = output.get("content")
        if isinstance(content, str) and marker in content:
            return True
    return False


@pytest.mark.asyncio
async def test_skill_without_args_displays_literal_command(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("/my-skill"))
        await pilot.pause(0.1)

        message = await _wait_for_user_message_containing(
            chartreux_app_with_skills, pilot, "/my-skill"
        )
        assert message._content == "/my-skill"
        assert "Do the thing." not in message._content


@pytest.mark.asyncio
async def test_skill_with_args_displays_literal_command_with_args(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("/my-skill foo bar"))
        await pilot.pause(0.1)

        message = await _wait_for_user_message_containing(
            chartreux_app_with_skills, pilot, "/my-skill foo bar"
        )
        assert message._content == "/my-skill foo bar"
        assert "Do the thing." not in message._content


@pytest.mark.asyncio
async def test_unknown_skill_falls_through_to_agent(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("/nonexistent-skill"))
        await pilot.pause(0.2)

        skill_errors = [
            e
            for e in chartreux_app_with_skills.query(ErrorMessage)
            if "skill" in str(getattr(e, "_error", "")).lower()
        ]
        assert not skill_errors


@pytest.mark.asyncio
async def test_bare_slash_falls_through(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("/"))
        await pilot.pause(0.2)

        assert not any(
            "Do the thing." in m._content
            for m in chartreux_app_with_skills.query(UserMessage)
        )


@pytest.mark.asyncio
async def test_skill_without_args_does_not_add_extra_text(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        await pilot.pause(0.1)
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        chat_input.post_message(ChatInputContainer.Submitted("/my-skill"))
        await pilot.pause(0.1)

        message = await _wait_for_user_message_containing(
            chartreux_app_with_skills, pilot, "/my-skill"
        )
        assert message._content == "/my-skill"


@pytest.mark.asyncio
async def test_queued_head_skill_injects_skill_tool_message(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        backend = await _block_agent_job(chartreux_app_with_skills, pilot)
        try:
            chat_input.post_message(ChatInputContainer.Submitted("/my-skill"))
            chat_input.post_message(ChatInputContainer.Submitted("follow-up prompt"))
            assert await _wait_until(
                pilot, lambda: len(chartreux_app_with_skills._queue) == 2
            )
        finally:
            await _release_agent_job(chartreux_app_with_skills, pilot, backend)

        assert await _wait_until(
            pilot,
            lambda: (
                len(chartreux_app_with_skills._queue) == 0
                and chartreux_app_with_skills._agent_task is None
                and any(
                    widget._tool_name == "skill"
                    for widget in chartreux_app_with_skills.query(ToolCallMessage)
                )
                and any(
                    widget.tool_name == "skill"
                    for widget in chartreux_app_with_skills.query(ToolResultMessage)
                )
            ),
            timeout=5.0,
        )

        assert _skill_effect_loaded(chartreux_app_with_skills, "my-skill")


@pytest.mark.asyncio
async def test_skill_prompt_runs_after_following_bash_is_rejected(
    chartreux_app_with_skills: ChartreuxApp,
) -> None:
    async with chartreux_app_with_skills.run_test() as pilot:
        chat_input = chartreux_app_with_skills.query_one(ChatInputContainer)
        backend = await _block_agent_job(chartreux_app_with_skills, pilot)
        try:
            chat_input.post_message(ChatInputContainer.Submitted("/my-skill"))
            chat_input.post_message(ChatInputContainer.Submitted("!echo queued"))
            assert await _wait_until(
                pilot,
                lambda: (
                    len(chartreux_app_with_skills._queue) == 1
                    and any(
                        "Shell commands cannot be queued" in notification.message
                        for notification in chartreux_app_with_skills._notifications
                    )
                ),
            )
            assert chat_input.value == "!echo queued"
        finally:
            await _release_agent_job(chartreux_app_with_skills, pilot, backend)

        assert await _wait_until(
            pilot,
            lambda: (
                len(chartreux_app_with_skills._queue) == 0
                and chartreux_app_with_skills._agent_task is None
                and chartreux_app_with_skills._bash_task is None
                and any(
                    widget._tool_name == "skill"
                    for widget in chartreux_app_with_skills.query(ToolCallMessage)
                )
                and any(
                    widget.tool_name == "skill"
                    for widget in chartreux_app_with_skills.query(ToolResultMessage)
                )
            ),
            timeout=5.0,
        )

        assert _skill_effect_loaded(chartreux_app_with_skills, "my-skill")
        assert not any(
            widget.tool_name == "shell"
            for widget in chartreux_app_with_skills.query(ToolResultMessage)
        )
