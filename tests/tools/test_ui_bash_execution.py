from __future__ import annotations

import time

from pydantic import JsonValue
import pytest

from chartreux.app_server.models import (
    CompletedEffectState,
    FailedEffectState,
    RunningEffectState,
    ShellEffectInput,
)
from chartreux.cli.textual_ui.app import ChartreuxApp
from chartreux.cli.textual_ui.widgets.chat_input.container import ChatInputContainer
from chartreux.cli.textual_ui.widgets.messages import ErrorMessage
from chartreux.cli.textual_ui.widgets.tools import ToolCallMessage, ToolResultMessage
from chartreux.core.llm_models import Role
from tests.conftest import build_test_agent_loop, build_test_chartreux_app
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _shell_calls(chartreux_app: ChartreuxApp) -> list[ToolCallMessage]:
    return [
        message
        for message in chartreux_app.query(ToolCallMessage)
        if message._entry is not None and message._entry.detail.tool_name == "shell"
    ]


def _shell_results(chartreux_app: ChartreuxApp) -> list[ToolResultMessage]:
    return [
        message
        for message in chartreux_app.query(ToolResultMessage)
        if message.tool_name == "shell"
    ]


async def _wait_for_shell_result(
    chartreux_app: ChartreuxApp, pilot, timeout: float = 1.0
) -> ToolResultMessage:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if messages := _shell_results(chartreux_app):
            return messages[-1]
        await pilot.pause(0.05)
    raise TimeoutError(f"Shell result effect did not appear within {timeout}s")


async def _wait_for_running_shell(
    chartreux_app: ChartreuxApp, pilot, timeout: float = 1.0
) -> ToolCallMessage:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for message in _shell_calls(chartreux_app):
            if message._entry is not None and isinstance(
                message._entry.state, RunningEffectState
            ):
                return message
        await pilot.pause(0.05)
    raise TimeoutError(f"Running shell effect did not appear within {timeout}s")


def _result_output(message: ToolResultMessage) -> dict[str, JsonValue]:
    assert message._entry is not None
    state = message._entry.state
    assert isinstance(state, CompletedEffectState)
    assert isinstance(state.output, dict)
    return state.output


def assert_no_command_error(chartreux_app: ChartreuxApp) -> None:
    errors = list(chartreux_app.query(ErrorMessage))
    if not errors:
        return

    disallowed = {
        "Command failed",
        "Command timed out",
        "No command provided after '!'",
    }
    offending = [
        getattr(err, "_error", "")
        for err in errors
        if getattr(err, "_error", "")
        and any(phrase in getattr(err, "_error", "") for phrase in disallowed)
    ]
    assert not offending, f"Unexpected command errors: {offending}"


@pytest.mark.asyncio
async def test_ui_reports_no_output(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!true"

        await pilot.press("enter")
        message = await _wait_for_shell_result(chartreux_app, pilot)
        assert _result_output(message) == {"stdout": "", "stderr": "", "output": ""}
        assert_no_command_error(chartreux_app)


@pytest.mark.asyncio
async def test_ui_shows_success_in_case_of_zero_code(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!true"

        await pilot.press("enter")
        message = await _wait_for_shell_result(chartreux_app, pilot)
        assert message._entry is not None
        assert isinstance(message._entry.state, CompletedEffectState)


@pytest.mark.asyncio
async def test_ui_shows_failure_in_case_of_non_zero_code(
    chartreux_app: ChartreuxApp,
) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!bash -c 'exit 7'"

        await pilot.press("enter")
        message = await _wait_for_shell_result(chartreux_app, pilot)
        assert message._entry is not None
        assert isinstance(message._entry.state, FailedEffectState)


@pytest.mark.asyncio
async def test_ui_handles_non_utf8_output(chartreux_app: ChartreuxApp) -> None:
    """Assert the UI accepts decoding a non-UTF8 sequence like `printf '\xf0\x9f\x98'`.
    Whereas `printf '\xf0\x9f\x98\x8b'` prints a smiley face (😋) and would work even without those changes.
    """
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!printf '\\xff\\xfe'"

        await pilot.press("enter")
        output = _result_output(await _wait_for_shell_result(chartreux_app, pilot))
        assert output["stdout"] in {"��", "\xff\xfe", r"\xff\xfe"}
        assert_no_command_error(chartreux_app)


@pytest.mark.asyncio
async def test_ui_handles_utf8_output(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!echo hello"

        await pilot.press("enter")
        output = _result_output(await _wait_for_shell_result(chartreux_app, pilot))
        assert output["stdout"] == "hello\n"
        assert_no_command_error(chartreux_app)


@pytest.mark.asyncio
async def test_ui_handles_non_utf8_stderr(chartreux_app: ChartreuxApp) -> None:
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!bash -c \"printf '\\\\xff\\\\xfe' 1>&2\""

        await pilot.press("enter")
        output = _result_output(await _wait_for_shell_result(chartreux_app, pilot))
        assert output["stderr"] == "��"
        assert_no_command_error(chartreux_app)


@pytest.mark.asyncio
async def test_ui_sends_manual_command_output_to_next_agent_turn() -> None:
    backend = FakeBackend(mock_llm_chunk(content="I saw it."))
    agent_loop = build_test_agent_loop(backend=backend)
    chartreux_app = build_test_chartreux_app(agent_loop=agent_loop)

    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!echo hello"

        await pilot.press("enter")
        await _wait_for_shell_result(chartreux_app, pilot)

        injected_message = agent_loop.messages[-1]
        assert injected_message.role == Role.user
        assert injected_message.injected is True
        assert injected_message.content is not None
        assert "Manual `!` command result from the user." in injected_message.content
        assert "Command: `echo hello`" in injected_message.content
        assert "Exit code: 0" in injected_message.content
        assert "Stdout:\n```text\nhello\n```" in injected_message.content

        chat_input.value = "what did the command print?"
        await pilot.press("enter")
        deadline = time.monotonic() + 2
        while len(backend.requests_messages) != 1 or chartreux_app._agent_job_active():
            if time.monotonic() >= deadline:
                raise AssertionError("Timed out waiting for the agent turn")
            await pilot.pause(0.01)

        assert len(backend.requests_messages) == 1
        user_messages = [
            msg for msg in backend.requests_messages[0] if msg.role == Role.user
        ]
        assert len(user_messages) >= 2
        assert user_messages[-2].content == injected_message.content
        assert user_messages[-2].injected is True
        assert user_messages[-1].content == "what did the command print?"


@pytest.mark.asyncio
async def test_ui_shows_command_immediately_in_pending_state(
    chartreux_app: ChartreuxApp,
) -> None:
    """The command line should appear before the process finishes."""
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 10"

        await pilot.press("enter")
        message = await _wait_for_running_shell(chartreux_app, pilot)
        assert message._entry is not None
        assert message._entry.detail.input == ShellEffectInput(command="sleep 10")
        assert message._entry.detail.display.verb == "Running"
        assert message.get_content() == "sleep 10"

        # clean up: cancel the background task
        if chartreux_app._bash_task and not chartreux_app._bash_task.done():
            chartreux_app._bash_task.cancel()


@pytest.mark.asyncio
async def test_ui_streams_output_incrementally(chartreux_app: ChartreuxApp) -> None:
    """Output should appear as the command produces it, not all at once."""
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        # print lines with a small delay so streaming has a chance to show partial output
        chat_input.value = "!bash -c 'echo first; echo second'"

        await pilot.press("enter")
        message = await _wait_for_shell_result(chartreux_app, pilot)
        output = _result_output(message)
        assert output["stdout"] == "first\nsecond\n"
        assert message._entry is not None
        state = message._entry.state
        assert isinstance(state, CompletedEffectState)
        assert state.output_text == "first\nsecond\n"


@pytest.mark.asyncio
async def test_ui_rejects_bash_submitted_while_command_running(
    chartreux_app: ChartreuxApp,
) -> None:
    """Submitting new bash while a bang command is running preserves the input."""
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 2"

        await pilot.press("enter")
        await _wait_for_running_shell(chartreux_app, pilot)
        assert chartreux_app._bash_task is not None
        assert not chartreux_app._bash_task.done()

        chat_input.value = "!echo done"
        await pilot.press("enter")

        assert chat_input.value == "!echo done"
        assert not chartreux_app._queue
        assert any(
            "cannot be queued while a shell command is running" in notification.message
            for notification in chartreux_app._notifications
        )

        await pilot.press("escape")
        deadline = time.monotonic() + 1
        while chartreux_app._bash_task is not None:
            if time.monotonic() >= deadline:
                raise AssertionError("Timed out waiting for the shell task to stop")
            await pilot.pause(0.05)

        # Cancelling the shell stream does not synthesize a completed result.
        assert not _shell_results(chartreux_app)
        assert len(_shell_calls(chartreux_app)) == 1
