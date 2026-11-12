from __future__ import annotations

from pathlib import Path
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


async def _wait_until(pilot, predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await pilot.pause(0.05)
    return False


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
async def test_wp2_ui_manual_shell_writes_outside_workspace(tmp_path: Path) -> None:
    """GAP-2 also reaches the manual executor through the CLI bang input."""
    project = tmp_path / "project"
    project.mkdir()
    target = tmp_path / "outside.txt"
    target.write_text("unchanged")
    agent_loop = build_test_agent_loop(cwd=project)
    chartreux_app = build_test_chartreux_app(agent_loop=agent_loop)
    try:
        async with chartreux_app.run_test() as pilot:
            chat_input = chartreux_app.query_one(ChatInputContainer)
            chat_input.value = f"!printf changed > {target}"
            await pilot.press("enter")
            message = await _wait_for_shell_result(chartreux_app, pilot)
            assert message._entry is not None
            assert isinstance(message._entry.state, CompletedEffectState)
            assert target.read_text() == "changed"
    finally:
        await agent_loop.aclose()


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
async def test_ui_manual_command_result_is_a_normal_user_turn() -> None:
    """Item-5 B.2: the record is the turn's user message, not an injection."""
    backend = FakeBackend(mock_llm_chunk(content="I saw it."))
    agent_loop = build_test_agent_loop(backend=backend)
    chartreux_app = build_test_chartreux_app(agent_loop=agent_loop)

    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!echo hello"

        await pilot.press("enter")
        await _wait_for_shell_result(chartreux_app, pilot)

        deadline = time.monotonic() + 3
        while not backend.requests_messages:
            if time.monotonic() >= deadline:
                raise AssertionError("Timed out waiting for the record turn")
            await pilot.pause(0.01)

        record = next(
            message
            for message in agent_loop.messages
            if message.role == Role.user
            and "Manual `!` command result from the user." in (message.content or "")
        )
        assert record.injected is False
        assert record.manual_shell is not None
        assert "Command: `echo hello`" in (record.content or "")
        assert "Exit code: 0" in (record.content or "")
        assert "Stdout:\n```text\nhello\n```" in (record.content or "")

        # The command started its own turn: the model received the record as
        # ordinary user input, with no typed prompt in between.
        first_request = backend.requests_messages[0]
        assert first_request[-1].role == Role.user
        assert first_request[-1].content == record.content
        assert first_request[-1].injected is False


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
        assert message.get_content() == "Running: sleep 10"

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
async def test_ui_queues_bash_submitted_while_command_running(
    chartreux_app: ChartreuxApp,
) -> None:
    """Item-5 A.4: shell inputs share the queue, so neither family is refused."""
    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!sleep 1"

        await pilot.press("enter")
        await _wait_for_running_shell(chartreux_app, pilot)
        assert chartreux_app._bash_task is not None
        assert not chartreux_app._bash_task.done()

        chat_input.value = "!echo done"
        await pilot.press("enter")

        assert chat_input.value == ""
        assert not any(
            "cannot be queued" in notification.message
            for notification in chartreux_app._notifications
        )

        # FIFO: the second command was accepted rather than refused, and it runs
        # once the first finishes — whether it was observed queued or already
        # executed by then, its own shell effect arrives.
        assert await _wait_until(
            pilot,
            lambda: any(
                isinstance(message._entry.state, CompletedEffectState)
                and message._entry.state.output
                == {"stdout": "done\n", "stderr": "", "output": "done\n"}
                for message in _shell_results(chartreux_app)
                if message._entry is not None
            ),
            timeout=5.0,
        )


@pytest.mark.asyncio
async def test_ui_empty_bash_produces_the_refused_record() -> None:
    """Item 5 B.1: a bare `!` is a record, never a client-side error."""
    backend = FakeBackend(mock_llm_chunk(content="Bad request."))
    agent_loop = build_test_agent_loop(backend=backend)
    chartreux_app = build_test_chartreux_app(agent_loop=agent_loop)

    async with chartreux_app.run_test() as pilot:
        chat_input = chartreux_app.query_one(ChatInputContainer)
        chat_input.value = "!"

        await pilot.press("enter")
        message = await _wait_for_shell_result(chartreux_app, pilot)

        deadline = time.monotonic() + 3
        while not backend.requests_messages:
            if time.monotonic() >= deadline:
                raise AssertionError("Timed out waiting for the refused record turn")
            await pilot.pause(0.01)

        record = next(
            message
            for message in agent_loop.messages
            if message.role == Role.user
            and "Manual `!` command result from the user." in (message.content or "")
        )
        assert record.injected is False
        assert record.manual_shell is not None
        assert record.manual_shell.status == "refused"
        assert "Status: refused" in (record.content or "")
        assert "Shell command cannot be empty" in (record.content or "")

        # The refusal still shows its own effect tile and mounts no error.
        assert message._entry is not None
        assert isinstance(message._entry.state, FailedEffectState)
        assert_no_command_error(chartreux_app)
