from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from chartreux.core.events import BaseEvent, RequestEvent, ToolResultEvent
from chartreux.core.tools.builtins.bash import CapturedShellResult
from tests.agent_loop.e2e.conftest import MistralAPI, build_e2e_agent_loop
from tests.backend.data.mistral import mistral_completion
from tests.conftest import build_test_vibe_config


def _bash_call(command: str, timeout: int | None = None) -> dict[str, Any]:
    arguments: dict[str, Any] = {"command": command}
    if timeout is not None:
        arguments["timeout"] = timeout
    return mistral_completion(
        "",
        tool_calls=[
            {
                "id": "call_bash",
                "function": {"name": "bash", "arguments": json.dumps(arguments)},
                "index": 0,
            }
        ],
    )


async def _run_bash(
    mistral_api: MistralAPI, command: str, *, timeout: int | None = None
) -> ToolResultEvent:
    mistral_api.reply(_bash_call(command, timeout), mistral_completion("done"))
    agent = build_e2e_agent_loop(config=build_test_vibe_config(enabled_tools=["bash"]))

    events: list[BaseEvent] = []
    async for event in agent.act("go"):
        events.append(event)
        assert not isinstance(event, RequestEvent)
    return next(e for e in events if isinstance(e, ToolResultEvent))


@pytest.mark.asyncio
async def test_bash_captures_stdout(mistral_api: MistralAPI) -> None:
    result = await _run_bash(mistral_api, "echo hello")

    bash_result = cast(CapturedShellResult, result.result)
    assert bash_result.exit_code == 0
    assert "hello" in bash_result.stdout


@pytest.mark.asyncio
async def test_bash_captures_stderr(mistral_api: MistralAPI) -> None:
    result = await _run_bash(
        mistral_api, "python -c \"import sys; sys.stderr.write('oops')\""
    )

    bash_result = cast(CapturedShellResult, result.result)
    assert "oops" in bash_result.stderr


@pytest.mark.asyncio
async def test_bash_nonzero_exit_surfaces_as_error(mistral_api: MistralAPI) -> None:
    result = await _run_bash(mistral_api, "exit 3")

    assert result.error is not None
    assert "Return code: 3" in result.error


@pytest.mark.asyncio
async def test_bash_timeout_surfaces_as_error(mistral_api: MistralAPI) -> None:
    result = await _run_bash(mistral_api, "sleep 5", timeout=1)

    assert result.error is not None
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_bash_output_truncated_to_max_bytes(mistral_api: MistralAPI) -> None:
    result = await _run_bash(mistral_api, "yes x | head -c 100000")

    bash_result = cast(CapturedShellResult, result.result)
    assert len(bash_result.stdout) <= 16_000


@pytest.mark.asyncio
async def test_bash_denylisted_command_is_skipped(mistral_api: MistralAPI) -> None:
    result = await _run_bash(mistral_api, "vim file.txt")

    assert result.skipped is True
    assert result.skip_reason is not None
    assert "denied" in result.skip_reason.lower()


@pytest.mark.asyncio
async def test_bash_allowlisted_command_runs_without_approval(
    mistral_api: MistralAPI,
) -> None:
    # No interaction responder is available; an allowlisted command still runs.
    result = await _run_bash(mistral_api, "echo allowed")

    assert result.skipped is False
    assert "allowed" in cast(CapturedShellResult, result.result).stdout


@pytest.mark.asyncio
async def test_bash_non_allowlisted_command_runs_without_approval(
    mistral_api: MistralAPI,
) -> None:
    result = await _run_bash(mistral_api, "touch newfile.txt")

    assert result.skipped is False
    assert (Path.cwd() / "newfile.txt").exists()


@pytest.mark.asyncio
async def test_bash_command_touching_outside_workdir_is_denied_without_prompt(
    mistral_api: MistralAPI, tmp_path: Path
) -> None:
    outside = tmp_path / "outside.txt"
    result = await _run_bash(mistral_api, f"touch {outside}")

    assert result.skipped is True
    assert result.skip_reason is not None
    assert "outside the authorized workspace" in result.skip_reason
    assert not outside.exists()
