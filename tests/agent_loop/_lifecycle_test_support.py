"""Cross-concern helpers for the dangerous-command lifecycle split (stage 20c).

Phase-(i) support module for the ``test_dangerous_command_lifecycle.py``
split (plan row 20c): the helpers and constants used by more than one
concern move here first, and every concern destination imports them from
this module. The single-concern helpers stay with their concern's tests
in the original file until the 20c phase-(ii) concern commits move them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
from pathlib import Path
import subprocess
from typing import Any

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.questions import UserAnswer, UserQuestionResult
from tests.conftest import build_test_vibe_config

VISION_MODEL = ModelConfig(
    name="vision", provider="mistral", alias="vision", supports_images=True
)

DECLINED_FEEDBACK = "user declined; do not retry this command"
PROMPT_TEXT = "Exercise dangerous-command grants"


def _config() -> ChartreuxConfigSchema:
    return build_test_vibe_config(active_model="vision", models=[VISION_MODEL])


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    return path


def scope_of(path: Path) -> RepositoryScope:
    scope = resolve_repository_scope(path, GitRepositoryRelocation())
    assert scope is not None
    return scope


def _call(name: str, args: dict[str, Any], call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name=name, arguments=json.dumps(args)),
    )


def _bash_args(command: str) -> dict[str, Any]:
    return {"command": command}


async def _drive(
    agent: AgentLoop,
    answer: str | Callable[[UserInputRequestEvent], str] = "Deny",
    *,
    on_prompt: Callable[[UserInputRequestEvent], Any] | None = None,
    respond: Callable[[AgentLoop, UserInputRequestEvent], None] | None = None,
    prompt_text: str = PROMPT_TEXT,
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one turn, answering every dangerous-command prompt.

    ``answer`` is a constant label or a per-request label function;
    ``on_prompt`` runs when a prompt arrives, before it is answered, so a test
    can change the world between the denial and the approval; ``respond``
    replaces the default answer entirely.
    """
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act(prompt_text):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                if on_prompt is not None:
                    await on_prompt(event)
                if respond is not None:
                    respond(agent, event)
                else:
                    label = answer(event) if callable(answer) else answer
                    agent.resolve_user_input_request(
                        event.request_id,
                        UserQuestionResult(
                            answers=[UserAnswer(question="q", answer=label)]
                        ),
                    )
            events.append(event)
    return events, requests


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _authorities(agent: AgentLoop) -> list[tuple[frozenset[str], RepositoryScope]]:
    return [
        (authority.operations, authority.scope)
        for authority in agent._dangerous_command_grants.dangerous_command_authorities()
    ]
