"""Package P4 lifecycle tests: the runtime dangerous-command approval flow.

The loop mirrors the dynamic-roots flow: a grantable dangerous_command denial
opens a user question through the request broker, the answer installs runtime
authority (once permit or session grant), the original call is fully
re-resolved, and only then does it resume. Refusals are remembered per call
identity; noninteractive sessions deny promptly with the remediation; the
persistence port reports "Always for this project" outcomes separately from
the live grant. No real git publication happens: the repositories under test
have no remote, so an admitted ``git push`` fails harmlessly on its own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import contextlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
from typing import Any
from unittest.mock import AsyncMock

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._dangerous_command_grants import (
    DangerousCommandCallIdentity,
)
from chartreux.core.config import (
    ChartreuxConfigSchema,
    ModelConfig,
    SessionLoggingConfig,
)
from chartreux.core.config._dangerous_command_persistence import SavedGrantsRead
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.dangerous_commands import RepositoryScope
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.tools.base import ToolError, ToolPermission
from chartreux.core.tools.builtins._shell_command_policy import (
    GitRepositoryRelocation,
    resolve_repository_scope,
)
from chartreux.core.tools.io_port import ShellCommandResult
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.workspace import Workspace
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio

VISION_MODEL = ModelConfig(
    name="vision", provider="mistral", alias="vision", supports_images=True
)

DECLINED_FEEDBACK = "user declined; do not retry this command"
PROMPT_TEXT = "Exercise dangerous-command grants"


def _config() -> ChartreuxConfigSchema:
    return build_test_vibe_config(active_model="vision", models=[VISION_MODEL])


def _logged_config(save_dir: Path) -> ChartreuxConfigSchema:
    """A config with session logging on, so a resume has a session to rebind."""
    return build_test_vibe_config(
        active_model="vision",
        models=[VISION_MODEL],
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(save_dir)),
    )


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


def _tool_messages(agent: AgentLoop) -> list[str]:
    return [
        message.content or ""
        for message in agent.messages
        if message.role == Role.tool and message.content
    ]


class _StubGrantPort:
    """Runtime persistence port stub: records save attempts, returns a result."""

    def __init__(self, result: ConfigSaveResult) -> None:
        self._result = result
        self.saves: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, project, operation_id, expected_revision))
        return self._result


class _NeverCalledGrantPort:
    """Runtime marker only; a session that never approves never saves."""

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("A session without an approval must not save grants")


class _WorldMutatingGrantPort:
    """Stub port that mutates the world inside the save await (TOCTOU probe)."""

    def __init__(self, agent: AgentLoop, mutate: Callable[[AgentLoop], None]) -> None:
        self._agent = agent
        self._mutate = mutate
        self.saves: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, project, operation_id, expected_revision))
        self._mutate(self._agent)
        return ConfigSaveResult("user", "saved", "unchanged", "rev-2")


class _BlockingGrantPort:
    """Stub port whose save await blocks until released (cancellation probe)."""

    def __init__(self, entered: asyncio.Event, release: asyncio.Event) -> None:
        self._entered = entered
        self._release = release
        self.saves: list[tuple[str, Path, str, str]] = []

    async def save_dangerous_command_grant(
        self, session_id: str, project: Path, operation_id: str, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, project, operation_id, expected_revision))
        self._entered.set()
        await self._release.wait()
        return ConfigSaveResult("user", "saved", "unchanged", "rev-2")


def _patch_saved_grants(
    monkeypatch: pytest.MonkeyPatch, revision: str = "user-rev-1"
) -> AsyncMock:
    """Point the loop's saved-grants read surface at a controlled revision."""
    read = AsyncMock(
        return_value=SavedGrantsRead(operations=(), user_revision=revision)
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_dangerous_command_grants", read
    )
    return read


def _authorities(agent: AgentLoop) -> list[tuple[frozenset[str], RepositoryScope]]:
    return [
        (authority.operations, authority.scope)
        for authority in agent._dangerous_command_grants.dangerous_command_authorities()
    ]


def _script_recheck(
    monkeypatch: pytest.MonkeyPatch, tool: Any, recheck: Callable[[Any], Any]
) -> list[int]:
    """Script only the post-approval recheck of a tool's resolution.

    The first resolution — the original denial that opens the approval
    prompt — stays real; every later resolution runs *recheck*, so the
    approval flow's full recheck is controlled without touching the denial
    that started it. Returns the resolution call counter.
    """
    real = tool.resolve_permission
    calls = [0]

    def resolve(args: Any) -> Any:
        calls[0] += 1
        if calls[0] == 1:
            return real(args)
        return recheck(args)

    monkeypatch.setattr(tool, "resolve_permission", resolve)
    return calls


def _spy_invalidations(monkeypatch: pytest.MonkeyPatch, agent: AgentLoop) -> list[int]:
    """Count dangerous-command authority invalidations on the tool manager."""
    manager = agent.tool_manager
    real = manager._invalidate_dangerous_command_authority
    calls = [0]

    def invalidate() -> None:
        calls[0] += 1
        real()

    monkeypatch.setattr(manager, "_invalidate_dangerous_command_authority", invalidate)
    return calls


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_allow_once_resumes_exactly_one_original_call(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    marker = project / "executed.txt"
    command = f"git push; touch {marker.name}"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "once-1")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "once-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow once")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert not results[0].skipped
        if tool == "bash":
            assert marker.exists()
        # The once permit authorized exactly this invocation: no authority
        # remains, so the identical call prompts again instead of resuming.
        assert _authorities(agent) == []
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_allow_session_grants_the_operation_for_the_session(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    command = "git push"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "sess-1")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "sess-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        # The second identical call resumes without prompting.
        events, requests = await _drive(agent)
        assert requests == []
        assert len(_results(events)) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
    finally:
        await agent.aclose()


async def test_session_grant_is_bound_to_the_equivalent_scope(tmp_path: Path) -> None:
    repo_a = make_repo(tmp_path / "a")
    make_repo(tmp_path / "b")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git -C a push"), "scope-1")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git -C b push"), "scope-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=tmp_path, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(repo_a))]
        # A sibling repository is a different scope: it still prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(repo_a))]
    finally:
        await agent.aclose()


async def test_sibling_worktree_scope_is_isolated(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    subprocess.run(
        [
            "git",
            "-C",
            str(project),
            "-c",
            "user.email=t@t",
            "-c",
            "user.name=t",
            "commit",
            "--allow-empty",
            "-q",
            "-m",
            "init",
        ],
        check=True,
    )
    worktree = project / "wt"
    subprocess.run(
        ["git", "-C", str(project), "worktree", "add", "-q", str(worktree), "-b", "wt"],
        check=True,
    )
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git -C wt push"), "wt-1")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "proj-1")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(worktree))]
        # The main worktree is a different repository identity: it prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
    finally:
        await agent.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_deny_records_refusal_and_never_reprompts(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    command = "git push"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "deny-1")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "deny-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        assert results[0].result is None
        assert _authorities(agent) == []
        # The decided call identity returns the honest refusal, no re-prompt.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_refusal_memory_does_not_block_unrelated_operations(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "ref-1")])],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "ref-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        # A different call identity is a different decision: it still prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_refusal_memory_matches_shlex_token_variants_only(tmp_path: Path) -> None:
    """Refusal identity is the whole-call shlex token tuple, nothing smarter.

    Whitespace and quoting variants of the denied command hit the same
    refusal record (no re-prompt); a semantically equivalent reformulation
    (-f vs --force) tokenizes differently and re-prompts.
    """
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "canon-1")])],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git  push"), "canon-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args('git "push"'), "canon-3")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git push -f"), "canon-4")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git push --force"), "canon-5")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        # Deny the plain push: the refusal is remembered for its token tuple.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        # Whitespace and quoting variants are the same call identity: the
        # decided refusal answers, no re-prompt.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        assert _results(events)[0].skipped
        assert _results(events)[0].skip_reason == DECLINED_FEEDBACK
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        assert _results(events)[0].skipped
        assert _results(events)[0].skip_reason == DECLINED_FEEDBACK
        # -f vs --force: semantically equivalent, different tokens, so the
        # reformulation is a new decision and prompts again.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_cancelled_prompt_denies_and_records_refusal(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    command = "git push"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "esc-1")])],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args(command), "esc-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events = []
        requests: list[UserInputRequestEvent] = []
        async with asyncio.timeout(10):
            async for event in agent.act(PROMPT_TEXT):
                if isinstance(event, UserInputRequestEvent):
                    requests.append(event)
                    agent.resolve_user_input_request(
                        event.request_id, UserQuestionResult(answers=[], cancelled=True)
                    )
                events.append(event)
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        # Escape counts as a decision: the refusal is remembered.
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        assert _results(events)[0].skipped
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_forged_consequential_always_is_rejected(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "forged")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        # The consequential class never offers the persistent choice.
        assert "Always for this project" not in {
            option.label for option in request.questions[0].options
        }
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == DECLINED_FEEDBACK
        # The forged answer installed no authority at all.
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_concurrent_identical_once_calls_need_two_approvals(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    command = "git push"
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(tool, _bash_args(command), "same-1", 0),
                    _call(tool, _bash_args(command), "same-2", 1),
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow once")
        # No roots-style coalescing: two identical concurrent calls, two prompts.
        assert len(requests) == 2
        assert {request.request_id for request in requests} == {
            requests[0].request_id,
            requests[1].request_id,
        }
        results = _results(events)
        assert len(results) == 2
        assert all(not result.skipped for result in results)
        # Each permit was consumed by its own invocation; nothing is reusable.
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_duplicate_callback_completion_cannot_execute_another_call(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "dup")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow once")
        assert len(requests) == 1
        assert len(_results(events)) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == []
        # A duplicate completion of the settled callback is a no-op: it cannot
        # install authority or execute another call.
        agent.resolve_user_input_request(
            requests[0].request_id,
            UserQuestionResult(
                answers=[UserAnswer(question="q", answer="Allow this session")]
            ),
        )
        assert _authorities(agent) == []
        assert len(_results(events)) == 1
    finally:
        await agent.aclose()


async def test_model_generated_question_installs_no_authority(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    question_call = _call(
        "ask_user_question",
        {
            "questions": [
                {
                    "question": "Approve the push?",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                }
            ]
        },
        "model-q",
        0,
    )
    push_call = _call("bash", _bash_args("git push"), "model-push", 1)
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[question_call, push_call])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )

    def respond(loop: AgentLoop, event: UserInputRequestEvent) -> None:
        args = event.args
        header = (
            args.questions[0].header if isinstance(args, UserQuestionRequest) else ""
        )
        if header == "Dangerous command":
            label = "Deny"
        else:
            # The model's own question is answered as if the user approved.
            label = "Yes"
        loop.resolve_user_input_request(
            event.request_id,
            UserQuestionResult(answers=[UserAnswer(question="q", answer=label)]),
        )

    try:
        events, requests = await _drive(agent, respond=respond)
        assert len(requests) == 2
        results = _results(events)
        push_results = [
            result for result in results if result.tool_call_id == "model-push"
        ]
        assert len(push_results) == 1
        assert push_results[0].skipped
        assert push_results[0].skip_reason == DECLINED_FEEDBACK
        # The conversational "Yes" installed no authority.
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_mixed_compound_call_is_fully_displayed(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    command = "git push && git push --force"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args(command), "mixed")])],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "mixed-2")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        question = request.questions[0]
        # The whole compound call is displayed, with every operation.
        assert "git push" in question.question
        assert "git push --force" in question.question
        assert "git.push" in question.question
        assert "git.push.force" in question.question
        # A mixed call offers only the consequential choices.
        assert [option.label for option in question.options] == [
            "Deny",
            "Allow once",
            "Allow this session",
        ]
        assert not _results(events)[0].skipped
        # The session approval covers exactly the proposed operations.
        assert _authorities(agent) == [
            (frozenset({"git.push", "git.push.force"}), scope_of(project))
        ]
        # An operation outside the approval still prompts.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
    finally:
        await agent.aclose()


async def test_consequential_session_grant_does_not_authorize_publication(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git push --force"), "cons-1")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "cons-2")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert _authorities(agent) == [
            (frozenset({"git.push.force"}), scope_of(project))
        ]
        # Ordinary publication was never approved: it prompts again.
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        assert _authorities(agent) == [
            (frozenset({"git.push.force"}), scope_of(project))
        ]
    finally:
        await agent.aclose()


async def test_multi_repo_approval_covers_all_operations_and_scopes(
    tmp_path: Path,
) -> None:
    repo_a = make_repo(tmp_path / "a")
    repo_b = make_repo(tmp_path / "b")
    command = "git -C a push && git -C b checkout"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args(command), "multi")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=tmp_path, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        question = request.questions[0]
        # The proposal is complete (every operation), the scope representative.
        assert "git.push" in question.question
        assert "git.checkout" in question.question
        assert not _results(events)[0].skipped
        # The approval covers every operation at every scope: the recheck
        # passes only when both repositories carry their operation.
        assert set(_authorities(agent)) == {
            (frozenset({"git.push"}), scope_of(repo_a)),
            (frozenset({"git.checkout"}), scope_of(repo_b)),
        }
    finally:
        await agent.aclose()


async def test_multi_repo_question_displays_every_repository(tmp_path: Path) -> None:
    """The approval question names every repository the call touches."""
    repo_a = make_repo(tmp_path / "a")
    repo_b = make_repo(tmp_path / "b")
    command = "git -C a push && git -C b checkout"
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args(command), "multi-q")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=tmp_path, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        question = request.questions[0]
        # Every repository the approval grants is displayed, not only the
        # proposal's representative scope.
        assert (
            f"Repositories: {scope_of(repo_a).work_tree}, "
            f"{scope_of(repo_b).work_tree}" in question.question
        )
        assert "Repository: " not in question.question
        assert str(tmp_path) in question.question
        # The session choice no longer claims a single repository.
        session = next(
            option
            for option in question.options
            if option.label == "Allow this session"
        )
        assert session.description == (
            "Allow these operations in these repositories for this session"
        )
        assert not _results(events)[0].skipped
        assert set(_authorities(agent)) == {
            (frozenset({"git.push"}), scope_of(repo_a)),
            (frozenset({"git.checkout"}), scope_of(repo_b)),
        }
    finally:
        await agent.aclose()


class TestRecheckFailure:
    """A world change between the denial and the approval executes nothing."""

    async def test_new_denylist_entry_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call(
                            "bash",
                            _bash_args(f"git push; touch {marker.name}"),
                            "recheck",
                        )
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )

        async def add_denylist(event: UserInputRequestEvent) -> None:
            agent.config.tools.setdefault("bash", {})["denylist"] = ["git"]

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=add_denylist)
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "denylist" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_new_disablement_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def disable(event: UserInputRequestEvent) -> None:
            agent.config.tools.setdefault("bash", {})["permission"] = "never"

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=disable)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_new_sensitivity_rule_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def sensitize(event: UserInputRequestEvent) -> None:
            agent.config.tools.setdefault("bash", {})["sensitive_patterns"] = ["git"]

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=sensitize)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert "sensitive" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_new_path_restriction_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        sub = project / "sub"
        sub.mkdir()
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git -C sub push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def relocate(event: UserInputRequestEvent) -> None:
            shutil.rmtree(sub)
            sub.symlink_to("/etc")

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=relocate)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_changed_repo_identity_fails_the_recheck(
        self, tmp_path: Path
    ) -> None:
        project = make_repo(tmp_path / "project")
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "-c",
                "user.email=t@t",
                "-c",
                "user.name=t",
                "commit",
                "--allow-empty",
                "-q",
                "-m",
                "init",
            ],
            check=True,
        )
        worktree = project / "wt"
        subprocess.run(
            [
                "git",
                "-C",
                str(project),
                "worktree",
                "add",
                "-q",
                str(worktree),
                "-b",
                "wt",
            ],
            check=True,
        )
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git -C wt push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def dissolve(event: UserInputRequestEvent) -> None:
            # The approved worktree identity no longer resolves to itself.
            (worktree / ".git").unlink()

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=dissolve)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()

    async def test_retired_manager_fails_the_recheck(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            _call(
                                "bash",
                                _bash_args(f"git push; touch {marker.name}"),
                                "recheck",
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        async def retire(event: UserInputRequestEvent) -> None:
            agent.tool_manager._authority_retired = True

        try:
            events, requests = await _drive(agent, "Allow once", on_prompt=retire)
            assert len(requests) == 1
            results = _results(events)
            assert results[0].skipped
            assert "retired" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert _authorities(agent) == []
        finally:
            await agent.aclose()


class TestPersistAwaitAdmissionGate:
    """The persist await is a world-change window; the recheck after it gates.

    For "Always for this project" the approval flow persists BETWEEN its
    recheck and execution, so the world can change while the save runs. The
    full resolution runs again after the await — the final admission gate,
    exactly as the roots flow rechecks last — and a mid-await change skips
    the call instead of executing it.
    """

    async def test_denylist_added_inside_the_save_await_skips_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args(command), "toctou")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        def deny_git(loop: AgentLoop) -> None:
            loop.config.tools.setdefault("bash", {})["denylist"] = ["git"]

        port = _WorldMutatingGrantPort(agent, deny_git)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            # git became denylisted during the save await: the post-await
            # recheck refuses the call and nothing executes.
            assert results[0].skipped
            assert "denylist" in (results[0].skip_reason or "")
            assert not marker.exists()
            # The approval still installed the live session grant and the
            # save still ran; only this invocation is refused.
            assert len(port.saves) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        finally:
            await agent.aclose()

    async def test_retired_manager_inside_the_save_await_skips_cleanly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args(command), "retire")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
        )

        def retire(loop: AgentLoop) -> None:
            loop.tool_manager._authority_retired = True

        port = _WorldMutatingGrantPort(agent, retire)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            # A mid-await manager retirement is a clean SKIP, not an
            # invoke-time error.
            assert results[0].skipped
            assert "retired" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert len(port.saves) == 1
        finally:
            await agent.aclose()


class _BlockingSnapshot:
    """Deterministically block the pre-invoke snapshot await, thread-side.

    The loop awaits ``get_file_snapshot`` through ``asyncio.to_thread``
    before invoking the tool, so blocking here holds the call inside the
    world-change window between the admission pre-check and execution.
    """

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def __call__(self, args: Any) -> Any:
        self.calls += 1
        self.entered.set()
        assert self.release.wait(10)
        return None


class _RecordingToolIO:
    """Client-terminal delegation port that records every shell request."""

    supports_terminal = True

    def __init__(self) -> None:
        self.shell_requests: list[Any] = []

    async def run_shell(self, request: Any) -> Any:
        self.shell_requests.append(request)
        return ShellCommandResult(stdout="host output", stderr="", returncode=0)


def _revoke_live_grants(agent: AgentLoop) -> None:
    """Revoke every live dangerous-command grant, as the endpoint does."""
    store = agent._dangerous_command_grants
    requirements = {
        (operation_id, authority.scope)
        for authority in store.dangerous_command_authorities()
        for operation_id in authority.operations
    }
    store.revoke_session(requirements)
    agent.tool_manager._invalidate_dangerous_command_authority()


async def _drive_through_window(
    agent: AgentLoop,
    answer: str | Callable[[UserInputRequestEvent], str],
    *,
    block: _BlockingSnapshot,
    mutate: Callable[[AgentLoop], None],
    prompt_text: str = PROMPT_TEXT,
    tool_io: Any = None,
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one turn; ``mutate`` runs while the snapshot await is blocked."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []

    async def consume() -> None:
        async for event in agent.act(prompt_text, tool_io=tool_io):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                label = answer(event) if callable(answer) else answer
                agent.resolve_user_input_request(
                    event.request_id,
                    UserQuestionResult(
                        answers=[UserAnswer(question="q", answer=label)]
                    ),
                )
            events.append(event)

    task = asyncio.create_task(consume())
    try:
        async with asyncio.timeout(10):
            while not block.entered.is_set():
                await asyncio.sleep(0.005)
            mutate(agent)
            block.release.set()
        await asyncio.wait_for(task, 10)
    finally:
        block.release.set()
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    return events, requests


class TestExecutionAdmissionGate:
    """The snapshot await is a revocation window; the handoff gate blocks.

    The approval flow's final recheck returns EXECUTE, but the loop then
    awaits the tool's file snapshot before invoking it. Authority revoked
    while that await is blocked must be enforced at the execution handoff —
    after the snapshot, immediately before invoke — for every grant type,
    with the denial accounted as a rejected call and never re-prompted.
    """

    @pytest.mark.parametrize("tool", ["bash", "bash_start"])
    async def test_session_grant_revoked_in_the_window_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call(tool, _bash_args(command), "gate-sess")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get(tool), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent, "Allow this session", block=block, mutate=_revoke_live_grants
            )
            # The revoked grant is not re-prompted: the gate fails closed.
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            # A POLICY-class denial: the blocked requirement stated plainly,
            # with no alternative offered.
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            assert "Instead," not in (results[0].skip_reason or "")
            assert block.calls == 1
            assert not marker.exists()
            assert _authorities(agent) == []
            # The denial is accounted as a rejected call, not an execution.
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
            assert agent.stats.tool_calls_failed == 0
        finally:
            await agent.aclose()

    async def test_project_grant_revoked_in_the_window_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-proj")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev-2"))
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent,
                "Always for this project",
                block=block,
                mutate=_revoke_live_grants,
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            assert not marker.exists()
            # The save still ran; only this invocation is refused.
            assert len(port.saves) == 1
            assert _authorities(agent) == []
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
        finally:
            await agent.aclose()

    async def test_previously_granted_call_revoked_in_the_window_blocks_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        first = project / "first.txt"
        second = project / "second.txt"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call(
                            "bash",
                            _bash_args(f"git push; touch {first.name}"),
                            "gate-prev-1",
                        )
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call(
                            "bash",
                            _bash_args(f"git push; touch {second.name}"),
                            "gate-prev-2",
                        )
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        try:
            # Turn 1: the session grant is approved and executes.
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert first.exists()
            # Turn 2: the operation is already granted, so no prompt arrives;
            # the revocation inside the snapshot window still blocks it.
            monkeypatch.setattr(
                agent.tool_manager.get("bash"), "get_file_snapshot", block
            )
            events, requests = await _drive_through_window(
                agent, "Deny", block=block, mutate=_revoke_live_grants
            )
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            assert not second.exists()
            assert first.exists()
            assert _authorities(agent) == []
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_succeeded == 1
        finally:
            await agent.aclose()

    @pytest.mark.parametrize(
        "answer", ["Allow once", "Allow this session", "Always for this project"]
    )
    async def test_a_still_valid_grant_executes_through_the_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-ok")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev-2"))
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent, answer, block=block, mutate=lambda loop: None
            )
            # One approval, no re-prompt at the gate.
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert marker.exists()
            assert block.calls == 1
            assert agent.stats.tool_calls_agreed == 1
            assert agent.stats.tool_calls_rejected == 0
            assert agent.stats.tool_calls_succeeded == 1
            if answer == "Allow once":
                # The gate consumed the once authority for exactly this
                # invocation: nothing is reusable.
                assert _authorities(agent) == []
                assert agent._dangerous_command_grants._once == []
            else:
                # A session/project grant is not consumed by the gate.
                assert _authorities(agent) == [
                    (frozenset({"git.push"}), scope_of(project))
                ]
        finally:
            await agent.aclose()

    async def test_concurrent_identical_call_is_not_authorized_in_the_once_window(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("bash", _bash_args(command), "once-a", 0),
                        _call("bash", _bash_args(command), "once-b", 1),
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        answers = iter(["Allow once", "Deny"])
        try:
            events, requests = await _drive_through_window(
                agent,
                lambda event: next(answers),
                block=block,
                mutate=lambda loop: None,
            )
            # Two prompts: the concurrent identical call did not inherit the
            # first call's once authority while its snapshot await was blocked.
            assert len(requests) == 2
            results = _results(events)
            assert len(results) == 2
            executed = [result for result in results if not result.skipped]
            denied = [result for result in results if result.skipped]
            assert len(executed) == 1
            assert len(denied) == 1
            assert denied[0].skip_reason == DECLINED_FEEDBACK
            assert marker.exists()
            # Only the authorized invocation reached the snapshot; the gate
            # consumed the once authority for exactly that invocation.
            assert block.calls == 1
            assert _authorities(agent) == []
            assert agent._dangerous_command_grants._once == []
            assert agent.stats.tool_calls_agreed == 1
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_succeeded == 1
        finally:
            await agent.aclose()

    async def test_client_delegation_gets_the_same_gate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-io")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        tool_io = _RecordingToolIO()
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent,
                "Allow this session",
                block=block,
                mutate=_revoke_live_grants,
                tool_io=tool_io,
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason or ""
            )
            # The delegated execution never ran.
            assert tool_io.shell_requests == []
            assert not marker.exists()
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
        finally:
            await agent.aclose()

    async def test_client_delegation_executes_when_the_grant_is_still_valid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        command = "git push"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-io-ok")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        tool_io = _RecordingToolIO()
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)
        try:
            events, requests = await _drive_through_window(
                agent,
                "Allow this session",
                block=block,
                mutate=lambda loop: None,
                tool_io=tool_io,
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            # The delegated execution ran exactly once through the gate.
            assert len(tool_io.shell_requests) == 1
            assert agent.stats.tool_calls_agreed == 1
            assert agent.stats.tool_calls_rejected == 0
            assert agent.stats.tool_calls_succeeded == 1
        finally:
            await agent.aclose()

    @pytest.mark.parametrize("answer", ["Allow once", "Allow this session"])
    @pytest.mark.parametrize(
        "kind, fragment",
        [
            ("disable", "disabled by policy"),
            ("denylist", "denylist"),
            ("retire", "retired"),
            ("sensitize", "sensitive"),
        ],
    )
    async def test_policy_revocation_in_the_window_blocks_execution(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        answer: str,
        kind: str,
        fragment: str,
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        command = f"git push; touch {marker.name}"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args(command), "gate-pol")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)

        def mutate(loop: AgentLoop) -> None:
            if kind == "disable":
                loop.config.tools.setdefault("bash", {})["permission"] = "never"
            elif kind == "denylist":
                loop.config.tools.setdefault("bash", {})["denylist"] = ["git"]
            elif kind == "retire":
                loop.tool_manager._authority_retired = True
            else:
                loop.config.tools.setdefault("bash", {})["sensitive_patterns"] = ["git"]

        try:
            events, requests = await _drive_through_window(
                agent, answer, block=block, mutate=mutate
            )
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert fragment in (results[0].skip_reason or "")
            assert not marker.exists()
            assert block.calls == 1
            # The denial is accounted, and no once authority survived the
            # gate whether the revalidation passed or failed.
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
            assert agent.stats.tool_calls_failed == 0
            assert agent._dangerous_command_grants._once == []
        finally:
            await agent.aclose()

    async def test_ungranted_call_is_denied_at_the_gate_when_policy_changes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("bash", _bash_args(f"touch {marker.name}"), "gate-plain")
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        block = _BlockingSnapshot()
        monkeypatch.setattr(agent.tool_manager.get("bash"), "get_file_snapshot", block)

        def mutate(loop: AgentLoop) -> None:
            loop.config.tools.setdefault("bash", {})["denylist"] = ["touch"]

        try:
            events, requests = await _drive_through_window(
                agent, "Deny", block=block, mutate=mutate
            )
            # A routine call never prompts; the gate still enforces the
            # world change that happened inside the snapshot window.
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert "denylist" in (results[0].skip_reason or "")
            assert not marker.exists()
            assert agent.stats.tool_calls_rejected == 1
            assert agent.stats.tool_calls_agreed == 0
            assert agent.stats.tool_calls_succeeded == 0
        finally:
            await agent.aclose()


class TestOncePermitCleanup:
    """An installed once permit never outlives the resume, whatever exits it.

    The approval flow installs a once permit, fully re-resolves the original
    call, and only a resumed call consumes the permit. A resolver that
    raises, a NEVER recheck, or a cancellation must all leave the store
    without the permit: a leaked permit would authorize any later matching
    call without a new grant.
    """

    async def _bound_tool_and_args(
        self, agent: AgentLoop, command: str
    ) -> tuple[Any, Any]:
        tool = agent.tool_manager.get("bash")
        args = tool.validate_arguments({"command": command})
        return tool, args

    async def _answer_prompt(
        self, agent: AgentLoop, queue: asyncio.Queue, label: str
    ) -> None:
        event = await asyncio.wait_for(queue.get(), 5)
        assert isinstance(event, UserInputRequestEvent)
        agent.resolve_user_input_request(
            event.request_id,
            UserQuestionResult(answers=[UserAnswer(question="q", answer=label)]),
        )

    async def test_resolver_exception_after_install_once_destroys_the_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recheck that raises must not leak the installed once permit."""
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(
                agent, f"git push; touch {marker.name}"
            )
            generation_before = agent._dangerous_command_grants.authority_generation
            counts_at_raise: list[int] = []

            def explode(_args: Any) -> Any:
                counts_at_raise.append(invalidations[0])
                raise RuntimeError("resolver exploded during the recheck")

            _script_recheck(monkeypatch, tool, explode)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "explode")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            with pytest.raises(RuntimeError, match="resolver exploded"):
                await task
            # The permit installed for the approval did not survive the
            # failed recheck: the store holds no once permit and no authority.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            # Install bumped the generation once and the cleanup destroy
            # bumped it again; a leaked permit would leave it one bump short.
            assert (
                agent._dangerous_command_grants.authority_generation
                == generation_before + 2
            )
            # The cached authority the permit fed was invalidated after the
            # failure, not only when the permit was installed.
            assert counts_at_raise == [1]
            assert invalidations[0] == 2
            assert not marker.exists()
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_never_verdict_at_the_recheck_destroys_the_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A recheck that denies executes nothing and leaves no permit."""
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(
                agent, f"git push; touch {marker.name}"
            )
            denial = PermissionContext(
                permission=ToolPermission.NEVER, reason="recheck denied"
            )
            _script_recheck(monkeypatch, tool, lambda _args: denial)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "recheck-never")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "SKIP"
            assert decision.feedback == "recheck denied"
            # Nothing executed and no once authority survived the denial.
            assert not marker.exists()
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            assert invalidations[0] == 2
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_cancellation_at_the_recheck_destroys_the_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CancelledError is not an Exception: the finally still cleans up.

        The recheck itself is synchronous, so a task cancellation is
        delivered at an await; raising CancelledError at the scripted
        recheck reproduces exactly that delivery while the permit is live.
        """
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(agent, "git push")
            counts_at_cancel: list[int] = []

            def cancel(_args: Any) -> Any:
                counts_at_cancel.append(invalidations[0])
                raise asyncio.CancelledError()

            _script_recheck(monkeypatch, tool, cancel)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "recheck-cancel")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            with pytest.raises(asyncio.CancelledError):
                await task
            # The permit installed for the approval did not survive the
            # cancelled recheck: no once permit, no authority, no execution.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            assert counts_at_cancel == [1]
            assert invalidations[0] == 2
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_cancellation_during_the_persist_await_leaves_no_once_permit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allow_project path installs no once permit, so cancelling the
        save await leaks no permit; the intentional session grant remains.
        """
        project = make_repo(tmp_path / "project")
        marker = project / "executed.txt"
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        save_entered = asyncio.Event()
        release = asyncio.Event()
        port = _BlockingGrantPort(save_entered, release)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        try:
            tool, args = await self._bound_tool_and_args(
                agent, f"git push; touch {marker.name}"
            )
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "persist-cancel")
            )
            await self._answer_prompt(agent, queue, "Always for this project")
            await asyncio.wait_for(save_entered.wait(), 5)
            # The approval installed a session grant, never a once permit:
            # there is no permit to leak while the save await is suspended.
            assert agent._dangerous_command_grants._once == []
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # Cleanup is total: still no once permit, and the session grant
            # is intentional and persistent — it is not rolled back.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            assert not marker.exists()
        finally:
            release.set()
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_matching_call_after_a_failed_recheck_needs_a_new_grant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression pin: a permit leaked by a failed recheck would
        authorize a later identical call without a new grant; it must not.
        """
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        try:
            tool, args = await self._bound_tool_and_args(agent, "git push")

            def explode(_args: Any) -> Any:
                raise RuntimeError("resolver exploded during the recheck")

            _script_recheck(monkeypatch, tool, explode)
            task = asyncio.create_task(agent._should_execute_tool(tool, args, "leak-1"))
            await self._answer_prompt(agent, queue, "Allow once")
            with pytest.raises(RuntimeError, match="resolver exploded"):
                await task
            assert agent._dangerous_command_grants._once == []
            # The leaked-permit scenario is over: the real resolver is back.
            monkeypatch.delattr(tool, "resolve_permission")
            # The identical call is denied again: with no residual authority
            # the resolver proposes a fresh approval instead of resuming, so
            # a prompt arrives (a leaked permit would skip it and execute).
            task = asyncio.create_task(agent._should_execute_tool(tool, args, "leak-2"))
            await self._answer_prompt(agent, queue, "Deny")
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "SKIP"
            assert decision.feedback == DECLINED_FEEDBACK
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_successful_allow_once_consumes_the_permit_before_resume(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Normal path unchanged: the recheck passes, the permit is consumed
        before the call resumes, and exactly one execution is authorized.
        """
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        invalidations = _spy_invalidations(monkeypatch, agent)
        try:
            tool, args = await self._bound_tool_and_args(agent, "git push")
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "allow-once-ok")
            )
            await self._answer_prompt(agent, queue, "Allow once")
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "EXECUTE"
            # The permit authorized exactly this invocation and is gone: no
            # authority remains for a later matching call.
            assert agent._dangerous_command_grants._once == []
            assert _authorities(agent) == []
            # Install and consume each invalidated once; the cleanup finally
            # added no extra invalidation on the success path.
            assert invalidations[0] == 2
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()


@pytest.mark.parametrize("tool", ["bash", "bash_start"])
async def test_noninteractive_session_denies_with_remediation(
    tmp_path: Path, tool: str
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, _bash_args("git push"), "no-cap")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        agent.bind_dangerous_command_grant_port(_NeverCalledGrantPort())
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert "dangerous_command_grants_by_project" in results[0].skip_reason
        assert "config.toml" in results[0].skip_reason
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_noninteractive_consequential_denial_names_no_persistent_knob(
    tmp_path: Path,
) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("bash", _bash_args("git stash drop"), "no-cap")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        agent.bind_dangerous_command_grant_port(_NeverCalledGrantPort())
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert "cannot be pre-authorized" in results[0].skip_reason
        assert "interactive session" in results[0].skip_reason
        assert "dangerous_command_grants_by_project" not in results[0].skip_reason
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_noninteractive_mixed_call_names_only_the_persistable_part(
    tmp_path: Path,
) -> None:
    """A mixed call: the config knob pre-authorizes only part of the call."""
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "bash",
                        _bash_args("git push && git push --force"),
                        "mixed-no-cap",
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        agent.bind_dangerous_command_grant_port(_NeverCalledGrantPort())
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        reason = results[0].skip_reason or ""
        # The persistable subset is named as partial pre-authorization only.
        assert "Add git.push to dangerous_command_grants_by_project" in reason
        assert "pre-authorize only part of this call" in reason
        assert "remaining operations (git.push.force)" in reason
        assert "cannot be pre-authorized" in reason
        assert "interactive session" in reason
        # The message must not claim the config edit authorizes the whole call.
        assert "to authorize it." not in reason
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_bare_loop_without_a_port_keeps_the_plain_denial(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "bare")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert (
            "requires approval for a dangerous git operation" in results[0].skip_reason
        )
        assert "git.push" in results[0].skip_reason
        assert _authorities(agent) == []
    finally:
        await agent.aclose()


async def test_prompt_shape_shows_the_frozen_choices(tmp_path: Path) -> None:
    project = make_repo(tmp_path / "project")
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("bash", _bash_args("git push"), "shape")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        request = requests[0].args
        assert isinstance(request, UserQuestionRequest)
        question = request.questions[0]
        # Deny is the default-selected (first) option; single-select.
        assert [option.label for option in question.options] == [
            "Deny",
            "Allow once",
            "Allow this session",
            "Always for this project",
        ]
        assert question.multi_select is False
        assert question.hide_other is True
        assert len(question.header) <= 20
        assert "git push" in question.question
        assert str(project.resolve()) in question.question
        assert str(scope_of(project).work_tree) in question.question
        assert "permission authority" in question.question
        assert "standing intent to publish" in question.question
        assert request.footer_note
    finally:
        await agent.aclose()


class TestPersistencePort:
    """Live approval and persistence outcomes are reported separately."""

    async def test_always_applies_the_grant_and_reports_the_saved_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push; true"), "always")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev-2"))
        agent.bind_dangerous_command_grant_port(port)
        read = _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            # The live session grant applies exactly like Allow-session does.
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            # The user revision is fetched from the read surface at decision time.
            assert read.call_count == 1
            assert read.call_args.kwargs["project"] == project.resolve()
            assert port.saves == [
                (agent.session_id, project.resolve(), "git.push", "user-rev-1")
            ]
            assert any(
                "applies to this session" in text
                and "saved in your user config" in text
                and str(project.resolve()) in text
                for text in _tool_messages(agent)
            )
        finally:
            await agent.aclose()

    async def test_always_reports_a_conflict_without_retrying(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push; true"), "conflict")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(
            ConfigSaveResult("user", "not_saved", "unchanged", error="conflict")
        )
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            # No automatic retry and no automatic re-prompt.
            assert len(requests) == 1
            assert len(port.saves) == 1
            assert not _results(events)[0].skipped
            # The live grant still applies; the save is reported session-only.
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            assert any(
                "applies to this session only" in text and "retry" in text
                for text in _tool_messages(agent)
            )
        finally:
            await agent.aclose()

    @pytest.mark.parametrize(
        ("result", "expected"),
        [
            (
                ConfigSaveResult("user", "not_saved", "unchanged", error="write"),
                "saving it to your user config failed",
            ),
            (
                ConfigSaveResult("user", "durability_uncertain", "unchanged", "rev"),
                "could not be confirmed",
            ),
            (
                ConfigSaveResult("user", "saved", "unchanged", "rev", "cancelled"),
                "was cancelled",
            ),
        ],
    )
    async def test_always_reports_save_failures_honestly(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        result: ConfigSaveResult,
        expected: str,
    ) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push; true"), "failure")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(result)
        agent.bind_dangerous_command_grant_port(port)
        _patch_saved_grants(monkeypatch)
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            # The live session grant still applies.
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            assert any(
                expected in text and "applies to this session" in text
                for text in _tool_messages(agent)
            )
        finally:
            await agent.aclose()

    async def test_always_without_a_save_transport_reports_it(
        self, tmp_path: Path
    ) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push; true"), "bare")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            assert any(
                "cannot save dangerous-command grants" in text
                for text in _tool_messages(agent)
            )
        finally:
            await agent.aclose()

    async def test_always_without_a_readable_saved_surface_skips_the_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("bash", _bash_args("git push; true"), "unreadable")
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev"))
        agent.bind_dangerous_command_grant_port(port)
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.read_saved_dangerous_command_grants",
            AsyncMock(return_value=SavedGrantsRead(unavailable="no_user_source")),
        )
        try:
            events, requests = await _drive(agent, "Always for this project")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            assert port.saves == []
            assert any("could not be read" in text for text in _tool_messages(agent))
        finally:
            await agent.aclose()

    async def test_always_fetches_the_user_revision_at_decision_time(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        make_repo(tmp_path / "a")
        make_repo(tmp_path / "b")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git -C a push; true"), "one")]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git -C b push; true"), "two")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=tmp_path, backend=backend, user_input_capability=True
        )
        port = _StubGrantPort(ConfigSaveResult("user", "saved", "unchanged", "rev"))
        agent.bind_dangerous_command_grant_port(port)
        read = _patch_saved_grants(monkeypatch)
        try:
            await _drive(agent, "Always for this project")
            await _drive(agent, "Always for this project")
            # The revision is read per decision and never cached across turns.
            assert read.call_count == 2
            assert [save[3] for save in port.saves] == ["user-rev-1", "user-rev-1"]
            # Both saves are keyed by the registered project (the loop's cwd).
            assert {save[1] for save in port.saves} == {tmp_path.resolve()}
        finally:
            await agent.aclose()


class TestSessionLifetime:
    """Runtime grants live exactly as long as the session."""

    async def test_session_grant_survives_compaction(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
            [mock_llm_chunk(content="<summary>summary</summary>")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            summary = await agent.compact()
            assert summary == "summary"
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            tool = agent.tool_manager.get("bash")
            decision = tool.resolve_permission(
                tool.validate_arguments({"command": "git push"})
            )
            assert decision is not None
            assert decision.permission == ToolPermission.ALWAYS
        finally:
            await agent.aclose()

    async def test_reload_preserves_runtime_grants(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            old_manager = agent.tool_manager
            await agent.reload_with_initial_messages()
            # The reload rebuilds the manager around the same runtime store.
            assert agent.tool_manager is not old_manager
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            tool = agent.tool_manager.get("bash")
            decision = tool.resolve_permission(
                tool.validate_arguments({"command": "git push"})
            )
            assert decision is not None
            assert decision.permission == ToolPermission.ALWAYS
        finally:
            await agent.aclose()

    async def test_resume_does_not_restore_runtime_grants(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) != []
            session_id = agent.session_id
        finally:
            await agent.aclose()
        # A resumed conversation is a new loop: the runtime store starts empty.
        resumed = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args("git push"), "resume")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            user_input_capability=True,
            session_id=session_id,
        )
        try:
            assert _authorities(resumed) == []
            events, requests = await _drive(resumed, "Deny")
            assert len(requests) == 1
            assert _results(events)[0].skipped
            assert _authorities(resumed) == []
        finally:
            await resumed.aclose()

    async def test_resume_rebind_clears_runtime_grants(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resume reuses the loop in place; session grants do not survive it."""
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "resume")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_logged_config(tmp_path / "logs"),
            cwd=project,
            backend=backend,
            user_input_capability=True,
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            # Refusal memory and a pending once permit are session-scoped too.
            refused = DangerousCommandCallIdentity(
                scope=scope_of(project), command=("git", "push", "--force")
            )
            agent._dangerous_command_grants.record_refusal(refused)
            agent._dangerous_command_grants.install_once(
                frozenset({("git.push", scope_of(project))})
            )
            # The runtime reuses this loop in place on resume.
            logger = agent.session_logger
            assert logger.session_dir is not None
            assert logger.session_metadata is not None
            agent.rebind_to_session(
                agent.session_id,
                logger.session_dir,
                list(agent.messages),
                session_metadata=logger.session_metadata,
                prepared_scratchpad=None,
            )
            assert _authorities(agent) == []
            assert not agent._dangerous_command_grants.is_refused(refused)
            # The same call re-prompts instead of resuming on stale authority.
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        finally:
            await agent.aclose()

    async def test_clear_history_clears_runtime_grants(self, tmp_path: Path) -> None:
        """A history clear starts a new conversation; session grants expire."""
        project = make_repo(tmp_path / "project")
        backend = FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "grant")]
                )
            ],
            [mock_llm_chunk(content="done")],
            [
                mock_llm_chunk(
                    tool_calls=[_call("bash", _bash_args("git push"), "again")]
                )
            ],
            [mock_llm_chunk(content="done")],
        ])
        agent = build_test_agent_loop(
            config=_config(), cwd=project, backend=backend, user_input_capability=True
        )
        try:
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
            refused = DangerousCommandCallIdentity(
                scope=scope_of(project), command=("git", "push", "--force")
            )
            agent._dangerous_command_grants.record_refusal(refused)
            agent._dangerous_command_grants.install_once(
                frozenset({("git.push", scope_of(project))})
            )
            await agent.clear_history()
            assert _authorities(agent) == []
            assert not agent._dangerous_command_grants.is_refused(refused)
            # The cached authority the grant fed is dropped too: the push is
            # denied again and proposes a fresh approval.
            tool = agent.tool_manager.get("bash")
            decision = tool.resolve_permission(
                tool.validate_arguments({"command": "git push"})
            )
            assert decision is not None
            assert decision.permission == ToolPermission.NEVER
            assert decision.dangerous_command_proposal is not None
            events, requests = await _drive(agent, "Allow this session")
            assert len(requests) == 1
            assert not _results(events)[0].skipped
            assert _authorities(agent) == [(frozenset({"git.push"}), scope_of(project))]
        finally:
            await agent.aclose()


class TestChildSessions:
    """A child session never prompts on its own; parent gaps are ceilings."""

    async def test_bare_child_never_prompts(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        parent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args("git push"), "child")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            is_subagent=True,
            inherited_workspace=Workspace.for_session(project),
            user_input_capability=True,
        )
        try:
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            assert results[0].skip_reason is not None
            assert "requires approval for a dangerous git operation" in (
                results[0].skip_reason
            )
            assert _authorities(child) == []
        finally:
            await child.aclose()
            await parent.aclose()

    async def test_parent_dangerous_gap_is_a_ceiling_for_the_child(
        self, tmp_path: Path
    ) -> None:
        project = make_repo(tmp_path / "project")
        parent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[_call("bash", _bash_args("git push"), "child")]
                    )
                ],
                [mock_llm_chunk(content="done")],
            ]),
            is_subagent=True,
            inherited_workspace=Workspace.for_session(project),
            parent_authority_getter=lambda: parent.tool_manager,
            user_input_capability=True,
        )
        try:
            events, requests = await _drive(child, "Allow this session")
            assert requests == []
            results = _results(events)
            assert len(results) == 1
            assert results[0].skipped
            # The parent's dangerous gap is a ceiling: no local grant, no prompt.
            assert _authorities(child) == []
            assert _authorities(parent) == []
        finally:
            await child.aclose()
            await parent.aclose()


class TestPromptFailureModes:
    """Cancellation, failed prompts, and closed batches grant nothing."""

    async def _bound_tool_and_args(
        self, agent: AgentLoop, project: Path
    ) -> tuple[Any, Any]:
        tool = agent.tool_manager.get("bash")
        args = tool.validate_arguments({"command": "git push"})
        return tool, args

    async def test_cancellation_during_prompt_grants_nothing(
        self, tmp_path: Path
    ) -> None:
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        try:
            tool, args = await self._bound_tool_and_args(agent, project)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "cancel-me")
            )
            event = await asyncio.wait_for(queue.get(), 5)
            assert isinstance(event, UserInputRequestEvent)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert _authorities(agent) == []
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_failed_prompt_denies_without_grant(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        queue: asyncio.Queue = asyncio.Queue()
        agent._request_broker.bind(queue)
        try:
            tool, args = await self._bound_tool_and_args(agent, project)
            task = asyncio.create_task(
                agent._should_execute_tool(tool, args, "reject-me")
            )
            event = await asyncio.wait_for(queue.get(), 5)
            assert isinstance(event, UserInputRequestEvent)
            agent.reject_request(event.request_id, ToolError("UI gone"))
            decision = await asyncio.wait_for(task, 5)
            assert decision.verdict.name == "SKIP"
            assert decision.feedback is not None
            assert "dangerous_command_grants_by_project" in decision.feedback
            assert _authorities(agent) == []
        finally:
            agent._request_broker.unbind(queue)
            await agent.aclose()

    async def test_closed_batch_denies_without_grant(self, tmp_path: Path) -> None:
        project = make_repo(tmp_path / "project")
        agent = build_test_agent_loop(
            config=_config(),
            cwd=project,
            backend=FakeBackend(),
            user_input_capability=True,
        )
        try:
            tool, args = await self._bound_tool_and_args(agent, project)
            # The broker is unbound: the batch that would carry the prompt is gone.
            decision = await agent._should_execute_tool(tool, args, "closed")
            assert decision.verdict.name == "SKIP"
            assert decision.feedback is not None
            assert "dangerous_command_grants_by_project" in decision.feedback
            assert _authorities(agent) == []
        finally:
            await agent.aclose()
