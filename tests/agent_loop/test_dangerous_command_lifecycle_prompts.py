"""Dangerous-command lifecycle — prompts concern (stage 20c split target).

Prompt content and shape, and prompt failure modes: a model-generated
question installs no authority, compound and multi-repo calls are fully
displayed in the approval question, and a cancelled, failed, or closed-batch
prompt denies without a grant. Moved here from
``test_dangerous_command_lifecycle.py`` in the 20c-ii-2 concern commit,
together with the class-local ``TestPromptFailureModes._bound_tool_and_args``
helper. The module re-registers the marks the population had in the
original — ``pytestmark = pytest.mark.asyncio``; the original file defines
no module-local fixtures, so there is no fixture registration to carry.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.events import UserInputRequestEvent
from chartreux.core.tools.base import ToolError
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from tests.agent_loop._lifecycle_test_support import (
    DECLINED_FEEDBACK,
    _authorities,
    _bash_args,
    _call,
    _config,
    _drive,
    _results,
    make_repo,
    scope_of,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


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
