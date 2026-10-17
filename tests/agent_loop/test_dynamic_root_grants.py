"""Mid-turn session root grants: the feature-specific remainder.

The shared grant-lifecycle twins migrated to
``tests/agent_loop/_grant_lifecycle_harness.py`` in stage 15a-ii:
allow-session resume, deny/refusal memory, the cancelled prompt, the prompt
shape, the Always-for-this-project persistence family, compaction survival,
the noninteractive and bare-loop denials, child bounding, and concurrent-call
coalescing. What stays here is root-specific: the sensitive-path
short-circuit, root proposal targets, the broad-root warning, the cached
workspace check, waiter cancellation, and the cancelled-save propagation.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.config._root_persistence import SavedRootsRead
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.base import ToolPermission
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio

VISION_MODEL = ModelConfig(
    name="vision", provider="mistral", alias="vision", supports_images=True
)


def _config() -> ChartreuxConfigSchema:
    return build_test_vibe_config(active_model="vision", models=[VISION_MODEL])


def _call(name: str, args: dict[str, str], call_id: str, index: int = 0) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=index,
        function=FunctionCall(name=name, arguments=json.dumps(args)),
    )


async def _drive(
    agent: AgentLoop, answer: str | None = None, *, cancelled: bool = False
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one turn, answering every root-grant prompt with *answer*."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act("Exercise root grants"):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                result = (
                    UserQuestionResult(answers=[], cancelled=True)
                    if cancelled
                    else UserQuestionResult(
                        answers=[UserAnswer(question="grant", answer=answer or "")]
                    )
                )
                agent.resolve_user_input_request(event.request_id, result)
            events.append(event)
    return events, requests


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


class _StubSavePort:
    """Runtime port stub: applies grants locally and records save attempts."""

    def __init__(self, loop: AgentLoop, result: ConfigSaveResult) -> None:
        self._loop = loop
        self._result = result
        self.saves: list[tuple[str, Path, str]] = []

    async def grant_root(self, session_id: str, root: Path) -> None:
        assert session_id == self._loop.session_id
        self._loop.apply_root_grant(root)

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        assert session_id == self._loop.session_id
        self.saves.append((session_id, root, expected_revision))
        return self._result


class _CancellingSavePort(_StubSavePort):
    def __init__(self, loop: AgentLoop) -> None:
        super().__init__(loop, ConfigSaveResult("user", "saved", "unchanged", "rev"))

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        self.saves.append((session_id, root, expected_revision))
        raise asyncio.CancelledError


def _patch_saved_roots(
    monkeypatch: pytest.MonkeyPatch, revision: str = "user-rev-1"
) -> AsyncMock:
    """Point the loop's saved-roots read surface at a controlled revision."""
    read = AsyncMock(return_value=SavedRootsRead(roots=(), user_revision=revision))
    monkeypatch.setattr("chartreux.core.agent_loop._loop.read_saved_roots", read)
    return read


async def test_sensitive_denial_never_prompts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / ".env"
    target.write_text("SECRET=1")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "env")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert "Sensitive" in results[0].skip_reason
        assert agent._session_root_grants.roots == set()
    finally:
        await agent.aclose()


async def test_one_waiters_cancellation_does_not_cancel_the_shared_decision(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    queue: asyncio.Queue = asyncio.Queue()
    agent._request_broker.bind(queue)
    try:
        root = outside.resolve()
        first = asyncio.create_task(agent._prompt_session_root_grant(root, None))
        event = await asyncio.wait_for(queue.get(), 5)
        assert isinstance(event, UserInputRequestEvent)
        second = asyncio.create_task(agent._prompt_session_root_grant(root, None))
        await asyncio.sleep(0)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        agent.resolve_user_input_request(
            event.request_id,
            UserQuestionResult(
                answers=[UserAnswer(question="grant", answer="Allow this session")]
            ),
        )
        assert await asyncio.wait_for(second, 5) == "session"
        assert agent._session_root_grants.pending == {}
        assert queue.empty()
    finally:
        agent._request_broker.unbind(queue)
        await agent.aclose()


async def test_cached_workspace_sees_the_grant_without_manager_rebuild(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "grant")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        tool = agent.tool_manager.get("read_file")
        args = tool.validate_arguments({"file_path": str(target)})
        decision = tool.resolve_permission(args)
        assert decision is not None and decision.permission == ToolPermission.NEVER
        cached_workspace = agent.tool_manager.workspace
        assert not cached_workspace.allows(target)
        manager = agent.tool_manager
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert agent.tool_manager is manager
        assert agent.tool_manager.workspace.allows(target)
        assert agent.tool_manager.workspace is not cached_workspace
        decision = tool.resolve_permission(args)
        assert decision is not None and decision.permission == ToolPermission.ALWAYS
        assert not _results(events)[0].skipped
    finally:
        await agent.aclose()


async def test_directory_target_proposes_the_directory_itself(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    nested = outside / "nested"
    nested.mkdir()
    (nested / "hits.txt").write_text("granted needle\n")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "grep", {"pattern": "needle", "path": str(nested)}, "grep-dir"
                    )
                ]
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
        assert question.question.startswith(f"Grant {nested.resolve()}")
        assert agent._session_root_grants.roots == {nested.resolve()}
        assert not _results(events)[0].skipped
    finally:
        await agent.aclose()


async def test_file_target_proposes_the_nearest_existing_ancestor(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    deep = outside / "deep"
    deep.mkdir()
    target = deep / "created.txt"
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "write_file",
                        {"file_path": str(target), "content": "granted write"},
                        "write-deep",
                    )
                ]
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
        assert str(deep.resolve()) in request.questions[0].question
        assert agent._session_root_grants.roots == {deep.resolve()}
        assert target.read_text() == "granted write"
    finally:
        await agent.aclose()


async def test_broad_root_prompt_carries_the_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    project = tmp_path / "project"
    project.mkdir()
    target = home / "note.txt"
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "write_file",
                        {"file_path": str(target), "content": "granted write"},
                        "write-home",
                    )
                ]
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
        assert "home directory or wider" in question.question
        assert agent._session_root_grants.roots == {home.resolve()}
    finally:
        await agent.aclose()


async def test_a_cancelled_save_propagates_with_the_grant_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    port = _CancellingSavePort(agent)
    agent.bind_root_grant_port(port)
    _patch_saved_roots(monkeypatch)
    try:
        # The grant is applied before the save, but a cancelled save cannot
        # resume the tool call: the cancellation propagates.
        with pytest.raises(asyncio.CancelledError):
            await agent._persist_project_root_grant(outside.resolve())
        assert len(port.saves) == 1
    finally:
        await agent.aclose()
