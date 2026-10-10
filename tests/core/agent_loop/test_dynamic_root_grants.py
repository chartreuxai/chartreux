"""Mid-turn session root grants: prompt, coalesce, apply, and deny-memory."""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema, ModelConfig
from chartreux.core.config._root_persistence import SavedRootsRead
from chartreux.core.config.types import ConfigSaveResult
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.tools.base import ToolPermission
from chartreux.core.workspace import Workspace
from chartreux.questions import UserAnswer, UserQuestionRequest, UserQuestionResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABAQAAAAA3bvkkAAAACklEQVQI12NoAAAAggCB3UNq9"
    "AAAAABJRU5ErkJggg=="
)

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


def _prepare_tool(tool: str, outside: Path) -> tuple[Path, dict[str, str]]:
    """Create the out-of-root target and tool arguments for one file tool."""
    if tool == "read_file":
        target = outside / "note.txt"
        target.write_text("granted content")
        return target, {"file_path": str(target)}
    if tool == "write_file":
        target = outside / "created.txt"
        return target, {"file_path": str(target), "content": "granted write"}
    if tool == "edit":
        target = outside / "edit.txt"
        target.write_text("old content")
        return target, {
            "file_path": str(target),
            "old_string": "old content",
            "new_string": "new content",
        }
    if tool == "grep":
        target = outside / "hits.txt"
        target.write_text("granted needle\n")
        return target, {"pattern": "needle", "path": str(target)}
    if tool == "read_image":
        target = outside / "pixel.png"
        target.write_bytes(PNG_BYTES)
        return target, {"file_path": str(target)}
    raise AssertionError(f"unknown tool {tool}")


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


@pytest.mark.parametrize(
    "tool", ["read_file", "write_file", "edit", "grep", "read_image"]
)
async def test_allow_session_grant_resumes_the_tool_call(
    tmp_path: Path, tool: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target, args = _prepare_tool(tool, outside)
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, args, "grant-me")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert not results[0].skipped
        assert results[0].result is not None
        assert agent._session_root_grants.roots == {outside.resolve()}
        assert agent._session_root_grants.denied_roots == set()
        assert agent._session_root_grants.pending == {}
        if tool == "write_file":
            assert target.read_text() == "granted write"
        elif tool == "edit":
            assert target.read_text() == "new content"
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "tool", ["read_file", "write_file", "edit", "grep", "read_image"]
)
async def test_deny_skips_the_tool_call(tmp_path: Path, tool: str) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target, args = _prepare_tool(tool, outside)
    before = target.read_bytes() if target.exists() else None
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call(tool, args, "deny-me")])],
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
        assert results[0].skip_reason == "user declined; do not retry this path"
        assert results[0].result is None
        assert agent._session_root_grants.roots == set()
        assert agent._session_root_grants.denied_roots == {outside.resolve()}
        if before is None:
            assert not target.exists()
        else:
            assert target.read_bytes() == before
    finally:
        await agent.aclose()


async def test_cancelled_prompt_denies_the_tool_call(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "esc")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, cancelled=True)
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == "user declined; do not retry this path"
        assert agent._session_root_grants.denied_roots == {outside.resolve()}
    finally:
        await agent.aclose()


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


async def test_concurrent_calls_coalesce_one_prompt_and_one_grant(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    first = outside / "first.txt"
    first.write_text("first granted content")
    second = outside / "second.txt"
    second.write_text("second granted content")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    _call("read_file", {"file_path": str(first)}, "read-first", 0),
                    _call("read_file", {"file_path": str(second)}, "read-second", 1),
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
        results = _results(events)
        assert len(results) == 2
        assert all(
            not result.skipped and result.result is not None for result in results
        )
        assert agent._session_root_grants.roots == {outside.resolve()}
        assert agent._session_root_grants.pending == {}
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


async def test_denied_root_fails_immediately_on_retry(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    args = {"file_path": str(target)}
    backend = FakeBackend([
        [mock_llm_chunk(tool_calls=[_call("read_file", args, "first")])],
        [mock_llm_chunk(content="denied once")],
        [mock_llm_chunk(tool_calls=[_call("read_file", args, "retry")])],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Deny")
        assert len(requests) == 1
        assert _results(events)[0].skipped
        events, requests = await _drive(agent)
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason == "user declined; do not retry this path"
        assert agent._session_root_grants.roots == set()
    finally:
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


async def test_grant_survives_compaction(tmp_path: Path) -> None:
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
        [mock_llm_chunk(content="<summary>summary</summary>")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    try:
        events, requests = await _drive(agent, "Allow this session")
        assert len(requests) == 1
        assert agent._session_root_grants.roots == {outside.resolve()}
        summary = await agent.compact()
        assert summary == "summary"
        assert agent._session_root_grants.roots == {outside.resolve()}
        tool = agent.tool_manager.get("read_file")
        args = tool.validate_arguments({"file_path": str(target)})
        decision = tool.resolve_permission(args)
        assert decision is not None and decision.permission == ToolPermission.ALWAYS
        assert agent.tool_manager.workspace.allows(target)
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


async def test_prompt_shape_states_access_and_options(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "shape")]
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
        assert len(question.header) <= 20
        assert question.hide_other is True
        assert [option.label for option in question.options] == [
            "Allow this session",
            "Always for this project",
            "Deny",
        ]
        assert str(outside.resolve()) in question.question
        assert "reading and writing" in question.question
        assert "shell" in question.question
        assert request.footer_note
        assert "home directory" not in question.question
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


async def test_always_applies_the_grant_and_reports_the_saved_outcome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                tool_calls=[_call("read_file", {"file_path": str(target)}, "always")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    port = _StubSavePort(agent, ConfigSaveResult("user", "saved", "unchanged", "rev-2"))
    agent.bind_root_grant_port(port)
    read = _patch_saved_roots(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        results = _results(events)
        assert len(results) == 1
        assert not results[0].skipped
        # The session grant applies exactly like Allow-session does.
        assert agent._session_root_grants.roots == {outside.resolve()}
        assert agent.tool_manager.workspace.allows(target)
        # The user revision is fetched from the read surface at decision time.
        assert read.call_count == 1
        assert read.call_args.kwargs["project"] == project.resolve()
        assert port.saves == [(agent.session_id, outside.resolve(), "user-rev-1")]
        tool_messages = [
            message.content or ""
            for message in agent.messages
            if message.role == Role.tool and message.content
        ]
        assert any(
            "applies to this session" in text
            and "saved in your user config" in text
            and str(project.resolve()) in text
            and str(outside.resolve()) in text
            for text in tool_messages
        )
    finally:
        await agent.aclose()


async def test_always_reports_a_conflict_without_retrying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                tool_calls=[_call("read_file", {"file_path": str(target)}, "conflict")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    port = _StubSavePort(
        agent, ConfigSaveResult("user", "not_saved", "unchanged", error="conflict")
    )
    agent.bind_root_grant_port(port)
    _patch_saved_roots(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        # No automatic retry and no automatic re-prompt.
        assert len(requests) == 1
        assert len(port.saves) == 1
        assert not _results(events)[0].skipped
        # The session grant is still applied; the save is reported session-only.
        assert agent._session_root_grants.roots == {outside.resolve()}
        tool_messages = [
            message.content or ""
            for message in agent.messages
            if message.role == Role.tool and message.content
        ]
        assert any(
            "applies to this session only" in text and "retry" in text
            for text in tool_messages
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
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: ConfigSaveResult,
    expected: str,
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
                tool_calls=[_call("read_file", {"file_path": str(target)}, "failure")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    port = _StubSavePort(agent, result)
    agent.bind_root_grant_port(port)
    _patch_saved_roots(monkeypatch)
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        # The session-only authorization still applies.
        assert agent._session_root_grants.roots == {outside.resolve()}
        tool_messages = [
            message.content or ""
            for message in agent.messages
            if message.role == Role.tool and message.content
        ]
        assert any(
            expected in text and "applies to this session" in text
            for text in tool_messages
        )
    finally:
        await agent.aclose()


async def test_always_without_a_save_transport_reports_it(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "bare")]
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
        assert agent._session_root_grants.roots == {outside.resolve()}
        tool_messages = [
            message.content or ""
            for message in agent.messages
            if message.role == Role.tool and message.content
        ]
        assert any("cannot save roots" in text for text in tool_messages)
    finally:
        await agent.aclose()


async def test_always_fetches_the_user_revision_at_decision_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    first_root = tmp_path / "first-outside"
    first_root.mkdir()
    second_root = tmp_path / "second-outside"
    second_root.mkdir()
    first = first_root / "note.txt"
    first.write_text("first content")
    second = second_root / "note.txt"
    second.write_text("second content")
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(first)}, "one")]
            )
        ],
        [mock_llm_chunk(content="done")],
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(second)}, "two")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    port = _StubSavePort(agent, ConfigSaveResult("user", "saved", "unchanged", "rev"))
    agent.bind_root_grant_port(port)
    read = _patch_saved_roots(monkeypatch)
    try:
        await _drive(agent, "Always for this project")
        await _drive(agent, "Always for this project")
        # The revision is read per decision and never cached across turns.
        assert read.call_count == 2
        assert [save[2] for save in port.saves] == ["user-rev-1", "user-rev-1"]
        assert agent._session_root_grants.roots == {
            first_root.resolve(),
            second_root.resolve(),
        }
    finally:
        await agent.aclose()


async def test_always_without_a_readable_saved_surface_skips_the_save(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
                tool_calls=[
                    _call("read_file", {"file_path": str(target)}, "unreadable")
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=True
    )
    port = _StubSavePort(agent, ConfigSaveResult("user", "saved", "unchanged", "rev"))
    agent.bind_root_grant_port(port)
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.read_saved_roots",
        AsyncMock(return_value=SavedRootsRead(unavailable="no_user_source")),
    )
    try:
        events, requests = await _drive(agent, "Always for this project")
        assert len(requests) == 1
        assert not _results(events)[0].skipped
        assert agent._session_root_grants.roots == {outside.resolve()}
        assert port.saves == []
        tool_messages = [
            message.content or ""
            for message in agent.messages
            if message.role == Role.tool and message.content
        ]
        assert any("could not be read" in text for text in tool_messages)
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


class _StubRootGrantPort:
    """Runtime marker only; a no-capability session never calls it."""

    async def grant_root(self, session_id: str, root: Path) -> None:
        raise AssertionError("A session without user input must not grant roots")

    async def save_root(
        self, session_id: str, root: Path, expected_revision: str
    ) -> ConfigSaveResult:
        raise AssertionError("A session without user input must not save roots")


async def test_without_user_input_capability_a_runtime_session_denies_actionably(
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
                tool_calls=[_call("read_file", {"file_path": str(target)}, "no-cap")]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    agent = build_test_agent_loop(
        config=_config(), cwd=project, backend=backend, user_input_capability=False
    )
    try:
        agent.bind_root_grant_port(_StubRootGrantPort())
        events, requests = await _drive(agent, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert "authorized_roots_by_project" in results[0].skip_reason
        assert "config.toml" in results[0].skip_reason
        assert agent._session_root_grants.roots == set()
    finally:
        await agent.aclose()


async def test_bare_loop_without_a_runtime_port_keeps_the_existing_denial_text(
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
                tool_calls=[_call("read_file", {"file_path": str(target)}, "bare")]
            )
        ],
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
        assert results[0].skip_reason == (
            "File read is outside authorized workspace and session scratch roots "
            "and is not an instruction file injected into this agent's context. "
            "Other paths require an explicit user scope change."
        )
        assert agent._session_root_grants.roots == set()
    finally:
        await agent.aclose()


async def test_child_sessions_are_bounded_and_never_prompt(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    child = AgentLoop(
        config_orchestrator=parent.config_orchestrator.copy(),
        cwd=project,
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    tool_calls=[
                        _call("read_file", {"file_path": str(target)}, "child-call")
                    ]
                )
            ],
            [mock_llm_chunk(content="done")],
        ]),
        is_subagent=True,
        inherited_workspace=Workspace.for_session(project, authorized_roots=[outside]),
        user_input_capability=True,
    )
    try:
        events, requests = await _drive(child, "Allow this session")
        assert requests == []
        results = _results(events)
        assert len(results) == 1
        assert results[0].skipped
        assert results[0].skip_reason is not None
        assert "outside authorized workspace" in results[0].skip_reason
        assert child._session_root_grants.roots == set()
    finally:
        await child.aclose()
        await parent.aclose()
