"""Child sessions inherit and originate session root grants end to end."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

from git import Repo
import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory, SessionRootGrantPort
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.session import AppServerSession
from chartreux.core.agent_loop import AgentLoop, AgentRuntimePolicy
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.events import BaseEvent, ToolResultEvent, UserInputRequestEvent
from chartreux.core.git.worktree import WorktreeRepository
from chartreux.core.llm_models import FunctionCall, LLMChunk, Role, ToolCall
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext, ToolPermission
from chartreux.core.trusted_folders import trusted_folders_manager
from chartreux.questions import (
    UserAnswer as QuestionAnswer,
    UserQuestionResult as QuestionResult,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


def _config() -> ChartreuxConfigSchema:
    tool_names = ["task", "read_file"]
    return build_test_vibe_config(
        enabled_tools=tool_names,
        tools={
            name: {"permission": ToolPermission.ALWAYS.value} for name in tool_names
        },
        session_logging=SessionLoggingConfig(enabled=False),
    )


def _read_call(target: Path, call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="read_file", arguments=json.dumps({"file_path": str(target)})
        ),
    )


def _read_turn(target: Path, call_id: str) -> list[list[LLMChunk]]:
    return [
        [mock_llm_chunk(tool_calls=[_read_call(target, call_id)])],
        [mock_llm_chunk(content="done")],
    ]


def _task_call(
    *, background: bool, agent_id: str | None = None, tool_call_id: str = "task-1"
) -> ToolCall:
    arguments: dict[str, object] = {
        "task": "Read the granted note",
        "agent_type": "worker",
        "background": background,
    }
    if agent_id is not None:
        arguments["agent_id"] = agent_id
    return ToolCall(
        id=tool_call_id,
        index=0,
        function=FunctionCall(name="task", arguments=json.dumps(arguments)),
    )


def _task_turn(
    *, background: bool, agent_id: str | None = None, tool_call_id: str = "task-1"
) -> list[list[LLMChunk]]:
    return [
        [
            mock_llm_chunk(
                tool_calls=[
                    _task_call(
                        background=background,
                        agent_id=agent_id,
                        tool_call_id=tool_call_id,
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="root done")],
    ]


def _streams(*turns: list[list[LLMChunk]]) -> list[list[LLMChunk]]:
    """Flatten scripted turns into one FakeBackend stream sequence."""
    return [stream for turn in turns for stream in turn]


async def _drive(session: AppServerSession, answer: str | None = None) -> list[str]:
    """Run one turn, answering every root-grant callback with *answer*."""
    callbacks: list[str] = []
    async for event in session.act("Exercise child grants"):
        if isinstance(event, CallbackRequested):
            callbacks.append(event.callback.callback_id)
            if answer is not None:
                await session.respond_to_callback(
                    event.callback.callback_id,
                    UserInputCallbackOutput(
                        result=UserQuestionResult(
                            answers=[UserAnswer(question="grant", answer=answer)]
                        )
                    ),
                )
    return callbacks


async def _drive_loop(
    agent: AgentLoop, answer: str | None = None
) -> tuple[list[BaseEvent], list[UserInputRequestEvent]]:
    """Run one child turn directly, answering every root-grant prompt."""
    events: list[BaseEvent] = []
    requests: list[UserInputRequestEvent] = []
    async with asyncio.timeout(10):
        async for event in agent.act("Exercise child grants"):
            if isinstance(event, UserInputRequestEvent):
                requests.append(event)
                agent.resolve_user_input_request(
                    event.request_id,
                    QuestionResult(
                        answers=[QuestionAnswer(question="grant", answer=answer or "")]
                    ),
                )
            events.append(event)
    return events, requests


def _results(events: list[BaseEvent]) -> list[ToolResultEvent]:
    return [event for event in events if isinstance(event, ToolResultEvent)]


def _tool_texts(loop: AgentLoop) -> list[str]:
    return [
        message.content or "" for message in loop.messages if message.role == Role.tool
    ]


def _capture_registries(
    monkeypatch: pytest.MonkeyPatch,
) -> list[SessionRuntimeRegistry]:
    registries: list[SessionRuntimeRegistry] = []
    bind = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: Any) -> None:
        bind(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    return registries


def _capture_children(monkeypatch: pytest.MonkeyPatch) -> list[AgentLoop]:
    children: list[AgentLoop] = []
    create_child = AgentRuntimeFactory.create_child

    async def capture(
        self: AgentRuntimeFactory, parent: AgentLoop, candidate: Any, **kwargs: Any
    ) -> AgentLoop:
        child = await create_child(self, parent, candidate, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", capture)
    return children


def _use_child_backend(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend) -> None:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )


class _GatedChildBackend(FakeBackend):
    """Block each run's final completion until the test releases it.

    A background run's completion notifies the parent with a fresh turn, so
    the test releases each run only once the parent's scripted turns and the
    notification turns that follow them are accounted for.
    """

    def __init__(self, runs: list[list[list[LLMChunk]]]) -> None:
        super().__init__([stream for run in runs for stream in run])
        self.releases = [asyncio.Event() for _ in runs]

    async def complete(self, **kwargs: Any) -> Any:
        call = len(self._requests_messages)
        if call % 2 == 1:
            await self.releases[call // 2].wait()
        return await super().complete(**kwargs)


def _child_loop(
    parent: AgentLoop,
    policy: AgentRuntimePolicy,
    backend: FakeBackend,
    *,
    cwd: Path | None = None,
) -> AgentLoop:
    """Build a child the way the runtime factory does, from a captured policy."""
    return AgentLoop(
        config_orchestrator=parent.config_orchestrator.copy(),
        cwd=cwd or parent.cwd,
        backend=backend,
        is_subagent=True,
        inherited_workspace=policy.inherited_workspace,
        inherited_restrictions=policy.inherited_restrictions,
        inherited_mode_restrictions=policy.inherited_mode_restrictions,
        inherited_plan_write_scopes=policy.inherited_plan_write_scopes,
        inherited_root_grants=policy.inherited_root_grants,
        user_input_capability=policy.user_input_capability,
        parent_authority_getter=policy.parent_authority_getter,
        parent_authority_revision_getter=policy.parent_authority_revision_getter,
    )


async def test_child_dispatched_after_a_root_grant_inherits_it(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    registries = _capture_registries(monkeypatch)
    children = _capture_children(monkeypatch)
    child_backend = FakeBackend([*_read_turn(target, "child-read")])
    _use_child_backend(monkeypatch, child_backend)
    parent_backend = FakeBackend([
        *_read_turn(target, "root-read"),
        *_task_turn(background=False),
    ])
    parent = build_test_agent_loop(
        config=_config(),
        cwd=project,
        backend=parent_backend,
        user_input_capability=True,
    )
    session = await create_test_app_server_session(parent)
    try:
        # Turn 1: the root prompts and grants the outside root.
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        assert registries
        registry = registries[0]
        root = registry._root
        assert root is not None
        root_loop = root.agent_loop
        assert root_loop._session_root_grants.roots == {outside.resolve()}
        assert any("granted content" in text for text in _tool_texts(root_loop))
        # Turn 2: a child dispatched after the grant inherits it.
        callbacks = await _drive(session, "Allow this session")
        assert callbacks == []
        assert len(children) == 1
        child = children[0]
        assert child._session_root_grants.roots == {outside.resolve()}
        assert child.tool_manager.workspace.allows(target)
        assert any("granted content" in text for text in _tool_texts(child))
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await parent.aclose()


async def _wait_for_root_idle(registry: SessionRuntimeRegistry) -> None:
    """Wait out a background run's automatic parent notification turn."""
    root = registry._root
    assert root is not None
    for _ in range(500):
        if (
            root.turns.active_turn is None
            and root.turns._active_task is None
            and not root.turns.queue_state.items
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Root session did not settle")


async def _background_result(
    registry: SessionRuntimeRegistry, args: TaskArgs, ctx: InvokeContext
) -> Any:
    return [result async for result in registry.run(args, ctx)][-1]


async def test_child_dispatched_before_a_root_grant_stays_frozen_across_retask(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    config_path = config_dir / "config.toml"
    disk = config_path.read_bytes()
    registries = _capture_registries(monkeypatch)
    children = _capture_children(monkeypatch)
    child_backend = _GatedChildBackend([
        _read_turn(target, "child-run-one"),
        _read_turn(target, "child-run-two"),
    ])
    _use_child_backend(monkeypatch, child_backend)
    parent_backend = FakeBackend(
        _streams(
            _task_turn(background=True, tool_call_id="task-launch"),
            _read_turn(target, "root-read"),
            [[mock_llm_chunk(content="noted")]],
            [[mock_llm_chunk(content="noted")]],
        )
    )
    parent = build_test_agent_loop(
        config=_config(),
        cwd=project,
        backend=parent_backend,
        user_input_capability=True,
    )
    session = await create_test_app_server_session(parent)
    try:
        registry = registries[0]
        # Turn 1: the child is dispatched before any grant and is denied.
        callbacks = await _drive(session, "Allow this session")
        assert callbacks == []
        assert len(children) == 1
        child = children[0]
        records = list(registry._agent_records.values())
        assert len(records) == 1
        record = records[0]
        assert child._session_root_grants.roots == set()
        # Turn 2: the root prompts and grants the outside root.
        callbacks = await _drive(session, "Allow this session")
        assert len(callbacks) == 1
        root = registry._root
        assert root is not None
        assert root.agent_loop._session_root_grants.roots == {outside.resolve()}
        # Let the first run finish; its notification turn is scripted above.
        child_backend.releases[0].set()
        await registry.wait_for_agent(record.agent_id)
        await _wait_for_root_idle(registry)
        assert len(_tool_texts(child)) == 1
        assert "outside authorized workspace" in _tool_texts(child)[0]
        assert child._session_root_grants.roots == set()
        # Retask the retained child through the same runner the task tool uses.
        ctx = InvokeContext(tool_call_id="task-retask", session_id=parent.session_id)
        launch = await _background_result(
            registry,
            TaskArgs(
                task="Read the granted note",
                agent_type="worker",
                background=True,
                agent_id=record.agent_id,
            ),
            ctx,
        )
        assert launch.agent_id == record.agent_id
        assert len(children) == 1
        child_backend.releases[1].set()
        await registry.wait_for_agent(record.agent_id)
        await _wait_for_root_idle(registry)
        assert child._session_root_grants.roots == set()
        assert not child.tool_manager.workspace.allows(target)
        texts = _tool_texts(child)
        assert len(texts) == 2
        assert all("outside authorized workspace" in text for text in texts)
        assert config_path.read_bytes() == disk
    finally:
        await session.close()
        await parent.aclose()


async def test_child_local_grant_applies_only_to_that_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    nested = project / "nested"
    nested.mkdir()
    target = project / "note.txt"
    target.write_text("parent project content")
    parent = build_test_agent_loop(
        config=_config(), cwd=project, backend=FakeBackend(), user_input_capability=True
    )
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    try:
        registry.bind_root(registry._build_child_runtime(parent))
        policy = parent.child_runtime_policy
        assert policy.inherited_root_grants == ()
        child = _child_loop(
            parent, policy, FakeBackend([*_read_turn(target, "child-read")]), cwd=nested
        )
        sibling = _child_loop(
            parent,
            policy,
            FakeBackend([*_read_turn(target, "sibling-read")]),
            cwd=nested,
        )
        try:
            registry._children[child.session_id] = registry._build_child_runtime(child)
            registry._children[sibling.session_id] = registry._build_child_runtime(
                sibling
            )
            assert isinstance(child._root_grant_port, SessionRootGrantPort)
            assert isinstance(sibling._root_grant_port, SessionRootGrantPort)
            # The child's own roots stop at its cwd; the frozen ceiling (the
            # parent's workspace) permits the parent project directory.
            assert not child.tool_manager.workspace.allows(target)
            events, requests = await _drive_loop(child, "Allow this session")
            assert len(requests) == 1
            results = _results(events)
            assert len(results) == 1
            assert not results[0].skipped
            assert results[0].result is not None
            assert "parent project content" in results[0].result.model_dump_json()
            assert child._session_root_grants.roots == {project.resolve()}
            # The grant is child-local: parent and sibling stores stay empty.
            assert parent._session_root_grants.roots == set()
            assert sibling._session_root_grants.roots == set()
            assert not sibling.tool_manager.workspace.allows(target)
            # A sibling asks for itself; the parent still receives nothing.
            events, requests = await _drive_loop(sibling, "Deny")
            assert len(requests) == 1
            assert _results(events)[0].skipped
            assert sibling._session_root_grants.roots == set()
            assert sibling._session_root_grants.denied_roots == {project.resolve()}
            assert parent._session_root_grants.roots == set()
        finally:
            await registry.close()
    finally:
        await parent.aclose()


def _init_repo(root: Path) -> None:
    repo = Repo.init(root, initial_branch="main")
    repo.config_writer().set_value("user", "name", "Tester").release()
    repo.config_writer().set_value("user", "email", "t@example.com").release()
    (root / "file.txt").write_text("hello\n")
    repo.index.add(["file.txt"])
    repo.index.commit("initial")


async def test_grants_survive_relocate_as_bare_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _init_repo(repo)
    with WorktreeRepository.open(repo) as repository:
        worktree = repository.prepare("feature").path.resolve()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "note.txt"
    target.write_text("granted content")
    children = _capture_children(monkeypatch)
    child_backend = FakeBackend([*_read_turn(target, "child-read")])
    _use_child_backend(monkeypatch, child_backend)
    parent_backend = FakeBackend([*_task_turn(background=False)])
    parent = build_test_agent_loop(
        config=_config(),
        cwd=repo,
        backend=parent_backend,
        user_input_capability=True,
        harness_files=HarnessFilesManager(sources=("project",)).for_session(
            repo, workspace_roots=[repo]
        ),
    )
    # Session grants are bare canonical paths, never keyed to the project.
    parent.apply_root_grant(outside)
    session = await create_test_app_server_session(parent)
    try:
        response = await session.resources.sessions.relocate(str(worktree))
        assert response.state.session.cwd == str(worktree)
        assert parent.cwd == worktree
        assert parent._session_root_grants.roots == {outside.resolve()}
        assert parent.tool_manager.workspace.allows(target)
        # A child dispatched after the move still inherits the bare-path grant.
        callbacks = await _drive(session, "Allow this session")
        assert callbacks == []
        assert len(children) == 1
        child = children[0]
        assert child.cwd == worktree
        assert child._session_root_grants.roots == {outside.resolve()}
        assert child.tool_manager.workspace.allows(target)
        assert any("granted content" in text for text in _tool_texts(child))
    finally:
        await session.close()
        trusted_folders_manager.revoke_session_trust(worktree)
        await parent.aclose()
