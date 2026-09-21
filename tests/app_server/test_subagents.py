from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import suppress
import json
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.app_server._model import validate_wire
from chartreux.app_server._projector import EventProjector, ProjectedUpdate
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntime,
    SessionRuntimeRegistry,
    StoredRunResult,
    _AgentState,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server.events import AgentsUpdate, CallbackRequested, TurnCompleted
from chartreux.app_server.models import (
    CompletedEffectState,
    EffectResultDisplay,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicSessionState,
    PublicTurn,
    PublicTurnStatus,
    SubagentEffectDetail,
    SubagentEffectInput,
    TextContentBlock,
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ProtocolErrorCode,
    SessionReadParams,
    SessionReadResponse,
    TurnStartParams,
)
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.cli.textual_ui.widgets.agent_sidebar import AgentSidebar
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, LLMChunk, LLMMessage, Role, ToolCall
from chartreux.core.message_list import MessageList
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.resolver import ModelResolver
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.session.session_loader import MESSAGES_FILENAME, SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import ChildSessionLink, CommittedModelIdentity
from chartreux.core.subagents import (
    AgentAvailability,
    AgentBusyError,
    AgentEvictedError,
    AgentProfileMismatchError,
    AgentResultExpiredError,
    AgentSummary,
    LaunchConfig,
    LaunchConfigError,
    LaunchToolOverride,
    RunStatus,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission, ToolPermissionError
from chartreux.utils.tool_presentation import (
    EffectCallDisplay,
    ToolCallPresentation,
    ToolEffectKind,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    create_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


def _task_call(
    agent: str = "worker",
    *,
    background: bool = True,
    agent_id: str | None = None,
    task_summary: str | None = None,
    tool_call_id: str = "task-1",
) -> ToolCall:
    arguments: dict[str, object] = {
        "task": "Inspect the project",
        "agent": agent,
        "background": background,
    }
    if agent_id is not None:
        arguments["agent_id"] = agent_id
    if task_summary is not None:
        arguments["task_summary"] = task_summary
    return ToolCall(
        id=tool_call_id,
        index=0,
        function=FunctionCall(name="task", arguments=json.dumps(arguments)),
    )


def _config(
    session_logging: SessionLoggingConfig | None = None,
    *,
    enabled_tools: list[str] | None = None,
):
    tool_names = enabled_tools or ["task"]
    return build_test_vibe_config(
        enabled_tools=tool_names,
        tools={
            name: {"permission": ToolPermission.ALWAYS.value} for name in tool_names
        },
        session_logging=session_logging or SessionLoggingConfig(enabled=False),
    )


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()

    async def complete(self, **_kwargs):
        self.started.set()
        try:
            await self.release.wait()
            return mock_llm_chunk(content="")
        finally:
            self.stopped.set()

    async def complete_streaming(self, **kwargs):
        yield await self.complete(**kwargs)


class ImmediateFailureBackend(FakeBackend):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception

    async def complete(self, **_kwargs):
        raise self.exception

    async def complete_streaming(self, **_kwargs) -> AsyncGenerator[LLMChunk, None]:
        raise self.exception
        yield mock_llm_chunk(content="")


class GatedFailureBackend(FakeBackend):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, **_kwargs):
        self.started.set()
        await self.release.wait()
        raise self.exception

    async def complete_streaming(self, **_kwargs) -> AsyncGenerator[LLMChunk, None]:
        self.started.set()
        await self.release.wait()
        raise self.exception
        yield mock_llm_chunk(content="")


class GatedSequenceBackend(FakeBackend):
    """Return one response per gate, keeping every selected child run observable."""

    def __init__(self, chunks) -> None:
        super().__init__(chunks)
        self.started: list[asyncio.Event] = []
        self.releases: list[asyncio.Event] = []

    def add_gate(self) -> tuple[asyncio.Event, asyncio.Event]:
        started = asyncio.Event()
        release = asyncio.Event()
        self.started.append(started)
        self.releases.append(release)
        return started, release

    async def complete(self, **kwargs):
        gate_index = len(self.requests_messages)
        if gate_index < len(self.started):
            self.started[gate_index].set()
            await self.releases[gate_index].wait()
        return await super().complete(**kwargs)


async def _background_result(
    registry: SessionRuntimeRegistry, args: TaskArgs, ctx: InvokeContext
) -> TaskResult:
    return cast(TaskResult, [result async for result in registry.run(args, ctx)][-1])


def _todo_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="todo", arguments='{"action":"read"}'),
    )


async def _consume(events) -> None:
    _ = [event async for event in events]


async def _wait_for_turn_completion(session: AppServerSession) -> None:
    async def wait() -> None:
        async for event in session.events():
            if isinstance(event, TurnCompleted):
                return

    await asyncio.wait_for(wait(), timeout=5)


async def _wait_for_agents_update(session: AppServerSession) -> AgentsUpdate:
    async def wait() -> AgentsUpdate:
        async for event in session.events():
            if isinstance(event, AgentsUpdate):
                return event
        raise RuntimeError("App-server event stream closed before agents update")

    return await asyncio.wait_for(wait(), timeout=5)


async def _read_child(
    session: AppServerSession, child_session_id: str
) -> PublicSessionState:
    client = session._connection.current
    assert client is not None
    response = validate_wire(
        SessionReadResponse,
        await client.request(
            "session/read", SessionReadParams(session_id=child_session_id)
        ),
    )
    return response.state


async def _persist_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SessionLoggingConfig, str, str, Path]:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=False)])],
        [mock_llm_chunk(content="Parent completed")],
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="Child completed")]),
    )
    parent = build_test_agent_loop(
        config=_config(logging), backend=parent_backend, enable_streaming=True
    )
    session = await create_test_app_server_session(parent)
    try:
        _ = [event async for event in session.act("Delegate this")]
        metadata = parent.session_logger.session_metadata
        parent_dir = parent.session_logger.session_dir
        assert metadata is not None
        assert parent_dir is not None
        link = metadata.child_sessions[0]
        assert link.relative_path is not None
        return (
            logging,
            session.session_id,
            link.session_id,
            parent_dir / link.relative_path,
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_root_handoff_updates_live_child_parent_routing() -> None:
    parent = build_test_agent_loop(config=_config())
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    child = await AgentRuntimeFactory().create_child(parent, "worker")
    assert child.enable_streaming is False
    runtime = registry._build_child_runtime(child)
    registry._children[child.session_id] = runtime
    old_parent_id = parent.session_id
    new_parent_id = f"{old_parent_id}-compacted"

    registry.handoff_root(old_parent_id, new_parent_id)

    assert registry.child_belongs_to(child.session_id, new_parent_id)
    assert not registry.child_belongs_to(child.session_id, old_parent_id)
    await registry.close()
    await parent.aclose()


@pytest.mark.asyncio
async def test_resumed_child_restores_cumulative_stats(tmp_path: Path) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent = build_test_agent_loop(config=_config(logging))
    await parent.persist_empty_session()
    factory = AgentRuntimeFactory()
    child = await factory.create_child(parent, "worker")
    await child.wait_until_ready()
    child.stats.session_prompt_tokens = 11
    child.stats.session_completion_tokens = 7
    child.stats.context_tokens = 18
    await child.persist_empty_session()
    child_session_id = child.session_id
    child_session_dir = child.session_logger.session_dir
    assert child_session_dir is not None
    await child.aclose()

    resumed = await factory.resume_child(
        parent, "worker", child_session_id, child_session_dir
    )
    try:
        assert resumed.stats.session_prompt_tokens == 11
        assert resumed.stats.session_completion_tokens == 7
        assert resumed.stats.context_tokens == 18
    finally:
        await resumed.aclose()
        await parent.aclose()


@pytest.mark.asyncio
async def test_task_creates_independently_readable_child_session(monkeypatch) -> None:
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=False)])],
        [mock_llm_chunk(content="Parent completed")],
    ])
    child_backend = FakeBackend([mock_llm_chunk(content="Child completed")])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(), backend=parent_backend, enable_streaming=True
    )
    session = await create_test_app_server_session(parent)

    try:
        _ = [event async for event in session.act("Delegate this")]
        effect = next(
            entry
            for entry in session.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        assert isinstance(effect.detail, SubagentEffectDetail)
        child_session_id = effect.detail.child_session_id
        assert child_session_id is not None
        child_state = await _read_child(session, child_session_id)
        assert child_state.session.parent_session_id == session.session_id
        assert any(
            isinstance(entry, PublicMessageEntry) and entry.text == "Child completed"
            for entry in child_state.history or []
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_child_auto_compaction_keeps_live_and_persisted_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=False)])],
        [mock_llm_chunk(content="Parent completed")],
    ])
    child_backend = FakeBackend([
        [mock_llm_chunk(content="<summary>Child summary</summary>")],
        [mock_llm_chunk(content="Child completed")],
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    original_child_ids: list[str] = []
    create_child = AgentRuntimeFactory.create_child

    async def create_compacting_child(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs
    ) -> AgentLoop:
        child = await create_child(factory, loop, agent_name, **kwargs)
        original_child_ids.append(child.session_id)
        set_agent_config(
            child, child.config.model_copy(update={"auto_compact_threshold": 1})
        )
        child.stats.context_tokens = 1_000_000
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", create_compacting_child)
    parent = build_test_agent_loop(
        config=_config(logging), backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    session = await attach_test_app_server_session(client)

    try:
        _ = [event async for event in session.act("Delegate this")]
        assert len(original_child_ids) == 1
        original_child_id = original_child_ids[0]
        effect = next(
            entry
            for entry in session.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        assert isinstance(effect.detail, SubagentEffectDetail)
        child_session_id = effect.detail.child_session_id
        assert child_session_id is not None
        assert child_session_id == original_child_id
        assert isinstance(effect.state, CompletedEffectState)

        child_state = await _read_child(session, child_session_id)
        assert child_state.session.parent_session_id == session.session_id
        assert child_state.session.root_session_id == session.session_id
        assert child_state.history is not None
        assert any(
            isinstance(entry, PublicCheckpointEntry) and entry.kind == "compaction"
            for entry in child_state.history
        )
        assert any(
            isinstance(entry, PublicMessageEntry) and entry.text == "Child completed"
            for entry in child_state.history
        )
        metadata = parent.session_logger.session_metadata
        assert metadata is not None
        assert len(metadata.child_sessions) == 1
        assert metadata.child_sessions[0].session_id == child_session_id
        parent_session_id = session.session_id
    finally:
        await session.close()

    resumed_loop = build_test_agent_loop(
        config=_config(logging), backend=FakeBackend(), enable_streaming=True
    )
    resumed = await attach_test_app_server_session(
        start_test_app_server(resumed_loop), resume_session_id=parent_session_id
    )
    try:
        resumed_effect = next(
            entry
            for entry in resumed.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        assert isinstance(resumed_effect.detail, SubagentEffectDetail)
        assert resumed_effect.detail.child_session_id == child_session_id
        child_state = await _read_child(resumed, child_session_id)
        assert child_state.session.parent_session_id == parent_session_id
        assert child_state.session.root_session_id == parent_session_id
        assert child_state.history is not None
        assert any(
            isinstance(entry, PublicCheckpointEntry) and entry.kind == "compaction"
            for entry in child_state.history
        )
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_child_registration_rolls_back_when_projection_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=False)])],
        [mock_llm_chunk(content="Parent completed")],
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="Child completed")]),
    )

    async def fail_link(
        _turns: TurnController, _tool_call_id: str, _child_session_id: str
    ) -> None:
        raise RuntimeError("projection failed")

    monkeypatch.setattr(TurnController, "link_subagent", fail_link)
    parent = build_test_agent_loop(
        config=_config(logging), backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        _ = [event async for event in session.act("Delegate this")]
        metadata = parent.session_logger.session_metadata
        parent_dir = parent.session_logger.session_dir
        assert metadata is not None
        assert parent_dir is not None
        assert metadata.child_sessions == []
        assert legacy_backend(server).children._children == {}
        agents_dir = parent_dir / "agents"
        assert not agents_dir.exists() or not any(agents_dir.iterdir())
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_child_callbacks_round_trip_using_child_session_id(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "interactive.toml").write_text(
        "\n".join([
            'agent_type = "subagent"',
            'enabled_tools = ["todo", "ask_user_question"]',
        ]),
        encoding="utf-8",
    )
    todo_call = ToolCall(
        id="todo-1",
        index=0,
        function=FunctionCall(name="todo", arguments='{"action":"read"}'),
    )
    question_call = ToolCall(
        id="question-1",
        index=0,
        function=FunctionCall(
            name="ask_user_question",
            arguments=json.dumps({
                "questions": [
                    {
                        "question": "Ship it?",
                        "options": [{"label": "Yes"}, {"label": "No"}],
                    }
                ]
            }),
        ),
    )
    child_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[todo_call])],
        [mock_llm_chunk(content="", tool_calls=[question_call])],
        [mock_llm_chunk(content="Child completed")],
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent_backend = FakeBackend([
        [
            mock_llm_chunk(
                content="", tool_calls=[_task_call("interactive", background=False)]
            )
        ],
        [mock_llm_chunk(content="Parent completed")],
    ])
    config = build_test_vibe_config(
        agent_paths=[tmp_path],
        enabled_tools=["task", "todo", "ask_user_question"],
        tools={
            "task": {"permission": ToolPermission.ALWAYS.value},
            "todo": {"permission": ToolPermission.ASK.value},
        },
    )
    parent = build_test_agent_loop(
        config=config, backend=parent_backend, enable_streaming=True
    )
    session = await create_test_app_server_session(parent)
    callbacks = []

    try:
        async for event in session.act("Delegate this"):
            if not isinstance(event, CallbackRequested):
                continue
            callbacks.append(event.callback)
            output = UserInputCallbackOutput(
                result=UserQuestionResult(
                    answers=[UserAnswer(question="Ship it?", answer="Yes")]
                )
            )
            await session.respond_to_callback(event.callback.callback_id, output)

        assert callbacks, session.history
        assert [callback.detail.kind for callback in callbacks] == ["user_input"]
        child_session_ids = {callback.session_id for callback in callbacks}
        assert len(child_session_ids) == 1
        child_session_id = child_session_ids.pop()
        assert child_session_id != session.session_id
        child_state = await _read_child(session, child_session_id)
        assert any(
            isinstance(entry, PublicMessageEntry) and entry.text == "Child completed"
            for entry in child_state.history or []
        )
        assert any(
            isinstance(entry, PublicEffectEntry)
            and isinstance(entry.state, CompletedEffectState)
            for entry in child_state.history or []
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_interrupting_task_detaches_and_closes_runtime(monkeypatch) -> None:
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=False)])]
    ])
    parent = build_test_agent_loop(
        config=_config(), backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    turn = asyncio.create_task(_consume(session.act("Delegate this")))

    try:
        await child_backend.started.wait()
        await session.interrupt()
        await asyncio.wait_for(turn, timeout=1)
        await asyncio.wait_for(child_backend.stopped.wait(), timeout=1)

        children = legacy_backend(server).children
        assert children._children == {}
        await asyncio.gather(*children._teardown_tasks)
        assert not [
            task
            for task in asyncio.all_tasks()
            if not task.done() and task.get_name().startswith("vibe-subagent-")
        ]
    finally:
        turn.cancel()
        await session.close()

    assert children._children == {}


@pytest.mark.asyncio
async def test_resuming_parent_restores_child_link_and_public_history(
    tmp_path, monkeypatch
) -> None:
    logging, parent_session_id, child_session_id, _ = await _persist_child(
        tmp_path, monkeypatch
    )

    resumed_loop = build_test_agent_loop(
        config=_config(logging), backend=FakeBackend(), enable_streaming=True
    )
    resumed = await attach_test_app_server_session(
        start_test_app_server(resumed_loop), resume_session_id=parent_session_id
    )
    try:
        resumed_effect = next(
            entry
            for entry in resumed.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        assert isinstance(resumed_effect.detail, SubagentEffectDetail)
        assert resumed_effect.detail.child_session_id == child_session_id
        child = await _read_child(resumed, child_session_id)
        assert child.session.parent_session_id == parent_session_id
        assert any(
            isinstance(entry, PublicMessageEntry) and entry.text == "Child completed"
            for entry in child.history or []
        )
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_corrupt_child_log_does_not_fail_parent_resume(
    tmp_path, monkeypatch
) -> None:
    logging, parent_session_id, child_session_id, child_dir = await _persist_child(
        tmp_path, monkeypatch
    )

    (child_dir / MESSAGES_FILENAME).write_text("{not-json", encoding="utf-8")
    resumed_loop = build_test_agent_loop(
        config=_config(logging), backend=FakeBackend(), enable_streaming=True
    )
    created_children = []
    create_child = AgentRuntimeFactory.create_child

    async def track_child(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs
    ) -> AgentLoop:
        child = await create_child(factory, loop, agent_name, **kwargs)
        created_children.append(child)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", track_child)
    resumed = await attach_test_app_server_session(
        start_test_app_server(resumed_loop), resume_session_id=parent_session_id
    )
    try:
        assert created_children == []
        effect = next(
            entry
            for entry in resumed.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        assert isinstance(effect.detail, SubagentEffectDetail)
        assert effect.detail.child_session_id == child_session_id
        with pytest.raises(AppServerResponseError) as exc_info:
            await _read_child(resumed, child_session_id)
        assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_repeated_reads_of_broken_child_stay_not_found(
    tmp_path, monkeypatch
) -> None:
    logging, parent_session_id, child_session_id, child_dir = await _persist_child(
        tmp_path, monkeypatch
    )

    (child_dir / MESSAGES_FILENAME).write_text("{not-json", encoding="utf-8")
    resumed_loop = build_test_agent_loop(
        config=_config(logging), backend=FakeBackend(), enable_streaming=True
    )
    created_children: list[AgentLoop] = []
    create_child = AgentRuntimeFactory.create_child

    async def track_child(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs
    ) -> AgentLoop:
        child = await create_child(factory, loop, agent_name, **kwargs)
        created_children.append(child)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", track_child)
    resumed = await attach_test_app_server_session(
        start_test_app_server(resumed_loop), resume_session_id=parent_session_id
    )
    try:
        for _ in range(3):
            with pytest.raises(AppServerResponseError) as exc_info:
                await _read_child(resumed, child_session_id)
            assert exc_info.value.error.code is ProtocolErrorCode.NOT_FOUND
        assert created_children == []
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_unprojectable_child_tool_call_is_tolerated_on_resume(
    tmp_path, monkeypatch
) -> None:
    logging, parent_session_id, child_session_id, child_dir = await _persist_child(
        tmp_path, monkeypatch
    )

    # A stored tool call whose arguments no longer match its presentation model.
    # Projection must degrade gracefully rather than discard the whole child.
    invalid_message = LLMMessage(
        role=Role.assistant,
        tool_calls=[
            ToolCall(
                id="invalid-task",
                index=0,
                function=FunctionCall(name="task", arguments="{}"),
                presentation=ToolCallPresentation(
                    kind=ToolEffectKind.FILE_READ,
                    display=EffectCallDisplay(summary="invalid", status_text="invalid"),
                ),
            )
        ],
    )
    with (child_dir / MESSAGES_FILENAME).open("a", encoding="utf-8") as messages:
        messages.write(json.dumps(invalid_message.model_dump(mode="json")) + "\n")

    resumed_loop = build_test_agent_loop(
        config=_config(logging), backend=FakeBackend(), enable_streaming=True
    )
    close_calls: list[AsyncMock] = []
    create_child = AgentRuntimeFactory.create_child

    async def track_child(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs
    ) -> AgentLoop:
        child = await create_child(factory, loop, agent_name, **kwargs)
        close = AsyncMock(wraps=child.aclose)
        monkeypatch.setattr(child, "aclose", close)
        close_calls.append(close)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", track_child)
    resumed = await attach_test_app_server_session(
        start_test_app_server(resumed_loop), resume_session_id=parent_session_id
    )
    try:
        # Stored child reads project the transcript directly without materializing
        # or retaining a runtime, and stay readable.
        with suppress(Exception):
            await _read_child(resumed, child_session_id)
        assert close_calls == []
        child = await _read_child(resumed, child_session_id)
        assert child.session.parent_session_id == parent_session_id
        assert any(
            isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.TOOL
            for entry in child.history or []
        )
    finally:
        await resumed.close()


@pytest.mark.asyncio
async def test_background_agent_management_survives_wait_timeout(monkeypatch) -> None:
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="Parent completed")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        launched_events = [event async for event in session.act("Delegate this")]
        await child_backend.started.wait()
        registry = legacy_backend(server).children
        summaries = await registry.check_agents()
        assert len(summaries) == 1
        summary = summaries[0]
        assert summary.availability.value == "running"
        launched_update = next(
            event
            for event in launched_events
            if isinstance(event, AgentsUpdate) and len(event.agents) == 1
        )
        assert launched_update.agents[0].agent_id == summary.agent_id
        assert launched_update.agents[0].availability == "running"
        assert launched_update.agents[0].current_run_id == summary.current_run_id
        assert launched_update.agents[0].current_run_status == "running"
        assert summary.current_run_id is not None
        assert (
            await registry.get_agent_result(summary.agent_id, summary.current_run_id)
            is None
        )
        with pytest.raises(TimeoutError):
            await registry.wait_for_agent(
                summary.agent_id, summary.current_run_id, timeout=0.01
            )
        assert not child_backend.stopped.is_set()

        child_backend.release.set()
        result = await asyncio.wait_for(
            registry.wait_for_agent(summary.agent_id, summary.current_run_id), timeout=1
        )
        assert result.completed
        assert result.agent_id == summary.agent_id
        assert result.run_id == summary.current_run_id
        assert (
            await registry.get_agent_result(summary.agent_id, summary.current_run_id)
            == result
        )
        idle_update = await _wait_for_agents_update(session)
        assert idle_update.agents[0].availability == "finalizing"
        idle_update = await _wait_for_agents_update(session)
        assert idle_update.agents[0].agent_id == summary.agent_id
        assert idle_update.agents[0].availability == "idle"
        assert idle_update.agents[0].current_run_id is None
        assert idle_update.agents[0].current_run_status is None
        assert (await registry.check_agents())[0].availability.value == "idle"

        runtime = registry._agent_records[summary.agent_id].runtime
        await registry.release_agent(summary.agent_id)
        released_update = await _wait_for_agents_update(session)
        assert released_update.agents == []
        assert runtime._closed
        assert await registry.check_agents() == []
    finally:
        child_backend.release.set()
        await session.close()


@pytest.mark.asyncio
async def test_background_completion_during_parent_final_save_is_delivered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
        [mock_llm_chunk(content="Parent completed")],
        [mock_llm_chunk(content="Notification received")],
    ])
    parent = build_test_agent_loop(
        config=_config(), backend=parent_backend, enable_streaming=True
    )
    finalizing = asyncio.Event()
    release_finalizer = asyncio.Event()
    original_save = parent._save_messages
    save_count = 0

    async def gated_save(*, allow_empty: bool = False) -> None:
        nonlocal save_count
        save_count += 1
        if save_count == 3:
            finalizing.set()
            await release_finalizer.wait()
        await original_save(allow_empty=allow_empty)

    monkeypatch.setattr(parent, "_save_messages", gated_save)
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        action = asyncio.create_task(_consume(session.act("Delegate this")))
        await asyncio.wait_for(child_backend.started.wait(), timeout=1)
        await asyncio.wait_for(finalizing.wait(), timeout=1)
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        assert summary.current_run_id is not None

        child_backend.release.set()
        await asyncio.wait_for(
            registry.wait_for_agent(summary.agent_id, summary.current_run_id), timeout=1
        )
        release_finalizer.set()
        await asyncio.wait_for(action, timeout=1)

        assert len(parent_backend.requests_messages) == 3
        assert any(
            message.role is Role.user
            and f"Background agent {summary.agent_id}" in str(message.content)
            for message in parent_backend.requests_messages[-1]
        )
    finally:
        child_backend.release.set()
        release_finalizer.set()
        await session.close()


@pytest.mark.asyncio
async def test_background_notification_survives_agents_update_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
        [mock_llm_chunk(content="Parent completed")],
        [mock_llm_chunk(content="Notification received")],
    ])
    parent = build_test_agent_loop(
        config=_config(), backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("Delegate this"))
        await asyncio.wait_for(child_backend.started.wait(), timeout=1)
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        assert summary.current_run_id is not None
        registry._emit_agents_update = AsyncMock(side_effect=RuntimeError("UI failed"))

        child_backend.release.set()
        await asyncio.wait_for(
            registry.wait_for_agent(summary.agent_id, summary.current_run_id), timeout=1
        )
        await _wait_for_turn_completion(session)

        assert len(parent_backend.requests_messages) == 3
        assert any(
            message.role is Role.user
            and f"Background agent {summary.agent_id}" in str(message.content)
            for message in parent_backend.requests_messages[-1]
        )
    finally:
        child_backend.release.set()
        await session.close()


@pytest.mark.asyncio
async def test_background_agent_reuse_preserves_run_results_and_restores_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([
        [mock_llm_chunk(content="First child response")],
        [mock_llm_chunk(content="Second child response")],
    ])
    first_started, first_release = child_backend.add_gate()
    second_started, second_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="First parent response")],
            [mock_llm_chunk(content="Noted.")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _task_call(
                            background=True, agent_id="agent-1", tool_call_id="task-2"
                        )
                    ],
                )
            ],
            [mock_llm_chunk(content="Second parent response")],
            [mock_llm_chunk(content="Noted.")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("Launch first"))
        registry = legacy_backend(server).children
        record = registry._agent_records["agent-1"]
        first_run_id = (
            record.current_run.run_id
            if record.current_run is not None
            else record.run_history[-1].run_id
        )
        assert first_run_id is not None
        runtime = record.runtime
        await asyncio.wait_for(first_started.wait(), timeout=1)
        first_release.set()
        await _wait_for_turn_completion(session)
        first = await registry.wait_for_agent(record.agent_id, first_run_id)
        assert first.turns_used == 1
        assert runtime.turns._event_sink is None

        await _consume(session.act("Launch second"))
        second_record = registry._agent_records[record.agent_id]
        second_run_id = (
            second_record.current_run.run_id
            if second_record.current_run is not None
            else second_record.run_history[-1].run_id
        )
        assert second_run_id != first_run_id
        await asyncio.wait_for(second_started.wait(), timeout=1)
        second_release.set()
        await _wait_for_turn_completion(session)
        second = await registry.wait_for_agent(record.agent_id, second_run_id)

        assert registry._agent_records[record.agent_id].runtime is runtime
        assert second.turns_used == 1
        assert await registry.get_agent_result(record.agent_id, first_run_id) == first
        assert runtime.turns._event_sink is None
        assert len(child_backend.requests_messages) == 2
    finally:
        first_release.set()
        second_release.set()
        await session.close()

    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _session_id: 0)
    failure = RuntimeError("first child failed to close")
    closed = []
    first = MagicMock()
    second = MagicMock()
    third = MagicMock()

    def close_runtime(runtime, error: Exception | None = None) -> None:
        closed.append(runtime)
        if error is not None:
            raise error

    first.close = AsyncMock(side_effect=lambda: close_runtime(first, failure))
    second.close = AsyncMock(side_effect=lambda: close_runtime(second))
    third.close = AsyncMock(side_effect=lambda: close_runtime(third))
    registry._children.update({
        "first": cast(Any, first),
        "second": cast(Any, second),
        "third": cast(Any, third),
    })

    with pytest.raises(RuntimeError) as exc_info:
        await registry.close()

    assert exc_info.value is failure
    assert closed == [first, second, third]
    assert registry._children == {}


@pytest.mark.asyncio
async def test_background_lifecycle_allows_parent_turns_and_reuse(monkeypatch) -> None:
    child_backend = GatedSequenceBackend([
        [mock_llm_chunk(content="child A complete")],
        [mock_llm_chunk(content="child B complete")],
    ])
    first_started, first_release = child_backend.add_gate()
    second_started, second_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="launch turn finished")],
            [mock_llm_chunk(content="second root turn finished")],
            [mock_llm_chunk(content="Noted A.")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _task_call(
                            background=True, agent_id="agent-1", tool_call_id="task-2"
                        )
                    ],
                )
            ],
            [mock_llm_chunk(content="reuse turn finished")],
            [mock_llm_chunk(content="Noted B.")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("launch A"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        registry = legacy_backend(server).children
        running = await registry.check_agents()
        assert len(running) == 1
        agent_id = running[0].agent_id
        first_run_id = running[0].current_run_id
        assert first_run_id is not None
        assert running[0].availability.value == "running"

        await _consume(session.act("continue while A runs"))
        assert await registry.get_agent_result(agent_id, first_run_id) is None
        assert [
            entry
            for entry in session.history
            if isinstance(entry, PublicEffectEntry) and entry.detail.tool_name == "task"
        ].__len__() == 1

        first_release.set()
        first = await asyncio.wait_for(
            registry.wait_for_agent(agent_id, first_run_id), timeout=1
        )
        assert first.response == "child A complete"
        await _wait_for_turn_completion(session)
        await _consume(session.act("retrieve A in a later root turn"))

        await _consume(session.act("reuse A"))
        await asyncio.wait_for(second_started.wait(), timeout=1)
        second_run_id = (await registry.check_agents())[0].current_run_id
        assert second_run_id is not None and second_run_id != first_run_id
        second_release.set()
        second = await asyncio.wait_for(
            registry.wait_for_agent(agent_id, second_run_id), timeout=1
        )
        assert second.response == "child B complete"
        assert await registry.get_agent_result(agent_id, first_run_id) == first
        assert await registry.get_agent_result(agent_id, second_run_id) == second
        assert "child A complete" not in [
            entry.text
            for entry in session.history
            if isinstance(entry, PublicMessageEntry) and entry.role == "assistant"
        ]
    finally:
        first_release.set()
        second_release.set()
        await session.close()


@pytest.mark.asyncio
async def test_background_progress_overflow_completes_without_a_consumer(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "noisy.toml").write_text(
        "\n".join(['agent_type = "subagent"', 'enabled_tools = ["todo"]']),
        encoding="utf-8",
    )
    progress_count = 65
    child_backend = FakeBackend(
        [
            [mock_llm_chunk(content="", tool_calls=[_todo_call(f"todo-{index}")])]
            for index in range(progress_count)
        ]
        + [[mock_llm_chunk(content="all progress complete")]]
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(
            agent_paths=[tmp_path],
            enabled_tools=["task", "todo"],
            tools={
                "task": {"permission": ToolPermission.ALWAYS.value},
                "todo": {"permission": ToolPermission.ALWAYS.value},
            },
        ),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call("noisy", background=True)]
                )
            ],
            [mock_llm_chunk(content="parent completed")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("launch noisy child"))
        registry = legacy_backend(server).children
        record = registry._agent_records["agent-1"]
        run_id = record.current_run.run_id if record.current_run is not None else None
        assert run_id is not None
        result = await asyncio.wait_for(
            registry.wait_for_agent("agent-1", run_id), timeout=2
        )
        assert result.completed
        assert result.response == "all progress complete"
        completed_run = registry._agent_records["agent-1"].run_history[-1]
        assert completed_run.progress_overflow
        assert len(completed_run.progress_summaries) == 32
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_concurrent_background_reuse_admits_exactly_one_run(monkeypatch) -> None:
    child_backend = GatedSequenceBackend([
        [mock_llm_chunk(content="first")],
        [mock_llm_chunk(content="second")],
    ])
    first_started, first_release = child_backend.add_gate()
    second_started, second_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="parent completed")],
            [mock_llm_chunk(content="Noted.")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _task_call(
                            background=True, agent_id="agent-1", tool_call_id="reuse"
                        )
                    ],
                )
            ],
            [mock_llm_chunk(content="parent completed")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("launch"))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        registry = legacy_backend(server).children
        first_release.set()
        first_run = (await registry.check_agents())[0].current_run_id
        assert first_run is not None
        await asyncio.wait_for(registry.wait_for_agent("agent-1", first_run), timeout=1)
        await _wait_for_turn_completion(session)

        ctx = InvokeContext(tool_call_id="reuse", session_id=session.session_id)
        args = TaskArgs(
            task="reuse", agent="worker", agent_id="agent-1", background=True
        )
        await _consume(session.act("reuse"))
        await asyncio.wait_for(second_started.wait(), timeout=1)
        rejected = asyncio.create_task(_background_result(registry, args, ctx))
        with pytest.raises(ValueError, match="already running"):
            await rejected
        second_release.set()
    finally:
        first_release.set()
        second_release.set()
        await session.close()


@pytest.mark.asyncio
async def test_background_waiters_compaction_and_shutdown_preserve_lifecycle(
    monkeypatch,
) -> None:
    child_backend = GatedSequenceBackend([
        mock_llm_chunk(content="complete after wait")
    ])
    started, release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="parent completed")],
            [mock_llm_chunk(content="<summary>root compacted</summary>")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("launch"))
        await asyncio.wait_for(started.wait(), timeout=1)
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        assert summary.current_run_id is not None
        timed_out = asyncio.create_task(
            registry.wait_for_agent(
                summary.agent_id, summary.current_run_id, timeout=0.01
            )
        )
        waiting = asyncio.create_task(
            registry.wait_for_agent(summary.agent_id, summary.current_run_id)
        )
        with pytest.raises(TimeoutError):
            await timed_out
        assert not child_backend.releases[0].is_set()
        assert (await registry.check_agents())[0].availability.value == "running"

        # Root compaction must not erase registry-owned handles while the child runs.
        await session.compact()
        assert (await registry.check_agents())[0].agent_id == summary.agent_id
        release.set()
        result = await asyncio.wait_for(waiting, timeout=1)
        assert (
            await registry.get_agent_result(summary.agent_id, summary.current_run_id)
            == result
        )

        await registry.close_children()
        assert await registry.check_agents() == []
        assert not [
            task
            for task in asyncio.all_tasks()
            if not task.done()
            and task.get_name().startswith("vibe-subagent-background:")
        ]
        with pytest.raises(RuntimeError, match="admission is closed"):
            await _background_result(
                registry,
                TaskArgs(task="new", background=True),
                InvokeContext(tool_call_id="closed", session_id=session.session_id),
            )
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_eager_background_monitor_reports_immediate_backend_failure(
    monkeypatch,
) -> None:
    child_backend = ImmediateFailureBackend(
        RuntimeError("child backend failed immediately")
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="parent completed")],
            [mock_llm_chunk(content="Noted.")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    loop = asyncio.get_running_loop()
    previous_task_factory = loop.get_task_factory()

    try:
        loop.set_task_factory(asyncio.eager_task_factory)
        await _consume(session.act("launch failing child"))
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        record = registry._agent_records["agent-1"]
        run_id = (
            record.current_run.run_id
            if record.current_run is not None
            else record.run_history[-1].run_id
        )
        failed = await asyncio.wait_for(
            registry.wait_for_agent(summary.agent_id, run_id), timeout=1
        )
        assert not failed.completed
        assert "child backend failed immediately" in failed.response
        assert "Turn did not complete" not in failed.response
    finally:
        loop.set_task_factory(previous_task_factory)
        await session.close()


@pytest.mark.asyncio
async def test_background_failure_and_root_shutdown_close_active_runs(
    monkeypatch,
) -> None:
    failing_child = GatedFailureBackend(RuntimeError("child backend failed"))
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: failing_child
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="parent completed")],
            [mock_llm_chunk(content="Noted.")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_task_call(background=True, tool_call_id="task-2")],
                )
            ],
            [mock_llm_chunk(content="parent completed")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    gated_child: BlockingBackend | None = None

    try:
        await _consume(session.act("launch failing child"))
        await asyncio.wait_for(failing_child.started.wait(), timeout=1)
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        assert summary.current_run_id is not None
        failing_child.release.set()
        failed = await asyncio.wait_for(
            registry.wait_for_agent(summary.agent_id, summary.current_run_id), timeout=1
        )
        assert not failed.completed
        assert "child backend failed" in failed.response
        await _wait_for_turn_completion(session)
        failed_summary = (await registry.check_agents())[0]
        assert failed_summary.availability is AgentAvailability.IDLE
        assert failed_summary.current_run_status is None
        assert failed_summary.last_run_status is RunStatus.FAILED

        gated_child = BlockingBackend()
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", lambda **_: gated_child
        )
        await _consume(session.act("launch then shut down"))
        await asyncio.wait_for(gated_child.started.wait(), timeout=1)
        await registry.close_children()
        await asyncio.wait_for(gated_child.stopped.wait(), timeout=1)
        assert await registry.check_agents() == []
        assert not [
            task
            for task in asyncio.all_tasks()
            if not task.done()
            and task.get_name().startswith("vibe-subagent-background:")
        ]
    finally:
        failing_child.release.set()
        if gated_child is not None:
            gated_child.release.set()
        await session.close()


@pytest.mark.asyncio
async def test_session_runtime_close_is_single_flight_retryable_and_cancellation_safe(
    monkeypatch,
) -> None:
    runtime = SessionRuntime(MagicMock(), MagicMock(), MagicMock(), MagicMock())
    runtime.agent_loop.session_id = "child"
    first_attempt = True
    started = asyncio.Event()
    release = asyncio.Event()
    close_agent = AsyncMock()

    async def close_turns() -> None:
        nonlocal first_attempt
        if first_attempt:
            first_attempt = False
            raise RuntimeError("turn cleanup failed")
        started.set()
        await release.wait()

    runtime.turns.close = close_turns
    monkeypatch.setattr("chartreux.app_server._sessions.close_agent_loop", close_agent)

    with pytest.raises(RuntimeError, match="turn cleanup failed"):
        await runtime.close()
    assert not runtime._closed
    assert close_agent.await_count == 1

    first_waiter = asyncio.create_task(runtime.close())
    await asyncio.wait_for(started.wait(), timeout=1)
    second_waiter = asyncio.create_task(runtime.close())
    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_waiter
    release.set()
    await asyncio.wait_for(second_waiter, timeout=1)

    assert runtime._closed
    assert close_agent.await_count == 2


class _ManualClock:
    def __init__(self, value: float = 0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _retention_registry(clock: _ManualClock | None = None) -> SessionRuntimeRegistry:
    return SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _session_id: 0, clock=clock or _ManualClock()
    )


def _stored_result(
    agent_id: str, run_id: str, *, completed_at: float = 0, generation: int = 0
) -> StoredRunResult:
    return StoredRunResult(
        agent_id=agent_id,
        run_id=run_id,
        result=TaskResult(
            response=f"result for {run_id}",
            turns_used=1,
            completed=True,
            agent_id=agent_id,
            run_id=run_id,
        ),
        completed_at=completed_at,
        root_generation=generation,
    )


def _idle_record(
    agent_id: str,
    *,
    clock: _ManualClock,
    summary: str = "initial",
    idle_ttl_seconds: int | None = None,
) -> AgentRecord:
    runtime = MagicMock()
    runtime.close = AsyncMock()
    return AgentRecord(
        agent_id=agent_id,
        profile="worker",
        session_id=f"session-{agent_id}",
        runtime=runtime,
        root_generation=0,
        idle_ttl_seconds=idle_ttl_seconds,
        initial_task_summary=summary,
        state=_AgentState.IDLE,
        idle_since=clock(),
        last_task_summary=summary,
    )


@pytest.mark.asyncio
async def test_evicted_result_is_available_by_the_exact_requested_run() -> None:
    registry = _retention_registry()
    stored = _stored_result("agent-1", "run-1")
    registry._result_store[(stored.agent_id, stored.run_id)] = stored
    cast(Any, registry._evicted_agents)[stored.agent_id] = AgentSummary(
        agent_id=stored.agent_id,
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id=stored.run_id,
        current_run_status=RunStatus.COMPLETED,
    )

    assert await registry.get_agent_result("agent-1", "run-1") == stored.result
    assert await registry.wait_for_agent("agent-1", "run-1") == stored.result


@pytest.mark.asyncio
async def test_result_cap_expires_oldest_but_keeps_pending_and_leased_results() -> None:
    registry = _retention_registry()
    entries = [_stored_result(f"agent-{index:02}", "run") for index in range(35)]
    for stored in entries:
        registry._result_store[(stored.agent_id, stored.run_id)] = stored
    registry._pending_notifications.add((entries[0].agent_id, entries[0].run_id))
    registry._wait_leases[(entries[1].agent_id, entries[1].run_id)] = 1

    registry._expire_results_locked()

    assert (entries[0].agent_id, entries[0].run_id) in registry._result_store
    assert (entries[1].agent_id, entries[1].run_id) in registry._result_store
    assert (entries[2].agent_id, entries[2].run_id) in registry._expired_results
    assert (entries[3].agent_id, entries[3].run_id) in registry._result_store


@pytest.mark.asyncio
async def test_result_cap_is_scoped_to_each_root_generation() -> None:
    registry = _retention_registry()
    for index in range(32):
        stored = _stored_result(f"old-{index:02}", "run", generation=1)
        registry._result_store[(stored.agent_id, stored.run_id)] = stored
    current = _stored_result("current", "run", generation=2)
    registry._result_store[(current.agent_id, current.run_id)] = current

    registry._expire_results_locked()

    assert len(registry._result_store) == 33
    assert registry._expired_results == set()


@pytest.mark.asyncio
async def test_expiry_keeps_resident_agent_but_marks_requested_run_expired() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    expired = _stored_result("agent-0", "run-0")
    run = RunRecord(
        run_id=expired.run_id,
        agent_id=expired.agent_id,
        profile="worker",
        status=RunStatus.COMPLETED,
        completion_task=asyncio.get_running_loop().create_future(),
        result=expired.result,
    )
    record = _idle_record(expired.agent_id, clock=clock)
    record.run_history.append(run)
    registry._agent_records[record.agent_id] = record
    registry._result_store[(expired.agent_id, expired.run_id)] = expired
    for index in range(32):
        stored = _stored_result(f"agent-{index + 1:02}", "run")
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    registry._expire_results_locked()

    assert record.agent_id in registry._agent_records
    assert run.result is None
    with pytest.raises(AgentResultExpiredError):
        await registry.get_agent_result(expired.agent_id, expired.run_id)
    summary = (await registry.check_agents())[0]
    assert summary.result_expired


@pytest.mark.asyncio
async def test_idle_summary_marks_the_latest_expired_result() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    record.latest_run_id = "run-1"
    registry._agent_records[record.agent_id] = record
    registry._expired_results.add((record.agent_id, record.latest_run_id))

    summary = (await registry.check_agents())[0]

    assert summary.result_expired


@pytest.mark.asyncio
async def test_requested_run_never_returns_a_newer_reused_run_result() -> None:
    registry = _retention_registry()
    first = _stored_result("agent-1", "first", completed_at=1)
    second = _stored_result("agent-1", "second", completed_at=2)
    registry._result_store[(first.agent_id, first.run_id)] = first
    registry._result_store[(second.agent_id, second.run_id)] = second

    assert await registry.get_agent_result("agent-1", "first") == first.result
    assert await registry.wait_for_agent("agent-1", "first") == first.result


@pytest.mark.asyncio
async def test_omitted_run_id_does_not_fall_back_after_latest_result_expires() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    record = _idle_record("agent-1", clock=clock)
    record.latest_run_id = "latest"
    older = _stored_result(record.agent_id, "older", completed_at=1)
    registry._agent_records[record.agent_id] = record
    registry._result_store[(older.agent_id, older.run_id)] = older
    registry._expired_results.add((record.agent_id, "latest"))
    assert await registry._evict_agent(record.agent_id, "ttl")
    assert record.agent_id not in registry._agent_records

    with pytest.raises(AgentResultExpiredError, match="latest"):
        await registry.get_agent_result(record.agent_id)


@pytest.mark.asyncio
async def test_expired_marker_is_distinct_from_unknown_until_generation_end() -> None:
    registry = _retention_registry()
    key = ("agent-1", "expired")
    registry._expired_results.add(key)

    with pytest.raises(AgentResultExpiredError):
        await registry.get_agent_result(*key)
    with pytest.raises(UnknownAgentError):
        await registry.get_agent_result("agent-1", "unknown")


@pytest.mark.asyncio
async def test_dispatch_and_eviction_race_has_a_deterministic_winner() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    root = MagicMock()
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = 0
    root.turns._projector = None
    registry._root = root
    registry._generation_identity = ("root", root.agent_loop._session_generation)
    registry._retention_policy = (60, 16)
    record = _idle_record("agent-1", clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None
    cast(MagicMock, record.runtime.turns.start).return_value = (
        MagicMock(id="child-turn"),
        lambda: None,
    )
    record.runtime.turns.wait_for_operation = AsyncMock(
        return_value=MagicMock(error=None, status=PublicTurnStatus.COMPLETED)
    )
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    args = TaskArgs(
        task="reuse", agent="worker", agent_id=record.agent_id, background=True
    )
    context = InvokeContext(tool_call_id="reuse", session_id="root")

    async with registry._registry_lock:
        dispatch = asyncio.create_task(_background_result(registry, args, context))
        eviction = asyncio.create_task(registry._evict_agent(record.agent_id, "ttl"))
        await asyncio.sleep(0)
    await dispatch
    assert not await eviction
    assert record.agent_id in registry._agent_records

    run = record.current_run or record.run_history[-1]
    monitor = run.completion_task
    assert isinstance(monitor, asyncio.Task)
    await monitor
    assert await registry._evict_agent(record.agent_id, "ttl")
    with pytest.raises(AgentEvictedError):
        await _background_result(registry, args, context)


@pytest.mark.asyncio
async def test_wait_and_expiry_race_keeps_a_leased_run_until_delivery() -> None:
    registry = _retention_registry()
    key = ("agent-0", "target")
    target = RunRecord(
        run_id=key[1],
        agent_id=key[0],
        profile="worker",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
    )
    record = _idle_record(key[0], clock=_ManualClock())
    record.current_run = target
    registry._agent_records[key[0]] = record
    registry._result_store[key] = _stored_result(*key)
    for index in range(33):
        stored = _stored_result(f"agent-{index + 1:02}", "run")
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    waiter = asyncio.create_task(registry.wait_for_agent(*key))
    await asyncio.sleep(0)
    registry._expire_results_locked()
    assert key not in registry._expired_results

    target.result = _stored_result(*key).result
    target.status = RunStatus.COMPLETED
    target.completion_task.set_result(None)
    assert await waiter == target.result

    registry._expire_results_locked()
    with pytest.raises(AgentResultExpiredError):
        await registry.get_agent_result(*key)


@pytest.mark.asyncio
async def test_release_running_agent_invalidates_waiters_without_leaking_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="done")]])
    started, release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()

    try:
        launched = await _background_result(registry, args, context)
        assert launched.run_id is not None
        run = record.current_run
        assert run is not None and isinstance(run.completion_task, asyncio.Task)
        await started.wait()
        waiter = asyncio.create_task(
            registry.wait_for_agent(record.agent_id, run.run_id)
        )
        cancelled_waiter = asyncio.create_task(
            registry.wait_for_agent(record.agent_id, run.run_id)
        )
        await asyncio.sleep(0)
        assert registry._wait_leases[(record.agent_id, run.run_id)] == 2

        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter

        await registry.release_agent(record.agent_id)
        with pytest.raises((UnknownAgentError, AgentResultExpiredError)):
            await waiter
        assert (record.agent_id, run.run_id) not in registry._result_write_tokens
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_wait_for_expired_run_raises_typed_error() -> None:
    registry = _retention_registry()
    key = ("agent-1", "run-1")
    registry._expired_results.add(key)

    with pytest.raises(AgentResultExpiredError):
        await registry.wait_for_agent(*key)


@pytest.mark.asyncio
async def test_reuse_lifecycle_errors_are_typed_and_do_not_substitute_agents() -> None:
    registry = _retention_registry()
    root = MagicMock()
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = 0
    root.turns._projector = None
    registry._root = root
    registry._generation_identity = ("root", root.agent_loop._session_generation)
    clock = _ManualClock()
    busy = _idle_record("busy", clock=clock)
    busy.state = _AgentState.RUNNING
    mismatch = _idle_record("mismatch", clock=clock)
    registry._agent_records.update({busy.agent_id: busy, mismatch.agent_id: mismatch})
    cast(Any, registry._evicted_agents)["evicted"] = AgentSummary(
        agent_id="evicted",
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id=None,
        current_run_status=None,
    )

    def reuse_args(agent_id: str, agent: str = "worker") -> TaskArgs:
        return TaskArgs(task="reuse", agent=agent, agent_id=agent_id, background=True)

    def reuse_context(agent_id: str) -> InvokeContext:
        return InvokeContext(tool_call_id=f"reuse-{agent_id}", session_id="root")

    with pytest.raises(AgentEvictedError):
        await _background_result(
            registry, reuse_args("evicted"), reuse_context("evicted")
        )
    with pytest.raises(AgentBusyError):
        await _background_result(registry, reuse_args("busy"), reuse_context("busy"))
    with pytest.raises(AgentProfileMismatchError):
        await _background_result(
            registry, reuse_args("mismatch", "review"), reuse_context("mismatch")
        )
    assert set(registry._agent_records) == {"busy", "mismatch"}


@pytest.mark.asyncio
async def test_observation_does_not_refresh_idle_age_or_idle_task_summary() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    registry._root = MagicMock()
    registry._root.agent_loop.config.subagents.idle_ttl_seconds = 10
    record = _idle_record("agent-1", clock=clock, summary="first task")
    record.last_task_summary = "latest task"
    registry._agent_records[record.agent_id] = record

    clock.advance(7)
    summary = (await registry.check_agents())[0]
    assert summary.idle_seconds == 7
    assert summary.ttl_remaining_seconds == 3
    assert summary.current_task_summary == "latest task"
    with pytest.raises(UnknownAgentError):
        await registry.get_agent_result("agent-1", "unknown")
    clock.advance(2)
    assert (await registry.check_agents())[0].idle_seconds == 9
    assert await registry._evict_agent("agent-1", "ttl")


@pytest.mark.asyncio
async def test_eviction_tombstone_contains_no_runtime_reference() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    registry._notify_agents = AsyncMock()
    notifier = cast(AsyncMock, registry._notify_agents)
    record = _idle_record("agent-1", clock=clock)
    record.effective_model = "strong"
    record.effective_thinking = "high"
    record.last_run_status = RunStatus.COMPLETED
    result = _stored_result("agent-1", "run-1")
    record.run_history.append(
        RunRecord(
            run_id=result.run_id,
            agent_id=result.agent_id,
            profile="worker",
            status=RunStatus.COMPLETED,
            completion_task=asyncio.get_running_loop().create_future(),
            result=result.result,
        )
    )
    registry._agent_records[record.agent_id] = record
    registry._result_store[(result.agent_id, result.run_id)] = result

    assert await registry._evict_agent(record.agent_id, "ttl")
    tombstone = registry._evicted_agents[record.agent_id].summary
    assert (tombstone.effective_model, tombstone.effective_thinking) == (
        "strong",
        "high",
    )
    assert tombstone.last_run_status is RunStatus.COMPLETED
    assert not tombstone.result_expired
    await registry._emit_agents_update()
    notification = notifier.await_args
    assert notification is not None
    preserved_model = notification.args[0][0]
    assert "child_session_id" not in preserved_model.model_dump()
    assert "parent_identity" not in preserved_model.model_dump()
    assert not preserved_model.result_expired
    sidebar = AgentSidebar()
    sidebar.update_agents((preserved_model,))
    assert "Evicted: result preserved" in str(sidebar.render())

    for index in range(32):
        stored_result = _stored_result(
            f"agent-{index:02}", "run", completed_at=index + 1
        )
        registry._result_store[(stored_result.agent_id, stored_result.run_id)] = (
            stored_result
        )
    registry._expire_results_locked()

    summary = (await registry.check_agents())[0]
    assert summary.result_expired
    await registry._emit_agents_update()
    notification = notifier.await_args
    assert notification is not None
    expired_model = notification.args[0][0]
    assert expired_model.result_expired
    sidebar.update_agents((expired_model,))
    assert "Evicted: result expired" in str(sidebar.render())
    stored = result
    assert set(StoredRunResult.__dataclass_fields__) == {
        "agent_id",
        "run_id",
        "result",
        "completed_at",
        "root_generation",
        "terminal_identity",
    }
    assert all(
        getattr(stored, field_name) is not record.runtime
        for field_name in StoredRunResult.__dataclass_fields__
    )


@pytest.mark.asyncio
async def test_equal_timestamp_result_cap_uses_agent_id_tie_break() -> None:
    registry = _retention_registry()
    for index in range(33):
        stored = _stored_result(f"agent-{32 - index:02}", "run", completed_at=1)
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    registry._expire_results_locked()

    assert ("agent-00", "run") in registry._expired_results


@pytest.mark.asyncio
async def test_release_agent_cleans_evicted_results_and_expiry_markers() -> None:
    registry = _retention_registry()
    agent_id = "agent-1"
    cast(Any, registry._evicted_agents)[agent_id] = AgentSummary(
        agent_id=agent_id,
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id="run-1",
        current_run_status=RunStatus.COMPLETED,
    )
    registry._result_store[(agent_id, "run-1")] = _stored_result(agent_id, "run-1")
    registry._expired_results.add((agent_id, "run-0"))

    registry._emit_agents_update = AsyncMock()
    await registry.release_agent(agent_id)

    assert agent_id not in registry._evicted_agents
    assert not [key for key in registry._result_store if key[0] == agent_id]
    assert not [key for key in registry._expired_results if key[0] == agent_id]
    cast(AsyncMock, registry._emit_agents_update).assert_awaited_once()


class _GatedWakeup:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.arrivals: asyncio.Queue[float] = asyncio.Queue()
        self.called = asyncio.Event()
        self.gates: list[asyncio.Event] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)
        self.arrivals.put_nowait(delay)
        gate = asyncio.Event()
        self.gates.append(gate)
        self.called.set()
        await gate.wait()


def _reaper_registry(
    clock: _ManualClock,
    *,
    ttl: float = 0,
    cap: int = 0,
    wakeup: _GatedWakeup | None = None,
    generation: int = 0,
) -> tuple[SessionRuntimeRegistry, MagicMock]:
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        notify_agents=AsyncMock(),
        clock=clock,
        wakeup=wakeup or _GatedWakeup(),
    )
    root = MagicMock()
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = generation
    root.agent_loop.config.subagents.idle_ttl_seconds = ttl
    root.agent_loop.config.subagents.max_idle_agents = cap
    registry.bind_root(root)
    return registry, root


async def _rearm(registry: SessionRuntimeRegistry) -> None:
    async with registry._registry_lock:
        registry._rearm_reaper_locked()


@pytest.mark.asyncio
async def test_reaper_ttl_ages_from_completion_not_observation() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    record = _idle_record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record

    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    assert wakeup.calls == [10]
    clock.advance(7)
    assert (await registry.check_agents())[0].idle_seconds == 7
    with pytest.raises(UnknownAgentError):
        await registry.get_agent_result(record.agent_id, "unknown")
    clock.advance(3)
    wakeup.gates[0].set()
    reaper_task = registry._reaper_task
    assert reaper_task is not None
    await asyncio.wait_for(reaper_task, timeout=1)

    assert record.agent_id not in registry._agent_records
    assert registry._evicted_agents[record.agent_id].summary.idle_seconds == 10


@pytest.mark.asyncio
async def test_reaper_uses_profile_ttl_before_global_ttl() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    profile = _idle_record("profile", clock=clock, idle_ttl_seconds=5)
    global_ttl = _idle_record("global", clock=clock)
    registry._agent_records = {
        profile.agent_id: profile,
        global_ttl.agent_id: global_ttl,
    }

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 5
    clock.advance(5)
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper
    assert set(registry._agent_records) == {"global"}
    assert await wakeup.arrivals.get() == 5
    await registry.drain_children()


@pytest.mark.asyncio
async def test_zero_profile_ttl_disables_ttl_eviction_but_not_cap_eviction() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, cap=1, wakeup=wakeup)
    exempt = _idle_record("exempt", clock=clock, idle_ttl_seconds=0)
    other = _idle_record("other", clock=clock)
    other.idle_since = 1
    registry._agent_records = {exempt.agent_id: exempt, other.agent_id: other}

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 0
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper
    assert set(registry._agent_records) == {"other"}
    assert (
        registry._evicted_agents["exempt"].summary.availability
        is AgentAvailability.EVICTED
    )
    await registry.drain_children()


@pytest.mark.asyncio
async def test_reaper_arms_for_earliest_mixed_ttl_and_not_all_exempt_agents() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    slow = _idle_record("slow", clock=clock, idle_ttl_seconds=20)
    fast = _idle_record("fast", clock=clock, idle_ttl_seconds=5)
    registry._agent_records = {slow.agent_id: slow, fast.agent_id: fast}

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 5
    slow.idle_ttl_seconds = 0
    fast.idle_ttl_seconds = 0
    await _rearm(registry)
    assert registry._reaper_task is None


@pytest.mark.asyncio
async def test_idle_summaries_use_each_record_ttl() -> None:
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=10)
    exempt = _idle_record("exempt", clock=clock, idle_ttl_seconds=0)
    overridden = _idle_record("overridden", clock=clock, idle_ttl_seconds=5)
    inherited = _idle_record("inherited", clock=clock)
    registry._agent_records = {
        record.agent_id: record for record in (exempt, overridden, inherited)
    }

    clock.advance(3)
    summaries = {summary.agent_id: summary for summary in await registry.check_agents()}
    assert summaries["exempt"].ttl_remaining_seconds is None
    assert summaries["overridden"].ttl_remaining_seconds == 2
    assert summaries["inherited"].ttl_remaining_seconds == 7


@pytest.mark.asyncio
async def test_reaper_evicts_oldest_eligible_agents_with_agent_id_tie_break() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, cap=1, wakeup=wakeup)
    for agent_id, idle_since in (("agent-b", 2), ("agent-a", 2), ("agent-c", 3)):
        record = _idle_record(agent_id, clock=clock)
        record.idle_since = idle_since
        registry._agent_records[agent_id] = record

    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 0
    assert wakeup.calls == [0]
    wakeup.gates[0].set()
    reaper_task = registry._reaper_task
    assert reaper_task is not None
    await asyncio.wait_for(reaper_task, timeout=1)

    assert set(registry._agent_records) == {"agent-c"}
    assert list(registry._evicted_agents) == ["agent-a", "agent-b"]


@pytest.mark.asyncio
async def test_idle_cap_eviction_revalidates_eligibility_at_commit() -> None:
    clock = _ManualClock(10)
    registry, _root = _reaper_registry(clock, cap=1)
    first = _idle_record("agent-a", clock=clock)
    second = _idle_record("agent-b", clock=clock)
    registry._agent_records.update({first.agent_id: first, second.agent_id: second})
    registry._draining_children = True
    await registry.release_agent(first.agent_id)
    registry._draining_children = False

    assert not await registry._evict_agent(second.agent_id, "idle_cap")
    assert second.agent_id in registry._agent_records


@pytest.mark.asyncio
async def test_reaper_never_evicts_running_permission_waiting_or_finalizing_agents() -> (
    None
):
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=1, cap=1)
    running = _idle_record("running", clock=clock)
    permission_waiting = _idle_record("permission-waiting", clock=clock)
    finalizing = _idle_record("finalizing", clock=clock)
    running.state = _AgentState.RUNNING
    permission_waiting.state = _AgentState.RUNNING
    finalizing.state = _AgentState.FINALIZING
    registry._agent_records.update({
        record.agent_id: record for record in (running, permission_waiting, finalizing)
    })

    await _rearm(registry)
    assert registry._reaper_task is None
    for record in (running, permission_waiting, finalizing):
        assert not await registry._evict_agent(record.agent_id, "ttl")
    assert set(registry._agent_records) == {
        "running",
        "permission-waiting",
        "finalizing",
    }


@pytest.mark.asyncio
async def test_reaper_has_one_timer_and_rearms_or_stops_with_eligibility() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    later = _idle_record("later", clock=clock)
    later.idle_since = 5
    registry._agent_records[later.agent_id] = later
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 5
    first = registry._reaper_task
    assert first is not None and wakeup.calls == [5]

    sooner = _idle_record("sooner", clock=clock)
    sooner.idle_since = 3
    registry._agent_records[sooner.agent_id] = sooner
    await _rearm(registry)
    await asyncio.wait_for(asyncio.gather(first, return_exceptions=True), timeout=1)
    assert registry._reaper_task is not first
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 3
    assert wakeup.calls == [5, 3]

    sooner.state = _AgentState.RUNNING
    later.state = _AgentState.RUNNING
    await _rearm(registry)
    assert registry._reaper_task is None
    await registry.release_agent("later")
    await registry.release_agent("sooner")
    assert registry._reaper_task is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ttl,cap,expect_timer", [(0, 1, True), (1, 0, True), (0, 0, False)]
)
async def test_reaper_zero_limit_combinations_disable_the_corresponding_policy(
    ttl: float, cap: int, expect_timer: bool
) -> None:
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=ttl, cap=cap)
    for agent_id in range(2 if ttl == 0 else 1):
        record = _idle_record(f"agent-{agent_id}", clock=clock)
        registry._agent_records[record.agent_id] = record

    await _rearm(registry)
    assert (registry._reaper_task is not None) is expect_timer
    await registry.drain_children()


@pytest.mark.asyncio
async def test_drain_joins_detached_eviction_teardown_without_double_close() -> None:
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock, ttl=1)
    record = _idle_record("agent-1", clock=clock)
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_count = 0

    async def close() -> None:
        nonlocal close_count
        close_count += 1
        close_started.set()
        await close_release.wait()

    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    eviction = asyncio.create_task(registry._evict_agent(record.agent_id, "ttl"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    assert record.agent_id not in registry._agent_records
    await registry.release_agent(record.agent_id)
    assert close_count == 1
    drain = asyncio.create_task(registry.drain_children())
    assert not drain.done()
    close_release.set()
    assert await eviction
    await asyncio.wait_for(drain, timeout=1)
    assert close_count == 1
    assert registry._teardown_tasks == set()


@pytest.mark.asyncio
async def test_root_replacement_drain_clears_generation_state_and_timer() -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    record = _idle_record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    cast(Any, registry._evicted_agents)["old"] = AgentSummary(
        agent_id="old",
        profile="worker",
        availability=AgentAvailability.EVICTED,
        current_run_id=None,
        current_run_status=None,
    )
    registry._expired_results.add(("old", "run"))
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10

    await registry.drain_children()
    root.agent_loop._session_generation = 1
    registry.begin_root_generation()

    assert registry._agent_records == {}
    assert registry._evicted_agents == {}
    assert registry._expired_results == set()
    assert registry._reaper_task is None


@pytest.mark.asyncio
async def test_generation_fences_pending_eviction_notification_and_reports_generation() -> (
    None
):
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=1, generation=4)
    notify_agents = cast(AsyncMock, registry._notify_agents)
    record = _idle_record("agent-1", clock=clock)
    record.root_generation = 4
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def close() -> None:
        close_started.set()
        await close_release.wait()

    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    eviction = asyncio.create_task(registry._evict_agent(record.agent_id, "ttl"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    root.agent_loop._session_generation = 5
    registry.begin_root_generation()
    close_release.set()
    assert await eviction
    notify_agents.assert_not_awaited()

    root.agent_loop._session_generation = 6
    registry.begin_root_generation()
    current = _idle_record("agent-2", clock=clock)
    current.root_generation = 6
    registry._agent_records[current.agent_id] = current
    assert await registry._evict_agent(current.agent_id, "ttl")
    await_args = notify_agents.await_args
    assert await_args is not None
    evictions = await_args.args[1]
    assert evictions[0].root_generation == 6


@pytest.mark.asyncio
async def test_retention_policy_is_snapshotted_per_root_generation() -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=10, cap=2, generation=1)
    assert registry._retention_policy == (10, 2)
    root.agent_loop.config.subagents.idle_ttl_seconds = 3
    root.agent_loop.config.subagents.max_idle_agents = 1
    assert registry._retention_policy == (10, 2)

    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    assert registry._retention_policy == (3, 1)


@pytest.mark.asyncio
async def test_registry_operations_sweep_unreferenced_result_store_entries() -> None:
    registry = _retention_registry()
    for index in range(33):
        stored = _stored_result(f"agent-{index:02}", "run")
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    await registry.check_agents()

    assert ("agent-00", "run") in registry._expired_results


@pytest.mark.asyncio
async def test_notification_failure_still_idles_agent_and_cleans_pending_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=10)
    root.turns._projector = None
    root.turns.active_turn = None
    root.turns._active_task = None
    root.turns.queue_state.items = []
    root.agent_loop._pending_injected_messages = []
    record = _idle_record("agent-1", clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None
    record.runtime.turns.wait_for_operation = AsyncMock(
        return_value=MagicMock(error=None, status=PublicTurnStatus.COMPLETED)
    )
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    monkeypatch.setattr(
        SessionRuntimeRegistry,
        "_start_child_turn",
        lambda *_args: ("turn-1", lambda: None),
    )
    monkeypatch.setattr(
        registry,
        "_emit_agents_update",
        AsyncMock(side_effect=[None, RuntimeError(), None, None]),
    )

    await _background_result(
        registry,
        TaskArgs(task="work", agent="worker", agent_id="agent-1", background=True),
        InvokeContext(tool_call_id="work", session_id="root"),
    )
    monitor = next(iter(registry._monitor_tasks))
    await monitor

    assert record.state is _AgentState.IDLE
    assert registry._pending_notifications == set()
    assert (
        record.agent_id,
        cast(str, record.latest_run_id),
    ) not in registry._pending_notifications
    await registry.drain_children()


@pytest.mark.asyncio
async def test_drain_cancels_finalizing_monitor_without_stale_notification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=10, generation=1)
    root.turns._projector = None
    root.turns.active_turn = None
    root.turns._active_task = None
    root.turns.queue_state.items = []
    root.agent_loop._pending_injected_messages = []
    record = _idle_record("agent-1", clock=clock)
    record.root_generation = 1
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None
    finish = asyncio.Event()
    finalizing = asyncio.Event()
    release_notification = asyncio.Event()

    async def wait_for_operation(initial_turn_id: str) -> PublicTurn:
        assert initial_turn_id == "turn-1"
        await finish.wait()
        return MagicMock(error=None, status=PublicTurnStatus.COMPLETED)

    emit_calls = 0

    async def emit_agents_update(*_args: object) -> None:
        nonlocal emit_calls
        emit_calls += 1
        if emit_calls == 2:
            finalizing.set()
            await release_notification.wait()

    record.runtime.turns.wait_for_operation = wait_for_operation
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    monkeypatch.setattr(
        SessionRuntimeRegistry,
        "_start_child_turn",
        lambda *_args: ("turn-1", lambda: None),
    )
    monkeypatch.setattr(registry, "_emit_agents_update", emit_agents_update)

    launch = await _background_result(
        registry,
        TaskArgs(task="work", agent="worker", agent_id="agent-1", background=True),
        InvokeContext(tool_call_id="work", session_id="root"),
    )
    assert launch.run_id is not None
    waiter = asyncio.create_task(registry.wait_for_agent("agent-1", launch.run_id))
    await asyncio.sleep(0)
    finish.set()
    await finalizing.wait()
    assert record.state is _AgentState.FINALIZING
    finalizing_summary = (await registry.check_agents())[0]
    assert finalizing_summary.availability is AgentAvailability.FINALIZING
    assert finalizing_summary.availability is not AgentAvailability.IDLE
    assert finalizing_summary.last_run_status is RunStatus.COMPLETED
    monitor = next(iter(registry._monitor_tasks))

    await registry.drain_children()
    waiter_result = await asyncio.gather(waiter, return_exceptions=True)

    assert monitor.done()
    assert isinstance(waiter_result[0], asyncio.CancelledError)
    assert root.agent_loop._pending_injected_messages == []
    assert registry._wait_leases == {}
    assert registry._pending_notifications == set()
    assert registry._reaper_task is None


@pytest.mark.asyncio
async def test_reused_launch_failure_restores_idle_bookkeeping_and_rearms_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock(side_effect=RuntimeError("projection failed"))
    record = _idle_record("agent-1", clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    def fail_start(*_args: object) -> tuple[str, object]:
        raise RuntimeError("start failed")

    monkeypatch.setattr(SessionRuntimeRegistry, "_start_child_turn", fail_start)
    with pytest.raises(RuntimeError, match="projection failed"):
        await _background_result(
            registry,
            TaskArgs(task="retry", agent="worker", agent_id="agent-1", background=True),
            InvokeContext(tool_call_id="retry", session_id="root"),
        )

    assert record.state is _AgentState.IDLE
    assert record.idle_since == 0
    assert registry._eligible_idle_locked() == [record]
    assert registry._reaper_task is not None
    await registry.drain_children()


@pytest.mark.asyncio
async def test_child_creation_cancellation_at_publication_lock_closes_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, root = _reaper_registry(_ManualClock(), generation=1)
    created = MagicMock()
    created.session_id = "unpublished-child"
    created_ready = asyncio.Event()
    lease_released = False
    runtime = MagicMock(spec=SessionRuntime)
    runtime.agent_loop = created
    runtime._closed = False
    runtime._close_task = None

    async def close() -> None:
        nonlocal lease_released
        lease_released = True
        runtime._closed = True

    runtime.close = close

    class Factory:
        async def create_child(self, *_args: object) -> MagicMock:
            created_ready.set()
            return created

    registry._runtime_factory = Factory()  # type: ignore[assignment]
    monkeypatch.setattr(registry, "_resolve_launch_candidate", MagicMock())
    monkeypatch.setattr(registry, "_build_child_runtime", lambda _child: runtime)
    await registry._ensure_child_lock.acquire()
    launch = asyncio.create_task(
        registry._create_registered_child(
            root,
            TaskArgs(task="new", agent="worker", background=True),
            InvokeContext(tool_call_id="new", session_id="root"),
        )
    )
    await created_ready.wait()
    await asyncio.sleep(0)
    launch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await launch
    registry._ensure_child_lock.release()
    await registry.drain_children()

    assert lease_released
    assert registry._children == {}
    assert registry._pending_child_closes == {}
    assert registry._teardown_tasks == set()


@pytest.mark.asyncio
async def test_child_creation_generation_change_discards_orphan_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, generation=1)
    created = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()

    class Factory:
        async def create_child(self, *_args: object) -> MagicMock:
            entered.set()
            await release.wait()
            return created

    registry._runtime_factory = Factory()  # type: ignore[assignment]
    candidate = MagicMock()
    resolve = MagicMock(return_value=candidate)
    monkeypatch.setattr(registry, "_resolve_launch_candidate", resolve)
    discard = AsyncMock()
    monkeypatch.setattr(registry, "_discard_child", discard)
    create = asyncio.create_task(
        registry._create_registered_child(
            root,
            TaskArgs(task="new", agent="worker", background=True),
            InvokeContext(tool_call_id="new", session_id="root"),
        )
    )
    await entered.wait()
    resolve.assert_called_once_with(
        root, TaskArgs(task="new", agent="worker", background=True)
    )
    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    release.set()

    with pytest.raises(RuntimeError, match="admission changed"):
        await create
    discard.assert_awaited_once_with(created)
    assert registry._children == {}
    assert registry._child_links == {}


@pytest.mark.asyncio
async def test_handoff_generation_allows_new_current_agent_to_be_evicted() -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, ttl=1, generation=4)
    stale = _idle_record("stale", clock=clock)
    stale.root_generation = 4
    registry._agent_records[stale.agent_id] = stale
    registry._children[stale.session_id] = stale.runtime
    root.agent_loop._session_generation = 5
    registry.handoff_root("old-root", "root")
    assert stale.agent_id not in registry._agent_records
    await asyncio.sleep(0)
    cast(AsyncMock, stale.runtime.close).assert_awaited_once()
    record = _idle_record("current", clock=clock)
    record.root_generation = root.agent_loop._session_generation
    registry._agent_records[record.agent_id] = record

    assert await registry._evict_agent(record.agent_id, "ttl")
    assert record.agent_id in registry._evicted_agents


@pytest.mark.asyncio
async def test_drain_successful_close_retry_retires_failed_teardown_wrapper() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    record.runtime._closed = False
    record.runtime._close_task = None
    failure = RuntimeError("first close failed")
    close_calls = 0

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            raise failure
        record.runtime._closed = True

    record.runtime.close = close
    wrapper = registry._track_child_close(record.runtime, name="failed-close")
    result = await asyncio.gather(wrapper, return_exceptions=True)
    assert result == [failure]
    assert wrapper in registry._teardown_tasks
    assert failure.__traceback__ is not None

    await registry.drain_children()

    assert close_calls == 2
    assert wrapper not in registry._teardown_tasks
    assert failure.__traceback__ is None
    assert registry._pending_child_closes == {}


@pytest.mark.asyncio
async def test_eviction_close_failure_does_not_skip_other_victims_or_reaper() -> None:
    clock = _ManualClock(1)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, cap=1, wakeup=wakeup)
    first = _idle_record("first", clock=clock)
    second = _idle_record("second", clock=clock)
    first.runtime.close = AsyncMock(side_effect=RuntimeError("close failed"))
    registry._agent_records.update({first.agent_id: first, second.agent_id: second})

    await _rearm(registry)
    await wakeup.arrivals.get()
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper

    assert set(registry._evicted_agents) == {"first"}
    assert set(registry._agent_records) == {"second"}
    assert registry._reaper_task is not None
    await registry.drain_children()


@pytest.mark.asyncio
async def test_committed_eviction_notification_survives_reaper_rearm() -> None:
    clock = _ManualClock(1)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, cap=1, wakeup=wakeup)
    first = _idle_record("first", clock=clock)
    second = _idle_record("second", clock=clock)
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def close() -> None:
        close_started.set()
        await close_release.wait()

    first.runtime.close = close
    registry._agent_records.update({first.agent_id: first, second.agent_id: second})
    await _rearm(registry)
    await wakeup.arrivals.get()
    wakeup.gates[0].set()
    await close_started.wait()
    await _rearm(registry)
    close_release.set()
    evictions = [task for task in registry._eviction_tasks]
    if evictions:
        await asyncio.gather(*evictions)

    assert "first" in registry._evicted_agents
    notify_agents = cast(AsyncMock, registry._notify_agents)
    assert any(call.args[1] for call in notify_agents.await_args_list)
    await registry.drain_children()


@pytest.mark.asyncio
async def test_background_launch_normalizes_and_caps_task_summary(monkeypatch) -> None:
    task_summary = "  " + "summary\n" * 100
    child_backend = GatedSequenceBackend([mock_llm_chunk(content="done")])
    started, release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(task_summary=task_summary)]
                )
            ],
            [mock_llm_chunk(content="parent complete")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("launch"))
        await asyncio.wait_for(started.wait(), timeout=1)
        registry = legacy_backend(server).children
        record = registry._agent_records["agent-1"]
        expected = " ".join(" ".join(task_summary.split())[:240].split())[:240]
        assert record.initial_task_summary == expected[:240]
        assert record.current_run is not None
        assert record.current_run.task_summary == expected[:240]
        release.set()
        await registry.wait_for_agent("agent-1", record.current_run.run_id)
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_post_commit_launch_cancellation_keeps_the_monitor_and_exact_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the acknowledgement cannot roll back a committed child run."""
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child done")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(), backend=FakeBackend(), enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    parent_runtime = registry._root
    assert parent_runtime is not None
    parent_runtime.turns._projector = MagicMock()
    parent_runtime.turns.link_subagent = AsyncMock()
    update_entered = asyncio.Event()
    update_release = asyncio.Event()
    calls = 0
    original_emit = registry._emit_agents_update

    async def gated_emit(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            update_entered.set()
            await update_release.wait()
        await cast(Any, original_emit)(*args)

    monkeypatch.setattr(registry, "_emit_agents_update", gated_emit)
    launch = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(task="work", agent="worker", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
    )
    try:
        await update_entered.wait()
        record = registry._agent_records["agent-1"]
        run = record.current_run
        assert run is not None and isinstance(run.completion_task, asyncio.Task)
        monitor = run.completion_task
        await child_started.wait()

        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch
        assert record.current_run is run
        assert record.latest_run_id == run.run_id
        assert record.state is _AgentState.RUNNING
        assert not monitor.done()

        child_release.set()
        await monitor
        result = await registry.get_agent_result(record.agent_id, run.run_id)
        assert result is not None
        assert result.completed is True
        assert result.run_id == run.run_id
    finally:
        update_release.set()
        child_release.set()
        await session.close()


@pytest.mark.asyncio
async def test_post_commit_reused_launch_cancellation_keeps_the_monitor_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child done")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    update_entered = asyncio.Event()
    update_release = asyncio.Event()
    original_emit = registry._emit_agents_update

    async def gated_emit(*values: object) -> None:
        update_entered.set()
        await update_release.wait()
        await cast(Any, original_emit)(*values)

    monkeypatch.setattr(registry, "_emit_agents_update", gated_emit)
    launch = asyncio.create_task(_background_result(registry, args, context))
    try:
        await update_entered.wait()
        run = record.current_run
        assert run is not None and isinstance(run.completion_task, asyncio.Task)
        await child_started.wait()
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch
        assert record.current_run is run
        assert record.state is _AgentState.RUNNING
        assert not run.completion_task.done()

        child_release.set()
        update_release.set()
        await run.completion_task
        result = await registry.get_agent_result(record.agent_id, run.run_id)
        assert result is not None and result.completed
    finally:
        update_release.set()
        child_release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_post_commit_update_failure_does_not_cancel_busy_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child done")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(), backend=FakeBackend(), enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    parent_runtime = registry._root
    assert parent_runtime is not None
    parent_runtime.turns._projector = MagicMock()
    parent_runtime.turns.link_subagent = AsyncMock()
    real_emit = registry._emit_agents_update
    emit_calls = 0

    async def fail_first_emit(*args: object) -> None:
        nonlocal emit_calls
        emit_calls += 1
        if emit_calls == 1:
            raise RuntimeError("initial update failed")
        await cast(Any, real_emit)(*args)

    monkeypatch.setattr(registry, "_emit_agents_update", fail_first_emit)
    try:
        acknowledgement = await _background_result(
            registry,
            TaskArgs(task="work", agent="worker", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert acknowledgement.agent_id == "agent-1"
        assert acknowledgement.run_id is not None
        record = registry._agent_records[acknowledgement.agent_id]
        run = record.current_run
        assert run is not None and run.run_id == acknowledgement.run_id
        assert isinstance(run.completion_task, asyncio.Task)
        monitor = run.completion_task
        await child_started.wait()
        assert record.state is _AgentState.RUNNING
        assert not monitor.done()

        child_release.set()
        await monitor
        result = await registry.get_agent_result(
            record.agent_id, acknowledgement.run_id
        )
        assert result is not None
        assert result.completed is True
    finally:
        child_release.set()
        await session.close()


@pytest.mark.asyncio
async def test_release_removes_pending_publication_tokens_for_all_agent_runs() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._result_write_tokens.update({
        (record.agent_id, "old"): object(),
        (record.agent_id, "current"): object(),
        ("other", "run"): object(),
    })
    registry._latest_run_ids[record.agent_id] = "current"

    await registry.release_agent(record.agent_id)

    assert not [
        key for key in registry._result_write_tokens if key[0] == record.agent_id
    ]
    assert record.agent_id not in registry._latest_run_ids
    assert ("other", "run") in registry._result_write_tokens
    cast(AsyncMock, record.runtime.close).assert_awaited_once()


@pytest.mark.asyncio
async def test_reaper_batch_evicts_all_required_victims_despite_close_failure() -> None:
    clock = _ManualClock(10)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=20, cap=1, wakeup=wakeup)
    records = [
        _idle_record(agent_id, clock=clock)
        for agent_id in ("agent-a", "agent-b", "agent-c")
    ]
    for offset, record in enumerate(records):
        record.idle_since = float(offset)
        run = RunRecord(
            run_id=f"{record.agent_id}-run",
            agent_id=record.agent_id,
            profile=record.profile,
            status=RunStatus.COMPLETED,
            completion_task=asyncio.get_running_loop().create_future(),
            result=_stored_result(record.agent_id, f"{record.agent_id}-run").result,
        )
        record.run_history.append(run)
        record.latest_run_id = run.run_id
        registry._agent_records[record.agent_id] = record
        registry._children[record.session_id] = record.runtime
        registry._result_store[(record.agent_id, run.run_id)] = _stored_result(
            record.agent_id, run.run_id
        )
    records[0].runtime.close = AsyncMock(side_effect=RuntimeError("close failed"))

    await _rearm(registry)
    assert await wakeup.arrivals.get() == 0
    wakeup.gates[0].set()
    reaper = registry._reaper_task
    assert reaper is not None
    await reaper

    assert set(registry._evicted_agents) == {"agent-a", "agent-b"}
    assert set(registry._agent_records) == {"agent-c"}
    cast(AsyncMock, records[0].runtime.close).assert_awaited_once()
    cast(AsyncMock, records[1].runtime.close).assert_awaited_once()
    cast(AsyncMock, records[2].runtime.close).assert_not_awaited()
    notifications = cast(AsyncMock, registry._notify_agents).await_args_list
    assert {
        eviction.agent_id for call in notifications for eviction in call.args[1]
    } == {"agent-a", "agent-b"}
    evicted_result = await registry.get_agent_result("agent-a", "agent-a-run")
    assert evicted_result is not None and evicted_result.completed
    assert wakeup.calls[-1] == 12
    await registry.drain_children()


async def _real_reused_background() -> tuple[
    SessionRuntimeRegistry, Any, AgentRecord, TaskArgs, InvokeContext
]:
    """Build a retained child with real turn controllers for launch rollback tests."""
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    child = await AgentRuntimeFactory().create_child(parent, "worker")
    registry = _retention_registry(_ManualClock())
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    runtime = registry._build_child_runtime(child)
    record = AgentRecord(
        agent_id="agent-1",
        profile="worker",
        session_id=child.session_id,
        runtime=runtime,
        root_generation=parent._session_generation,
        initial_task_summary="initial",
        state=_AgentState.IDLE,
        idle_since=0,
        last_task_summary="initial",
    )
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = runtime
    return (
        registry,
        parent,
        record,
        TaskArgs(
            task="retry", agent="worker", agent_id=record.agent_id, background=True
        ),
        InvokeContext(tool_call_id="retry", session_id=parent.session_id),
    )


@pytest.mark.asyncio
async def test_reused_projection_linkage_failure_never_prepares_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock(side_effect=RuntimeError("link failed"))
    start = MagicMock(wraps=record.runtime.turns.start)
    monkeypatch.setattr(record.runtime.turns, "start", start)
    try:
        with pytest.raises(RuntimeError, match="link failed"):
            await _background_result(registry, args, context)
        start.assert_not_called()
        assert record.state is _AgentState.IDLE
        assert record.idle_since == 0
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reused_start_failure_aborts_real_prepared_turn_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    original_start = record.runtime.turns.start
    failing_start = MagicMock(side_effect=RuntimeError("start failed"))

    def prepare_then_fail(*values: Any, **kwargs: Any) -> tuple[Any, Any]:
        response, action = original_start(*values, **kwargs)
        action._run = failing_start  # type: ignore[attr-defined]
        return response, action

    monkeypatch.setattr(record.runtime.turns, "start", prepare_then_fail)
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            await _background_result(registry, args, context)
        failing_start.assert_called_once()
        assert record.runtime.execution.active is None
        assert record.runtime.turns._pending_start is None
        assert record.runtime.turns.active_turn is None
        assert record.runtime.turns._active_task is None
        assert not registry._monitor_tasks
        monkeypatch.setattr(record.runtime.turns, "start", original_start)
        result = await _background_result(registry, args, context)
        assert result.run_id is not None
        run = record.current_run
        assert run is not None
        await run.completion_task
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_monitor_creation_failure_aborts_prepared_turn_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, args, context = await _real_reused_background()
    original_create_task = asyncio.create_task

    def fail_background_monitor(
        coro: Any, *, name: str | None = None, **kwargs: Any
    ) -> Any:
        if name is not None and name.startswith("vibe-subagent-background:"):
            raise RuntimeError("monitor setup failed")
        return original_create_task(coro, name=name, **kwargs)

    monkeypatch.setattr(asyncio, "create_task", fail_background_monitor)
    try:
        with pytest.raises(RuntimeError, match="monitor setup failed"):
            await _background_result(registry, args, context)
        assert record.runtime.execution.active is None
        assert record.runtime.turns._pending_start is None
        assert record.runtime.turns.active_turn is None
        assert record.runtime.turns._active_task is None
        assert record.runtime.turns._event_sink is None
        assert record.current_run is None
        assert record.latest_run_id is None
        assert record.idle_since == 0
        assert not registry._monitor_tasks
        monkeypatch.setattr(asyncio, "create_task", original_create_task)
        result = await _background_result(registry, args, context)
        assert result.run_id is not None
        run = record.current_run
        assert run is not None
        await cast(asyncio.Future[Any], run.completion_task)
    finally:
        monkeypatch.setattr(asyncio, "create_task", original_create_task)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_turn_task_creation_failure_closes_run_and_aborts_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _registry, parent, record, _args, _context = await _real_reused_background()
    turns = record.runtime.turns
    captured: list[Any] = []

    def fail_create_task(coro: Any, **_kwargs: Any) -> Any:
        captured.append(coro)
        raise RuntimeError("task creation failed")

    original_create_task = asyncio.create_task
    monkeypatch.setattr(asyncio, "create_task", fail_create_task)
    try:
        _response, action = turns.start(
            TurnStartParams(
                session_id=record.session_id, message=[TextContentBlock(text="work")]
            )
        )
        with pytest.raises(RuntimeError, match="task creation failed"):
            action()
        assert len(captured) == 1
        assert captured[0].cr_frame is None
        assert turns._pending_start is None
        assert turns.active_turn is None
        assert turns._active_task is None
        assert record.runtime.execution.active is None
    finally:
        monkeypatch.setattr(asyncio, "create_task", original_create_task)
        await parent.aclose()


@pytest.mark.asyncio
async def test_prelaunch_registry_lock_cancellation_aborts_real_prepared_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    prepared = asyncio.Event()
    final_lock_entered = asyncio.Event()
    release_final_lock = asyncio.Event()
    original_start_child_turn = SessionRuntimeRegistry._start_child_turn

    def signal_prepared(*values: Any) -> tuple[str, Any]:
        result = original_start_child_turn(*values)
        prepared.set()
        return result

    class GatedRegistryLock:
        def __init__(self, lock: asyncio.Lock) -> None:
            self.lock = lock
            self.entries = 0

        async def __aenter__(self) -> None:
            await self.lock.acquire()
            self.entries += 1
            if self.entries == 3:
                final_lock_entered.set()
                self.lock.release()
                await release_final_lock.wait()
                await self.lock.acquire()

        async def __aexit__(self, *exc: object) -> None:
            self.lock.release()

    monkeypatch.setattr(SessionRuntimeRegistry, "_start_child_turn", signal_prepared)
    registry._registry_lock = GatedRegistryLock(registry._registry_lock)  # type: ignore[assignment]
    launch = asyncio.create_task(_background_result(registry, args, context))
    try:
        await final_lock_entered.wait()
        assert prepared.is_set()
        launch.cancel()
        release_final_lock.set()
        with pytest.raises(asyncio.CancelledError):
            await launch
        assert record.runtime.execution.active is None
        assert record.runtime.turns._pending_start is None
        assert record.runtime.turns.active_turn is None
        assert record.runtime.turns._active_task is None
        assert not registry._monitor_tasks
    finally:
        release_final_lock.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_release_cancellation_waits_for_cancelled_monitor_cleanup() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    cancelled = asyncio.Event()
    settle = asyncio.Event()
    close_started = asyncio.Event()
    close_release = asyncio.Event()

    async def monitor() -> None:
        try:
            await asyncio.Future[None]()
        except asyncio.CancelledError:
            cancelled.set()
            await settle.wait()
            raise

    async def close() -> None:
        close_started.set()
        await close_release.wait()

    task = asyncio.create_task(monitor())
    run = RunRecord("run-1", "agent-1", "worker", RunStatus.RUNNING, task)
    record.current_run = run
    record.runtime.close = close
    cast(Any, record.runtime.turns).active_turn = None
    record.runtime.turns.close = AsyncMock()
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._monitor_tasks.add(task)
    release = asyncio.create_task(registry.release_agent(record.agent_id))
    await cancelled.wait()
    release.cancel()
    assert registry._teardown_tasks
    assert record.agent_id in registry._suppressed_notifications
    assert not close_started.is_set()
    settle.set()
    await close_started.wait()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await release
    await registry.drain_children()
    assert not registry._suppressed_notifications
    assert registry._children == {}


@pytest.mark.asyncio
async def test_repeated_release_cancellation_still_closes_once() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    close_calls = 0

    async def close() -> None:
        nonlocal close_calls
        close_calls += 1
        close_started.set()
        await close_release.wait()

    record.runtime.close = close
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    release = asyncio.create_task(registry.release_agent(record.agent_id))
    await close_started.wait()
    release.cancel()
    await asyncio.sleep(0)
    release.cancel()
    close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await release
    await registry.drain_children()
    assert close_calls == 1
    assert registry._teardown_tasks == set()


@pytest.mark.asyncio
async def test_released_retired_results_are_invalidated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([
        [mock_llm_chunk(content="first result")],
        [mock_llm_chunk(content="reused result")],
    ])
    first_started, first_release = child_backend.add_gate()
    reused_started, _reused_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry, parent, record, args, context = await _real_reused_background()
    record.root_generation = 1
    parent._session_generation = 1
    registry._generation_identity = (parent.session_id, 1)
    first = await _background_result(registry, args, context)
    assert first.run_id is not None
    first_run = record.current_run
    assert first_run is not None
    await first_started.wait()
    first_release.set()
    await cast(asyncio.Task[Any], first_run.completion_task)
    registry._emit_agents_update = AsyncMock()

    original_wait = record.runtime.turns.wait_for_operation
    close_calls = 0
    original_close = type(record.runtime).close
    cancellation_acknowledged = asyncio.Event()
    resume_finalization = asyncio.Event()

    async def counted_close(runtime: Any) -> None:
        nonlocal close_calls
        if runtime is record.runtime:
            close_calls += 1
        await original_close(runtime)

    monkeypatch.setattr(type(record.runtime), "close", counted_close)

    async def gate_retiring_monitor(turn_id: str) -> Any:
        try:
            return await original_wait(turn_id)
        except asyncio.CancelledError:
            cancellation_acknowledged.set()
            await resume_finalization.wait()
            raise

    monkeypatch.setattr(
        record.runtime.turns, "wait_for_operation", gate_retiring_monitor
    )
    reused = await _background_result(registry, args, context)
    assert reused.run_id is not None
    reused_run = record.current_run
    assert reused_run is not None
    await reused_started.wait()
    cast(AsyncMock, registry._emit_agents_update).reset_mock()

    parent._session_generation = 2
    registry.begin_root_generation()
    await cancellation_acknowledged.wait()
    assert record.agent_id not in registry._agent_records
    await registry.release_agent(record.agent_id)
    resume_finalization.set()
    await asyncio.gather(*registry._teardown_tasks)

    for run_id in (first.run_id, reused.run_id, None):
        with pytest.raises((UnknownAgentError, AgentResultExpiredError)):
            await registry.get_agent_result(record.agent_id, run_id)
    assert not [key for key in registry._result_store if key[0] == record.agent_id]
    assert not [
        key for key in registry._result_write_tokens if key[0] == record.agent_id
    ]
    assert not [key for key in registry._expired_results if key[0] == record.agent_id]
    cast(AsyncMock, registry._emit_agents_update).assert_not_awaited()
    assert close_calls == 1
    assert record.runtime.turns._event_sink is None
    await registry.drain_children()
    await parent.aclose()


@pytest.mark.asyncio
async def test_retirement_preserves_unreleased_terminal_results() -> None:
    registry, root = _reaper_registry(_ManualClock(), generation=1)
    record = _idle_record("agent-1", clock=_ManualClock())
    record.root_generation = 1
    result = _stored_result(record.agent_id, "run-1", generation=1)
    record.latest_run_id = result.run_id
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    registry._result_store[(record.agent_id, result.run_id)] = result
    registry._latest_run_ids[record.agent_id] = result.run_id

    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    await asyncio.gather(*registry._teardown_tasks)

    assert (
        await registry.get_agent_result(record.agent_id, result.run_id) == result.result
    )
    assert await registry.get_agent_result(record.agent_id) == result.result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "factory",
        "persistence",
        "metadata",
        "projection",
        "ensure_lock",
        "registry_lock",
    ],
)
@pytest.mark.parametrize("identity_change", ["session_id", "generation"])
async def test_background_creation_fences_live_identity_at_each_admission_boundary(
    boundary: str, identity_change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every awaited creation boundary rejects a changed live root identity."""
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    root.turns._projector = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()
    original_factory = registry._runtime_factory.create_child

    async def gate(awaitable):
        result = await awaitable
        entered.set()
        await release.wait()
        return result

    if boundary == "factory":

        async def create_child(
            parent: AgentLoop,
            agent_name: str,
            *,
            session_id: str | None = None,
            session_dir: Path | None = None,
        ) -> AgentLoop:
            return await gate(
                original_factory(
                    parent, agent_name, session_id=session_id, session_dir=session_dir
                )
            )

        monkeypatch.setattr(registry._runtime_factory, "create_child", create_child)
    elif boundary == "persistence":
        original = AgentLoop.persist_empty_session

        async def persist(child: AgentLoop) -> None:
            await gate(original(child))

        monkeypatch.setattr(AgentLoop, "persist_empty_session", persist)
    elif boundary == "metadata":
        original = AgentLoop.record_child_session

        async def record(
            parent: AgentLoop, child: AgentLoop, tool_call_id: str, agent_name: str
        ) -> None:
            await gate(original(parent, child, tool_call_id, agent_name))

        monkeypatch.setattr(AgentLoop, "record_child_session", record)
    elif boundary == "projection":
        original = TurnController.link_subagent

        async def link(
            turns: TurnController, tool_call_id: str, child_session_id: str
        ) -> None:
            await gate(original(turns, tool_call_id, child_session_id))

        monkeypatch.setattr(TurnController, "link_subagent", link)
    else:
        lock_name = (
            "_ensure_child_lock" if boundary == "ensure_lock" else "_registry_lock"
        )
        lock = getattr(registry, lock_name)

        class GatedLock:
            async def __aenter__(self) -> None:
                await lock.acquire()
                entered.set()
                await release.wait()

            async def __aexit__(self, *exc: object) -> None:
                lock.release()

        setattr(registry, lock_name, GatedLock())

    launch = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(task="work", agent="worker", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        if identity_change == "session_id":
            parent.session_id = f"{parent.session_id}-changed"
        else:
            parent._session_generation += 1
        release.set()
        with pytest.raises(RuntimeError, match="admission changed"):
            await launch
        assert registry._agent_records == {}
        assert registry._children == {}
        assert registry._child_links == {}
        assert registry._monitor_tasks == set()
    finally:
        release.set()
        if not launch.done():
            launch.cancel()
            await asyncio.gather(launch, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_clear_history_handoff_preserves_stale_completion_and_new_generation_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completion in the real clear handoff is retained but never published."""
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    child_backend = GatedSequenceBackend([
        [mock_llm_chunk(content="old generation result")],
        [mock_llm_chunk(content="new generation result")],
    ])
    old_started, old_release = child_backend.add_gate()
    new_started, new_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="", tool_calls=[_task_call(background=True)])],
            [mock_llm_chunk(content="launch complete")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[_task_call(background=True, tool_call_id="task-2")],
                )
            ],
            [mock_llm_chunk(content="new launch complete")],
            [mock_llm_chunk(content="new completion noted")],
        ]),
        enable_streaming=True,
    )
    parent.config.subagents.idle_ttl_seconds = 10
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    registry._clock = clock
    registry._wakeup = wakeup
    identity_changed = asyncio.Event()
    resume_clear = asyncio.Event()
    original_clear_history = parent.clear_history

    async def gated_clear_history() -> None:
        await original_clear_history()
        identity_changed.set()
        await resume_clear.wait()

    monkeypatch.setattr(parent, "clear_history", gated_clear_history)
    try:
        await _consume(session.act("launch old generation"))
        await old_started.wait()
        old_record = registry._agent_records["agent-1"]
        old_run = old_record.current_run
        assert old_run is not None and isinstance(old_run.completion_task, asyncio.Task)
        old_monitor = old_run.completion_task
        old_runtime = old_record.runtime
        old_close_calls = 0
        original_runtime_close = type(old_runtime).close

        async def counted_close(runtime: Any) -> None:
            nonlocal old_close_calls
            if runtime is old_runtime:
                old_close_calls += 1
            await original_runtime_close(runtime)

        monkeypatch.setattr(type(old_runtime), "close", counted_close)

        clear = asyncio.create_task(session.clear_history())
        await identity_changed.wait()
        old_release.set()
        await old_monitor

        old_result = await registry.get_agent_result(
            old_record.agent_id, old_run.run_id
        )
        assert old_result is not None
        assert old_result.response == "old generation result"
        assert parent._pending_injected_messages == []
        root = registry._root
        assert root is not None and root.turns.active_turn is None

        resume_clear.set()
        await clear
        teardown_tasks = set(registry._teardown_tasks)
        await asyncio.gather(*teardown_tasks)
        assert old_close_calls == 1
        assert (
            await registry.get_agent_result(old_record.agent_id, old_run.run_id)
            == old_result
        )

        await _consume(session.act("launch new generation"))
        await new_started.wait()
        new_record = (await registry.check_agents())[0]
        assert new_record.current_run_id is not None
        new_agent_id = new_record.agent_id
        new_live_record = registry._agent_records[new_agent_id]
        new_run = new_live_record.current_run
        assert new_run is not None and new_run.run_id == new_record.current_run_id
        assert isinstance(new_run.completion_task, asyncio.Task)
        new_release.set()
        await new_run.completion_task
        assert await wakeup.arrivals.get() == 10
        clock.advance(10)
        wakeup.gates[0].set()
        reaper = registry._reaper_task
        assert reaper is not None
        await reaper
        assert new_agent_id not in registry._agent_records
        assert registry._evicted_agents[new_agent_id].summary.idle_seconds == 10
    finally:
        old_release.set()
        new_release.set()
        resume_clear.set()
        await session.close()


_DYNAMIC_PROFILE = AgentProfile(
    name="dynamic",
    display_name="Dynamic",
    description="Dynamic launch acceptance profile",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
    instructions="Retained role instruction.",
    idle_ttl_seconds=300,
)


def _dynamic_config():
    return build_test_vibe_config(**{
        "active_model": "small",
        "system_prompt_id": "tests",
        "include_project_context": False,
        "include_prompt_detail": False,
        "enabled_tools": ["task", "read_file"],
        "tools": {
            "task": {"permission": "always"},
            "read_file": {"permission": "always"},
        },
        "providers": [
            {"name": "one", "api_base": "https://one.test", "backend": "generic"},
            {"name": "two", "api_base": "https://two.test", "backend": "generic"},
        ],
        "models": [
            {
                "name": "small",
                "alias": "small",
                "provider": "one",
                "thinking": "off",
                "supported_thinking_levels": ["off", "high"],
            },
            {
                "name": "large",
                "alias": "large",
                "provider": "one",
                "thinking": "high",
                "supported_thinking_levels": ["off", "high"],
            },
            {
                "name": "other",
                "alias": "other",
                "provider": "two",
                "thinking": "high",
                "supported_thinking_levels": ["off", "high"],
            },
        ],
    })


async def _dynamic_registry(monkeypatch: pytest.MonkeyPatch):
    """Real retained-agent machinery with inspectable fake provider backends."""
    backends: list[tuple[str, FakeBackend]] = []

    def create_backend(*, provider, **_kwargs):
        backend = FakeBackend([mock_llm_chunk(content=provider.name)])
        backends.append((provider.name, backend))
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=_dynamic_config(), backend=FakeBackend())
    parent.agent_manager._discovered[_DYNAMIC_PROFILE.name] = _DYNAMIC_PROFILE
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="dynamic", session_id=parent.session_id)
    first = await _background_result(
        registry,
        TaskArgs(
            task="first",
            agent="dynamic",
            background=True,
            config=LaunchConfig(
                model="small",
                thinking="off",
                enabled_tools=["read_file"],
                instructions="Retained role instruction.",
            ),
        ),
        context,
    )
    assert first.agent_id is not None and first.run_id is not None
    await registry.wait_for_agent(first.agent_id, first.run_id)
    return (
        registry,
        parent,
        registry._agent_records[first.agent_id],
        context,
        first,
        backends,
    )


@pytest.mark.asyncio
async def test_real_child_creation_carries_profile_idle_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, _context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        assert record.idle_ttl_seconds == 300
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_ttl_seconds", [300, 0])
async def test_profile_ttl_overrides_zero_global_retention_at_completion(
    monkeypatch: pytest.MonkeyPatch, idle_ttl_seconds: int
) -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, parent, _notifications = await _zero_retention_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    parent.agent_manager._discovered["persistent"] = AgentProfile(
        name="persistent",
        display_name="Persistent",
        description="Retained under zero global retention.",
        safety=AgentSafety.NEUTRAL,
        agent_type=AgentType.SUBAGENT,
        idle_ttl_seconds=idle_ttl_seconds,
    )
    registry._clock = clock
    registry._wakeup = wakeup
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="persist", agent="persistent", background=True),
            InvokeContext(tool_call_id="persistent", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.idle_ttl_seconds == idle_ttl_seconds
        completion = record.current_run.completion_task if record.current_run else None
        assert isinstance(completion, asyncio.Task)
        await completion
        assert record.state is _AgentState.IDLE

        if idle_ttl_seconds == 0:
            assert registry._reaper_task is None
            assert launch.agent_id in registry._agent_records
        else:
            assert await wakeup.arrivals.get() == 300
            clock.advance(300)
            wakeup.gates[0].set()
            reaper = registry._reaper_task
            assert reaper is not None
            await reaper
            assert launch.agent_id not in registry._agent_records
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retask_escalation_reuses_handle_history_and_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        child = record.runtime.agent_loop
        initial_summary = (await registry.check_agents())[0]
        assert (
            initial_summary.effective_model,
            initial_summary.effective_thinking,
        ) == ("one/default/small", "off")
        history_before = len(child.messages)
        second = await _background_result(
            registry,
            TaskArgs(
                task="second",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(
                    model="large", thinking="high", disabled_tools=["read_file"]
                ),
            ),
            context,
        )
        assert second.agent_id == first.agent_id and second.run_id != first.run_id
        await registry.wait_for_agent(record.agent_id, second.run_id)
        assert registry._agent_records[record.agent_id].runtime.agent_loop is child
        assert len(child.messages) > history_before
        assert child.config.active_model == "large"
        assert child.config.get_active_model().thinking == "high"
        summary = (await registry.check_agents())[0]
        assert (summary.effective_model, summary.effective_thinking) == (
            "one/default/large",
            "high",
        )
        assert "read_file" not in child.tool_manager.available_tools
        assert (
            await registry.get_agent_result(record.agent_id, first.run_id) is not None
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retask_cross_provider_uses_replacement_backend_and_converts_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, _first, backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        old_backend = record.runtime.agent_loop.backend
        result = await _background_result(
            registry,
            TaskArgs(
                task="cross provider",
                agent_id=record.agent_id,
                background=True,
                config=cast(LaunchConfig, {"model": "other", "thinking": "high"}),
            ),
            context,
        )
        assert result.run_id is not None
        await registry.wait_for_agent(record.agent_id, result.run_id)
        child = record.runtime.agent_loop
        assert child.backend is not old_backend
        assert child.config.active_model == "other"
        assert backends[-1][0] == "two/default"
        assert backends[-1][
            1
        ].requests_messages  # The retained transcript reached the new backend.
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retask_refreshes_model_prompt_but_retains_frozen_role_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    child = record.runtime.agent_loop
    skill_manager = child.skill_manager
    mcp_pool = child._mcp_pool
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="refresh",
                agent_id=record.agent_id,
                background=True,
                config=cast(LaunchConfig, {"model": "large"}),
            ),
            context,
        )
        assert result.run_id is not None
        await registry.wait_for_agent(record.agent_id, result.run_id)
        prompt = str(child.messages[0].content)
        assert "Retained role instruction." in prompt
        assert "large" in prompt
        assert child.skill_manager is skill_manager
        assert child._mcp_pool is mcp_pool
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_concurrent_retask_accepts_one_run_and_preserves_failed_attempt_pointers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, first, _backends = await _dynamic_registry(
        monkeypatch
    )
    entered = asyncio.Event()
    release = asyncio.Event()
    original = record.runtime.agent_loop.prepare_launch_reconfiguration

    async def gated(candidate, **kwargs):
        entered.set()
        await release.wait()
        return await original(candidate, **kwargs)

    monkeypatch.setattr(
        record.runtime.agent_loop, "prepare_launch_reconfiguration", gated
    )
    attempt = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(
                task="one",
                agent_id=record.agent_id,
                background=True,
                config=cast(LaunchConfig, {"model": "large"}),
            ),
            context,
        )
    )
    try:
        await entered.wait()
        with pytest.raises(AgentBusyError):
            await _background_result(
                registry,
                TaskArgs(
                    task="two",
                    agent_id=record.agent_id,
                    background=True,
                    config=cast(LaunchConfig, {"model": "other"}),
                ),
                context,
            )
        assert record.latest_run_id == first.run_id
        release.set()
        accepted = await attempt
        assert accepted.run_id is not None and accepted.run_id != first.run_id
        await registry.wait_for_agent(record.agent_id, accepted.run_id)
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["prepare", "turn_start", "publication"])
async def test_failed_retask_rolls_back_configuration_resources_and_idle_ttl(
    boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, context, first, _backends = await _dynamic_registry(
        monkeypatch
    )
    child = record.runtime.agent_loop
    old_manager, old_pool, old_config, old_idle = (
        child.tool_manager,
        child.mcp_registry,
        child.config,
        record.idle_since,
    )
    if boundary == "prepare":
        monkeypatch.setattr(
            child,
            "prepare_launch_reconfiguration",
            AsyncMock(side_effect=RuntimeError("prepare failed")),
        )
    elif boundary == "turn_start":
        monkeypatch.setattr(
            SessionRuntimeRegistry,
            "_start_child_turn",
            MagicMock(side_effect=RuntimeError("turn start failed")),
        )
    else:
        monkeypatch.setattr(
            child,
            "publish_launch_reconfiguration",
            MagicMock(side_effect=RuntimeError("publish failed")),
        )
    try:
        with pytest.raises(RuntimeError):
            await _background_result(
                registry,
                TaskArgs(
                    task="fail",
                    agent_id=record.agent_id,
                    background=True,
                    config=cast(LaunchConfig, {"model": "large"}),
                ),
                context,
            )
        assert record.state is _AgentState.IDLE and record.idle_since == old_idle
        assert record.current_run is None and record.latest_run_id == first.run_id
        assert child.config is old_config and child.tool_manager is old_manager
        assert child.mcp_registry is old_pool
        summary = (await registry.check_agents())[0]
        assert (summary.effective_model, summary.effective_thinking) == (
            "one/default/small",
            "off",
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["disabled", "never"])
async def test_direct_runner_enforces_effective_task_admission(
    monkeypatch: pytest.MonkeyPatch, admission: str
) -> None:
    registry, parent, _record, _context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    create_child = AsyncMock()
    monkeypatch.setattr(registry._runtime_factory, "create_child", create_child)
    if admission == "disabled":
        denied_config = parent.tool_manager._config.model_copy(
            update={"disabled_tools": ["task"]}
        )
    else:
        tools = dict(parent.tool_manager._config.tools)
        tools["task"] = {"permission": ToolPermission.NEVER.value}
        denied_config = parent.tool_manager._config.model_copy(update={"tools": tools})
    monkeypatch.setattr(parent.tool_manager, "_config_getter", lambda: denied_config)
    try:
        with pytest.raises(ToolPermissionError, match=f"Task tool .*{admission}"):
            await _background_result(
                registry,
                TaskArgs(task="blocked", agent="dynamic", background=False),
                InvokeContext(
                    tool_call_id=f"direct-{admission}", session_id=parent.session_id
                ),
            )
        create_child.assert_not_awaited()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_dynamic_launch_uses_resolver_and_rejects_retained_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        with pytest.raises(ValueError, match="agent_id"):
            await _background_result(
                registry,
                TaskArgs(task="no reuse", agent_id=record.agent_id, background=False),
                context,
            )
        result = await _background_result(
            registry,
            TaskArgs(
                task="foreground",
                agent="dynamic",
                background=False,
                config=cast(LaunchConfig, {"model": "large", "thinking": "high"}),
            ),
            InvokeContext(
                tool_call_id="dynamic-foreground", session_id=parent.session_id
            ),
        )
        assert result.agent_id is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_turn_start_failure_rolls_back_persisted_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        registry,
        parent,
        _record,
        _context,
        _first,
        _backends,
    ) = await _persistent_dynamic_registry(tmp_path, monkeypatch)
    children_before = set(registry._children)
    metadata = parent.session_logger.session_metadata
    assert metadata is not None
    links_before = list(metadata.child_sessions)
    child_dirs: list[Path] = []
    original_create_child = registry._runtime_factory.create_child

    async def capture_created_child(*args, **kwargs):
        child = await original_create_child(*args, **kwargs)
        assert child.session_logger.session_dir is not None
        child_dirs.append(child.session_logger.session_dir)
        return child

    monkeypatch.setattr(
        registry._runtime_factory, "create_child", capture_created_child
    )
    monkeypatch.setattr(
        SessionRuntimeRegistry,
        "_start_child_turn",
        MagicMock(side_effect=RuntimeError("foreground turn start failed")),
    )
    try:
        with pytest.raises(RuntimeError, match="foreground turn start failed"):
            await _background_result(
                registry,
                TaskArgs(task="fail", agent="dynamic", background=False),
                InvokeContext(
                    tool_call_id="foreground-failure", session_id=parent.session_id
                ),
            )
        assert set(registry._children) == children_before
        assert metadata.child_sessions == links_before
        assert len(child_dirs) == 1
        assert not child_dirs[0].exists()
        with pytest.raises(ValueError, match="Session metadata not found"):
            SessionLoader.load_metadata(child_dirs[0])
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_omitted_config_retask_revalidates_parent_authority_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, first, _backends = await _dynamic_registry(
        monkeypatch
    )
    child = record.runtime.agent_loop
    old_config = child.config
    original = child.prepare_launch_reconfiguration

    async def tighten_during_prepare(candidate, **kwargs):
        parent._authority_revision += 1
        return await original(candidate, **kwargs)

    monkeypatch.setattr(child, "prepare_launch_reconfiguration", tighten_during_prepare)
    try:
        with pytest.raises(Exception, match="authority changed"):
            await _background_result(
                registry,
                TaskArgs(
                    task="omitted config", agent_id=record.agent_id, background=True
                ),
                context,
            )
        assert child.config is old_config
        assert record.latest_run_id == first.run_id
    finally:
        await registry.drain_children()
        await parent.aclose()


async def _persistent_dynamic_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the real retained-agent path with a real child-session directory."""
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )

    def create_backend(*, provider, **_kwargs):
        return FakeBackend([mock_llm_chunk(content=provider.name)])

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(
        config=_dynamic_config().model_copy(update={"session_logging": logging}),
        backend=FakeBackend(),
    )
    parent.agent_manager._discovered[_DYNAMIC_PROFILE.name] = _DYNAMIC_PROFILE
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="dynamic", session_id=parent.session_id)
    first = await _background_result(
        registry,
        TaskArgs(
            task="first",
            agent="dynamic",
            background=True,
            config=LaunchConfig(
                model="small",
                thinking="off",
                enabled_tools=["read_file"],
                instructions="Retained role instruction.",
            ),
        ),
        context,
    )
    assert first.agent_id is not None and first.run_id is not None
    await registry.wait_for_agent(first.agent_id, first.run_id)
    return registry, parent, registry._agent_records[first.agent_id], context, first, []


@pytest.mark.asyncio
async def test_concurrent_agent_id_issuance_reserves_unique_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    await parent.persist_empty_session()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    entered = 0
    both_persisting = asyncio.Event()
    release = asyncio.Event()

    async def gated_persist(_field: str, _value: object) -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_persisting.set()
        await release.wait()

    monkeypatch.setattr(parent.session_logger, "_persist_metadata_field", gated_persist)
    try:
        issued = asyncio.gather(
            registry._issue_agent_id(root), registry._issue_agent_id(root)
        )
        await asyncio.wait_for(both_persisting.wait(), timeout=1)
        release.set()
        assert await issued == ["agent-1", "agent-2"]
    finally:
        release.set()
        await parent.aclose()


@pytest.mark.asyncio
async def test_resume_preserves_sparse_released_agent_id_high_water_mark(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    await parent.persist_empty_session()
    session_id = parent.session_id
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    try:
        assert await registry._issue_agent_id(root) == "agent-1"
        # Simulate sparse history after both issued handles have been released.
        registry._next_agent_seq = 4
        assert await registry._issue_agent_id(root) == "agent-5"
        assert registry._agent_records == {}
    finally:
        await parent.aclose()

    async def resume_registry() -> tuple[
        SessionRuntimeRegistry, AgentLoop, InvokeContext
    ]:
        resumed_parent = build_test_agent_loop(
            config=_config(logging), backend=FakeBackend()
        )
        await AgentRuntimeFactory().resume_root(resumed_parent, session_id)
        resumed_registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
        resumed_registry.bind_root(
            resumed_registry._build_child_runtime(resumed_parent)
        )
        return (
            resumed_registry,
            resumed_parent,
            InvokeContext(tool_call_id="resumed", session_id=resumed_parent.session_id),
        )

    resumed_registry, resumed_parent, resumed_context = await resume_registry()
    try:
        resumed_root = resumed_registry._root
        assert resumed_root is not None
        assert await resumed_registry._issue_agent_id(resumed_root) == "agent-6"
        for historical_agent_id in ("agent-1", "agent-5"):
            with pytest.raises(UnknownAgentError):
                await _background_result(
                    resumed_registry,
                    TaskArgs(
                        task="historical", agent_id=historical_agent_id, background=True
                    ),
                    resumed_context,
                )
    finally:
        await resumed_parent.aclose()

    resumed_registry, resumed_parent, _resumed_context = await resume_registry()
    try:
        resumed_root = resumed_registry._root
        assert resumed_root is not None
        assert await resumed_registry._issue_agent_id(resumed_root) == "agent-7"
    finally:
        await resumed_parent.aclose()


@pytest.mark.asyncio
async def test_resume_reconstructs_accumulated_launch_state_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        registry,
        parent,
        record,
        context,
        _first,
        _backends,
    ) = await _persistent_dynamic_registry(tmp_path, monkeypatch)
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        retask = await _background_result(
            registry,
            TaskArgs(
                task="raise thinking",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(thinking="high"),
            ),
            context,
        )
        assert retask.run_id is not None
        await registry.wait_for_agent(record.agent_id, retask.run_id)

        persisted = SessionLoader.load_metadata(child_dir).launch_config
        assert persisted is not None
        assert persisted.overrides.model == "small"
        assert persisted.overrides.thinking == "high"
        assert persisted.overrides.enabled_tools == ["read_file"]
        assert persisted.persona.instructions == "Retained role instruction."

        # A normal turn save must retain the committed dedicated envelope.
        await child.session_logger.save_interaction(
            child.messages,
            child.stats,
            child.config,
            child.tool_manager,
            _DYNAMIC_PROFILE,
        )
        assert SessionLoader.load_metadata(child_dir).launch_config == persisted

        child_id = child.session_id
        await registry.drain_children()
        resumed = await AgentRuntimeFactory().resume_child(
            parent, "dynamic", child_id, child_dir
        )
        try:
            await resumed.wait_until_ready()
            assert resumed.config.active_model == "small"
            assert resumed.config.get_active_model().thinking == "high"
            assert set(resumed.tool_manager.available_tools) == {"read_file"}
            assert "Retained role instruction." in str(resumed.messages[0].content)
        finally:
            await resumed.aclose()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            lambda metadata: metadata["launch_config"]["persona"].update(
                system_prompt_id="gone"
            ),
            "config.system_prompt_id",
        ),
        (
            lambda metadata: metadata["launch_config"].update(version=99),
            "launch_config",
        ),
        (lambda metadata: metadata.update(launch_config=None), "launch_config"),
    ],
)
async def test_ensure_child_fails_closed_for_invalid_persisted_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, field: str
) -> None:
    (
        registry,
        parent,
        record,
        _context,
        _first,
        _backends,
    ) = await _persistent_dynamic_registry(tmp_path, monkeypatch)
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    parent_dir = parent.session_logger.session_dir
    assert child_dir is not None and parent_dir is not None
    child_id = child.session_id
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        mutation(raw)
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        await registry.drain_children()

        # ensure_child is the real session-access boundary, not resume_child directly.
        metadata = parent.session_logger.session_metadata
        assert metadata is not None
        metadata.child_sessions = [
            ChildSessionLink(
                session_id=child_id,
                tool_call_id="persisted",
                agent="dynamic",
                relative_path=str(child_dir.relative_to(parent_dir)),
            )
        ]
        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == field
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_persisted_child_launch_error_is_reported_as_protocol_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    logging = parent.config.session_logging
    await registry.drain_children()
    await registry.close()
    await parent.aclose()

    base_config = _dynamic_config()
    resumed_config = base_config.model_copy(
        update={
            "active_model": "large",
            "allowed_models": ["large", "other"],
            "session_logging": logging,
        }
    )
    resumed_loop = build_test_agent_loop(config=resumed_config, backend=FakeBackend())
    resumed_loop.agent_manager._discovered[_DYNAMIC_PROFILE.name] = _DYNAMIC_PROFILE
    with pytest.raises(AppServerResponseError) as exc_info:
        await attach_test_app_server_session(
            start_test_app_server(resumed_loop), resume_session_id=parent.session_id
        )
    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_legacy_no_config_child_resume_uses_profile_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent (legacy) envelope is distinct from a malformed present one."""
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        raw.pop("launch_config")
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        child_id = child.session_id
        await registry.drain_children()

        assert await registry.ensure_child(child_id)
        resumed = registry._children[child_id].agent_loop
        assert resumed.launch_overrides is not None
        assert resumed.launch_overrides.model_fields_set == set()
        assert resumed.config.active_model == "small"
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rechecks_tightened_parent_model_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        child_id = child.session_id
        object.__setattr__(parent.config, "allowed_models", ["other"])
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "config.model"
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda raw: raw["launch_config"].update(unexpected=True), "launch_config"),
        (lambda raw: raw["launch_config"].update(overrides=[]), "launch_config"),
        (lambda raw: raw.update(launch_config="not-an-envelope"), "launch_config"),
    ],
)
async def test_ensure_child_rejects_present_malformed_envelope_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, field: str
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        mutation(raw)
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        child_id = child.session_id
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == field
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rejects_corrupt_present_envelope_as_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        (child_dir / "meta.json").write_text("{corrupt", encoding="utf-8")
        child_id = child.session_id
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "launch_config"
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("keep_envelope", [True, False])
async def test_ensure_child_rejects_deleted_profile_without_worker_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keep_envelope: bool
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        if not keep_envelope:
            raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
            raw.pop("launch_config")
            (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        child_id = child.session_id
        parent.agent_manager._discovered.pop("dynamic")
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "agent"
        assert "dynamic" in str(raised.value)
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rejects_removed_builtin_explore_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    parent_dir = parent.session_logger.session_dir
    assert child_dir is not None and parent_dir is not None
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        raw.pop("launch_config")
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        child_id = child.session_id
        metadata = parent.session_logger.session_metadata
        assert metadata is not None
        metadata.child_sessions = [
            ChildSessionLink(
                session_id=child_id,
                tool_call_id="persisted",
                agent="explore",
                relative_path=str(child_dir.relative_to(parent_dir)),
            )
        ]
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "agent"
        assert "explore" in str(raised.value)
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retask_accumulation_round_trips_and_failed_patch_keeps_disk_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, context, first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        first_patch = await _background_result(
            registry,
            TaskArgs(
                task="grant paths",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(
                    tools={"read_file": LaunchToolOverride(allowlist=["*.py"])}
                ),
            ),
            context,
        )
        assert first_patch.run_id is not None
        await registry.wait_for_agent(record.agent_id, first_patch.run_id)
        second_patch = await _background_result(
            registry,
            TaskArgs(
                task="change permission only",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(
                    tools={
                        "read_file": LaunchToolOverride(permission=ToolPermission.ASK)
                    }
                ),
            ),
            context,
        )
        assert second_patch.run_id is not None
        await registry.wait_for_agent(record.agent_id, second_patch.run_id)
        committed = SessionLoader.load_metadata(child_dir).launch_config
        assert committed is not None
        assert committed.overrides.tools is not None
        read_file = committed.overrides.tools["read_file"]
        assert read_file.allowlist == ["*.py"]
        assert read_file.permission == "ask"

        omitted = TaskArgs(
            task="no semantic patch", agent_id=record.agent_id, background=True
        )
        for args in (
            omitted,
            TaskArgs(
                task="no semantic patch",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(),
            ),
        ):
            result = await _background_result(registry, args, context)
            assert result.run_id is not None
            await registry.wait_for_agent(record.agent_id, result.run_id)
        assert SessionLoader.load_metadata(child_dir).launch_config == committed

        monkeypatch.setattr(
            child,
            "prepare_launch_reconfiguration",
            AsyncMock(side_effect=RuntimeError("fail")),
        )
        with pytest.raises(RuntimeError, match="fail"):
            await _background_result(
                registry,
                TaskArgs(
                    task="failed patch",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(thinking="high"),
                ),
                context,
            )
        assert SessionLoader.load_metadata(child_dir).launch_config == committed
        assert record.latest_run_id != first.run_id
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_preacceptance_retask_failure_keeps_prior_launch_envelope_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, context, first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    old_manager, old_config = child.tool_manager, child.config
    old_envelope = SessionLoader.load_metadata(child_dir).launch_config
    assert old_envelope is not None

    def fail_before_acceptance(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected preacceptance failure")

    monkeypatch.setattr(
        SessionRuntimeRegistry, "_start_child_turn", fail_before_acceptance
    )
    try:
        with pytest.raises(RuntimeError, match="injected preacceptance failure"):
            await _background_result(
                registry,
                TaskArgs(
                    task="failed before acceptance",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(model="large"),
                ),
                context,
            )
        assert record.state is _AgentState.IDLE
        assert record.current_run is None and record.latest_run_id == first.run_id
        assert child.config is old_config and child.tool_manager is old_manager
        assert SessionLoader.load_metadata(child_dir).launch_config == old_envelope
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_postacceptance_launch_envelope_write_failure_keeps_accepted_retask_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, context, first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    old_envelope = SessionLoader.load_metadata(child_dir).launch_config
    assert old_envelope is not None
    persist_metadata = SessionLogger.persist_metadata

    async def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected metadata failure after acceptance")

    monkeypatch.setattr(SessionLogger, "persist_metadata", fail_write)
    try:
        with pytest.raises(RuntimeError, match="injected metadata failure"):
            await _background_result(
                registry,
                TaskArgs(
                    task="accepted but dirty",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(model="large"),
                ),
                context,
            )
        accepted_run_id = record.latest_run_id
        assert accepted_run_id is not None and accepted_run_id != first.run_id
        assert child.config.active_model == "large"
        assert child.session_logger.session_metadata is not None
        in_memory = child.session_logger.session_metadata.launch_config
        assert in_memory is not None and in_memory.overrides.model == "large"
        assert SessionLoader.load_metadata(child_dir).launch_config == old_envelope

        # The next ordinary save repairs the dirty accepted envelope.
        monkeypatch.setattr(SessionLogger, "persist_metadata", persist_metadata)
        await child.session_logger.save_interaction(
            child.messages,
            child.stats,
            child.config,
            child.tool_manager,
            _DYNAMIC_PROFILE,
        )
        persisted = SessionLoader.load_metadata(child_dir).launch_config
        assert persisted is not None and persisted.overrides.model == "large"
        await registry.wait_for_agent(record.agent_id, accepted_run_id)
    finally:
        await registry.close()
        await parent.aclose()


def test_cross_provider_dedicated_compaction_model_is_catalog_resolved() -> None:
    config = _dynamic_config().model_copy(update={"compaction_model": "other"})
    assert config.get_compaction_model().provider == "two/default"


async def _foreground_registry(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend):
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


async def _foreground_result(
    registry: SessionRuntimeRegistry,
    parent: AgentLoop,
    tool_call_id: str = "foreground",
) -> TaskResult:
    return cast(
        TaskResult,
        [
            item
            async for item in registry.run(
                TaskArgs(task="foreground", agent="worker", background=False),
                InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id),
            )
        ][-1],
    )


@pytest.mark.asyncio
async def test_foreground_completions_detach_and_close_every_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    closed: list[SessionRuntime] = []
    original_close = SessionRuntime.close

    async def counted_close(runtime: SessionRuntime) -> None:
        closed.append(runtime)
        await original_close(runtime)

    monkeypatch.setattr(SessionRuntime, "close", counted_close)
    try:
        results = await asyncio.gather(
            *(
                _foreground_result(registry, parent, f"foreground-{index}")
                for index in range(3)
            )
        )
        assert all(result.completed for result in results)
        await registry.drain_children()
        assert registry._children == {}
        assert len(closed) == 3
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_terminal_generator_close_hands_off_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    stream = registry.run(
        TaskArgs(task="foreground", agent="worker", background=False),
        InvokeContext(tool_call_id="terminal-close", session_id=parent.session_id),
    )
    try:
        terminal = None
        async for item in stream:
            if isinstance(item, TaskResult):
                terminal = item
                break
        assert terminal is not None
        await stream.aclose()
        assert registry._children == {}
        assert registry._teardown_tasks
        await registry.drain_children()
        assert registry._teardown_tasks == set()
    finally:
        await stream.aclose()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_accepted_turn_failure_detaches_closes_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = GatedFailureBackend(RuntimeError("backend exploded"))
    registry, parent = await _foreground_registry(monkeypatch, backend)
    task = asyncio.create_task(_foreground_result(registry, parent))
    try:
        await backend.started.wait()
        backend.release.set()
        with pytest.raises(RuntimeError, match="backend exploded"):
            await task
        assert registry._children == {}
        await registry.drain_children()
        assert registry._teardown_tasks == set()
    finally:
        backend.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_repeated_foreground_wait_cancellation_tracks_one_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent = await _foreground_registry(monkeypatch, backend)
    task = asyncio.create_task(_foreground_result(registry, parent))
    try:
        await backend.started.wait()
        task.cancel()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert registry._children == {}
        assert len(registry._teardown_tasks) == 1
        backend.release.set()
        await registry.drain_children()
        assert registry._teardown_tasks == set()
    finally:
        backend.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_completion_racing_close_children_closes_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent = await _foreground_registry(monkeypatch, backend)
    close_calls = 0
    original_close = AgentLoop.aclose

    async def counted_close(loop: AgentLoop) -> None:
        nonlocal close_calls
        close_calls += 1
        await original_close(loop)

    monkeypatch.setattr(AgentLoop, "aclose", counted_close)
    task = asyncio.create_task(_foreground_result(registry, parent))
    try:
        await backend.started.wait()
        backend.release.set()
        await asyncio.gather(task, registry.close_children())
        await registry.drain_children()
        assert close_calls == 1
    finally:
        backend.release.set()
        await asyncio.gather(task, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_joins_blocked_foreground_teardown_before_rematerializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    original_close = SessionRuntime.close

    async def blocked_close(runtime: SessionRuntime) -> None:
        if runtime is not root:
            close_started.set()
            await close_release.wait()
        await original_close(runtime)

    monkeypatch.setattr(SessionRuntime, "close", blocked_close)
    try:
        result = await _foreground_result(registry, parent)
        child_session_id = next(iter(registry._stored_children))
        assert result.completed
        await asyncio.wait_for(close_started.wait(), timeout=1)

        rematerialize = asyncio.create_task(registry.ensure_child(child_session_id))
        await asyncio.sleep(0)
        assert not rematerialize.done()
        close_release.set()
        assert await asyncio.wait_for(rematerialize, timeout=1)
        child = registry._children[child_session_id]
        assert not child._closed
        assert (
            registry.public_state(child_session_id, history_limit=10).session.id
            == child_session_id
        )
    finally:
        close_release.set()
        await registry.close()
        await parent.aclose()


async def _zero_retention_registry(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> tuple[SessionRuntimeRegistry, AgentLoop, list[str]]:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    config = _config()
    config = config.model_copy(
        update={
            "subagents": config.subagents.model_copy(
                update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
            )
        }
    )
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    notifications: list[str] = []
    start = root.turns.start

    def record_notification(params: TurnStartParams):
        notifications.extend(
            block.text
            for block in params.message
            if isinstance(block, TextContentBlock)
        )
        return start(params)

    monkeypatch.setattr(root.turns, "start", record_notification)
    registry.bind_root(root)
    return registry, parent, notifications


async def _assert_zero_retention_outcome(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, *, cancelled: bool = False
) -> None:
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    runtime: SessionRuntime | None = None
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="zero retention", agent="worker", background=True),
            InvokeContext(tool_call_id="zero-retention", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        runtime = record.runtime
        completion = record.current_run.completion_task if record.current_run else None
        assert isinstance(completion, asyncio.Task)
        if cancelled:
            assert isinstance(backend, BlockingBackend)
            await asyncio.wait_for(backend.started.wait(), timeout=1)
            completion.cancel()
        await asyncio.gather(completion, return_exceptions=True)
        await asyncio.gather(*tuple(registry._teardown_tasks))

        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert result is not None and result.run_id == launch.run_id
        assert any("Background agent" in message for message in notifications)
        with pytest.raises(AgentEvictedError):
            await _background_result(
                registry,
                TaskArgs(
                    task="reuse",
                    agent="worker",
                    agent_id=launch.agent_id,
                    background=True,
                ),
                InvokeContext(tool_call_id="reuse", session_id=parent.session_id),
            )
        assert runtime._closed
    finally:
        if isinstance(backend, BlockingBackend):
            backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_zero_retention_success_publishes_result_then_evicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_zero_retention_outcome(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )


@pytest.mark.asyncio
async def test_zero_retention_failure_publishes_result_then_evicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_zero_retention_outcome(
        monkeypatch, ImmediateFailureBackend(RuntimeError("failed"))
    )


@pytest.mark.asyncio
async def test_zero_retention_cancellation_publishes_tombstone_then_evicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _assert_zero_retention_outcome(monkeypatch, BlockingBackend(), cancelled=True)


@pytest.mark.asyncio
async def test_foreground_teardown_fd_count_does_not_grow_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    try:
        counts: list[int] = []
        for batch in range(3):
            results = await asyncio.gather(
                *(
                    _foreground_result(registry, parent, f"fd-{batch}-{index}")
                    for index in range(10)
                )
            )
            assert all(result.completed for result in results)
            await asyncio.gather(*tuple(registry._teardown_tasks))
            assert registry._teardown_tasks == set()
            counts.append(len(os.listdir("/proc/self/fd")))
        assert max(counts) - min(counts) <= 5, counts
    finally:
        await registry.drain_children()
        await parent.aclose()


def _fan_out_snapshot(
    *, tag_members: list[str], disabled: set[str] | None = None
) -> CatalogSnapshot:
    disabled = disabled or set()
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test/first": {"api_base": "https://first.test", "backend": "generic"},
                "test/second": {
                    "api_base": "https://second.test",
                    "backend": "generic",
                },
            },
            "models": {
                "small": {
                    "disabled": "small" in disabled,
                    "deployments": [
                        {"provider": "test/first", "name": "small-wire"},
                        {"provider": "test/second", "name": "small-failover-wire"},
                    ],
                },
                "large": {
                    "disabled": "large" in disabled,
                    "deployments": [{"provider": "test/first", "name": "large-wire"}],
                },
                "missing": {
                    "deployments": [{"provider": "test/second", "name": "missing-wire"}]
                },
                "other": {
                    "disabled": "other" in disabled,
                    "deployments": [{"provider": "test/second", "name": "other-wire"}],
                },
            },
            "tags": {"panel": tag_members},
        }),
        "fan-out-test",
    )


async def _fan_out_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tag_members: list[str] | None = None,
    disabled: set[str] | None = None,
    allowed_models: list[str] | None = None,
    failures: set[str] | None = None,
) -> tuple[SessionRuntimeRegistry, AgentLoop, InvokeContext]:
    tag_members = tag_members or ["small", "large", "other"]
    disabled = disabled or set()
    failures = failures or set()
    snapshot = _fan_out_snapshot(tag_members=tag_members, disabled=disabled)
    config = (
        _dynamic_config()
        .model_copy(update={"allowed_models": allowed_models or []})
        .attach_catalog_snapshot(snapshot)
    )

    def create_backend(*, provider, **_kwargs):
        if provider.name in failures:
            return ImmediateFailureBackend(RuntimeError(f"{provider.name} failed"))
        return FakeBackend([mock_llm_chunk(content=provider.name)])

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = None
    registry.bind_root(root)
    return (
        registry,
        parent,
        InvokeContext(tool_call_id="fan-out", session_id=parent.session_id),
    )


@pytest.mark.asyncio
async def test_fan_out_requires_explicit_tag_model_even_when_model_is_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(monkeypatch)
    try:
        for args in (
            TaskArgs(task="panel", fan_out=True),
            TaskArgs(task="panel", fan_out=True, config=LaunchConfig()),
        ):
            with pytest.raises(LaunchConfigError) as raised:
                await _background_result(registry, args, context)
            assert raised.value.field == "config.model"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_rejects_array_scalar_and_agent_reuse_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(monkeypatch)
    try:
        cases = (
            (
                TaskArgs(
                    task="panel",
                    fan_out=True,
                    config=LaunchConfig.model_construct(
                        model=["small"], _fields_set={"model"}
                    ),
                ),
                "config.model",
            ),
            (
                TaskArgs(
                    task="panel", fan_out=True, config=LaunchConfig(model="small")
                ),
                "config.model",
            ),
            (
                TaskArgs(
                    task="panel",
                    fan_out=True,
                    agent_id="agent-1",
                    config=LaunchConfig(model="@panel"),
                ),
                "agent_id",
            ),
        )
        for args, field in cases:
            with pytest.raises(LaunchConfigError) as raised:
                await _background_result(registry, args, context)
            assert raised.value.field == field
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_preflights_all_members_before_launching_any(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "missing"], allowed_models=["small"]
    )
    try:
        with pytest.raises(LaunchConfigError, match="missing"):
            await _background_result(
                registry,
                TaskArgs(
                    task="panel", fan_out=True, config=LaunchConfig(model="@panel")
                ),
                context,
            )
        assert registry._agent_records == {}
        assert registry._children == {}
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_launches_ordered_retained_members_with_exact_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(monkeypatch)
    try:
        result = await _background_result(
            registry,
            TaskArgs(task="panel", fan_out=True, config=LaunchConfig(model="@panel")),
            context,
        )
        assert result.completed and result.turns_used == 0
        assert [member.base_model for member in result.members or []] == [
            "small",
            "large",
            "other",
        ]
        assert len(registry._agent_records) == 3
        for index, member in enumerate(result.members or []):
            assert member.status == "running"
            assert set(member.model_dump(exclude_none=True)) == {
                "index",
                "base_model",
                "provider",
                "display_name",
                "status",
                "agent_id",
                "run_id",
            }
            assert (
                member.index == index
                and member.agent_id is not None
                and member.run_id is not None
            )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_links_members_during_a_projected_parent_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    root = registry._root
    assert root is not None
    projector = EventProjector(parent.session_id, "parent-turn")
    projector.start_effect(
        context.tool_call_id,
        title="task",
        detail=SubagentEffectDetail(
            tool_name="task",
            input=SubagentEffectInput(task="panel", agent="worker"),
            display=EffectCallDisplay(summary="panel", status_text="Running"),
        ),
    )
    root.turns._projector = projector
    root.turns._emit_projected = AsyncMock()
    try:
        result = await _background_result(
            registry,
            TaskArgs(task="panel", fan_out=True, config=LaunchConfig(model="@panel")),
            context,
        )
        assert result.members is not None
        member = result.members[0]
        assert member.agent_id is not None
        detail = projector.effect_detail("fan-out:fan-out:0")
        assert isinstance(detail, SubagentEffectDetail)
        assert (
            detail.child_session_id
            == registry._agent_records[member.agent_id].session_id
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_completes_projected_member_effect_when_background_run_finishes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    root = registry._root
    assert root is not None
    projector = EventProjector(parent.session_id, "parent-turn")
    projector.start_effect(
        context.tool_call_id,
        title="task",
        detail=SubagentEffectDetail(
            tool_name="task",
            input=SubagentEffectInput(task="panel", agent="worker"),
            display=EffectCallDisplay(summary="panel", status_text="Running"),
        ),
    )
    root.turns._projector = projector
    root.turns._emit_projected = AsyncMock()
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
        assert result.members is not None
        member = result.members[0]
        assert member.agent_id is not None and member.run_id is not None
        await registry.wait_for_agent(member.agent_id, member.run_id)
        for _ in range(10):
            if any(
                getattr(update.params, "entry_id", None) == "fan-out:fan-out:0"
                and any(
                    patch.path == "/state"
                    and isinstance(patch.value, dict)
                    and patch.value["status"] == "completed"
                    for patch in update.params.patch
                )
                for (update,), _kwargs in root.turns._emit_projected.call_args_list
            ):
                break
            await asyncio.sleep(0)
        assert any(
            getattr(update.params, "entry_id", None) == "fan-out:fan-out:0"
            and any(
                patch.path == "/state"
                and isinstance(patch.value, dict)
                and patch.value["status"] == "completed"
                for patch in update.params.patch
            )
            for (update,), _kwargs in root.turns._emit_projected.call_args_list
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


def test_fan_out_member_effect_remains_updatable_after_parent_finalization() -> None:
    projector = EventProjector("parent", "turn")
    projector.start_effect(
        "task",
        title="task",
        detail=SubagentEffectDetail(
            tool_name="task",
            input=SubagentEffectInput(task="panel", agent="worker"),
            display=EffectCallDisplay(summary="panel", status_text="Running"),
        ),
    )
    projector.start_fan_out_member_effect("task", "task:fan-out:0")

    projector.finalize()
    update = projector.complete_effect(
        "task:fan-out:0",
        CompletedEffectState(
            output=None,
            display=EffectResultDisplay(success=True, message="task completed"),
        ),
    )

    assert getattr(update.params, "entry_id", None) == "task:fan-out:0"


@pytest.mark.asyncio
async def test_fan_out_effect_is_registered_before_publication_can_be_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    root = registry._root
    assert root is not None
    projector = EventProjector(parent.session_id, "parent-turn")
    projector.start_effect(
        context.tool_call_id,
        title="task",
        detail=SubagentEffectDetail(
            tool_name="task",
            input=SubagentEffectInput(task="panel", agent="worker"),
            display=EffectCallDisplay(summary="panel", status_text="Running"),
        ),
    )
    root.turns._projector = projector
    publication_started = asyncio.Event()
    release_publication = asyncio.Event()

    async def emit_projected(update: ProjectedUpdate) -> None:
        _ = update
        publication_started.set()
        await release_publication.wait()

    root.turns._emit_projected = emit_projected
    member_id = f"{context.tool_call_id}:fan-out:0"
    try:
        start = asyncio.create_task(
            registry._start_fan_out_member_effect(
                registry._runtime(parent.session_id), context.tool_call_id, member_id
            )
        )
        await publication_started.wait()
        assert registry._fan_out_effect_projectors[member_id] is projector
        start.cancel()
        with pytest.raises(asyncio.CancelledError):
            await start

        updates = projector.finalize(cancelled=True)
        assert any(
            getattr(update.params, "entry_id", None) == member_id
            and any(
                patch.path == "/state"
                and isinstance(patch.value, dict)
                and patch.value["status"] == "cancelled"
                for patch in getattr(update.params, "patch", ())
            )
            for update in updates
        )
    finally:
        release_publication.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_runtime_failure_does_not_cancel_siblings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "other"], failures={"test/second"}
    )
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
        assert not result.completed
        assert [member.status for member in result.members or []] == [
            "completed",
            "failed",
        ]
        assert result.members is not None
        assert result.members[0].result == "test/first"
        assert result.members[1].error is not None
        assert result.members[1].error["code"] == "runtime_failed"
        assert "test/second failed" in result.members[1].error["message"]
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_background_fan_out_reports_cancelled_member_as_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    cancelled = asyncio.Event()

    async def cancelled_wait(
        agent_id: str, run_id: str, **_kwargs: object
    ) -> TaskResult:
        record = registry._agent_records[agent_id]
        run = next(
            run
            for run in (record.current_run, *record.run_history)
            if run is not None and run.run_id == run_id
        )
        run.status = RunStatus.CANCELLED
        cancelled.set()
        return TaskResult(
            response="partial output",
            turns_used=0,
            completed=False,
            agent_id=agent_id,
            run_id=run_id,
        )

    completed_statuses: list[str] = []

    async def capture_completion(
        _parent: SessionRuntime, _member_tool_call_id: str, status: str, _message: str
    ) -> None:
        completed_statuses.append(status)

    monkeypatch.setattr(registry, "wait_for_agent", cancelled_wait)
    monkeypatch.setattr(registry, "_complete_fan_out_member_effect", capture_completion)
    try:
        await _background_result(
            registry,
            TaskArgs(task="panel", fan_out=True, config=LaunchConfig(model="@panel")),
            context,
        )
        await cancelled.wait()
        for _ in range(10):
            if completed_statuses:
                break
            await asyncio.sleep(0)
        assert completed_statuses == ["cancelled"]
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_fan_out_reports_partial_cancelled_member_as_cancelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )

    async def cancelled_wait(
        agent_id: str, run_id: str, **_kwargs: object
    ) -> TaskResult:
        record = registry._agent_records[agent_id]
        run = next(
            run
            for run in (record.current_run, *record.run_history)
            if run is not None and run.run_id == run_id
        )
        run.status = RunStatus.CANCELLED
        return TaskResult(
            response="partial output",
            turns_used=0,
            completed=False,
            agent_id=agent_id,
            run_id=run_id,
        )

    monkeypatch.setattr(registry, "wait_for_agent", cancelled_wait)
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )

        assert result.members is not None
        assert result.members[0].status == "cancelled"
        assert result.members[0].error is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_fan_out_preserves_member_completion_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )

    async def completed_wait(
        agent_id: str, run_id: str, **_kwargs: object
    ) -> TaskResult:
        return TaskResult(
            response="done",
            turns_used=1,
            completed=True,
            agent_id=agent_id,
            run_id=run_id,
            metadata={
                "providers_used": [["test/first", "test/second"]],
                "switch_notices": [{"base_model": "small"}],
            },
        )

    monkeypatch.setattr(registry, "wait_for_agent", completed_wait)
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
        assert result.members is not None
        assert result.members[0].metadata == {
            "providers_used": [["test/first", "test/second"]],
            "switch_notices": [{"base_model": "small"}],
        }
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_fan_out_waits_for_each_exact_handle_without_sibling_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "large"]
    )
    original_wait = registry.wait_for_agent
    waited: list[tuple[str, str | None]] = []

    async def track_wait(
        agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ):
        waited.append((agent_id, run_id))
        return await original_wait(agent_id, run_id, timeout=timeout)

    monkeypatch.setattr(registry, "wait_for_agent", track_wait)
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
        assert result.completed and result.members is not None
        assert waited == [(member.agent_id, member.run_id) for member in result.members]
        assert [member.status for member in result.members] == [
            "completed",
            "completed",
        ]
        assert result.turns_used == 2
    finally:
        await registry.drain_children()
        await parent.aclose()


def test_single_tag_selection_uses_order_and_retained_assignment_is_committed() -> None:
    selected = ModelResolver(_fan_out_snapshot(tag_members=["small", "large"])).resolve(
        "@panel"
    )
    changed = ModelResolver(_fan_out_snapshot(tag_members=["large", "small"]))

    assert selected.base_model == "small"
    assert changed.resolve_committed(selected.identity).base_model == "small"


@pytest.mark.asyncio
async def test_new_child_inherits_parent_failover_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _context = await _fan_out_registry(monkeypatch)
    committed = CommittedModelIdentity(
        base_model="small",
        provider="test/second",
        wire_name="small-failover-wire",
        catalog_revision="fan-out-test",
    )
    parent.committed_model = committed
    parent.config.attach_committed_model(committed)
    try:
        candidate = registry._resolve_launch_candidate(
            registry._runtime(parent.session_id),
            TaskArgs(task="child", agent="worker", background=True),
        )

        assert candidate.committed_model == committed
        assert candidate.effective_model.provider == "test/second"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_propagates_launch_cancellation_without_launching_later_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "large"]
    )
    launched: list[str] = []

    async def cancelled_run(args: TaskArgs, ctx: InvokeContext):
        launched.append(cast(str, args.config and args.config.model))
        raise asyncio.CancelledError()
        yield TaskResult(response="", turns_used=0, completed=False)

    monkeypatch.setattr(registry, "run", cancelled_run)
    try:
        with pytest.raises(asyncio.CancelledError):
            await registry._run_fan_out(
                registry._runtime(parent.session_id),
                TaskArgs(
                    task="panel", fan_out=True, config=LaunchConfig(model="@panel")
                ),
                context,
            )
        assert launched == ["small"]
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_failed_launch_discards_pending_result_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )

    async def failed_run(_args: TaskArgs, _ctx: InvokeContext):
        raise RuntimeError("launch failed")
        yield TaskResult(response="", turns_used=0, completed=False)

    monkeypatch.setattr(registry, "run", failed_run)
    try:
        for index in range(2):
            result = await registry._run_fan_out(
                registry._runtime(parent.session_id),
                TaskArgs(
                    task="panel", fan_out=True, config=LaunchConfig(model="@panel")
                ),
                InvokeContext(
                    tool_call_id=f"{context.tool_call_id}-{index}",
                    session_id=parent.session_id,
                ),
            )
            assert result.members is not None
            assert result.members[0].status == "failed"
            assert registry._pending_fan_out_result_leases == set()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_launch_cancellation_releases_already_started_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "large"]
    )
    original_run = registry.run

    async def cancel_second(args: TaskArgs, ctx: InvokeContext):
        if args.config is not None and args.config.model == "large":
            raise asyncio.CancelledError()
        async for event in original_run(args, ctx):
            yield event

    monkeypatch.setattr(registry, "run", cancel_second)
    try:
        with pytest.raises(asyncio.CancelledError):
            await registry._run_fan_out(
                registry._runtime(parent.session_id),
                TaskArgs(
                    task="panel", fan_out=True, config=LaunchConfig(model="@panel")
                ),
                context,
            )
        assert registry._agent_records == {}
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_fan_out_does_not_use_identity_mutated_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    original_wait = registry.wait_for_agent

    async def wait_after_failover(
        agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> TaskResult:
        result = await original_wait(agent_id, run_id, timeout=timeout)
        registry._agent_records[
            agent_id
        ].runtime.agent_loop.committed_model = CommittedModelIdentity(
            base_model="small",
            provider="test/second",
            wire_name="small-failover-wire",
            catalog_revision="fan-out-test",
        )
        return result

    monkeypatch.setattr(registry, "wait_for_agent", wait_after_failover)
    try:
        result = await _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
        assert result.members is not None
        assert result.members[0].provider == "test/first"
        assert result.members[0].display_name == "test/first/small-wire"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_background_fan_out_does_not_use_identity_mutated_after_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    original_wait = registry.wait_for_agent
    identity_updated = asyncio.Event()

    async def wait_after_failover(
        agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> TaskResult:
        result = await original_wait(agent_id, run_id, timeout=timeout)
        registry._agent_records[
            agent_id
        ].runtime.agent_loop.committed_model = CommittedModelIdentity(
            base_model="small",
            provider="test/second",
            wire_name="small-failover-wire",
            catalog_revision="fan-out-test",
        )
        identity_updated.set()
        return result

    monkeypatch.setattr(registry, "wait_for_agent", wait_after_failover)
    try:
        result = await _background_result(
            registry,
            TaskArgs(task="panel", fan_out=True, config=LaunchConfig(model="@panel")),
            context,
        )
        await identity_updated.wait()
        await asyncio.sleep(0)
        assert result.members is not None
        assert result.members[0].provider == "test/first"
        assert result.members[0].display_name == "test/first/small-wire"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_fan_out_does_not_treat_idle_retention_cap_as_total_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "large"]
    )
    registry._retention_policy = (3600, 1)
    try:
        result = await _background_result(
            registry,
            TaskArgs(task="panel", fan_out=True, config=LaunchConfig(model="@panel")),
            context,
        )
        assert result.members is not None
        assert len(result.members) == 2
        assert len(registry._agent_records) == 2
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_fan_out_cancellation_releases_every_started_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small", "large"]
    )
    original_wait = registry.wait_for_agent
    original_release = registry.release_agent
    waiting = asyncio.Event()
    release_wait = asyncio.Event()
    released: list[str] = []

    async def block_wait(
        agent_id: str, run_id: str | None = None, *, timeout: float | None = None
    ) -> TaskResult:
        waiting.set()
        await release_wait.wait()
        return await original_wait(agent_id, run_id, timeout=timeout)

    async def track_release(agent_id: str) -> None:
        released.append(agent_id)
        await original_release(agent_id)

    monkeypatch.setattr(registry, "wait_for_agent", block_wait)
    monkeypatch.setattr(registry, "release_agent", track_release)
    task = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
    )
    try:
        await asyncio.wait_for(waiting.wait(), timeout=1)
        for _ in range(10):
            if len(registry._agent_records) == 2:
                break
            await asyncio.sleep(0)
        started = set(registry._agent_records)
        assert len(started) == 2
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert set(released) == started
        assert registry._agent_records == {}
    finally:
        release_wait.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_fan_out_uses_terminal_identity_after_zero_retention_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, context = await _fan_out_registry(
        monkeypatch, tag_members=["small"]
    )
    registry._retention_policy = (0, 0)
    backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    task = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(
                task="panel",
                fan_out=True,
                background=False,
                config=LaunchConfig(model="@panel"),
            ),
            context,
        )
    )
    try:
        await asyncio.wait_for(backend.started.wait(), timeout=1)
        record = next(iter(registry._agent_records.values()))
        record.runtime.agent_loop.committed_model = CommittedModelIdentity(
            base_model="small",
            provider="test/second",
            wire_name="small-failover-wire",
            catalog_revision="fan-out-test",
        )
        backend.release.set()
        result = await asyncio.wait_for(task, timeout=1)
        assert result.members is not None
        assert result.members[0].provider == "test/second"
        assert result.members[0].display_name == "test/second/small-failover-wire"
        assert registry._agent_records == {}
    finally:
        backend.release.set()
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await registry.drain_children()
        await parent.aclose()


def _transcript_registry(tmp_path: Path) -> tuple[SessionRuntimeRegistry, AgentRecord]:
    registry = _retention_registry()
    root = MagicMock()
    root.agent_loop.session_id = "parent"
    root.agent_loop._session_generation = 7
    root.agent_loop.session_logger.session_dir = tmp_path / "parent"
    root.agent_loop.session_logger.session_metadata = MagicMock(child_sessions=[])
    registry._root = root
    registry._generation_identity = ("parent", 7)
    record = _idle_record("agent-1", clock=_ManualClock())
    record.session_id = "child-1"
    record.parent_session_id = "parent"
    record.root_generation = 7
    registry._agent_records[record.agent_id] = record
    return registry, record


@pytest.mark.asyncio
async def test_transcript_read_resolution_is_authorized_and_non_perturbing(
    tmp_path: Path,
) -> None:
    registry, record = _transcript_registry(tmp_path)
    root = registry._root
    assert root is not None
    metadata = root.agent_loop.session_logger.session_metadata
    assert metadata is not None
    metadata.child_sessions = [
        ChildSessionLink(
            session_id=record.session_id,
            tool_call_id="task-1",
            agent="worker",
            relative_path="children/child-1",
        )
    ]
    idle_since = record.idle_since
    reaper = registry._reaper_task

    snapshot = await registry.resolve_transcript_read(record.agent_id)

    assert snapshot.child_session_id == record.session_id
    assert snapshot.child_dir == tmp_path / "parent" / "children" / "child-1"
    assert snapshot.parent_dir == tmp_path / "parent"
    assert snapshot.has_saved_transcript
    assert await registry.transcript_read_is_current(snapshot)
    assert record.idle_since == idle_since
    assert registry._reaper_task is reaper
    assert registry._agent_records[record.agent_id] is record


@pytest.mark.asyncio
async def test_transcript_read_survives_eviction_but_not_release(
    tmp_path: Path,
) -> None:
    registry, record = _transcript_registry(tmp_path)
    root = registry._root
    assert root is not None
    metadata = root.agent_loop.session_logger.session_metadata
    assert metadata is not None
    metadata.child_sessions = [
        ChildSessionLink(
            session_id=record.session_id,
            tool_call_id="task-1",
            agent="worker",
            relative_path="children/child-1",
        )
    ]
    snapshot = await registry.resolve_transcript_read(record.agent_id)

    assert await registry._evict_agent(record.agent_id, "ttl")
    tombstone = registry._evicted_agents[record.agent_id]
    assert tombstone.child_session_id == record.session_id
    assert tombstone.parent_identity == ("parent", 7)
    assert await registry.transcript_read_is_current(snapshot)
    assert (
        await registry.resolve_transcript_read(record.agent_id)
    ).child_session_id == record.session_id

    await registry.release_agent(record.agent_id)
    assert not await registry.transcript_read_is_current(snapshot)
    with pytest.raises(UnknownAgentError):
        await registry.resolve_transcript_read(record.agent_id)


@pytest.mark.asyncio
async def test_transcript_read_no_saved_and_invalid_links(tmp_path: Path) -> None:
    registry, record = _transcript_registry(tmp_path)
    root = registry._root
    assert root is not None
    metadata = root.agent_loop.session_logger.session_metadata
    assert metadata is not None

    no_saved = await registry.resolve_transcript_read(record.agent_id)
    assert not no_saved.has_saved_transcript
    assert await registry.transcript_read_is_current(no_saved)
    with pytest.raises(UnknownAgentError):
        await registry.resolve_transcript_read(record.session_id)

    for relative_path in ("../escape", "/absolute"):
        metadata.child_sessions = [
            ChildSessionLink(
                session_id=record.session_id,
                tool_call_id="task-1",
                agent="worker",
                relative_path=relative_path,
            )
        ]
        with pytest.raises(UnknownAgentError):
            await registry.resolve_transcript_read(record.agent_id)


@pytest.mark.asyncio
async def test_transcript_read_invalidates_on_link_or_parent_change(
    tmp_path: Path,
) -> None:
    registry, record = _transcript_registry(tmp_path)
    root = registry._root
    assert root is not None
    metadata = root.agent_loop.session_logger.session_metadata
    assert metadata is not None
    metadata.child_sessions = [
        ChildSessionLink(
            session_id=record.session_id,
            tool_call_id="task-1",
            agent="worker",
            relative_path="child",
        )
    ]
    snapshot = await registry.resolve_transcript_read(record.agent_id)
    metadata.child_sessions[0] = ChildSessionLink(
        session_id=record.session_id,
        tool_call_id="task-2",
        agent="worker",
        relative_path="child",
    )
    assert not await registry.transcript_read_is_current(snapshot)
    root.agent_loop._session_generation = 8
    assert not await registry.transcript_read_is_current(snapshot)
