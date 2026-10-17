"""Tests for child-session assembly, transcripts, drain, and close."""

from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntime,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server.events import CallbackRequested
from chartreux.app_server.models import (
    CompletedEffectState,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicTurn,
    PublicTurnStatus,
    SubagentEffectDetail,
    UserAnswer,
    UserInputCallbackOutput,
    UserQuestionResult,
)
from chartreux.app_server.protocol import AppServerResponseError, ProtocolErrorCode
from chartreux.app_server.server import AppServer
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.events import AssistantEvent
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.message_list import MessageList
from chartreux.core.session.session_loader import MESSAGES_FILENAME
from chartreux.core.session_types import ChildSessionLink
from chartreux.core.subagents import (
    AgentAvailability,
    RunStatus,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from chartreux.utils.tool_presentation import (
    EffectCallDisplay,
    ToolCallPresentation,
    ToolEffectKind,
)
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _config,
    _consume,
    _evicted_tombstone,
    _GatedWakeup,
    _idle_record,
    _ManualClock,
    _read_child,
    _reaper_registry,
    _rearm,
    _retention_registry,
    _task_call,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    create_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


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
            "todo": {"permission": ToolPermission.ALWAYS.value},
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
    registry._evicted_agents["old"] = _evicted_tombstone(
        "old", run_id=None, status=None
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
        assert record.runtime.turns._event_sink is not None
        await record.runtime.turns._event_sink(
            AssistantEvent(content="Child completed")
        )
        return MagicMock(
            error=None, status=PublicTurnStatus.COMPLETED, stop_reason=None
        )

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
        TaskArgs(task="work", agent_type="worker", agent_id="agent-1", background=True),
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
async def test_agent_summary_projects_live_frozen_and_absent_turn_counts() -> None:
    registry = _retention_registry()
    record = _idle_record("agent-1", clock=_ManualClock())
    record.runtime.agent_loop.messages = MessageList([
        LLMMessage(role=Role.user, content="task"),
        LLMMessage(role=Role.assistant, content="first"),
        LLMMessage(role=Role.assistant, content="second"),
    ])
    registry._agent_records[record.agent_id] = record

    assert registry._agent_summaries()[0].turns_used is None
    run = RunRecord(
        "run-1",
        "agent-1",
        "worker",
        RunStatus.RUNNING,
        asyncio.get_running_loop().create_future(),
        start_count=1,
    )
    record.state = _AgentState.RUNNING
    record.current_run = run
    assert registry._agent_summaries()[0].turns_used == 1

    record.current_run = None
    record.state = _AgentState.IDLE
    run.result = TaskResult(response="done", turns_used=3, completed=True)
    record.run_history.append(run)
    assert registry._agent_summaries()[0].turns_used == 3


@pytest.mark.asyncio
async def test_resident_transcript_snapshot_is_current_only_while_runtime_is_resident(
    tmp_path: Path,
) -> None:
    registry, record = _transcript_registry(tmp_path)
    record.state = _AgentState.RUNNING
    record.runtime.agent_loop.messages = MessageList([
        LLMMessage(role=Role.system, content="instructions"),
        LLMMessage(role=Role.assistant, content="live"),
    ])
    registry._children[record.session_id] = record.runtime

    snapshot = await registry.resolve_resident_transcript_read(record.agent_id)

    assert snapshot is not None
    assert [message.content for message in snapshot.messages] == [
        "instructions",
        "live",
    ]
    assert await registry.resident_transcript_read_is_current(snapshot)
    registry._generation_identity = ("parent", 8)
    assert not await registry.resident_transcript_read_is_current(snapshot)

    registry._generation_identity = ("parent", 7)
    released = await registry.resolve_resident_transcript_read(record.agent_id)
    assert released is not None
    registry._children.pop(record.session_id)
    assert not await registry.resident_transcript_read_is_current(released)


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
