"""Background and foreground execution tests for subagents."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server._turns import TurnController
from chartreux.app_server.models import (
    CompletedEffectState,
    PublicCheckpointEntry,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicTurnStatus,
    SubagentEffectDetail,
    TextContentBlock,
)
from chartreux.app_server.protocol import TurnStartParams
from chartreux.app_server.server import AppServer
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import Role
from chartreux.core.middleware import AutoCompactMiddleware
from chartreux.core.subagents import (
    AgentAvailability,
    EmptySubagentResponseError,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    WaitStatus,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    GatedFailureBackend,
    GatedSequenceBackend,
    ImmediateFailureBackend,
    _background_result,
    _config,
    _consume,
    _foreground_registry,
    _foreground_result,
    _read_child,
    _real_reused_background,
    _task_call,
    _todo_call,
    _wait_for_turn_completion,
    _wait_result,
    _zero_retention_registry,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.timeout(60)
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
        # The gates enforce completion during final save, not a latency bound.
        action = asyncio.create_task(_consume(session.act("Delegate this")))
        await asyncio.wait_for(child_backend.started.wait(), timeout=30)
        await asyncio.wait_for(finalizing.wait(), timeout=30)
        registry = legacy_backend(server).children
        summary = (await registry.check_agents())[0]
        assert summary.current_run_id is not None

        child_backend.release.set()
        await asyncio.wait_for(
            registry.wait_for_agent(summary.agent_id, summary.current_run_id),
            timeout=30,
        )
        release_finalizer.set()
        await asyncio.wait_for(action, timeout=30)

        assert len(parent_backend.requests_messages) == 3
        completed_summary = (await registry.check_agents())[0]
        assert any(
            message.role is Role.user
            and f"Background agent {completed_summary.name} ({summary.agent_id}"
            in str(message.content)
            for message in parent_backend.requests_messages[-1]
        )
    finally:
        child_backend.release.set()
        release_finalizer.set()
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
            # The failing child is inspected through the resident record
            # below, so the launch opts into retention explicitly.
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
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
            _wait_result(registry, summary.agent_id, summary.current_run_id), timeout=1
        )
        assert not failed.completed
        assert "child backend failed" in failed.response
        await _wait_for_turn_completion(session)
        failed_summary = (await registry.check_agents())[0]
        assert failed_summary.availability is AgentAvailability.IDLE
        assert failed_summary.current_run_status is None
        assert failed_summary.last_run_status is RunStatus.FAILED
        assert failed_summary.stop_reason is RunStopReason.ERROR
        assert not failed_summary.compacting

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
            # The child is reused below, so the first launch opts into
            # retention explicitly instead of closing on completion.
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
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
            _wait_result(registry, agent_id, first_run_id), timeout=1
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
            _wait_result(registry, agent_id, second_run_id), timeout=1
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
        completed_summary = (await registry.check_agents())[0]
        assert any(
            message.role is Role.user
            and f"Background agent {completed_summary.name} ({summary.agent_id}"
            in str(message.content)
            for message in parent_backend.requests_messages[-1]
        )
    finally:
        child_backend.release.set()
        await session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
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
                    content="",
                    tool_calls=[_task_call("noisy", background=True, retain=True)],
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
        # Overflow must not block completion without a consumer; this is not
        # a speed assertion, so allow for durable writes under parallel load.
        result = await asyncio.wait_for(
            _wait_result(registry, "agent-1", run_id), timeout=30
        )
        assert result.completed
        assert result.response == "all progress complete"
        completed_run = registry._agent_records["agent-1"].run_history[-1]
        assert completed_run.progress_overflow
        assert len(completed_run.progress_summaries) == 32
    finally:
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
            _wait_result(registry, summary.agent_id, summary.current_run_id)
        )
        still_running = await timed_out
        assert still_running.status is WaitStatus.STILL_RUNNING
        assert still_running.run_id == summary.current_run_id
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
async def test_child_runtime_compaction_notifications_never_expose_stale_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="initial child run")],
        [mock_llm_chunk(content="<summary>compacted child</summary>")],
        [mock_llm_chunk(content="child done")],
    ])
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=100, max_idle_agents=10, retain_agents=True
    )
    snapshots = AsyncMock()
    registry._notify_agents = snapshots
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="compact then work", background=True),
            InvokeContext(tool_call_id="compact", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        first_run = record.current_run
        assert first_run is not None
        await first_run.completion_task
        record.runtime.agent_loop.stats.context_tokens = 500_000
        record.context_tokens = 500_000
        before_turn = AutoCompactMiddleware.before_turn
        forced = False

        async def force_compaction(middleware, context):
            nonlocal forced
            if not forced and context.stats is record.runtime.agent_loop.stats:
                context.stats.context_tokens = 500_000
                forced = True
            return await before_turn(middleware, context)

        monkeypatch.setattr(AutoCompactMiddleware, "before_turn", force_compaction)
        snapshots.reset_mock()
        await _background_result(
            registry,
            TaskArgs(task="reuse", agent_id=record.agent_id, background=True),
            InvokeContext(tool_call_id="compact-reuse", session_id=parent.session_id),
        )
        run = record.current_run
        assert run is not None
        await run.completion_task
        assert forced
        states = [call.args[0][0] for call in snapshots.await_args_list if call.args[0]]
        start = next(index for index, state in enumerate(states) if state.compacting)
        end = next(
            index
            for index in range(start + 1, len(states))
            if not states[index].compacting
        )
        assert states[start].context_tokens == 500_000
        assert states[end].context_tokens is None
        assert all(state.context_tokens != 500_000 for state in states[end:])
        assert any(state.context_tokens is not None for state in states[end + 1 :])
        assert not record.compacting
    finally:
        await registry.drain_children()
        await parent.aclose()


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
            # The record is inspected after the eager run may have completed,
            # so the launch opts into retention explicitly.
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
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
            _wait_result(registry, summary.agent_id, run_id), timeout=1
        )
        assert not failed.completed
        assert "child backend failed immediately" in failed.response
        assert "Turn did not complete" not in failed.response
    finally:
        loop.set_task_factory(previous_task_factory)
        await session.close()


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
async def test_foreground_completions_detach_and_close_every_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([[mock_llm_chunk(content="done")] for _ in range(3)])
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
@pytest.mark.parametrize(
    "status", [PublicTurnStatus.COMPLETED, PublicTurnStatus.INTERRUPTED]
)
async def test_foreground_simulated_empty_child_result(
    monkeypatch: pytest.MonkeyPatch, status: PublicTurnStatus
) -> None:
    registry, parent = await _foreground_registry(monkeypatch, FakeBackend())
    monkeypatch.setattr(
        SessionRuntimeRegistry,
        "_start_child_turn",
        lambda *_args: ("turn-1", lambda: None),
    )
    monkeypatch.setattr(
        TurnController,
        "wait_for_operation",
        AsyncMock(return_value=MagicMock(error=None, status=status)),
    )
    closed: list[SessionRuntime] = []
    original_close = SessionRuntime.close

    async def counted_close(runtime: SessionRuntime) -> None:
        closed.append(runtime)
        await original_close(runtime)

    monkeypatch.setattr(SessionRuntime, "close", counted_close)
    try:
        if status is PublicTurnStatus.COMPLETED:
            with pytest.raises(EmptySubagentResponseError, match="empty response"):
                await _foreground_result(registry, parent)
        else:
            result = await _foreground_result(registry, parent)
            assert result.response == ""
            assert not result.completed
        await registry.drain_children()
        assert registry._children == {}
        assert len(closed) == 1
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_teardown_fd_count_does_not_grow_across_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([[mock_llm_chunk(content="done")] for _ in range(30)])
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


@pytest.mark.asyncio
async def test_foreground_terminal_generator_close_hands_off_teardown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent = await _foreground_registry(
        monkeypatch, FakeBackend([mock_llm_chunk(content="done")])
    )
    stream = registry.run(
        TaskArgs(task="foreground", agent_type="worker", background=False),
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
@pytest.mark.timeout(30)
async def test_monitor_creation_failure_aborts_prepared_turn_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The child gets a fresh backend rather than the parent's injected FakeBackend.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="Child completed")]),
    )
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
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
        completion_task = cast(asyncio.Future[Any], run.completion_task)
        try:
            await asyncio.wait_for(asyncio.shield(completion_task), timeout=20)
        except TimeoutError as exc:
            pending_tasks = [
                f"{task.get_name()}: {task.get_coro()!r}"
                for task in asyncio.all_tasks()
                if not task.done() and task is not asyncio.current_task()
            ]
            init_thread = record.runtime.agent_loop._deferred_init_thread
            thread_status = (
                "not started"
                if init_thread is None
                else f"alive={init_thread.is_alive()}"
            )
            raise AssertionError(
                "Retry completion timed out after 20 seconds.\n"
                f"Pending asyncio tasks: {pending_tasks!r}\n"
                f"Child deferred-init thread: {thread_status}"
            ) from exc
    finally:
        monkeypatch.setattr(asyncio, "create_task", original_create_task)
        await registry.drain_children()
        await root.turns.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_mode", [False, True])
@pytest.mark.parametrize("unverifiable", [False, True])
async def test_runtime_turn_limit_preserves_budget_outcome_and_parent_wording(
    monkeypatch: pytest.MonkeyPatch, retained_mode: bool, unverifiable: bool
) -> None:
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, FakeBackend()
    )
    if retained_mode:
        registry._retention_policy = RetentionPolicy(
            idle_ttl_seconds=100, max_idle_agents=10, retain_agents=True
        )
    create_child = AgentRuntimeFactory.create_child

    async def limited_child(factory, loop, agent_name, **kwargs):
        child = await create_child(factory, loop, agent_name, **kwargs)
        if unverifiable:
            child.stats.has_unknown_cost = True
            child._max_price = 1.0
            child._setup_middleware()
        else:
            child.set_max_turns(0)
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", limited_child)
    snapshots = AsyncMock()
    registry._notify_agents = snapshots
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="budget", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await run.completion_task
        result = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert result is not None and not result.completed
        summary = (await registry.check_agents())[0]
        expected_reason = (
            RunStopReason.BUDGET_UNVERIFIABLE
            if unverifiable
            else RunStopReason.BUDGET_EXCEEDED
        )
        assert summary.stop_reason is expected_reason
        wording = "budget unverifiable" if unverifiable else "budget exceeded"
        assert any(f"budget-stopped ({wording})" in text for text in notifications)
        finishing = [
            call.args[0][0]
            for call in snapshots.await_args_list
            if call.args[0] and call.args[0][0].availability == "finalizing"
        ]
        assert finishing and finishing[0].stop_reason == expected_reason.value
        assert finishing[0].last_run_status == "completed"
    finally:
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
