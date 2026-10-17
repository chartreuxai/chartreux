"""Wait, cancellation, and result behavior tests for subagents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import (
    RunRecord,
    SessionRuntime,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server.events import AgentsUpdate
from chartreux.app_server.models import (
    CompletedEffectState,
    FailedEffectState,
    PublicEffectEntry,
    PublicMessageEntry,
    PublicTurnStatus,
    TextContentBlock,
    TurnErrorCode,
)
from chartreux.app_server.protocol import TurnStartParams
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.subagents import (
    AgentBusyError,
    AgentEvictedError,
    AgentProfileMismatchError,
    AgentResultExpiredError,
    CancelOutcome,
    RetentionPolicy,
    RunStatus,
    RunStopReason,
    TaskArgs,
    UnknownAgentError,
    WaitStatus,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    GatedFailureBackend,
    GatedSequenceBackend,
    _background_result,
    _config,
    _consume,
    _evicted_tombstone,
    _foreground_registry,
    _foreground_result,
    _idle_record,
    _ManualClock,
    _real_reused_background,
    _reaper_registry,
    _retention_registry,
    _stored_result,
    _task_call,
    _todo_call,
    _wait_result,
    _zero_retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("closed", ["_admission_closed", "_draining_children"])
async def test_cancel_unknown_agent_precedes_closed_admission(
    monkeypatch: pytest.MonkeyPatch, closed: str
) -> None:
    registry, parent, _ = await _zero_retention_registry(monkeypatch, BlockingBackend())
    try:
        setattr(registry, closed, True)
        result = await registry.cancel_run(
            "unknown",
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert result.outcome is CancelOutcome.UNKNOWN_RUN
    finally:
        setattr(registry, closed, False)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_requested_cancel_cannot_rewrite_genuine_provider_outcome(
    monkeypatch: pytest.MonkeyPatch, failed: bool
) -> None:
    class CompletionWinsBackend(BlockingBackend):
        async def complete(self, **_kwargs):
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                pass  # Model completion/error wins a concurrent interrupt.
            if failed:
                raise RuntimeError("genuine error")
            return mock_llm_chunk(content="genuine completion")

    backend = CompletionWinsBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work"),
            InvokeContext(tool_call_id="wins", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await backend.started.wait()
        outcome = await registry.cancel_run(
            record.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        result = await asyncio.wait_for(_wait_result(registry, record.agent_id), 2)
        assert run.requested_stop_reason is RunStopReason.USER_CANCELLED
        assert run.status is (RunStatus.FAILED if failed else RunStatus.COMPLETED)
        assert result.stop_reason is (RunStopReason.ERROR if failed else None)
        assert result.completed is not failed
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("active_parent", [False, True])
async def test_serialized_user_cancel_injects_without_waking_parent(
    monkeypatch: pytest.MonkeyPatch, active_parent: bool
) -> None:
    child_backend = BlockingBackend()
    parent_backend = GatedSequenceBackend([
        [mock_llm_chunk(content="parent done")],
        [mock_llm_chunk(content="parent continuation done")],
    ])
    started, release = parent_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(config=_config(), backend=parent_backend)
    client = start_test_app_server(parent)
    assert client._run_peer is not None
    server = cast(AppServer, cast(Any, client._run_peer).__self__)
    session = await attach_test_app_server_session(client)
    root = legacy_backend(server)
    registry = root.children
    root.session.turns.link_subagent = AsyncMock()
    try:
        parent_turn_id = None
        if active_parent:
            response, action = root.session.turns.start(
                TurnStartParams(
                    session_id=parent.session_id,
                    message=[TextContentBlock(text="parent work")],
                )
            )
            parent_turn_id = response.turn.id
            action()
            await asyncio.wait_for(started.wait(), 2)
        launch = await _background_result(
            registry,
            # The late cancel below must find a resident agent, so the launch
            # opts into retention explicitly.
            TaskArgs(task="child work", retain=True),
            InvokeContext(tool_call_id="child", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child_backend.started.wait(), 2)
        response = await session.cancel_agent(launch.agent_id, launch.run_id)
        assert response.outcome.value == "stop_requested"
        assert response.run_id == launch.run_id
        assert response.stop_reason is not None
        assert response.stop_reason.value == "user_cancelled"
        result = await asyncio.wait_for(_wait_result(registry, launch.agent_id), 2)
        assert result.stop_reason is RunStopReason.USER_CANCELLED
        assert parent._pending_injected_messages
        assert "cancelled (by user)" in (
            parent._pending_injected_messages[-1].content or ""
        )
        if active_parent:
            assert root.session.turns.active_turn is not None
            assert root.session.turns.active_turn.id == parent_turn_id
            release.set()
            assert parent_turn_id is not None
            await root.session.turns.wait_for_operation(parent_turn_id)
        else:
            assert not started.is_set()
            assert root.session.turns.active_turn is None
            assert parent_backend.requests_messages == []
            release.set()
            _ = [event async for event in session.act("next user turn")]
            # Injection is consumed by the user-started operation's continuation.
            assert len(parent_backend.requests_messages) == 2
            assert any(
                "cancelled (by user)" in (message.content or "")
                for message in parent_backend.requests_messages[-1]
            )
        late = await session.cancel_agent(launch.agent_id, launch.run_id)
        assert late.outcome.value == "not_running"
    finally:
        release.set()
        child_backend.release.set()
        await session.close()


@pytest.mark.asyncio
async def test_user_cancel_bypasses_tool_ownership_and_respects_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work"),
            InvokeContext(tool_call_id="child", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        await asyncio.wait_for(backend.started.wait(), 2)
        registry._admission_closed = True
        denied = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id="user",
        )
        assert denied.outcome is CancelOutcome.FORBIDDEN
        assert not backend.stopped.is_set()
        registry._admission_closed = False
        accepted = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id="user",
        )
        assert accepted.outcome is CancelOutcome.STOP_REQUESTED
        await asyncio.wait_for(registry.wait_for_agent(launch.agent_id), 2)
    finally:
        backend.release.set()
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_user_cancel_does_not_interrupt_executing_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent_backend, child_backend = BlockingBackend(), BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(config=_config(), backend=parent_backend)
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        response, action = root.turns.start(
            TurnStartParams(
                session_id=parent.session_id,
                message=[TextContentBlock(text="parent work")],
            )
        )
        action()
        await parent_backend.started.wait()
        launch = await _background_result(
            registry,
            TaskArgs(task="child work"),
            InvokeContext(tool_call_id="child", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        await child_backend.started.wait()
        await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        await asyncio.wait_for(registry.wait_for_agent(launch.agent_id), 2)
        assert (
            root.turns.active_turn is not None
            and root.turns.active_turn.id == response.turn.id
        )
        assert not parent_backend.stopped.is_set()
        assert parent._pending_injected_messages
        parent_backend.release.set()
        await root.turns.wait_for_operation(response.turn.id)
    finally:
        parent_backend.release.set()
        child_backend.release.set()
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_mode", [False, True])
@pytest.mark.parametrize(
    "reason",
    [
        RunStopReason.USER_CANCELLED,
        RunStopReason.ORCHESTRATOR_CANCELLED,
        RunStopReason.RETASKED,
    ],
)
async def test_cancel_preserves_waiter_result_identity_and_notification(
    monkeypatch: pytest.MonkeyPatch, retained_mode: bool, reason: RunStopReason
) -> None:
    backend = BlockingBackend()
    registry, parent, notifications = await _zero_retention_registry(
        monkeypatch, backend
    )
    if retained_mode:
        registry._retention_policy = RetentionPolicy(
            idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
        )
    ctx = InvokeContext(tool_call_id="cancel", session_id=parent.session_id)
    try:
        launch = await _background_result(registry, TaskArgs(task="work"), ctx)
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await asyncio.wait_for(backend.started.wait(), 2)
        waiter = asyncio.create_task(
            _wait_result(registry, launch.agent_id, launch.run_id)
        )
        await asyncio.sleep(0)
        key = (launch.agent_id, launch.run_id)
        assert registry._wait_leases[key] == 1
        outcome = await registry.cancel_run(
            launch.agent_id, reason=reason, requester_session_id=parent.session_id
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        assert outcome.run_id == launch.run_id and outcome.stop_reason is reason
        assert not run.completion_task.cancelled()
        assert registry._wait_leases[key] == 1
        result = await asyncio.wait_for(waiter, 2)
        assert result.stop_reason is reason and not result.completed
        assert run.status is RunStatus.CANCELLED and backend.stopped.is_set()
        assert await registry.get_agent_result(launch.agent_id, launch.run_id) == result
        late = await registry.cancel_run(
            launch.agent_id, reason=reason, requester_session_id=parent.session_id
        )
        assert late.outcome is CancelOutcome.NOT_RUNNING
        assert late.run_id == launch.run_id and late.stop_reason is reason
        if reason is RunStopReason.USER_CANCELLED:
            assert notifications == []
            assert parent._pending_injected_messages[-1].content is not None
            assert (
                "cancelled (by user)" in parent._pending_injected_messages[-1].content
            )
            assert (
                registry._root is not None and registry._root.turns.active_turn is None
            )
        else:
            assert any(
                ("retasked" if reason is RunStopReason.RETASKED else "cancelled")
                in text
                for text in notifications
            )
        if retained_mode:
            assert registry._agent_records[launch.agent_id] is record
            assert not record.runtime._closed
            backend.release.set()
            next_launch = await _background_result(
                registry, TaskArgs(task="reuse", agent_id=launch.agent_id), ctx
            )
            next_run = record.current_run
            assert next_run is not None and next_run.requested_stop_reason is None
            assert next_launch.run_id != launch.run_id
            stale = await registry.cancel_run(
                launch.agent_id,
                launch.run_id,
                reason=reason,
                requester_session_id=parent.session_id,
            )
            assert stale.outcome is CancelOutcome.NOT_RUNNING
            await asyncio.wait_for(next_run.completion_task, 2)
        else:
            await asyncio.gather(*tuple(registry._teardown_tasks))
            assert record.runtime._closed
            assert launch.agent_id in registry._evicted_agents
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_reason", [RunStopReason.USER_CANCELLED, RunStopReason.ORCHESTRATOR_CANCELLED]
)
async def test_cancel_first_writer_ownership_and_sibling_isolation(
    monkeypatch: pytest.MonkeyPatch, first_reason: RunStopReason
) -> None:
    backends = [BlockingBackend(), BlockingBackend()]
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backends[0])
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    root = registry._root
    assert root is not None
    parent_start = MagicMock(side_effect=RuntimeError("parent must stay idle"))
    monkeypatch.setattr(root.turns, "start", parent_start)
    try:
        ctx = InvokeContext(tool_call_id="one", session_id=parent.session_id)
        first = await _background_result(registry, TaskArgs(task="one"), ctx)
        await asyncio.wait_for(backends[0].started.wait(), 2)
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", lambda **_: backends[1]
        )
        second = await _background_result(
            registry,
            TaskArgs(task="two"),
            InvokeContext(tool_call_id="two", session_id=parent.session_id),
        )
        assert first.agent_id and second.agent_id
        assert first.agent_id != second.agent_id
        await asyncio.wait_for(backends[1].started.wait(), 2)
        first_record = registry._agent_records[first.agent_id]
        run = first_record.current_run
        assert run is not None
        interrupt = MagicMock(wraps=first_record.runtime.turns.interrupt_operation)
        monkeypatch.setattr(
            first_record.runtime.turns, "interrupt_operation", interrupt
        )
        denied = await registry.cancel_run(
            first.agent_id,
            reason=RunStopReason.ORCHESTRATOR_CANCELLED,
            requester_session_id="intruder",
        )
        unknown = await registry.cancel_run(
            first.agent_id,
            "never-launched",
            reason=first_reason,
            requester_session_id=parent.session_id,
        )
        missing = await registry.cancel_run(
            "missing", reason=first_reason, requester_session_id=parent.session_id
        )
        assert [denied.outcome, unknown.outcome, missing.outcome] == [
            CancelOutcome.FORBIDDEN,
            CancelOutcome.UNKNOWN_RUN,
            CancelOutcome.UNKNOWN_RUN,
        ]
        assert run.requested_stop_reason is None
        other_reason = (
            RunStopReason.ORCHESTRATOR_CANCELLED
            if first_reason is RunStopReason.USER_CANCELLED
            else RunStopReason.USER_CANCELLED
        )
        outcomes = await asyncio.gather(
            registry.cancel_run(
                first.agent_id,
                reason=first_reason,
                requester_session_id=parent.session_id,
            ),
            registry.cancel_run(
                first.agent_id,
                reason=other_reason,
                requester_session_id=parent.session_id,
            ),
        )
        assert [result.outcome for result in outcomes] == [
            CancelOutcome.STOP_REQUESTED,
            CancelOutcome.ALREADY_STOPPING,
        ]
        assert all(result.stop_reason is first_reason for result in outcomes)
        interrupt.assert_called_once_with(run.initial_turn_id)
        await asyncio.wait_for(run.completion_task, 2)
        assert not backends[1].stopped.is_set()
        assert not registry._agent_records[second.agent_id].runtime._closed
        assert parent.session_id == root.agent_loop.session_id
        if first_reason is RunStopReason.USER_CANCELLED:
            parent_start.assert_not_called()
        registry._result_store.pop((first.agent_id, run.run_id))
        registry._expired_results.add((first.agent_id, run.run_id))
        expired = await registry.cancel_run(
            first.agent_id,
            run.run_id,
            reason=other_reason,
            requester_session_id=parent.session_id,
        )
        assert expired.outcome is CancelOutcome.NOT_RUNNING
    finally:
        for backend in backends:
            backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "window", ["execution_terminal", "outcome_recorded", "finalizing"]
)
@pytest.mark.parametrize("failed", [False, True])
async def test_cancel_rejects_finishing_windows_and_preserves_outcome(
    monkeypatch: pytest.MonkeyPatch, window: str, failed: bool
) -> None:
    backend = (
        GatedFailureBackend(RuntimeError("genuine failure"))
        if failed
        else BlockingBackend()
    )
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work"),
            InvokeContext(tool_call_id="finish", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        run = record.current_run
        assert run is not None
        await backend.started.wait()
        if window != "finalizing":
            notify = record.runtime.turns._notify

            async def gate(method, params):
                if window == "outcome_recorded":
                    await notify(method, params)
                if method == "turn/completed":
                    entered.set()
                    await release.wait()
                if window != "outcome_recorded":
                    await notify(method, params)

            monkeypatch.setattr(record.runtime.turns, "_notify", gate)
        else:

            async def gate_update(*_args):
                if record.state is _AgentState.FINALIZING:
                    entered.set()
                    await release.wait()

            registry._notify_agents = gate_update
        backend.release.set()
        await asyncio.wait_for(entered.wait(), 2)
        assert (record.current_run is None) == (window == "finalizing")
        if window == "outcome_recorded":
            # Model the monitor having recorded the real terminal outcome before
            # its final registry transition; the controller is genuinely done.
            assert run.initial_turn_id is not None
            run.status, run.stop_reason = registry._resolve_run_outcome(
                run, record.runtime.turns.operation_terminal_turn(run.initial_turn_id)
            )
            assert run.status is not RunStatus.RUNNING
        cancelled = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert cancelled.outcome is CancelOutcome.ALREADY_FINISHING
        assert cancelled.run_id == run.run_id and run.requested_stop_reason is None
        release.set()
        await asyncio.wait_for(run.completion_task, 2)
        assert run.status is (RunStatus.FAILED if failed else RunStatus.COMPLETED)
        assert run.stop_reason is (RunStopReason.ERROR if failed else None)
    finally:
        release.set()
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup", ["release", "shutdown"])
async def test_pending_stop_release_and_shutdown_races(
    monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work"),
            InvokeContext(tool_call_id="race", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        await backend.started.wait()
        result = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert result.outcome is CancelOutcome.STOP_REQUESTED
        if cleanup == "release":
            await asyncio.wait_for(registry.release_agent(launch.agent_id), 2)
        else:
            await asyncio.wait_for(registry.drain_children(), 2)
        assert record.runtime._closed and backend.stopped.is_set()
        assert not registry._wait_leases and not registry._pending_notifications
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["provider", "tool"])
async def test_cancel_publishes_accumulated_partial_and_joins_tool_cleanup(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    from functools import wraps

    from chartreux.core.tools.builtins.bash import Bash

    backend = GatedSequenceBackend([
        [
            mock_llm_chunk(
                content="partial",
                tool_calls=[
                    ToolCall(
                        id="bash",
                        index=0,
                        function=FunctionCall(
                            name="bash", arguments='{"command":"pwd"}'
                        ),
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    started, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()
    if phase == "provider":
        backend.add_gate()[1].set()
        started, release = backend.add_gate()
    else:

        @wraps(Bash.run)
        async def gated_tool(self, args, ctx=None):
            started.set()
            try:
                await release.wait()
            finally:
                stopped.set()
            yield "unused"

        monkeypatch.setattr(Bash, "run", gated_tool)
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "bash"]), backend=FakeBackend()
    )
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work"),
            InvokeContext(tool_call_id="partial", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        await asyncio.wait_for(started.wait(), 3)
        run = record.current_run
        assert run is not None
        await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        result = await asyncio.wait_for(_wait_result(registry, launch.agent_id), 3)
        assert "partial" in result.response and not result.completed
        if phase == "tool":
            assert stopped.is_set()
        assert result.stop_reason is RunStopReason.USER_CANCELLED
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


async def _wait_for_agents_update(session: AppServerSession) -> AgentsUpdate:
    async def wait() -> AgentsUpdate:
        async for event in session.events():
            if isinstance(event, AgentsUpdate):
                return event
        raise RuntimeError("App-server event stream closed before agents update")

    return await asyncio.wait_for(wait(), timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [False, True])
@pytest.mark.parametrize("empty_final", [False, True])
async def test_child_final_response_outcome_through_app_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, background: bool, empty_final: bool
) -> None:
    (tmp_path / "final-response.toml").write_text(
        "\n".join(['agent_type = "subagent"', 'enabled_tools = ["todo"]']),
        encoding="utf-8",
    )
    final = mock_llm_chunk(content="Child completed")
    if empty_final:
        final = mock_llm_chunk(content="").model_copy(
            update={
                "message": LLMMessage(role=Role.assistant, content="", tool_calls=None)
            }
        )
    child_backend = FakeBackend(
        [
            [
                mock_llm_chunk(
                    content="Earlier child prose", tool_calls=[_todo_call("read")]
                )
            ]
        ]
        + [[final]] * (2 if empty_final else 1)
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    config = _config(enabled_tools=["task", "todo"]).model_copy(
        update={"agent_paths": [tmp_path]}
    )
    parent_backend = FakeBackend([
        [
            mock_llm_chunk(
                content="",
                tool_calls=[
                    _task_call(
                        "final-response",
                        background=background,
                        retain=True if background else None,
                    )
                ],
            )
        ],
        [mock_llm_chunk(content="Parent completed")],
    ])
    parent = build_test_agent_loop(
        config=config, backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    closed: list[SessionRuntime] = []
    original_close = SessionRuntime.close

    async def counted_close(runtime: SessionRuntime) -> None:
        if runtime.agent_loop is not parent:
            closed.append(runtime)
        await original_close(runtime)

    monkeypatch.setattr(SessionRuntime, "close", counted_close)
    try:
        await _consume(session.act("Delegate this"))
        effect = next(
            entry
            for entry in session.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        if background:
            assert isinstance(effect.state, CompletedEffectState)
            record = registry._agent_records["agent-1"]
            run_id = record.latest_run_id
            assert run_id is not None
            result = await asyncio.wait_for(
                _wait_result(registry, record.agent_id, run_id), timeout=30
            )
            run = record.run_history[-1]
            assert run.run_id == run_id
            assert run.status is (
                RunStatus.FAILED if empty_final else RunStatus.COMPLETED
            )
            assert run.stop_reason is (RunStopReason.ERROR if empty_final else None)
            assert result.completed is (not empty_final)
            assert await registry.get_agent_result(record.agent_id, run_id) == result
            assert result.agent_id == record.agent_id
            assert result.run_id == run_id
            assert "Earlier child prose" in result.response
            if empty_final:
                assert "empty assistant response" in result.response
            else:
                assert "Child completed" in result.response
            child_runtime = record.runtime
        else:
            await registry.drain_children()
            assert registry._children == {}
            assert len(closed) == 1
            child_runtime = closed[0]
            if empty_final:
                assert isinstance(effect.state, FailedEffectState)
                assert "empty assistant response" in effect.state.error.message
            else:
                assert isinstance(effect.state, CompletedEffectState)
                assert isinstance(effect.state.output, dict)
                assert effect.state.output["completed"] is True
                assert (
                    effect.state.output["response"]
                    == "Earlier child proseChild completed"
                )
            await registry.drain_children()
            assert len(closed) == 1

        turn = child_runtime.turns.completed_turns[-1]
        assert turn.status is (
            PublicTurnStatus.FAILED if empty_final else PublicTurnStatus.COMPLETED
        )
        if empty_final:
            assert turn.error is not None
            assert turn.error.code == TurnErrorCode.EMPTY_LLM_RESPONSE
            assert len(child_backend.requests_messages) == 3
            assert (
                child_backend.requests_messages[1] == child_backend.requests_messages[2]
            )
        else:
            assert turn.error is None
            assert len(child_backend.requests_messages) == 2
        assistant_messages = [
            message
            for message in child_runtime.agent_loop.messages
            if message.role is Role.assistant
        ]
        assert assistant_messages[0].content == "Earlier child prose"
        assert all((message.content or "").strip() for message in assistant_messages)
        assert len(assistant_messages) == (1 if empty_final else 2)
        task_result = next(
            message
            for message in parent_backend.requests_messages[1]
            if message.role is Role.tool and message.tool_call_id == "task-1"
        )
        if empty_final and not background:
            assert "empty assistant response" in (task_result.content or "")
        assert any(
            isinstance(entry, PublicMessageEntry) and entry.text == "Parent completed"
            for entry in session.history
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_background_agent_management_survives_wait_timeout(monkeypatch) -> None:
    child_backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            # The agent is released through the management surface below, so
            # the launch opts into retention instead of closing on completion.
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
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
        timed_out = await registry.wait_for_agent(
            summary.agent_id, summary.current_run_id, timeout=0.01
        )
        assert timed_out.status is WaitStatus.STILL_RUNNING
        assert timed_out.run_id == summary.current_run_id
        assert timed_out.run_status is RunStatus.RUNNING
        assert "End your turn" in (timed_out.guidance or "")
        assert not child_backend.stopped.is_set()

        child_backend.release.set()
        result = await asyncio.wait_for(
            _wait_result(registry, summary.agent_id, summary.current_run_id), timeout=1
        )
        assert result.completed
        assert result.agent_id == summary.agent_id
        assert result.run_id == summary.current_run_id
        assert (
            await registry.get_agent_result(summary.agent_id, summary.current_run_id)
            == result
        )
        idle_update = await _wait_for_agents_update(session)
        while idle_update.agents[0].availability == "running":
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
async def test_evicted_result_is_available_by_the_exact_requested_run() -> None:
    registry = _retention_registry()
    stored = _stored_result("agent-1", "run-1")
    registry._result_store[(stored.agent_id, stored.run_id)] = stored
    registry._evicted_agents[stored.agent_id] = _evicted_tombstone(
        stored.agent_id, run_id=stored.run_id, status=RunStatus.COMPLETED
    )

    assert await registry.get_agent_result("agent-1", "run-1") == stored.result
    assert await _wait_result(registry, "agent-1", "run-1") == stored.result


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
async def test_requested_run_never_returns_a_newer_reused_run_result() -> None:
    registry = _retention_registry()
    first = _stored_result("agent-1", "first", completed_at=1)
    second = _stored_result("agent-1", "second", completed_at=2)
    registry._result_store[(first.agent_id, first.run_id)] = first
    registry._result_store[(second.agent_id, second.run_id)] = second

    assert await registry.get_agent_result("agent-1", "first") == first.result
    assert await _wait_result(registry, "agent-1", "first") == first.result


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
    await registry.drain_children()


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

    waiter = asyncio.create_task(_wait_result(registry, *key))
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
    root.agent_loop.config.subagents.max_running_subagents = 16
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
    registry._evicted_agents["evicted"] = _evicted_tombstone(
        "evicted", run_id=None, status=None
    )

    def reuse_args(agent_id: str, agent: str = "worker") -> TaskArgs:
        return TaskArgs(
            task="reuse", agent_type=agent, agent_id=agent_id, background=True
        )

    def reuse_context(agent_id: str) -> InvokeContext:
        return InvokeContext(tool_call_id=f"reuse-{agent_id}", session_id="root")

    with pytest.raises(
        UnknownAgentError, match="Unknown agent_id instance handle"
    ) as unknown:
        await _background_result(
            registry, reuse_args("agent-999"), reuse_context("agent-999")
        )
    assert "Check the agent_id with check_agents" in str(unknown.value)
    assert not registry._active_work_slots
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
async def test_equal_timestamp_result_cap_uses_agent_id_tie_break() -> None:
    registry = _retention_registry()
    for index in range(33):
        stored = _stored_result(f"agent-{32 - index:02}", "run", completed_at=1)
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    registry._expire_results_locked()

    assert ("agent-00", "run") in registry._expired_results


@pytest.mark.asyncio
async def test_registry_operations_sweep_unreferenced_result_store_entries() -> None:
    registry = _retention_registry()
    for index in range(33):
        stored = _stored_result(f"agent-{index:02}", "run")
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    await registry.check_agents()

    assert ("agent-00", "run") in registry._expired_results


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
    # The retirement above removed a live record, which publishes the reduced
    # inventory (anything removed -> publish); that publication has settled by
    # the time the cancellation acknowledgment arrives. Reset the mock so the
    # final assertion pins only the release path: releasing the already
    # retired (now unknown) handle publishes nothing.
    cast(AsyncMock, registry._emit_agents_update).reset_mock()
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
