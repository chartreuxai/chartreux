"""Replacement and retask behavior tests for subagents."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry, _AgentState
from chartreux.app_server.server import AppServer
from chartreux.core.middleware import AutoCompactMiddleware
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.subagents import (
    AgentBusyError,
    AgentProfileMismatchError,
    LaunchConfig,
    LaunchConfigError,
    LaunchOutcome,
    LaunchToolOverride,
    RetentionPolicy,
    RunStopReason,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission, ToolPermissionError
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    GatedSequenceBackend,
    RetaskBackend,
    _background_result,
    _config,
    _consume,
    _dynamic_registry,
    _GatedWakeup,
    _ManualClock,
    _persistent_dynamic_registry,
    _task_call,
    _wait_for_turn_completion,
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
@pytest.mark.parametrize("retained_mode", [False, True])
@pytest.mark.parametrize("compacting", [False, True])
async def test_busy_retask_transfers_capacity_and_preserves_conversation(
    monkeypatch: pytest.MonkeyPatch, retained_mode: bool, compacting: bool
) -> None:
    backend = RetaskBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._max_running_subagents = 1
    if retained_mode:
        registry._retention_policy = RetentionPolicy(
            idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
        )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    replacement = None
    finalized, finalize_release = asyncio.Event(), asyncio.Event()
    try:
        old = await _background_result(registry, TaskArgs(task="old task"), ctx)
        assert old.agent_id is not None and old.run_id is not None
        record = registry._agent_records[old.agent_id]
        record.compacting = compacting
        old_run, slot, session = record.current_run, record.work_slot, record.session_id
        assert old_run is not None
        await asyncio.wait_for(backend.started.wait(), 2)

        async def gate_update(*_args):
            if record.reserved_for_replacement and record.current_run is None:
                finalized.set()
                await finalize_release.wait()

        registry._notify_agents = gate_update
        replacement = asyncio.create_task(
            _background_result(
                registry,
                TaskArgs(task="new task", agent_id=old.agent_id, replace_run=True),
                ctx,
            )
        )
        await asyncio.wait_for(backend.cleanup_started.wait(), 2)
        assert record.reserved_for_replacement
        assert not backend.replacement_started.is_set()
        with pytest.raises(RuntimeError, match="cap 1"):
            await _background_result(registry, TaskArgs(task="competitor"), ctx)
        competing = await _background_result(
            registry,
            TaskArgs(task="competitor", agent_id=old.agent_id, replace_run=True),
            ctx,
        )
        assert competing.launch_outcome is LaunchOutcome.REJECTED_RESERVATION
        with pytest.raises(AgentBusyError):
            await _background_result(
                registry, TaskArgs(task="reuse", agent_id=old.agent_id), ctx
            )
        backend.cleanup_release.set()
        await asyncio.wait_for(finalized.wait(), 2)
        assert not replacement.done() and record.state is _AgentState.RUNNING
        assert registry._active_work_slots == {slot}
        assert not registry._eligible_idle_locked()
        assert not await registry._evict_agent(record.agent_id, "ttl")
        assert await registry.get_agent_result(old.agent_id, old.run_id) is not None
        with pytest.raises(RuntimeError, match="cap 1"):
            await _background_result(registry, TaskArgs(task="gap competitor"), ctx)
        assert registry._root is not None
        with pytest.raises(RuntimeError, match="background agents are running"):
            await registry.reserve_resume_admission(registry._root)
        finalize_release.set()
        new = await asyncio.wait_for(replacement, 2)
        await asyncio.wait_for(backend.replacement_started.wait(), 2)
        assert new.status == "launched" and new.launch_outcome is LaunchOutcome.LAUNCHED
        assert new.metadata is not None
        assert new.metadata["replaced_run_id"] == old.run_id
        assert new.metadata["replacement_run_id"] == new.run_id != old.run_id
        assert registry._agent_records[old.agent_id] is record
        assert record.session_id == session and record.work_slot is slot
        assert not record.reserved_for_replacement
        assert registry._active_work_slots == {slot}
        prompts = [
            str(message.content) for message in record.runtime.agent_loop.messages
        ]
        assert any("old task" in prompt for prompt in prompts)
        assert any(
            "This task supersedes the interrupted task.\n\nnew task" in prompt
            for prompt in prompts
        )
        terminal = await _wait_result(registry, old.agent_id, old.run_id)
        assert terminal.stop_reason is RunStopReason.RETASKED and not terminal.completed
        backend.release.set()
        await registry.wait_for_agent(old.agent_id, new.run_id)
        assert not registry._active_work_slots
    finally:
        backend.release.set()
        backend.cleanup_release.set()
        finalize_release.set()
        if replacement is not None and not replacement.done():
            replacement.cancel()
            await asyncio.gather(replacement, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_overlapping_post_ack_retask_finalizers_are_owner_fenced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = RetaskBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    first = None
    second = None
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        await backend.started.wait()
        backend.cleanup_release.set()
        first = registry.run(
            TaskArgs(task="first", agent_id=old.agent_id, replace_run=True), ctx
        )
        ack = await anext(first)
        assert isinstance(ack, TaskResult) and ack.status == "launched"
        await backend.replacement_started.wait()

        async def gate_update(*_args):
            if record.reserved_for_replacement and record.current_run is None:
                entered.set()
                await release.wait()

        registry._notify_agents = gate_update
        second = asyncio.create_task(
            _background_result(
                registry,
                TaskArgs(task="second", agent_id=old.agent_id, replace_run=True),
                ctx,
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        owner, slot = record.replacement_owner, record.work_slot
        assert owner is not None
        # First generator is still suspended at its launch acknowledgment.
        await first.aclose()
        assert record.replacement_owner is owner and record.reserved_for_replacement
        assert registry._active_work_slots == {slot}
        release.set()
        result = await asyncio.wait_for(second, 2)
        assert result.status == "launched" and result.run_id != ack.run_id
    finally:
        release.set()
        backend.release.set()
        backend.cleanup_release.set()
        if first is not None:
            await first.aclose()
        if second is not None:
            await asyncio.gather(second, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["cleanup", "publication", "final_update"])
@pytest.mark.parametrize("cancellations", [1, 4])
async def test_retask_caller_cancel_unwinds_without_launch(
    monkeypatch: pytest.MonkeyPatch, phase: str, cancellations: int
) -> None:
    backend = RetaskBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    entered, release = asyncio.Event(), asyncio.Event()
    replacement = None
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        run = record.current_run
        assert run is not None
        await backend.started.wait()

        publications = 0

        async def gate_update(*_args):
            nonlocal publications
            if record.current_run is None:
                publications += 1
                if phase != "final_update" or publications == 2:
                    entered.set()
                    await release.wait()

        registry._notify_agents = gate_update
        replacement = asyncio.create_task(
            _background_result(
                registry,
                TaskArgs(task="replacement", agent_id=old.agent_id, replace_run=True),
                ctx,
            )
        )
        await asyncio.wait_for(backend.cleanup_started.wait(), 2)
        if phase != "cleanup":
            backend.cleanup_release.set()
            await asyncio.wait_for(entered.wait(), 2)
        async with registry._registry_lock:
            for _ in range(cancellations):
                replacement.cancel()
                # Deliver each interrupt while unwind is waiting on this lock.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            assert record.reserved_for_replacement
        with pytest.raises(asyncio.CancelledError):
            await replacement
        assert (
            not record.reserved_for_replacement and not run.completion_task.cancelled()
        )
        backend.cleanup_release.set()
        release.set()
        await asyncio.wait_for(run.completion_task, 2)
        await asyncio.gather(*tuple(registry._teardown_tasks))
        assert record.state is _AgentState.IDLE
        assert not registry._active_work_slots and backend.calls == 1
        assert run.stop_reason is RunStopReason.RETASKED
        new = await _background_result(
            registry, TaskArgs(task="reuse", agent_id=old.agent_id), ctx
        )
        assert new.status == "launched"
    finally:
        backend.release.set()
        backend.cleanup_release.set()
        release.set()
        if replacement is not None:
            await asyncio.gather(replacement, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "revocation", ["release", "drain", "authority", "generation", "admission"]
)
async def test_retask_handoff_revalidates_revocation(
    monkeypatch: pytest.MonkeyPatch, revocation: str
) -> None:
    backend = RetaskBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    entered, release = asyncio.Event(), asyncio.Event()
    replacement = None
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        run = record.current_run
        assert run is not None
        await backend.started.wait()

        async def gate_update(*_args):
            if record.reserved_for_replacement and record.current_run is None:
                entered.set()
                await release.wait()

        registry._notify_agents = gate_update
        replacement = asyncio.create_task(
            _background_result(
                registry,
                TaskArgs(task="replacement", agent_id=old.agent_id, replace_run=True),
                ctx,
            )
        )
        await asyncio.wait_for(backend.cleanup_started.wait(), 2)
        backend.cleanup_release.set()
        await asyncio.wait_for(entered.wait(), 2)
        if revocation == "release":
            await asyncio.wait_for(registry.release_agent(old.agent_id), 2)
        elif revocation == "drain":
            await asyncio.wait_for(registry.drain_children(), 2)
        elif revocation == "authority":
            parent._authority_revision += 1
        elif revocation == "generation":
            record.runtime.agent_loop._session_generation += 1
        else:
            registry._admission_closed = True
        release.set()
        result = (
            await asyncio.wait_for(
                asyncio.gather(replacement, return_exceptions=True), 2
            )
        )[0]
        assert isinstance(result, BaseException)
        assert not record.reserved_for_replacement and not registry._active_work_slots
        assert backend.calls == 1 and run.stop_reason is RunStopReason.RETASKED
    finally:
        release.set()
        backend.release.set()
        backend.cleanup_release.set()
        if replacement is not None:
            await asyncio.gather(replacement, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("restriction", ["owner", "profile", "config", "permission"])
async def test_retask_validation_precedes_stop(
    monkeypatch: pytest.MonkeyPatch, restriction: str
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        run = record.current_run
        assert run is not None
        await backend.started.wait()
        values: dict[str, Any] = {
            "task": "replacement",
            "agent_id": old.agent_id,
            "replace_run": True,
        }
        if restriction == "owner":
            record.parent_session_id = "another-parent"
        elif restriction == "profile":
            values["agent_type"] = "reviewer"
        elif restriction == "config":
            values["config"] = {}
        else:
            monkeypatch.setattr(
                registry,
                "_require_task_profile_allowed",
                MagicMock(side_effect=ToolPermissionError("disabled")),
            )
        with pytest.raises((
            ToolPermissionError,
            AgentProfileMismatchError,
            LaunchConfigError,
        )):
            await _background_result(registry, TaskArgs(**values), ctx)
        assert run.requested_stop_reason is None and not record.reserved_for_replacement
        assert not backend.stopped.is_set() and len(registry._active_work_slots) == 1
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["projection", "start"])
async def test_retask_admission_failure_keeps_old_result_and_rolls_back_slot(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    backend = RetaskBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        await backend.started.wait()
        assert registry._root is not None
        monkeypatch.setattr(
            registry._root.turns,
            "start",
            MagicMock(side_effect=RuntimeError("parent busy")),
        )
        if failure == "projection":
            registry._root.turns.link_subagent = AsyncMock(
                side_effect=RuntimeError("admission failed")
            )
        else:
            monkeypatch.setattr(
                SessionRuntimeRegistry,
                "_start_child_turn",
                MagicMock(side_effect=RuntimeError("admission failed")),
            )
        replacement = asyncio.create_task(
            _background_result(
                registry,
                TaskArgs(task="replacement", agent_id=old.agent_id, replace_run=True),
                ctx,
            )
        )
        await asyncio.wait_for(backend.cleanup_started.wait(), 2)
        backend.cleanup_release.set()
        with pytest.raises(RuntimeError, match="admission failed"):
            await asyncio.wait_for(replacement, 2)
        assert not record.reserved_for_replacement and record.state is _AgentState.IDLE
        assert not registry._active_work_slots and backend.calls == 1
        terminal = await _wait_result(registry, old.agent_id, old.run_id)
        assert terminal.stop_reason is RunStopReason.RETASKED
        assert record.latest_run_id == old.run_id
    finally:
        backend.release.set()
        backend.cleanup_release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("window", ["stopping", "execution_terminal", "finalizing"])
@pytest.mark.parametrize("config", [None, LaunchConfig(model="unused")])
async def test_busy_retask_typed_refusal_matrix(
    monkeypatch: pytest.MonkeyPatch, window: str, config: LaunchConfig | None
) -> None:
    backend = RetaskBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="retask", session_id=parent.session_id)
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        run = record.current_run
        assert run is not None
        await backend.started.wait()
        if window == "stopping":
            await registry.cancel_run(
                old.agent_id,
                reason=RunStopReason.USER_CANCELLED,
                requester_session_id=parent.session_id,
            )
            await backend.cleanup_started.wait()
        elif window == "execution_terminal":
            notify = record.runtime.turns._notify

            async def gate(method, params):
                if method == "turn/completed":
                    entered.set()
                    await release.wait()
                await notify(method, params)

            monkeypatch.setattr(record.runtime.turns, "_notify", gate)
            backend.cleanup_release.set()
            backend.release.set()
            await asyncio.wait_for(entered.wait(), 2)
        else:

            async def gate_update(*_args):
                if record.state is _AgentState.FINALIZING:
                    entered.set()
                    await release.wait()

            registry._notify_agents = gate_update
            backend.cleanup_release.set()
            backend.release.set()
            await asyncio.wait_for(entered.wait(), 2)
        result = await _background_result(
            registry,
            TaskArgs(
                task="new", agent_id=old.agent_id, replace_run=True, config=config
            ),
            ctx,
        )
        expected = (
            LaunchOutcome.ALREADY_STOPPING
            if window == "stopping"
            else LaunchOutcome.ALREADY_FINISHING
        )
        assert result.launch_outcome is expected and result.status is None
        assert not record.reserved_for_replacement and backend.calls == 1
        assert run.requested_stop_reason is (
            RunStopReason.USER_CANCELLED if window == "stopping" else None
        )
    finally:
        release.set()
        backend.release.set()
        backend.cleanup_release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retask_between_turns_interrupts_continuation_before_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="chain", session_id=parent.session_id)
    entered, joined = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        record = registry._agent_records[old.agent_id]
        run = record.current_run
        assert run is not None and run.initial_turn_id is not None
        turns = record.runtime.turns
        finalize = turns._finalize_turn

        async def linked_finalize(
            turn, status, error, stop_reason, execution, *, next_turn_id=None
        ):
            if turn.id != run.initial_turn_id:
                assert joined.is_set()
                return await finalize(
                    turn,
                    status,
                    error,
                    stop_reason,
                    execution,
                    next_turn_id=next_turn_id,
                )
            await finalize(
                turn, status, error, stop_reason, execution, next_turn_id="successor"
            )
            entered.set()
            try:
                await release.wait()
            finally:
                joined.set()

        monkeypatch.setattr(turns, "_finalize_turn", linked_finalize)
        await backend.started.wait()
        backend.release.set()
        await asyncio.wait_for(entered.wait(), 2)
        assert turns.operation_pending_turn_id(run.initial_turn_id) == "successor"
        new = await asyncio.wait_for(
            _background_result(
                registry,
                TaskArgs(task="new", agent_id=old.agent_id, replace_run=True),
                ctx,
            ),
            2,
        )
        assert joined.is_set() and new.status == "launched"
        result = await _wait_result(registry, old.agent_id, old.run_id)
        assert result.stop_reason is RunStopReason.RETASKED and not result.completed
        await registry.wait_for_agent(old.agent_id, new.run_id)
    finally:
        release.set()
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["generation", "persistence"])
async def test_retask_during_actual_compaction_joins_cleanup(
    monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="initial")],
        [mock_llm_chunk(content="<summary>compacted</summary>")],
        [mock_llm_chunk(content="replacement")],
    ])
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="compact", session_id=parent.session_id)
    entered, stopped, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    try:
        first = await _background_result(registry, TaskArgs(task="initial"), ctx)
        assert first.agent_id is not None
        await registry.wait_for_agent(first.agent_id, first.run_id)
        record = registry._agent_records[first.agent_id]
        child = record.runtime.agent_loop
        before = AutoCompactMiddleware.before_turn
        forced = False

        async def force(middleware, context):
            nonlocal forced
            if context.stats is child.stats and not forced:
                forced = True
                context.stats.context_tokens = 500_000
            return await before(middleware, context)

        monkeypatch.setattr(AutoCompactMiddleware, "before_turn", force)
        manager = child.compaction_manager
        method = "_complete" if phase == "generation" else "_save"
        original = getattr(manager, method)

        async def gated(*args, **kwargs):
            entered.set()
            try:
                await release.wait()
                return await original(*args, **kwargs)
            finally:
                stopped.set()

        monkeypatch.setattr(manager, method, gated)
        old = await _background_result(
            registry, TaskArgs(task="compact", agent_id=record.agent_id), ctx
        )
        await asyncio.wait_for(entered.wait(), 3)
        assert record.compacting
        # Do not force another compaction when the replacement enters its turn.
        monkeypatch.setattr(AutoCompactMiddleware, "before_turn", before)
        new = await asyncio.wait_for(
            _background_result(
                registry,
                TaskArgs(task="new", agent_id=record.agent_id, replace_run=True),
                ctx,
            ),
            3,
        )
        assert stopped.is_set() and not record.compacting
        result = await _wait_result(registry, record.agent_id, old.run_id)
        assert result.stop_reason is RunStopReason.RETASKED and not result.completed
        release.set()
        await asyncio.wait_for(registry.wait_for_agent(record.agent_id, new.run_id), 3)
        assert any("supersedes" in str(message.content) for message in child.messages)
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_idle_replace_is_plain_reuse_with_reconfiguration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _ = await _zero_retention_registry(
        monkeypatch, FakeBackend([[mock_llm_chunk(content="child done")]] * 2)
    )
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    ctx = InvokeContext(tool_call_id="idle", session_id=parent.session_id)
    try:
        old = await _background_result(registry, TaskArgs(task="old"), ctx)
        assert old.agent_id is not None
        await registry.wait_for_agent(old.agent_id, old.run_id)
        record = registry._agent_records[old.agent_id]
        new = await _background_result(
            registry,
            TaskArgs(
                task="idle continuation",
                agent_id=old.agent_id,
                replace_run=True,
                config=LaunchConfig(thinking="high"),
            ),
            ctx,
        )
        assert new.launch_outcome is LaunchOutcome.LAUNCHED
        assert new.metadata is not None and "replaced_run_id" not in new.metadata
        assert record.effective_thinking == "high"
        assert not any(
            "supersedes" in str(message.content)
            for message in record.runtime.agent_loop.messages
        )
        terminal = await _wait_result(registry, old.agent_id, old.run_id)
        assert terminal.completed and terminal.stop_reason is None
    finally:
        await registry.drain_children()
        await parent.aclose()


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
            # The child is reused below, so the first launch opts into
            # retention explicitly instead of closing on completion.
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
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
        first = await _wait_result(registry, record.agent_id, first_run_id)
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
        second = await _wait_result(registry, record.agent_id, second_run_id)

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
            # The child is reused below, so the first launch opts into
            # retention explicitly instead of closing on completion.
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
            task="reuse", agent_type="worker", agent_id="agent-1", background=True
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
            # Both agents stay resident across their completions: the old
            # generation's completion must not add a second runtime close,
            # and the new agent is reaped through the idle TTL below.
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(background=True, retain=True)]
                )
            ],
            [mock_llm_chunk(content="launch complete")],
            [
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _task_call(background=True, retain=True, tool_call_id="task-2")
                    ],
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
        ) == ("one/small", "off")
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
            "one/large",
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
        assert backends[-1][0] == "two"
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
            "one/small",
            "off",
        )
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
                        "read_file": LaunchToolOverride(
                            permission=ToolPermission.ALWAYS
                        )
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
        assert read_file.permission == "always"

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
