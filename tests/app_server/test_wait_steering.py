from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server import _runtime as runtime_module
from chartreux.app_server._execution import SessionExecutionKind
from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._turns import StaleTurnError, SteerReceipt, TurnConflictError
from chartreux.app_server.client import AppServerClient, AppServerResponseError
from chartreux.app_server.events import ClientProjection, HistoryEntryAdded, TurnUpdated
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicEffectEntry,
    PublicSession,
    PublicSessionState,
    PublicTurnStatus,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    HistoryEntryUpdatedParams,
    Notification,
    ProtocolErrorCode,
    SessionSnapshotParams,
    TurnStartParams,
    TurnSteerParams,
    TurnSteerResponse,
)
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import FunctionCall, Role, ToolCall
from chartreux.core.subagents import SubagentRunnerPort, TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.core.usage import UsageOutcome, UsagePurpose, UsageRecord
from tests.agent_loop.test_agent_wait_steering import WaitManager, make_loop, wait_call
from tests.app_server.test_subagents import BlockingBackend, _background_result
from tests.app_server.test_usage_attribution import AttemptBackend, open_test_root
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend


async def until(predicate) -> None:
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0)


async def waiting_runtime(loop: AgentLoop | None = None):
    if loop is None:
        loop, _ = make_loop([wait_call("child")])
    manager = WaitManager()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(loop)
    root.turns._subagent_runner = cast(SubagentRunnerPort, manager)
    response, action = root.turns.start(
        TurnStartParams(
            session_id=loop.session_id, message=[TextContentBlock(text="start")]
        )
    )
    action()
    await until(lambda: loop.is_waiting_only(response.turn.id))
    return loop, manager, root, response.turn.id


def steer_params(loop, turn_id: str, **kwargs) -> TurnSteerParams:
    return TurnSteerParams(
        session_id=loop.session_id,
        expected_turn_id=turn_id,
        message=[TextContentBlock(text="steering")],
        idempotency_key="steer:test",
        inject_invoked_skill=False,
        require_waiting_only=True,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rejection",
    ["stale", "session", "mixed", "lifecycle", "shell", "handoff", "closing", "no_key"],
)
async def test_rejected_admission_never_cancels_waits(rejection: str) -> None:
    loop, manager, root, turn_id = await waiting_runtime()
    params = steer_params(loop, turn_id)
    execution = root.execution.active
    assert execution is not None
    original = loop.is_waiting_only
    try:
        if rejection == "stale":
            params.expected_turn_id = "retired"
        elif rejection == "session":
            params.session_id = "another-parent"
        elif rejection == "mixed":
            # A stale true hint is never admission authority.
            assert root.turns.active_turn is not None
            root.turns.active_turn.waiting_only = True
            loop.is_waiting_only = lambda turn_id: False
        elif rejection in {"lifecycle", "shell"}:
            root.execution.finish(execution)
            root.execution.begin(SessionExecutionKind(rejection), turn_id)
        elif rejection == "handoff":
            root.turns._handoff_pending = True
        elif rejection == "closing":
            root.turns._closing = True
        elif rejection == "no_key":
            params.idempotency_key = None
        with pytest.raises((StaleTurnError, TurnConflictError)):
            await root.turns.steer(params)
        assert manager.released == []
        assert loop.outstanding_wait_call_ids(turn_id) == ("child",)
        assert not any(m.content == "steering" for m in loop.messages)
        if rejection == "mixed":
            loop.is_waiting_only = original
    finally:
        root.turns._handoff_pending = False
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_double_conditional_retry_replays_after_completion_without_second_cancel() -> (
    None
):
    loop, manager, root, turn_id = await waiting_runtime()
    params = steer_params(loop, turn_id)
    try:
        first, retry = await asyncio.gather(
            root.turns.steer(params), root.turns.steer(params)
        )
        assert first.accepted and retry.accepted
        terminal = await root.turns.wait_for_operation(turn_id)
        assert terminal.id == turn_id
        assert terminal.status is PublicTurnStatus.COMPLETED
        assert not terminal.waiting_only
        assert (await root.turns.steer(params)).accepted
        assert manager.released == ["child"]
        assert not manager.results["child"].cancelled()
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
        tool_results = [m for m in loop.messages if m.role is Role.tool]
        assert len(tool_results) == 1
        assert tool_results[0].tool_result is not None
        assert tool_results[0].tool_result.cancelled
        changed = params.model_copy(
            update={"message": [TextContentBlock(text="other")]}
        )
        with pytest.raises(TurnConflictError, match="different payload"):
            await root.turns.steer(changed)
        assert root.turns.steer_committed(params) is True
        changed_session = params.model_copy(update={"session_id": "compacted-session"})
        assert root.turns.steer_committed(changed_session) is None
        with pytest.raises(TurnConflictError, match="different payload"):
            await root.turns.steer(changed_session)
        assert root.turns.steer_committed(changed_session) is None
        assert manager.released == ["child"]
    finally:
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["completion", "mixed"])
async def test_final_commit_revalidates_after_initial_admission(
    monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    loop, manager, root, turn_id = await waiting_runtime()
    original = loop.inject_user_context
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(loop, "inject_user_context", delayed)
    try:
        task = asyncio.create_task(root.turns.steer(steer_params(loop, turn_id)))
        await entered.wait()
        if race == "completion":
            root.turns._active_turn = None
        else:
            monkeypatch.setattr(loop, "is_waiting_only", lambda _: False)
        release.set()
        with pytest.raises(TurnConflictError):
            await task
        assert manager.released == []
        assert not any(m.content == "steering" for m in loop.messages)
    finally:
        release.set()
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["persistence", "notification", "disconnect"])
async def test_committed_ambiguity_replays_without_append_or_projection_repeat(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    loop, manager, root, turn_id = await waiting_runtime()
    params = steer_params(loop, turn_id)
    entered, release = asyncio.Event(), asyncio.Event()
    original_save = loop._save_messages
    original_emit = root.turns._emit_projected

    async def save():
        if any(m.content == "steering" for m in loop.messages):
            entered.set()
            if failure == "disconnect":
                await release.wait()
            elif failure == "persistence":
                raise OSError("disk failure after commit")
        await original_save()

    monkeypatch.setattr(loop, "_save_messages", save)
    if failure == "notification":

        async def fail_notify(*_):
            raise OSError("notification lost")

        monkeypatch.setattr(root.turns, "_emit_projected", fail_notify)
    try:
        task = asyncio.create_task(root.turns.steer(params))
        if failure == "disconnect":
            await entered.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            release.set()
        else:
            with pytest.raises(RuntimeError, match="Steering committed"):
                await task
        assert params.idempotency_key is not None
        receipt = root.turns._steer_receipts[params.idempotency_key]
        assert receipt.task is not None
        await until(lambda: receipt.task is not None and receipt.task.done())
        before = root.turns.history.copy()
        emitted = AsyncMock(wraps=original_emit)
        monkeypatch.setattr(root.turns, "_emit_projected", emitted)
        pending_count = len(receipt.pending_updates)
        if failure == "notification":
            assert pending_count > 0
        assert (await root.turns.steer(params)).accepted
        assert emitted.await_count >= pending_count
        assert not receipt.pending_updates
        assert root.turns.history == before
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
        assert receipt.event is not None
        assert manager.released == ["child"]
    finally:
        release.set()
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_committed_ambiguity_survives_compaction_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop, manager, root, turn_id = await waiting_runtime()
    params = steer_params(loop, turn_id)
    original_save = loop._save_messages

    async def fail_save():
        raise OSError("accepted delivery could not be confirmed")

    monkeypatch.setattr(loop, "_save_messages", fail_save)
    try:
        with pytest.raises(RuntimeError, match="Steering committed"):
            await root.turns.steer(params)
        assert root.turns.steer_committed(params) is True
        monkeypatch.setattr(loop, "_save_messages", original_save)
        # Idle compaction calls this real reset after summarizing the context.
        # It closes the operation and clears both receipts and completed turns.
        await root.turns.reset()
        assert not root.turns._steer_receipts
        assert not root.turns.completed_turns
        for retry in (params, params.model_copy(update={"session_id": "compacted"})):
            with pytest.raises(TurnConflictError, match="No active turn"):
                await root.turns.steer(retry)
            # A queue may re-enqueue only on a definitive False, never None.
            assert root.turns.steer_committed(retry) is None
        await root.turns.reset()
        assert root.turns.steer_committed(params) is None
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
        assert manager.released == ["child"]
    finally:
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_stranded_pending_receipts_are_bounded_and_eviction_is_ambiguous() -> (
    None
):
    loop, _, root, turn_id = await waiting_runtime()
    params = steer_params(loop, turn_id)
    try:
        await root.turns.steer(params)
        await root.turns.wait_for_operation(turn_id)
        original = root.turns._steer_receipts["steer:test"]
        for index in range(256):
            stranded = params.model_copy(
                update={"idempotency_key": f"stranded:{index}"}
            )
            root.turns._steer_receipts[stranded.idempotency_key or ""] = SteerReceipt(
                params=stranded,
                task=original.task,
                event=original.event,
                pending_updates=[AsyncMock()],
            )
            root.turns._prune_steer_receipts()
            assert len(root.turns._steer_receipts) <= 128
        evicted = params.model_copy(update={"idempotency_key": "stranded:0"})
        assert not root.turns.known_steer_receipt(evicted)
        assert root.turns.steer_committed(evicted) is None
        with pytest.raises(TurnConflictError, match="No active turn"):
            await root.turns.steer(evicted)
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
    finally:
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_receipt_capacity_rejects_new_work_without_evicting_active_receipts() -> (
    None
):
    loop, manager, root, turn_id = await waiting_runtime()
    settled = asyncio.create_task(asyncio.sleep(0, result=TurnSteerResponse()))
    await settled
    try:
        for index in range(128):
            params = steer_params(loop, turn_id).model_copy(
                update={"idempotency_key": f"active:{index}"}
            )
            root.turns._steer_receipts[params.idempotency_key or ""] = SteerReceipt(
                params=params, task=settled
            )
        with pytest.raises(TurnConflictError, match="capacity"):
            await root.turns.steer(steer_params(loop, turn_id))
        assert len(root.turns._steer_receipts) == 128
        assert manager.released == []
        assert not any(m.content == "steering" for m in loop.messages)
        assert (
            await root.turns.steer(root.turns._steer_receipts["active:0"].params)
        ).accepted
    finally:
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("recover_snapshot", [False, True])
@pytest.mark.parametrize("dedupe_history_additions", [False, True])
async def test_pending_addition_replay_after_partial_delivery_or_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    recover_snapshot: bool,
    dedupe_history_additions: bool,
) -> None:
    loop, _, root, turn_id = await waiting_runtime()
    params = steer_params(loop, turn_id)
    projection = ClientProjection(
        PublicSessionState(
            event_id=0,
            session=PublicSession(
                id=loop.session_id,
                status=IdleSessionStatus(),
                created_at=1,
                updated_at=1,
            ),
            history=[],
        )
    )
    sequence = 0
    delivered = []

    async def deliver(update):
        nonlocal sequence
        sequence += 1
        event = projection.consume(
            Notification(
                method=update.method,
                params=update.params.model_copy(
                    update={"event_id": sequence}
                ).model_dump(mode="json", by_alias=True),
            ),
            dedupe_history_additions=dedupe_history_additions,
        )
        if event is not None:
            delivered.append(event)

    async def partial_delivery(update):
        await deliver(update)
        raise OSError("delivery applied before transport failure")

    monkeypatch.setattr(root.turns, "_emit_projected", partial_delivery)
    try:
        with pytest.raises(RuntimeError, match="Steering committed"):
            await root.turns.steer(params)
        if recover_snapshot:
            sequence += 1
            state = projection.state.model_copy(deep=True)
            state.event_id = sequence
            state.history = [
                entry.model_copy(deep=True) for entry in root.turns.history
            ]
            projection.consume(
                Notification(
                    method="session/snapshot",
                    params=SessionSnapshotParams(
                        session_id=loop.session_id,
                        event_id=sequence,
                        emitted_at=1,
                        state=state,
                    ).model_dump(mode="json", by_alias=True),
                )
            )
        before = projection.state.history.copy() if projection.state.history else []
        count = len(delivered)
        monkeypatch.setattr(root.turns, "_emit_projected", deliver)
        assert (await root.turns.steer(params)).accepted
        assert projection.state.history == before
        if dedupe_history_additions:
            assert len(delivered) == count
        else:
            assert len(delivered) > count
            assert all(
                isinstance(event, HistoryEntryAdded) for event in delivered[count:]
            )
        assert projection.last_event_id == sequence
        assert not root.turns._steer_receipts["steer:test"].pending_updates
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
    finally:
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["persistence", "partial", "snapshot", "conflict"])
async def test_completed_implicit_events_survive_persistence_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, recovery: str
) -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["wait_for_agent", "read_file"],
            tools={"read_file": {"permission": "always"}},
        ),
        cwd=tmp_path,
        backend=FakeBackend([
            [mock_llm_chunk(tool_calls=[wait_call("child")])],
            [mock_llm_chunk(content="Continued")],
        ]),
    )
    loop, _, root, turn_id = await waiting_runtime(loop)
    path = tmp_path / "context.txt"
    path.write_text("completed file expansion")
    params = steer_params(loop, turn_id).model_copy(
        update={
            "message": [TextContentBlock(text=f"@{path}")],
            "inject_invoked_skill": True,
        }
    )
    original_save = loop._save_messages

    async def fail_save():
        raise OSError("disk failure after expansion")

    monkeypatch.setattr(loop, "_save_messages", fail_save)
    try:
        with pytest.raises(RuntimeError, match="Steering committed"):
            await root.turns.steer(params)
        receipt = root.turns._steer_receipts["steer:test"]
        assert len(receipt.events) >= 3
        effects = [
            entry
            for entry in root.turns.history
            if isinstance(entry, PublicEffectEntry)
        ]
        assert any(entry.state.status == "completed" for entry in effects)
        pending = len(receipt.pending_updates)
        assert pending >= 3
        projection = ClientProjection(
            PublicSessionState(
                event_id=0,
                session=PublicSession(
                    id=loop.session_id,
                    status=IdleSessionStatus(),
                    created_at=1,
                    updated_at=1,
                ),
                history=[],
            )
        )
        sequence = 0

        async def deliver(update):
            nonlocal sequence
            sequence += 1
            projection.consume(
                Notification(
                    method=update.method,
                    params=update.params.model_copy(
                        update={"event_id": sequence}
                    ).model_dump(mode="json", by_alias=True),
                )
            )

        emitted = AsyncMock(wraps=deliver)
        monkeypatch.setattr(loop, "_save_messages", original_save)
        messages = list(loop.messages)
        before = projection.state.model_copy(deep=True)
        if recovery != "persistence":

            async def fail_completion(update):
                if update.method == "history/entryUpdated":
                    if recovery != "snapshot":
                        await deliver(update)
                    raise OSError("completion delivery failed")
                await deliver(update)

            monkeypatch.setattr(root.turns, "_emit_projected", fail_completion)
            with pytest.raises(OSError, match="completion delivery failed"):
                await root.turns.steer(params)
            assert receipt.pending_updates
            completion = receipt.pending_updates[0]
            assert completion.method == "history/entryUpdated"
            assert isinstance(completion.params, HistoryEntryUpdatedParams)
            assert any(op.path == "/updatedAt" for op in completion.params.patch)
            if recovery == "snapshot":
                sequence += 1
                state = projection.state.model_copy(deep=True)
                state.event_id = sequence
                state.history = [
                    entry.model_copy(deep=True) for entry in root.turns.history
                ]
                projection.consume(
                    Notification(
                        method="session/snapshot",
                        params=SessionSnapshotParams(
                            session_id=loop.session_id,
                            event_id=sequence,
                            emitted_at=1,
                            state=state,
                        ).model_dump(mode="json", by_alias=True),
                    )
                )
            completion_entry_id = completion.params.entry_id
            if recovery == "conflict":
                effect = next(
                    entry
                    for entry in projection.history
                    if isinstance(entry, PublicEffectEntry)
                    and entry.id == completion_entry_id
                )
                effect.state = effect.state.model_copy(update={"output": "conflict"})
            before = projection.state.model_copy(deep=True)
            pending = len(receipt.pending_updates)
        monkeypatch.setattr(root.turns, "_emit_projected", emitted)
        if recovery == "conflict":
            with pytest.raises(ValueError, match="frozen"):
                await root.turns.steer(params)
            assert len(receipt.pending_updates) == pending
            assert projection.history == before.history
            assert list(loop.messages) == messages
            return
        assert (await root.turns.steer(params)).accepted
        assert emitted.await_count == pending
        if recovery != "persistence":
            assert projection.history == before.history
        assert list(loop.messages) == messages
        assert any(
            isinstance(entry, PublicEffectEntry) and entry.state.status == "completed"
            for entry in projection.history
        )
        assert not receipt.pending_updates
    finally:
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_ordered_wait_updates_and_definitive_rejection_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from chartreux.app_server import _handler as handler_module

    validations = []
    original_validate = handler_module.validate_wire

    def validate(model, value):
        if model is TurnSteerParams:
            validations.append(value)
        return original_validate(model, value)

    monkeypatch.setattr(handler_module, "validate_wire", validate)
    loop, _ = make_loop([wait_call("child")])
    manager = WaitManager()
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(loop, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    turns = legacy_backend(server).session.turns
    turns._subagent_runner = cast(SubagentRunnerPort, manager)
    received = []

    async def consume():
        async for event in session.act("start"):
            received.append(event)

    consumer = asyncio.create_task(consume())
    try:
        await until(lambda: session.waiting_only)
        assert turns.active_turn is not None
        turn_id = turns.active_turn.id
        with pytest.raises(AppServerResponseError) as error:
            await session.inject_user_context(
                "rejected",
                require_waiting_only=True,
                expected_turn_id="retired",
                idempotency_key="steer:stale",
            )
        assert error.value.error.code is ProtocolErrorCode.STALE_TURN
        assert isinstance(error.value.error.data, dict)
        assert error.value.error.data["steerCommitted"] is False
        assert manager.released == []
        await session.inject_user_context(
            "steering",
            require_waiting_only=True,
            expected_turn_id=turn_id,
            idempotency_key="steer:ordered",
        )
        assert len(validations) == 2  # one validation per steering dispatch
        await asyncio.wait_for(consumer, 3)
        assert [
            e.turn.waiting_only for e in received if isinstance(e, TurnUpdated)
        ] == [True, False]
        assert not session.waiting_only
        # A known receipt bypasses the lifecycle guard and the active-turn route.
        execution = legacy_backend(server).session.execution
        reserved = execution.begin(SessionExecutionKind.LIFECYCLE, "next")
        try:
            await session.inject_user_context(
                "steering",
                require_waiting_only=True,
                expected_turn_id=turn_id,
                idempotency_key="steer:ordered",
            )
            with pytest.raises(AppServerResponseError) as rejected:
                await session.inject_user_context(
                    "new",
                    require_waiting_only=True,
                    expected_turn_id=turn_id,
                    idempotency_key="steer:new",
                )
            assert isinstance(rejected.value.error.data, dict)
            assert rejected.value.error.data["steerCommitted"] is False
        finally:
            execution.finish(reserved)
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
        # A compacted client's replay uses its new session identity. The frozen
        # dispatch path must not turn the original acceptance into a rejection.
        session.state.session.id = "compacted-session"
        with pytest.raises(AppServerResponseError) as changed_session:
            await session.inject_user_context(
                "steering",
                require_waiting_only=True,
                expected_turn_id=turn_id,
                idempotency_key="steer:ordered",
            )
        data = changed_session.value.error.data
        assert not isinstance(data, dict) or data.get("steerCommitted") is not False
        assert len([m for m in loop.messages if m.content == "steering"]) == 1
    finally:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        await session.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_mixed_batch_admission_tracks_remaining_invocations(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[
                    wait_call("child"),
                    ToolCall(
                        id="other",
                        index=1,
                        function=FunctionCall(
                            name="bash", arguments='{"command": "printf unrelated"}'
                        ),
                    ),
                ]
            )
        ],
        [mock_llm_chunk(content="Continued")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["wait_for_agent", "bash"],
            tools={"bash": {"permission": "always"}},
        ),
        backend=backend,
    )
    manager = WaitManager()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(loop)
    root.turns._subagent_runner = cast(SubagentRunnerPort, manager)
    entered, release = asyncio.Event(), asyncio.Event()
    original = loop._run_post_tool_hooks

    async def collect(call, **kwargs):
        if call.call_id == "other":
            entered.set()
            await release.wait()
        async for event in original(call, **kwargs):
            yield event

    monkeypatch.setattr(loop, "_run_post_tool_hooks", collect)
    try:
        response, action = root.turns.start(
            TurnStartParams(
                session_id=loop.session_id, message=[TextContentBlock(text="start")]
            )
        )
        action()
        turn_id = response.turn.id
        await asyncio.wait_for(entered.wait(), 3)
        await manager.entered.wait()
        params = steer_params(loop, turn_id)
        assert not loop.is_waiting_only(turn_id)
        with pytest.raises(TurnConflictError, match="not waiting-only"):
            await root.turns.steer(params)
        assert manager.released == []
        if explicit:
            params.require_waiting_only = False
            params.idempotency_key = "steer:explicit"
            await root.turns.steer(params)
            await until(lambda: manager.released == ["child"])
            unrelated = loop._tool_invocations[(turn_id, "other")]
            assert not unrelated.task.done() and not unrelated.task.cancelling()
            assert root.turns.active_turn is not None
            assert root.turns.active_turn.id == turn_id
            release.set()
        else:
            release.set()
            await until(lambda: loop.is_waiting_only(turn_id))
            params.idempotency_key = "steer:remaining"
            await root.turns.steer(params)
        terminal = await root.turns.wait_for_operation(turn_id)
        assert terminal.status is PublicTurnStatus.COMPLETED
        results = [m for m in loop.messages if m.role is Role.tool]
        assert len(results) == 2
        wait_result = next(m for m in results if m.tool_call_id == "child").tool_result
        other_result = next(m for m in results if m.tool_call_id == "other").tool_result
        assert wait_result is not None and wait_result.cancelled
        assert other_result is not None and not other_result.cancelled
        assert manager.released == ["child"]
    finally:
        release.set()
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_pending_callback_guards_only_conditional_steering(
    explicit: bool,
) -> None:
    loop, manager, root, turn_id = await waiting_runtime()
    root.turns._callbacks["pending"] = AsyncMock(core_resolved=False)
    params = steer_params(loop, turn_id)
    params.require_waiting_only = not explicit
    try:
        if explicit:
            assert (await root.turns.steer(params)).accepted
            assert manager.released == ["child"]
        else:
            with pytest.raises(TurnConflictError, match="cannot accept steering"):
                await root.turns.steer(params)
            assert manager.released == []
    finally:
        root.turns._callbacks.clear()
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["waiting", "completion", "interrupt"])
async def test_delayed_skill_finishes_before_wait_cancellation(
    monkeypatch: pytest.MonkeyPatch, race: str
) -> None:
    from chartreux.core.agent_loop import _loop as loop_module
    from chartreux.core.skills.models import ParsedSkillCommand, SkillInfo

    loop, manager, root, turn_id = await waiting_runtime()
    skill = SkillInfo(name="slow", description="test", prompt="Full skill context")
    monkeypatch.setattr(
        loop.skill_manager,
        "parse_skill_command",
        lambda _: ParsedSkillCommand(name=skill.name, content=""),
    )
    monkeypatch.setattr(loop.skill_manager, "get_skill", lambda _: skill)
    original = loop_module.build_skill_result
    provider_requests = []
    perform = loop._perform_llm_turn

    async def next_provider():
        assert any(
            m.name == "skill" and "Full skill context" in (m.content or "")
            for m in loop.messages
        )
        provider_requests.append(True)
        async for event in perform():
            yield event

    monkeypatch.setattr(loop, "_perform_llm_turn", next_provider)
    entered, release = asyncio.Event(), asyncio.Event()

    async def delayed(*args, **kwargs):
        entered.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(loop_module, "build_skill_result", delayed)
    params = steer_params(loop, turn_id)
    params.inject_invoked_skill = True
    params.message = [TextContentBlock(text="/slow")]
    task = asyncio.create_task(root.turns.steer(params))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        assert manager.released == []
        assert loop.outstanding_wait_call_ids(turn_id) == ("child",)
        assert root.turns.active_turn is not None
        if race == "interrupt":
            from chartreux.app_server.protocol import TurnInterruptParams

            root.turns.interrupt(
                TurnInterruptParams(
                    session_id=loop.session_id, expected_turn_id=turn_id
                )
            )
            await root.turns.wait_for_operation(turn_id)
            assert not loop._steering_injections
            assert not any(m.name == "skill" for m in loop.messages)
            assert task.done()
            return
        if race == "completion":
            from chartreux.core.subagents import TaskResult

            barrier_entered = asyncio.Event()
            settle = loop._settle_steering_injections

            async def barrier(**kwargs):
                barrier_entered.set()
                await settle(**kwargs)

            monkeypatch.setattr(loop, "_settle_steering_injections", barrier)
            manager.results["child"].set_result(
                TaskResult(response="done", turns_used=1, completed=True)
            )
            await asyncio.wait_for(barrier_entered.wait(), 3)
            assert root.turns.active_turn is not None
            assert not task.done()
        release.set()
        assert (await task).accepted
        await root.turns.wait_for_operation(turn_id)
        assert manager.released == ["child"]
        skill_result = next(m for m in loop.messages if m.name == "skill")
        assert "Full skill context" in (skill_result.content or "")
        assert provider_requests == [True]
        assert not loop._steering_injections
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await root.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_receipt_window_evicts_retired_not_active_or_inflight() -> None:
    loop, _, root, turn_id = await waiting_runtime()

    async def finish(delay: float) -> TurnSteerResponse:
        await asyncio.sleep(delay)
        return TurnSteerResponse()

    settled = asyncio.create_task(finish(0))
    await settled
    inflight = asyncio.create_task(finish(60))
    try:
        assert root.turns.active_turn is not None
        for index in range(70):
            params = steer_params(loop, f"retired-{index}")
            root.turns._completed_turns.append(
                root.turns.active_turn.model_copy(
                    update={
                        "id": params.expected_turn_id,
                        "status": PublicTurnStatus.COMPLETED,
                    }
                )
            )
            root.turns._steer_receipts[str(index)] = SteerReceipt(
                params=params, task=settled
            )
        root.turns._steer_receipts["active"] = SteerReceipt(
            params=steer_params(loop, turn_id), task=settled
        )
        root.turns._steer_receipts["inflight"] = SteerReceipt(
            params=steer_params(loop, "retired"), task=inflight
        )
        root.turns._steer_receipts["undelivered"] = SteerReceipt(
            params=steer_params(loop, "retired-0"),
            task=settled,
            pending_updates=[AsyncMock()],
        )
        root.turns._prune_steer_receipts()
        assert len(root.turns._steer_receipts) == 67
        assert "0" not in root.turns._steer_receipts
        assert root.turns.steer_committed(steer_params(loop, "retired-0")) is None
        assert root.turns.steer_committed(steer_params(loop, "retired-69")) is False
        assert "active" in root.turns._steer_receipts
        assert "inflight" in root.turns._steer_receipts
    finally:
        inflight.cancel()
        await asyncio.gather(inflight, return_exceptions=True)
        root.turns._steer_receipts.clear()
        await root.close()
        await loop.aclose()


class AccountedBlockingBackend(BlockingBackend):
    async def complete(self, **kwargs):
        from chartreux.core.llm.backend.generic import notify_request_started

        notify_request_started()
        return await super().complete(**kwargs)


@pytest.mark.asyncio
async def test_real_child_survives_parent_wait_steering_and_ledger_is_unchanged(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    process = runtime_module.HarnessProcess()
    parent_backend = AttemptBackend()
    parent = await open_test_root(
        process, monkeypatch, tmp_path, parent_backend, logging_enabled=False
    )
    child_backend = AccountedBlockingBackend()
    monkeypatch.setattr(
        runtime_module,
        "AgentLoop",
        lambda **kwargs: AgentLoop(backend=child_backend, **kwargs),
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _: 0, runtime_factory=process.runtime_factory
    )
    root = registry._build_child_runtime(parent)
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="child work"),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child_backend.started.wait(), 3)
        call = wait_call(launch.agent_id)
        parent_backend._streams.extend([
            [mock_llm_chunk(content="Waiting", tool_calls=[call])],
            [mock_llm_chunk(content="Parent continued")],
        ])
        response, action = root.turns.start(
            TurnStartParams(
                session_id=parent.session_id, message=[TextContentBlock(text="start")]
            )
        )
        action()
        turn_id = response.turn.id
        await until(lambda: parent.is_waiting_only(turn_id))
        key = (launch.agent_id, launch.run_id)
        await until(lambda: key in registry._wait_leases)
        assert registry._wait_leases[key] == 1
        path = config_dir / "usage" / parent.session_id / "usage.jsonl"
        preceding = [
            UsageRecord.model_validate_json(s) for s in path.read_text().splitlines()
        ]
        assert len(preceding) == 1
        assert preceding[0].session_id == parent.session_id
        assert preceding[0].outcome is UsageOutcome.COMPLETED
        await root.turns.steer(steer_params(parent, turn_id))
        terminal = await root.turns.wait_for_operation(turn_id)
        assert terminal.status is PublicTurnStatus.COMPLETED
        await until(lambda: key not in registry._wait_leases)
        assert not child_backend.stopped.is_set()
        assert await registry.get_agent_result(launch.agent_id, launch.run_id) is None
        child_backend.release.set()
        result = await asyncio.wait_for(registry.wait_for_agent(*key), 3)
        assert result.response == "Child completed"
        assert await registry.get_agent_result(*key) is result
        records = [
            UsageRecord.model_validate_json(s) for s in path.read_text().splitlines()
        ]
        assert (
            len(records) == 3
        )  # two parent provider calls, one child call; no wait record
        parents = [r for r in records if r.session_id == parent.session_id]
        children = [r for r in records if r.session_id != parent.session_id]
        assert parents[0] == preceding[0]
        assert len(parents) == 2 and len(children) == 1
        assert all(r.outcome is UsageOutcome.COMPLETED for r in records)
        assert all(r.purpose is UsagePurpose.CONVERSATION for r in records)
        assert all(r.root_session_id == parent.session_id for r in records)
        assert children[0].parent_session_id == parent.session_id
        assert children[0].agent_profile == "worker"
        assert len({r.record_id for r in records}) == 3
    finally:
        child_backend.release.set()
        await root.close()
        await registry.close()
        await process.close()
