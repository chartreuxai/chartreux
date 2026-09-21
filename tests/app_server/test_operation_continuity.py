from __future__ import annotations

import asyncio
from collections.abc import Callable
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.app_server._turns import TurnController
from chartreux.app_server.events import AppServerEvent, TurnCompleted, TurnStarted
from chartreux.app_server.models import PublicError, PublicTurn, PublicTurnStatus
from chartreux.app_server.protocol import (
    TurnCompletedParams,
    TurnStartedParams,
    TurnStartParams,
    TurnStartResponse,
)
from chartreux.app_server.session import AppServerSession, AppServerTurnError
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.conftest import build_test_agent_loop
from tests.stubs.app_server import create_test_app_server_session


def _turn(turn_id: str, session_id: str = "session") -> PublicTurn:
    return PublicTurn(
        id=turn_id,
        session_id=session_id,
        status=PublicTurnStatus.IN_PROGRESS,
        started_at=1,
    )


def test_next_turn_wire_default_and_roundtrip() -> None:
    turn = _turn("plan")
    wire = turn.model_dump(mode="json", by_alias=True)
    wire.pop("nextTurnId")
    assert PublicTurn.model_validate(wire).next_turn_id is None
    turn.next_turn_id = "implementation"
    assert (
        PublicTurn.model_validate_json(turn.model_dump_json(by_alias=True)).next_turn_id
        == "implementation"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [PublicTurnStatus.COMPLETED, PublicTurnStatus.FAILED, PublicTurnStatus.INTERRUPTED],
)
async def test_serialized_act_follows_successor(
    monkeypatch: pytest.MonkeyPatch, status: PublicTurnStatus
) -> None:
    release = asyncio.Event()
    successor_seen = asyncio.Event()
    events: list[AppServerEvent] = []

    def start(
        self: TurnController, params: TurnStartParams
    ) -> tuple[TurnStartResponse, Callable[[], None]]:
        plan = _turn("plan", params.session_id)
        successor = _turn("implementation", params.session_id)
        self._active_turn = plan

        async def drive() -> None:
            await self._notify(
                "turn/started",
                TurnStartedParams(
                    event_id=0, emitted_at=1, session_id=params.session_id, turn=plan
                ),
            )
            plan.status = PublicTurnStatus.COMPLETED
            plan.next_turn_id = successor.id
            self._completed_turns.append(plan)
            await self._notify(
                "turn/completed",
                TurnCompletedParams(
                    event_id=0, emitted_at=1, session_id=params.session_id, turn=plan
                ),
            )
            self._active_turn = successor
            await self._notify(
                "turn/started",
                TurnStartedParams(
                    event_id=0,
                    emitted_at=1,
                    session_id=params.session_id,
                    turn=successor,
                ),
            )
            try:
                await release.wait()
            except asyncio.CancelledError:
                assert status is PublicTurnStatus.INTERRUPTED
            successor.status = status
            if status is PublicTurnStatus.FAILED:
                successor.error = PublicError(message="successor failed")
            self._completed_turns.append(successor)
            self._active_turn = None
            await self._notify(
                "turn/completed",
                TurnCompletedParams(
                    event_id=0,
                    emitted_at=1,
                    session_id=params.session_id,
                    turn=successor,
                ),
            )

        def launch() -> None:
            self._active_task = asyncio.create_task(drive())

        return TurnStartResponse(turn=plan), launch

    monkeypatch.setattr(TurnController, "start", start)
    session = await create_test_app_server_session(build_test_agent_loop())
    refresh = AsyncMock()
    monkeypatch.setattr(session.resources.runtime, "refresh", refresh)

    async def consume() -> None:
        async for event in session.act("plan and implement"):
            events.append(event)
            if isinstance(event, TurnStarted) and event.turn.id == "implementation":
                successor_seen.set()

    task = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(successor_seen.wait(), 3)
        assert not task.done()
        assert session._active_public_turn_id() == "implementation"
        assert session._unsolicited_events.empty()
        assert [
            event.turn.id for event in events if isinstance(event, TurnCompleted)
        ] == ["plan"]
        refresh.assert_not_awaited()
        if status is PublicTurnStatus.INTERRUPTED:
            await session.interrupt()
        else:
            release.set()
        if status is PublicTurnStatus.FAILED:
            with pytest.raises(AppServerTurnError, match="successor failed"):
                await asyncio.wait_for(task, 3)
        else:
            await asyncio.wait_for(task, 3)
        refresh.assert_awaited_once()
        assert session._consumed_turn_id is None
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [PublicTurnStatus.COMPLETED, PublicTurnStatus.FAILED, PublicTurnStatus.INTERRUPTED],
)
async def test_operation_wait_joins_successor_task(status: PublicTurnStatus) -> None:
    controller = object.__new__(TurnController)
    plan = _turn("plan")
    successor = _turn("implementation")
    controller._active_turn = plan
    controller._completed_turns = []
    release = asyncio.Event()
    started = asyncio.Event()
    teardown = asyncio.Event()

    async def implementation() -> None:
        started.set()
        await release.wait()
        successor.status = status
        controller._completed_turns.append(successor)
        controller._active_turn = None
        await teardown.wait()

    async def planning() -> None:
        plan.status = PublicTurnStatus.COMPLETED
        plan.next_turn_id = successor.id
        controller._completed_turns.append(plan)
        controller._active_turn = successor
        controller._active_task = asyncio.create_task(implementation())

    controller._active_task = asyncio.create_task(planning())
    operation = asyncio.create_task(controller.wait_for_operation(plan.id))
    await started.wait()
    assert not operation.done()
    release.set()
    await asyncio.sleep(0)
    assert not operation.done()
    teardown.set()
    assert await asyncio.wait_for(operation, 1) is successor


@pytest.mark.asyncio
async def test_operation_wait_cancellation_does_not_cancel_controller() -> None:
    controller = object.__new__(TurnController)
    turn = _turn("implementation")
    controller._active_turn = turn
    controller._completed_turns = []
    release = asyncio.Event()

    async def execute() -> None:
        await release.wait()

    controller._active_task = asyncio.create_task(execute())
    waiter = asyncio.create_task(controller.wait_for_operation(turn.id))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    assert not controller._active_task.done()
    release.set()
    await controller._active_task


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_child_waits_for_operation_and_interrupts_successor(cancel: bool) -> None:
    registry = MagicMock(spec=SessionRuntimeRegistry)
    runtime = MagicMock()
    runtime.agent_loop.session_id = "child"
    runtime.agent_loop.messages = []
    runtime.agent_loop.session_logger.session_dir = None
    runtime.close = AsyncMock()
    registry._children = {"child": runtime}
    registry._child_links = {"child": (MagicMock(), "task")}
    registry._stored_children = {}
    registry._readable_children = {}
    registry._detached_child_ids = set()
    registry._pending_child_closes = {}
    registry._teardown_tasks = set()
    registry._track_child_close = SessionRuntimeRegistry._track_child_close.__get__(
        registry
    )
    registry._create_registered_child = AsyncMock(return_value=runtime)
    plan = _turn("plan", "child")
    successor = _turn("implementation", "replacement-child")
    runtime.turns.start.return_value = (TurnStartResponse(turn=plan), lambda: None)
    runtime.turns.active_turn = successor
    waiting = asyncio.Event()
    release = asyncio.Event()

    async def wait(turn_id: str) -> PublicTurn:
        assert turn_id == plan.id
        waiting.set()
        await release.wait()
        successor.status = PublicTurnStatus.COMPLETED
        return successor

    runtime.turns.wait_for_operation = AsyncMock(side_effect=wait)
    runtime.turns.interrupt.side_effect = lambda _params: release.set()

    context = MagicMock(
        spec=InvokeContext, tool_call_id="task", session_id="parent", is_subagent=False
    )

    async def consume() -> list[object]:
        return [
            item
            async for item in SessionRuntimeRegistry.run(
                registry,
                TaskArgs(task="implement", agent="worker", background=False),
                context,
            )
        ]

    operation = asyncio.create_task(consume())
    await asyncio.sleep(0)
    if operation.done():
        await operation
    await asyncio.wait_for(waiting.wait(), 1)
    assert not operation.done()
    if cancel:
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        params = runtime.turns.interrupt.call_args.args[0]
        assert params.expected_turn_id == successor.id
        assert params.session_id == successor.session_id
    else:
        release.set()
        result = await asyncio.wait_for(operation, 1)
        assert len(result) == 1
        runtime.turns.interrupt.assert_not_called()
    runtime.turns.wait_for_operation.assert_awaited_once_with(plan.id)
    runtime.turns.wait_for_turn.assert_not_called()


@pytest.mark.asyncio
async def test_publication_advances_routing_before_backpressure() -> None:
    session = object.__new__(AppServerSession)
    session._consumed_turn_id = "plan"
    session._event_generation = 0
    plan = _turn("plan")
    plan.status = PublicTurnStatus.COMPLETED
    plan.next_turn_id = "implementation"
    entered = asyncio.Event()
    release = asyncio.Event()

    async def put(_item: object) -> None:
        entered.set()
        await release.wait()

    session._events = MagicMock()
    session._events.put = AsyncMock(side_effect=put)
    publish = asyncio.create_task(session._publish_event(TurnCompleted(plan)))
    await entered.wait()
    assert session._consumed_turn_id == "implementation"
    assert not publish.done()
    release.set()
    await publish


@pytest.mark.asyncio
async def test_completed_operation_does_not_wait_for_unrelated_turn() -> None:
    controller = object.__new__(TurnController)
    plan = _turn("plan")
    plan.status = PublicTurnStatus.COMPLETED
    plan.next_turn_id = "implementation"
    successor = _turn("implementation")
    successor.status = PublicTurnStatus.COMPLETED
    controller._completed_turns = [plan, successor]
    controller._active_turn = _turn("queued")
    release = asyncio.Event()

    async def unrelated() -> None:
        await release.wait()

    controller._active_task = asyncio.create_task(unrelated())
    try:
        assert (
            await asyncio.wait_for(controller.wait_for_operation(plan.id), 1)
            is successor
        )
        assert not controller._active_task.done()
    finally:
        release.set()
        await controller._active_task
