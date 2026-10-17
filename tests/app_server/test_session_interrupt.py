"""Interrupt, cancellation, legacy queue, and rewind session tests."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

from pydantic import ValidationError
import pytest

from chartreux.app_server._turns import TurnController
from chartreux.app_server.events import (
    CallbackRequested,
    TurnCompleted,
    TurnQueueUpdated,
    TurnStarted,
)
from chartreux.app_server.models import (
    ImageAttachment,
    InlineImageSource,
    PublicQueuedTurn,
    PublicTurnQueue,
    PublicTurnStatus,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentsCancelParams,
    AgentsCancelResponse,
    AppServerResponseError,
    ProtocolError,
    ProtocolErrorCode,
    SessionTextContentBlock,
    TurnQueueRemoveParams,
    TurnStartParams,
    TurnUserInputEntry,
)
from chartreux.app_server.session import AppServerSession
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.events import AssistantEvent, UserMessageEvent
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.user_content import UserDisplayContent
from tests.app_server._session_runtime_support import (
    _consume,
    _isolate_recovery_registry,  # noqa: F401 - autouse fixture registered via module namespace
    _next_event,
    _question_tool_call,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
async def test_legacy_interrupt_rejects_the_open_callback_in_core(monkeypatch) -> None:
    tool_call = _question_tool_call()
    backend = FakeBackend([[mock_llm_chunk(content="", tool_calls=[tool_call])]])
    config = build_test_vibe_config(enabled_tools=["ask_user_question"])
    agent_loop = build_test_agent_loop(
        config=config, backend=backend, enable_streaming=True
    )
    reject_request = Mock(wraps=agent_loop.reject_request)
    monkeypatch.setattr(agent_loop, "reject_request", reject_request)
    session = await create_test_app_server_session(agent_loop)
    callback_id: str | None = None

    try:
        async for event in session.act("read todos"):
            if isinstance(event, CallbackRequested):
                callback_id = event.callback.callback_id
                await session.interrupt()
    finally:
        await session.close()
        await agent_loop.aclose()

    assert callback_id is not None
    assert reject_request.call_count == 1
    assert reject_request.call_args.args[0] == callback_id
    assert isinstance(reject_request.call_args.args[1], RuntimeError)


@pytest.mark.asyncio
@pytest.mark.parametrize("run_id", [None, "captured-run"])
async def test_cancel_agent_forwards_target_and_validates_response(
    run_id: str | None,
) -> None:
    session = object.__new__(AppServerSession)
    client = AsyncMock()
    client.request.return_value = {
        "outcome": "not_running",
        "runId": run_id,
        "stopReason": None,
    }
    session._ensure_attached = AsyncMock(return_value=client)
    response = await session.cancel_agent("agent", run_id)
    assert isinstance(response, AgentsCancelResponse)
    assert response.run_id == run_id
    client.request.assert_awaited_once_with(
        "agents/cancel", AgentsCancelParams(agent_id="agent", run_id=run_id)
    )
    client.request.return_value = {"outcome": "not_running", "run_id": "private"}
    with pytest.raises(ValidationError):
        await session.cancel_agent("agent", run_id)


@pytest.mark.asyncio
async def test_interrupt_waits_for_pending_turn_events() -> None:
    session = _interrupt_session()
    client = _InterruptClient({"accepted": True, "lastEventId": 1})
    session._ensure_attached = AsyncMock(return_value=client)

    await session.interrupt()

    assert client.requests == [
        {
            "method": "turn/interrupt",
            "session_id": "session-1",
            "expected_turn_id": "turn-1",
            "wait_for_incoming": True,
        }
    ]


@pytest.mark.asyncio
async def test_inject_user_context_require_active_turn_does_not_idle_inject() -> None:
    session = object.__new__(AppServerSession)
    session._active_public_turn_id = Mock(return_value=None)
    client = _InterruptClient({"entries": []})
    session._ensure_attached = AsyncMock(return_value=client)

    with pytest.raises(AppServerResponseError, match="No active turn") as exc_info:
        await session.inject_user_context("follow up", require_active_turn=True)

    assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    assert client.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        ProtocolError(code=ProtocolErrorCode.STALE_TURN, message="Stale turn"),
        ProtocolError(code=ProtocolErrorCode.CONFLICT, message="No active turn"),
        ProtocolError(
            code=ProtocolErrorCode.CONFLICT, message="No matching active turn"
        ),
    ],
)
async def test_interrupt_ignores_already_settled_turn_errors(
    error: ProtocolError,
) -> None:
    session = _interrupt_session()
    client = _InterruptClient(AppServerResponseError(error))
    session._ensure_attached = AsyncMock(return_value=client)

    await session.interrupt()

    # The settled-turn error was swallowed, but the interrupt request was
    # still issued to the backend.
    assert client.requests == [
        {
            "method": "turn/interrupt",
            "session_id": "session-1",
            "expected_turn_id": "turn-1",
            "wait_for_incoming": True,
        }
    ]


@pytest.mark.asyncio
async def test_legacy_queue_exposes_canonical_procedures_and_minimal_results() -> None:
    agent_loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        started.set()
        await release.wait()
        yield UserMessageEvent(content=msg, message_id="active-user")
        yield AssistantEvent(content="done", message_id="active-assistant")

    agent_loop.act = blocking_act
    session = await create_test_app_server_session(agent_loop)
    active_turn = asyncio.create_task(_consume(session.act("active")))
    try:
        await started.wait()
        client = session._connection.current
        assert client is not None
        enqueue_result = await client.request(
            "app_server/session/turn/enqueue",
            {
                "idempotencyKey": "enqueue-1",
                "sessionId": session.session_id,
                "entries": [
                    {
                        "role": "user",
                        "entryId": "message-queued",
                        "content": [{"type": "text", "text": "queued"}],
                    }
                ],
            },
        )
        queue_item_id = enqueue_result["queueItemId"]

        read_result = await client.request(
            "app_server/session/turn/queue/read", {"sessionId": session.session_id}
        )
        item = read_result["queue"]["items"][0]
        remove_result = await client.request(
            "app_server/session/turn/queue/remove",
            {"sessionId": session.session_id, "queueItemId": queue_item_id},
        )
        resume_result = await client.request(
            "app_server/session/turn/queue/resume", {"sessionId": session.session_id}
        )

        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "turn/enqueue",
                {
                    "idempotencyKey": "old-route",
                    "sessionId": session.session_id,
                    "message": [{"type": "text", "text": "old"}],
                },
            )
    finally:
        release.set()
        await active_turn
        await session.close()

    assert enqueue_result == {"queueItemId": queue_item_id}
    assert isinstance(item["createdAt"], int)
    assert {key: value for key, value in item.items() if key != "createdAt"} == {
        "id": queue_item_id,
        "entries": [
            {
                "role": "user",
                "entryId": "message-queued",
                "content": [{"type": "text", "text": "queued"}],
                "annotations": {},
            }
        ],
    }
    assert remove_result == {}
    assert resume_result == {}
    assert exc_info.value.error.code is ProtocolErrorCode.METHOD_NOT_FOUND


@pytest.mark.asyncio
async def test_legacy_queue_injects_context_before_promoting_the_user_entry() -> None:
    agent_loop = build_test_agent_loop()
    calls: list[tuple[str, object]] = []
    display = UserDisplayContent(
        version="1", host="vibe", content=[{"type": "text", "text": "shown prompt"}]
    )

    async def inject_user_context(content: str, **_kwargs: object) -> list[object]:
        calls.append(("context", content))
        return []

    async def queued_act(msg: str, **kwargs: object):
        calls.append(("user", (msg, kwargs["user_display_content"])))
        yield UserMessageEvent(content=msg, message_id="queued-user")
        yield AssistantEvent(content="done", message_id="queued-assistant")

    agent_loop.inject_user_context = inject_user_context
    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    try:
        client = session._connection.current
        assert client is not None
        await client.request(
            "app_server/session/turn/enqueue",
            {
                "sessionId": session.session_id,
                "entries": [
                    {
                        "role": "context",
                        "entryId": "context-1",
                        "content": [{"type": "text", "text": "hidden context"}],
                    },
                    {
                        "role": "user",
                        "entryId": "user-1",
                        "content": [{"type": "text", "text": "visible prompt"}],
                        "annotations": {
                            "vibe.userDisplayContent": display.model_dump(mode="json")
                        },
                    },
                ],
            },
        )

        async with asyncio.timeout(2):
            while (
                session.state.latest_turn is None
                or session.state.latest_turn.status is PublicTurnStatus.IN_PROGRESS
            ):
                await asyncio.sleep(0)
    finally:
        await session.close()

    assert calls == [
        ("context", "hidden context"),
        ("user", ("visible prompt", display)),
    ]


@pytest.mark.asyncio
async def test_enqueue_while_idle_promotes_without_leaving_stale_queue_state() -> None:
    agent_loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def queued_act(msg: str, **_kwargs):
        started.set()
        await release.wait()
        yield UserMessageEvent(content=msg, message_id="queued-user")
        yield AssistantEvent(content="done", message_id="queued-assistant")

    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    try:
        queued = await session.enqueue("start from queue")
        await started.wait()
        async with asyncio.timeout(2):
            while (
                session.state.latest_turn is None
                or session.state.latest_turn.queue_item_id != queued.id
            ):
                await asyncio.sleep(0)
        assert session.turn_queue.items == []

        release.set()
        async with asyncio.timeout(2):
            while session.state.latest_turn.status is PublicTurnStatus.IN_PROGRESS:
                await asyncio.sleep(0)
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_idle_queued_turn_refreshes_persisted_session_exit_summary(
    tmp_path: Path,
) -> None:
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(enabled=True, save_dir=str(tmp_path))
    )
    backend = FakeBackend([mock_llm_chunk(content="done")])
    session = await create_test_app_server_session(
        build_test_agent_loop(config=config, backend=backend, enable_streaming=True)
    )
    events = session.events()

    try:
        assert session.exit_summary().session_id is None

        await session.enqueue("start from queue")
        await asyncio.wait_for(_next_event(events, TurnCompleted), timeout=2)

        assert session.exit_summary().session_id == session.session_id
    finally:
        await events.aclose()
        await session.close()


@pytest.mark.asyncio
async def test_enqueue_response_does_not_restore_a_turn_promoted_by_newer_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """*Prepare*: An idle enqueue whose promotion events finish before its response resumes.
    *Do*: Let the client finish reconciling the enqueue response.
    *Assert*: The already-promoted turn is not restored to the local queue projection.
    """
    # Prepare
    agent_loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def queued_act(msg: str, **_kwargs):
        started.set()
        await release.wait()
        yield UserMessageEvent(content=msg, message_id="queued-user")
        yield AssistantEvent(content="done", message_id="queued-assistant")

    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    client = session._connection.current
    assert client is not None
    request = client.request

    async def wait_for_promotion_before_returning(
        method, params=None, *, wait_for_incoming=False
    ):
        response = await request(method, params, wait_for_incoming=wait_for_incoming)
        if method != "app_server/session/turn/enqueue":
            return response
        queue_item_id = response["queueItemId"]
        async with asyncio.timeout(2):
            while (
                session.state.latest_turn is None
                or session.state.latest_turn.queue_item_id != queue_item_id
                or session.turn_queue.items
            ):
                await asyncio.sleep(0)
        return response

    monkeypatch.setattr(client, "request", wait_for_promotion_before_returning)

    try:
        # Do
        queued = await session.enqueue("start from queue")

        # Assert
        assert session.state.latest_turn is not None
        assert session.state.latest_turn.queue_item_id == queued.id
        assert session.turn_queue.items == []
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_enqueue_during_turn_finalization_waits_for_finishing_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    prompts: list[str] = []
    finalizing = asyncio.Event()
    release_finalizer = asyncio.Event()
    queue_promotion_attempted = asyncio.Event()
    gated = False
    original_after_turn_terminal = TurnController._after_turn_terminal
    original_promote_next = TurnController._promote_next

    async def queued_act(msg: str, **_kwargs):
        prompts.append(msg)
        yield UserMessageEvent(content=msg, message_id=f"user-{msg}")
        yield AssistantEvent(content=f"response {msg}", message_id=f"assistant-{msg}")

    async def gate_first_finalizer(
        self: TurnController, status: PublicTurnStatus
    ) -> None:
        nonlocal gated
        if not gated:
            gated = True
            finalizing.set()
            await release_finalizer.wait()
        await original_after_turn_terminal(self, status)

    async def observe_promotion(self: TurnController) -> None:
        if asyncio.current_task() is not self._active_task:
            queue_promotion_attempted.set()
        await original_promote_next(self)

    monkeypatch.setattr(TurnController, "_after_turn_terminal", gate_first_finalizer)
    monkeypatch.setattr(TurnController, "_promote_next", observe_promotion)
    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    first_turn = asyncio.create_task(_consume(session.act("A")))
    try:
        await finalizing.wait()
        queued = await session.enqueue("B")
        await queue_promotion_attempted.wait()

        assert [item.id for item in session.turn_queue.items] == [queued.id]

        release_finalizer.set()
        await first_turn
        async with asyncio.timeout(2):
            while (
                session.state.latest_turn is None
                or session.state.latest_turn.queue_item_id != queued.id
                or session.state.latest_turn.status is PublicTurnStatus.IN_PROGRESS
            ):
                await asyncio.sleep(0)
    finally:
        release_finalizer.set()
        first_turn.cancel()
        await session.close()

    assert prompts == ["A", "B"]
    assert session.state.turn_queue.items == []


@pytest.mark.asyncio
async def test_invalid_queue_input_is_rejected_before_acceptance() -> None:
    agent_loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        started.set()
        await release.wait()
        yield UserMessageEvent(content=msg, message_id="active-user")
        yield AssistantEvent(content="done", message_id="active-assistant")

    agent_loop.act = blocking_act
    session = await create_test_app_server_session(agent_loop)
    active_turn = asyncio.create_task(_consume(session.act("active")))
    invalid_image = ImageAttachment(
        source=InlineImageSource(data="invalid-base64"),
        alias="invalid.png",
        mime_type="image/png",
    )
    try:
        await started.wait()
        with pytest.raises(
            AppServerResponseError, match="Invalid base64 image data"
        ) as exc_info:
            await session.enqueue(
                "invalid image", images=[invalid_image], idempotency_key="reusable-key"
            )

        assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        assert session.turn_queue.items == []
        queued = await session.enqueue("valid input", idempotency_key="reusable-key")
        assert [item.id for item in session.turn_queue.items] == [queued.id]
    finally:
        release.set()
        await active_turn
        await session.close()


@pytest.mark.asyncio
async def test_queued_turn_events_follow_a_directly_consumed_turn() -> None:
    agent_loop = build_test_agent_loop()
    first_started = asyncio.Event()
    release_first = asyncio.Event()

    async def queued_act(msg: str, **_kwargs):
        if msg == "A":
            first_started.set()
            await release_first.wait()
        yield UserMessageEvent(content=msg, message_id=f"user-{msg}")
        yield AssistantEvent(content=f"response {msg}", message_id=f"assistant-{msg}")

    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    first_turn = asyncio.create_task(_consume(session.act("A")))
    events = session.events()
    try:
        await first_started.wait()
        queued = await session.enqueue("B")
        release_first.set()
        await first_turn

        received = []
        async with asyncio.timeout(2):
            while True:
                event = await anext(events)
                received.append(event)
                if (
                    isinstance(event, TurnStarted)
                    and event.turn.queue_item_id == queued.id
                ):
                    break
    finally:
        release_first.set()
        first_turn.cancel()
        await events.aclose()
        await session.close()

    assert any(isinstance(event, TurnQueueUpdated) for event in received)


@pytest.mark.asyncio
async def test_queued_turns_run_fifo_and_continue_after_failure() -> None:
    agent_loop = build_test_agent_loop()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    prompts: list[str] = []

    async def queued_act(msg: str, **_kwargs):
        prompts.append(msg)
        if msg == "A":
            first_started.set()
            await release_first.wait()
        if msg == "B":
            raise RuntimeError("B failed")
        yield UserMessageEvent(content=msg, message_id=f"user-{msg}")
        yield AssistantEvent(content=f"response {msg}", message_id=f"assistant-{msg}")

    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    first_turn = asyncio.create_task(_consume(session.act("A")))
    try:
        await first_started.wait()
        queued_b = await session.enqueue("B", message_entry_id="queue-b")
        replayed_b = await session.enqueue("B", message_entry_id="queue-b")
        queued_c = await session.enqueue(
            "C", message_entry_id="queue-c", idempotency_key="explicit-queue-c"
        )
        assert replayed_b.id == queued_b.id
        assert [item.id for item in session.turn_queue.items] == [
            queued_b.id,
            queued_c.id,
        ]

        release_first.set()
        await first_turn
        async with asyncio.timeout(2):
            while (
                session.state.latest_turn is None
                or session.state.latest_turn.queue_item_id != queued_c.id
                or session.state.latest_turn.status is not PublicTurnStatus.COMPLETED
            ):
                await asyncio.sleep(0)
    finally:
        release_first.set()
        first_turn.cancel()
        await session.close()

    assert prompts == ["A", "B", "C"]
    assert session.state.turn_queue.items == []
    queued_turns = [
        turn for turn in session.state.turns or [] if turn.queue_item_id is not None
    ]
    assert [turn.queue_item_id for turn in queued_turns] == [queued_b.id, queued_c.id]
    assert [turn.status for turn in queued_turns] == [
        PublicTurnStatus.FAILED,
        PublicTurnStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_interrupt_pauses_queue_until_remove_and_resume() -> None:
    agent_loop = build_test_agent_loop()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    prompts: list[str] = []

    async def queued_act(msg: str, **_kwargs):
        prompts.append(msg)
        if msg == "A":
            first_started.set()
            await release_first.wait()
        yield UserMessageEvent(content=msg, message_id=f"user-{msg}")
        yield AssistantEvent(content=f"response {msg}", message_id=f"assistant-{msg}")

    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    first_turn = asyncio.create_task(_consume(session.act("A")))
    try:
        await first_started.wait()
        client = session._connection.current
        assert client is not None
        queued_b = await session.enqueue("B", idempotency_key="queue-b")
        queued_c = await session.enqueue("C", idempotency_key="queue-c")

        await session.interrupt()
        await first_turn
        queue = await session.read_turn_queue()
        assert queue.paused
        assert [item.id for item in queue.items] == [queued_b.id, queued_c.id]
        assert prompts == ["A"]
        with pytest.raises(AppServerResponseError) as exc_info:
            await client.request(
                "turn/start",
                TurnStartParams(
                    session_id=session.session_id,
                    message=[TextContentBlock(text="overtake")],
                ),
            )
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT

        assert await session.remove_queued_turn(queued_b.id)
        assert [item.id for item in session.turn_queue.items] == [queued_c.id]
        resumed = await session.resume_turn_queue()
        assert not resumed.paused

        async with asyncio.timeout(2):
            while (
                session.state.latest_turn is None
                or session.state.latest_turn.queue_item_id != queued_c.id
                or session.state.latest_turn.status is not PublicTurnStatus.COMPLETED
            ):
                await asyncio.sleep(0)
    finally:
        release_first.set()
        first_turn.cancel()
        await session.close()

    assert prompts == ["A", "C"]
    assert session.state.turn_queue.items == []
    assert not session.state.turn_queue.paused


@pytest.mark.asyncio
async def test_replace_queued_turn_preserves_identity_and_fifo_order() -> None:
    agent_loop = build_test_agent_loop()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    prompts: list[str] = []

    async def queued_act(msg: str, **_kwargs):
        prompts.append(msg)
        if msg == "active":
            first_started.set()
            await release_first.wait()
        yield UserMessageEvent(content=msg, message_id=f"user-{msg}")
        yield AssistantEvent(content="done", message_id=f"assistant-{msg}")

    agent_loop.act = queued_act
    session = await create_test_app_server_session(agent_loop)
    active_turn = asyncio.create_task(_consume(session.act("active")))
    try:
        await first_started.wait()
        queued_a = await session.enqueue("A", message_entry_id="queue-a")
        queued_b = await session.enqueue("B", message_entry_id="queue-b")

        replaced = await session.replace_queued_turn(
            queued_a.id, "A-edited", message_entry_id="queue-a"
        )

        assert replaced is not None
        assert replaced.id == queued_a.id
        assert [item.id for item in session.turn_queue.items] == [
            queued_a.id,
            queued_b.id,
        ]
        assert session.turn_queue.items[0].entries == [
            TurnUserInputEntry(
                entry_id="queue-a", content=[SessionTextContentBlock(text="A-edited")]
            )
        ]

        release_first.set()
        await active_turn
        async with asyncio.timeout(2):
            while prompts != ["active", "A-edited", "B"]:
                await asyncio.sleep(0)
    finally:
        release_first.set()
        active_turn.cancel()
        with suppress(asyncio.CancelledError):
            await active_turn
        await session.close()

    assert prompts == ["active", "A-edited", "B"]


@pytest.mark.asyncio
async def test_replace_queued_turn_returns_none_after_removal() -> None:
    agent_loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        started.set()
        await release.wait()
        yield UserMessageEvent(content=msg, message_id="active-user")

    agent_loop.act = blocking_act
    session = await create_test_app_server_session(agent_loop)
    active_turn = asyncio.create_task(_consume(session.act("active")))
    try:
        await started.wait()
        queued = await session.enqueue("queued")
        assert await session.remove_queued_turn(queued.id)

        replaced = await session.replace_queued_turn(queued.id, "edited")

        assert replaced is None
    finally:
        release.set()
        await active_turn
        await session.close()


@pytest.mark.asyncio
async def test_queue_read_does_not_restore_an_item_removed_by_a_newer_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        started.set()
        await release.wait()
        yield UserMessageEvent(content=msg, message_id="active-user")
        yield AssistantEvent(content="done", message_id="active-assistant")

    agent_loop.act = blocking_act
    session = await create_test_app_server_session(agent_loop)
    active_turn = asyncio.create_task(_consume(session.act("active")))
    read_captured = asyncio.Event()
    release_read = asyncio.Event()
    read_task: asyncio.Task[PublicTurnQueue] | None = None
    try:
        await started.wait()
        queued = await session.enqueue("queued")
        client = session._connection.current
        assert client is not None
        request = client.request

        async def delay_queue_read(method, params=None, *, wait_for_incoming=False):
            response = await request(
                method, params, wait_for_incoming=wait_for_incoming
            )
            if method == "app_server/session/turn/queue/read":
                read_captured.set()
                await release_read.wait()
            return response

        monkeypatch.setattr(client, "request", delay_queue_read)
        read_task = asyncio.create_task(session.read_turn_queue())
        await read_captured.wait()

        await request(
            "app_server/session/turn/queue/remove",
            TurnQueueRemoveParams(
                session_id=session.session_id, queue_item_id=queued.id
            ),
        )
        async with asyncio.timeout(2):
            while session.turn_queue.items:
                await asyncio.sleep(0)

        release_read.set()
        queue = await read_task

        assert queue.items == []
        assert session.turn_queue.items == []
    finally:
        release_read.set()
        release.set()
        if read_task is not None and not read_task.done():
            read_task.cancel()
            with suppress(asyncio.CancelledError):
                await read_task
        await active_turn
        await session.close()


@pytest.mark.asyncio
async def test_remove_queued_turn_reports_concurrent_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_loop = build_test_agent_loop()
    first_started = asyncio.Event()
    second_started = asyncio.Event()
    release_first = asyncio.Event()
    release_second = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        if msg == "A":
            first_started.set()
            await release_first.wait()
        else:
            second_started.set()
            await release_second.wait()
        yield UserMessageEvent(content=msg, message_id=f"user-{msg}")
        yield AssistantEvent(content="done", message_id=f"assistant-{msg}")

    agent_loop.act = blocking_act
    session = await create_test_app_server_session(agent_loop)
    first_turn = asyncio.create_task(_consume(session.act("A")))
    queued: PublicQueuedTurn | None = None
    try:
        await first_started.wait()
        queued = await session.enqueue("B")
        client = session._connection.current
        assert client is not None
        request = client.request

        async def promote_before_remove(
            method, params=None, *, wait_for_incoming=False
        ):
            if method == "app_server/session/turn/queue/remove":
                release_first.set()
                await second_started.wait()
            return await request(method, params, wait_for_incoming=wait_for_incoming)

        monkeypatch.setattr(client, "request", promote_before_remove)

        assert not await session.remove_queued_turn(queued.id)
    finally:
        release_first.set()
        release_second.set()
        await first_turn
        if queued is not None:
            async with asyncio.timeout(2):
                while (
                    session.state.latest_turn is None
                    or session.state.latest_turn.queue_item_id != queued.id
                    or session.state.latest_turn.status is PublicTurnStatus.IN_PROGRESS
                ):
                    await asyncio.sleep(0)
        await session.close()


@pytest.mark.asyncio
async def test_rewind_clears_paused_queue_and_emits_queue_update() -> None:
    agent_loop = build_test_agent_loop()
    original_session_id = agent_loop.session_id
    replacement_session_id = f"{original_session_id}-rewind"
    user_index = len(agent_loop.messages)
    agent_loop.messages.extend([
        LLMMessage(role=Role.user, content="rewind me", message_id="user-1"),
        LLMMessage(role=Role.assistant, content="response", message_id="assistant-1"),
    ])
    started = asyncio.Event()

    async def blocking_act(msg: str, **_kwargs):
        started.set()
        await asyncio.Event().wait()
        yield UserMessageEvent(content=msg, message_id="active-user")

    async def rewind(
        message_index: int, *, restore_files: bool, inplace: bool = False
    ) -> tuple[str, list[str], list[str]]:
        assert message_index == user_index
        assert not restore_files
        assert not inplace
        agent_loop.session_id = replacement_session_id
        agent_loop.parent_session_id = original_session_id
        agent_loop.session_logger.reset_session(
            replacement_session_id, parent_session_id=original_session_id
        )
        return "rewind me", [], []

    agent_loop.act = blocking_act
    agent_loop.rewind_manager.rewind_to_message = rewind
    session = await create_test_app_server_session(agent_loop)
    events = session.events()
    active_turn = asyncio.create_task(_consume(session.act("active")))
    try:
        await started.wait()
        queued = await session.enqueue("queued")
        await session.interrupt()
        await active_turn

        paused = await asyncio.wait_for(
            _next_event(events, TurnQueueUpdated), timeout=2
        )
        assert paused.queue.paused
        assert [item.id for item in paused.queue.items] == [queued.id]

        result = await session.resources.sessions.rewind("user-1", restore_files=False)
        cleared = await asyncio.wait_for(
            _next_event(events, TurnQueueUpdated), timeout=2
        )

        assert result.message == "rewind me"
        assert cleared.queue.items == []
        assert not cleared.queue.paused
        assert session.session_id == replacement_session_id
        assert session.turn_queue.items == []
    finally:
        active_turn.cancel()
        with suppress(asyncio.CancelledError):
            await active_turn
        await events.aclose()
        await session.close()


@pytest.mark.asyncio
async def test_rewind_uses_public_history_entry_id() -> None:
    agent_loop = build_test_agent_loop()
    user_index = len(agent_loop.messages)
    agent_loop.messages.append(
        LLMMessage(role=Role.user, content="rewind me", message_id="user-1")
    )
    agent_loop.messages.append(
        LLMMessage(role=Role.assistant, content="response", message_id="assistant-1")
    )
    agent_loop.rewind_manager.restorable_paths_at = lambda message_index: (
        ["x.py"] if message_index == user_index else []
    )
    rewind = AsyncMock(return_value=("rewind me", ["restore warning"], ["x.py"]))
    agent_loop.rewind_manager.rewind_to_message = rewind
    session = await create_test_app_server_session(agent_loop)

    try:
        assert await session.resources.sessions.rewind_preview("user-1") == ["x.py"]
        assert await session.resources.sessions.rewind_has_file_changes("user-1")
        result = await session.resources.sessions.rewind(
            "user-1", restore_files=True, inplace=True
        )
    finally:
        await session.close()

    rewind.assert_awaited_once_with(user_index, restore_files=True, inplace=True)
    assert result.message == "rewind me"
    assert result.restore_errors == ["restore warning"]
    assert result.restored_paths == ["x.py"]


@pytest.mark.asyncio
async def test_rewind_reserves_session_before_restoring_files() -> None:
    agent_loop = build_test_agent_loop()
    agent_loop.messages.extend([
        LLMMessage(role=Role.user, content="rewind me", message_id="user-1"),
        LLMMessage(role=Role.assistant, content="response", message_id="assistant-1"),
    ])
    entered = asyncio.Event()
    release = asyncio.Event()

    async def rewind(
        message_index: int, *, restore_files: bool, inplace: bool = False
    ) -> tuple[str, list[str], list[str]]:
        del message_index, restore_files, inplace
        entered.set()
        await release.wait()
        return "rewind me", [], []

    agent_loop.rewind_manager.rewind_to_message = rewind
    session = await create_test_app_server_session(agent_loop)
    rewinding = asyncio.create_task(
        session.resources.sessions.rewind("user-1", restore_files=True)
    )
    await entered.wait()
    try:
        with pytest.raises(AppServerResponseError) as exc_info:
            await _consume(session.act("racing prompt"))
        assert exc_info.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        release.set()
        await rewinding
        await session.close()


def _interrupt_session() -> AppServerSession:
    session = object.__new__(AppServerSession)
    session._state = cast(Any, SimpleNamespace(session_id="session-1"))
    session._active_public_turn_id = Mock(return_value="turn-1")
    return cast(AppServerSession, session)


class _InterruptClient:
    def __init__(self, result: dict[str, Any] | Exception) -> None:
        self._result = result
        self.requests: list[dict[str, Any]] = []

    async def request(
        self, method: str, params: Any, *, wait_for_incoming: bool = False
    ) -> dict[str, Any]:
        self.requests.append({
            "method": method,
            "session_id": params.session_id,
            "expected_turn_id": params.expected_turn_id,
            "wait_for_incoming": wait_for_incoming,
        })
        if isinstance(self._result, Exception):
            raise self._result
        return self._result
