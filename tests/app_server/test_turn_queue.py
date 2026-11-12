from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import time
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server._execution import SessionExecution
from chartreux.app_server._shell_record import (
    ManualShellOutcome,
    ManualShellRecord,
    build_manual_shell_record,
)
from chartreux.app_server._turn_queue import (
    TurnQueue,
    TurnQueueFullError,
    TurnQueueIdempotencyConflictError,
    TurnQueueItemNotFoundError,
)
from chartreux.app_server._turns import (
    ModelChoicePendingError,
    TurnConflictError,
    TurnController,
)
from chartreux.app_server.models import PublicTurnStatus
from chartreux.app_server.protocol import (
    SessionTextContentBlock,
    TurnEnqueueParams,
    TurnShellInputEntry,
    TurnUserInputEntry,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import LLMChunk, Role
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _params(key: str, text: str = "next prompt") -> TurnEnqueueParams:
    return TurnEnqueueParams(
        idempotency_key=key,
        session_id="session-1",
        entries=[
            TurnUserInputEntry(
                entry_id=f"user-{key}", content=[SessionTextContentBlock(text=text)]
            )
        ],
    )


def _queue(*ids: str, max_items: int = 32) -> TurnQueue:
    item_ids = iter(ids or ("queue-1", "queue-2", "queue-3"))
    return TurnQueue(
        max_items=max_items, item_id_factory=lambda: next(item_ids), clock_ms=lambda: 10
    )


def test_enqueue_builds_public_state_from_a_private_copy() -> None:
    queue = _queue()
    params = _params("one")

    result = queue.enqueue(params)
    params.entries[0].content.append(SessionTextContentBlock(text="mutated"))
    state = queue.state
    state.items.clear()

    assert not result.duplicate
    assert result.record.queued_turn.id == "queue-1"
    assert result.record.queued_turn.created_at == 10
    queued_entry = result.record.queued_turn.entries[0]
    assert isinstance(queued_entry, TurnUserInputEntry)
    assert queued_entry.entry_id == "user-one"
    assert queued_entry.content == [SessionTextContentBlock(text="next prompt")]
    assert result.record.params.entries[0].content == [
        SessionTextContentBlock(text="next prompt")
    ]
    assert len(queue) == 1
    assert queue.state.items == [result.record.queued_turn]


def test_pop_next_preserves_fifo_order() -> None:
    queue = _queue("queue-a", "queue-b")
    queue.enqueue(_params("a"))
    queue.enqueue(_params("b"))

    peeked = queue.peek_next()
    first = queue.pop_next()
    second = queue.pop_next()

    assert peeked is first
    assert first is not None and first.queued_turn.id == "queue-a"
    assert second is not None and second.queued_turn.id == "queue-b"
    assert queue.pop_next() is None


def test_duplicate_enqueue_returns_the_original_record() -> None:
    queue = _queue()
    first = queue.enqueue(_params("same"))
    duplicate = queue.enqueue(_params("same"))

    assert duplicate.duplicate
    assert duplicate.record is first.record
    assert len(queue) == 1


def test_enqueue_without_idempotency_key_creates_distinct_items() -> None:
    queue = _queue("queue-a", "queue-b")
    params = TurnEnqueueParams(
        session_id="session-1",
        entries=[
            TurnUserInputEntry(content=[SessionTextContentBlock(text="next prompt")])
        ],
    )

    first = queue.enqueue(params)
    second = queue.enqueue(params)

    assert first.record.queued_turn.id == "queue-a"
    assert second.record.queued_turn.id == "queue-b"
    assert len(queue) == 2


def test_idempotency_key_rejects_different_input() -> None:
    queue = _queue()
    queue.enqueue(_params("same", "first"))

    with pytest.raises(TurnQueueIdempotencyConflictError) as exc_info:
        queue.enqueue(_params("same", "different"))

    assert exc_info.value.idempotency_key == "same"
    assert len(queue) == 1


def test_full_queue_accepts_replay_but_rejects_new_input() -> None:
    queue = _queue(max_items=1)
    queue.enqueue(_params("one"))

    assert queue.enqueue(_params("one")).duplicate
    with pytest.raises(TurnQueueFullError) as exc_info:
        queue.enqueue(_params("two"))

    assert exc_info.value.max_items == 1
    assert len(queue) == 1


def test_enqueue_validation_failure_does_not_accept_or_reserve_input() -> None:
    queue = _queue("queue-a")
    params = _params("one")

    def reject(_params: TurnEnqueueParams) -> None:
        raise ValueError("invalid input")

    with pytest.raises(ValueError, match="invalid input"):
        queue.enqueue(params, validate=reject)

    accepted = queue.enqueue(params)
    replayed = queue.enqueue(params, validate=reject)

    assert accepted.record.queued_turn.id == "queue-a"
    assert replayed.duplicate
    assert replayed.record is accepted.record
    assert len(queue) == 1


def test_replace_preserves_queue_identity_position_and_idempotency_receipts() -> None:
    queue = _queue("queue-a", "queue-b", "queue-c")
    original = queue.enqueue(_params("a", "A"))
    queue.enqueue(_params("b", "B"))
    replacement_params = TurnEnqueueParams(
        idempotency_key="edit-a",
        session_id="session-1",
        entries=[
            TurnUserInputEntry(
                entry_id=original.record.params.message_entry_id,
                content=[SessionTextContentBlock(text="A-edited")],
            )
        ],
    )

    replaced = queue.replace("queue-a", replacement_params)
    replayed_original = queue.enqueue(_params("a", "A"))
    replayed_edit = queue.replace("queue-a", replacement_params)

    assert not replaced.duplicate
    assert [item.id for item in queue.state.items] == ["queue-a", "queue-b", "queue-c"]
    assert queue.state.items[0].created_at == 10
    assert queue.state.items[0].entries[0].content == [
        SessionTextContentBlock(text="A-edited")
    ]
    assert not replayed_original.duplicate
    assert replayed_original.record.queued_turn.id == "queue-c"
    assert replayed_edit.duplicate
    assert replayed_edit.record is replaced.record


@pytest.mark.parametrize("replacement_key", [None, "replacement"])
@pytest.mark.parametrize("retirement", [None, "promotion", "removal"])
def test_live_replacement_retains_only_current_idempotency(
    replacement_key: str | None, retirement: str | None
) -> None:
    live_ids = iter(("queue-original", "queue-new-original", "queue-new-intermediate"))
    live = TurnQueue(item_id_factory=lambda: next(live_ids), clock_ms=lambda: 10)
    live.enqueue(_params("original"))
    live.replace("queue-original", _params("intermediate", "first edit"))
    replacement = _params("replacement", "second edit").model_copy(
        update={"idempotency_key": replacement_key}
    )
    live.replace("queue-original", replacement)
    if retirement == "promotion":
        live.pop_next()
    elif retirement == "removal":
        live.remove("queue-original")

    if replacement_key is not None:
        replay = live.enqueue(replacement)
        assert replay.duplicate
        assert replay.record.queued_turn.id == "queue-original"

    for key in ("original", "intermediate"):
        assert not live.enqueue(_params(key)).duplicate
        with pytest.raises(TurnQueueIdempotencyConflictError):
            live.enqueue(_params(key, "different input"))


def test_replace_replay_rejects_a_queue_item_that_already_started() -> None:
    """*Prepare*: A queued turn is edited once, then promoted out of the queue.
    *Do*: Replay the accepted edit with the same idempotency key.
    *Assert*: The retired queue item is reported as missing instead of replayed.
    """
    # Prepare
    queue = _queue("queue-a")
    original = queue.enqueue(_params("a", "A"))
    replacement_params = TurnEnqueueParams(
        idempotency_key="edit-a",
        session_id="session-1",
        entries=[
            TurnUserInputEntry(
                entry_id=original.record.params.message_entry_id,
                content=[SessionTextContentBlock(text="A-edited")],
            )
        ],
    )
    queue.replace("queue-a", replacement_params)
    queue.pop_next()

    # Do
    with pytest.raises(TurnQueueItemNotFoundError) as exc_info:
        queue.replace("queue-a", replacement_params)

    # Assert
    assert exc_info.value.queue_item_id == "queue-a"


def test_replace_replay_rejects_a_removed_queue_item() -> None:
    """*Prepare*: A queued turn is edited once, then explicitly removed.
    *Do*: Replay the accepted edit with the same idempotency key.
    *Assert*: The retired queue item is reported as missing instead of replayed.
    """
    # Prepare
    queue = _queue("queue-a")
    original = queue.enqueue(_params("a", "A"))
    replacement_params = TurnEnqueueParams(
        idempotency_key="edit-a",
        session_id="session-1",
        entries=[
            TurnUserInputEntry(
                entry_id=original.record.params.message_entry_id,
                content=[SessionTextContentBlock(text="A-edited")],
            )
        ],
    )
    queue.replace("queue-a", replacement_params)
    queue.remove("queue-a")

    # Do
    with pytest.raises(TurnQueueItemNotFoundError) as exc_info:
        queue.replace("queue-a", replacement_params)

    # Assert
    assert exc_info.value.queue_item_id == "queue-a"


def test_remove_is_idempotent_and_clears_pause_when_empty() -> None:
    queue = _queue("queue-a", "queue-b")
    queue.enqueue(_params("a"))
    queue.enqueue(_params("b"))
    assert queue.pause()

    assert not queue.remove("missing")
    assert queue.remove("queue-b")
    assert queue.paused
    assert queue.remove("queue-a")
    assert not queue.paused
    assert not queue.remove("queue-a")


def test_pause_and_resume_are_idempotent() -> None:
    queue = _queue()

    assert not queue.pause()
    queue.enqueue(_params("one"))
    assert queue.pause()
    assert not queue.pause()
    assert queue.pop_next() is None
    assert queue.resume()
    assert not queue.resume()
    assert queue.pop_next() is not None


def test_retired_idempotency_key_replays_without_requeuing() -> None:
    queue = _queue()
    first = queue.enqueue(_params("one"))
    queue.pop_next()

    replay = queue.enqueue(_params("one"))

    assert replay.duplicate
    assert replay.record is first.record
    assert not queue


def test_retired_idempotency_key_survives_many_later_items() -> None:
    """*Prepare*: An enqueue receipt retired before 256 later keyed items.
    *Do*: Replay the original enqueue request in the same live queue.
    *Assert*: The original receipt is returned without adding another item.
    """
    # Prepare
    queue = _queue(*(f"queue-{index}" for index in range(258)))
    original = queue.enqueue(_params("original"))
    queue.pop_next()
    for index in range(256):
        queue.enqueue(_params(f"later-{index}"))
        queue.pop_next()

    # Do
    replay = queue.enqueue(_params("original"))

    # Assert
    assert replay.duplicate
    assert replay.record is original.record
    assert not queue


def test_reset_clears_queue_pause_and_idempotency() -> None:
    queue = _queue("queue-a", "queue-b")
    queue.enqueue(_params("same"))
    queue.pause()

    queue.reset()
    result = queue.enqueue(_params("same"))

    assert not result.duplicate
    assert result.record.queued_turn.id == "queue-b"
    assert not queue.paused


def test_rebind_session_updates_active_and_retired_records() -> None:
    queue = _queue("queue-a", "queue-b")
    retired = queue.enqueue(_params("retired"))
    queue.pop_next()
    active = queue.enqueue(_params("active"))

    queue.rebind_session("session-2")

    assert active.record.params.session_id == "session-2"
    replay = queue.enqueue(
        _params("retired").model_copy(update={"session_id": "session-2"})
    )
    assert replay.duplicate
    assert replay.record is retired.record


def _shell_params(operation_id: str = "op-1", command: str = "echo hi"):
    return TurnEnqueueParams(
        session_id="session-1",
        entries=[
            TurnShellInputEntry(
                operation_id=operation_id,
                command=command,
                cwd="/tmp/workspace",
                timeout_seconds=5.0,
            )
        ],
    )


def test_shell_input_is_a_queue_item_kind_of_its_own() -> None:
    queue = _queue()
    record = queue.enqueue(_shell_params()).record

    assert record.queued_turn.entries[0].role == "shell"
    assert record.queued_turn.entries[0].operation_id == "op-1"
    assert record.queued_turn.entries[0].content == []
    assert queue.state.items[0].entries[0].role == "shell"


def test_a_shell_entry_cannot_share_a_turn_queue_item() -> None:
    with pytest.raises(ValueError):
        TurnEnqueueParams(
            session_id="session-1",
            entries=[
                TurnShellInputEntry(operation_id="op-1", command="echo hi"),
                TurnUserInputEntry(content=[SessionTextContentBlock(text="prompt")]),
            ],
        )

    with pytest.raises(ValueError):
        TurnEnqueueParams(
            session_id="session-1",
            entries=[
                TurnShellInputEntry(operation_id="op-1", command="echo one"),
                TurnShellInputEntry(operation_id="op-2", command="echo two"),
            ],
        )


def _controller(backend: FakeBackend | None = None) -> tuple[TurnController, AgentLoop]:
    backend = backend or FakeBackend([[mock_llm_chunk(content="ok")]] * 4)
    loop = build_test_agent_loop(backend=backend)
    controller = TurnController(
        loop,
        AsyncMock(),
        AsyncMock(),
        SessionExecution(),
        Mock(),
        snapshot_state=Mock(),
    )
    return controller, loop


class _GatedFailureBackend(FakeBackend):
    """A provider whose completions fail once the test releases the gate."""

    def __init__(self) -> None:
        super().__init__([[mock_llm_chunk(content="ok")]] * 4)
        self.release = asyncio.Event()
        self.failing = True

    async def _gate(self) -> None:
        if not self.failing:
            return
        await self.release.wait()
        if self.failing:
            raise RuntimeError("provider unavailable")

    async def complete(self, **kwargs: Any) -> LLMChunk:
        await self._gate()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        await self._gate()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


class _InterleavedFailureBackend(FakeBackend):
    """One failing completion, then completions the test releases on demand.

    The first completion is the record turn the retention case needs; the
    later ones are the turns that run after it, so the test can observe the
    retained record while a later turn is still in flight.
    """

    def __init__(self) -> None:
        super().__init__([[mock_llm_chunk(content="ok")]] * 6)
        self.release_failure = asyncio.Event()
        self.release_turns = asyncio.Event()
        self._calls = 0

    async def _enter(self) -> None:
        self._calls += 1
        if self._calls == 1:
            await self.release_failure.wait()
            raise RuntimeError("provider unavailable")
        await self.release_turns.wait()

    async def complete(self, **kwargs: Any) -> LLMChunk:
        await self._enter()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        await self._enter()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


class _TwoFailureBackend(FakeBackend):
    """The record turn and one intervening ordinary turn both fail.

    Each failure waits for its own release, so the test can observe the record
    retained beside a full queue before the intervening turn fails.
    """

    def __init__(self) -> None:
        super().__init__([[mock_llm_chunk(content="ok")]] * 6)
        self.release_record_failure = asyncio.Event()
        self.release_prompt_failure = asyncio.Event()
        self._calls = 0

    async def _enter(self) -> None:
        self._calls += 1
        if self._calls == 1:
            await self.release_record_failure.wait()
        elif self._calls == 2:
            await self.release_prompt_failure.wait()
        if self._calls <= 2:
            raise RuntimeError("provider unavailable")

    async def complete(self, **kwargs: Any) -> LLMChunk:
        await self._enter()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        await self._enter()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


def _outcome(entry: TurnShellInputEntry) -> ManualShellOutcome:
    failed = "fails" in entry.command
    return ManualShellOutcome(
        operation_id=entry.operation_id,
        command=entry.command,
        cwd=entry.cwd or "",
        status="failed" if failed else "ok",
        exit_code=1 if failed else 0,
        stdout="" if failed else "hi\n",
        output_text="" if failed else "hi\n",
        duration_ms=1.0,
        created_at=1,
    )


def _bind_executor(controller: TurnController) -> list[ManualShellRecord]:
    """Bind a fake effect runner; the returned list records what it ran."""
    records: list[ManualShellRecord] = []

    async def execute(entry: TurnShellInputEntry, session_id: str) -> ManualShellRecord:
        record = build_manual_shell_record(_outcome(entry), max_output_bytes=16_000)
        records.append(record)
        return record

    controller.bind_shell_executor(execute)
    return records


async def _settle(controller: TurnController, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (
            controller.active_turn is None
            and not controller.has_queued_turns
            and controller._active_task is None
            and not controller._queue_tasks
        ):
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Turn queue did not settle")


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("Condition not met in time")


def _user_texts(loop: AgentLoop) -> list[str]:
    return [
        message.content or "" for message in loop.messages if message.role is Role.user
    ]


def _queued_user_texts(controller: TurnController) -> list[str]:
    return [
        block.text
        for item in controller.queue_state.items
        for entry in item.entries
        if entry.role == "user"
        for block in entry.content
        if isinstance(block, SessionTextContentBlock)
    ]


@pytest.mark.asyncio
async def test_shell_input_queues_beside_prompts_in_fifo_order() -> None:
    controller, loop = _controller()
    controller._turn_queue.pause()
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        controller.enqueue(_params("prompt", "typed prompt"))

        assert [
            entry.role
            for item in controller.queue_state.items
            for entry in item.entries
        ] == ["shell", "user"]

        after_resume = controller.resume_queue()
        assert after_resume is not None
        after_resume()

        await asyncio.wait_for(shell_future, timeout=5.0)
        await _settle(controller)

        # The command ran once, before the prompt queued behind it, and its
        # record is the first user message the model received.
        assert [record.context.command for record in records] == ["echo hi"]
        assert _user_texts(loop) == [records[0].text, "typed prompt"]
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_interrupt_before_execution_drops_the_queued_command() -> None:
    controller, loop = _controller()
    controller._turn_queue.pause()
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-drop",
            command="echo never",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )

        assert controller.discard_queued_shell_input("op-drop") is True
        assert await asyncio.wait_for(shell_future, timeout=5.0) is None

        after_resume = controller.resume_queue()
        if after_resume is not None:
            after_resume()
        await _settle(controller)

        assert records == []
        assert not controller.has_queued_turns
        assert _user_texts(loop) == []
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_admission_rejection_retains_the_record_without_rerunning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller, loop = _controller()
    records = _bind_executor(controller)
    real_start = controller.start
    pending = {"reject": True}

    def flaky_start(*args, **kwargs):
        if pending["reject"]:
            pending["reject"] = False
            raise ModelChoicePendingError("choose a model")
        return real_start(*args, **kwargs)

    monkeypatch.setattr(controller, "start", flaky_start)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _wait_for(lambda: _queued_user_texts(controller) == [records[0].text])

        # The effect ran once and its record is a ready user turn; admission
        # rejected the turn, so the record waits instead of rerunning.
        assert len(records) == 1
        assert _user_texts(loop) == []

        after_rejection = controller.resume_queue()
        assert after_rejection is not None
        after_rejection()
        await _settle(controller)

        assert len(records) == 1
        assert _user_texts(loop) == [records[0].text]
        record_message = next(
            message for message in loop.messages if message.role is Role.user
        )
        assert record_message.manual_shell is not None
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_provider_failure_retains_the_record_without_rerunning() -> None:
    backend = FakeBackend(
        [[mock_llm_chunk(content="ok")]] * 4,
        exception_to_raise=RuntimeError("provider unavailable"),
    )
    controller, loop = _controller(backend)
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _wait_for(
            lambda: (
                bool(controller.completed_turns)
                and controller.completed_turns[-1].status is PublicTurnStatus.FAILED
                and _queued_user_texts(controller) == [records[0].text]
            )
        )

        # The provider failed after the record was committed: the effect is not
        # rerun, and the record waits as a ready user turn instead of being lost.
        assert len(records) == 1
        assert controller.completed_turns[-1].status is PublicTurnStatus.FAILED
        assert _user_texts(loop) == [records[0].text]
        assert _queued_user_texts(controller) == [records[0].text]

        backend._exception_to_raise = None
        after_failure = controller.resume_queue()
        assert after_failure is not None
        after_failure()
        await _settle(controller)

        # The retained record became the next turn from the message already in
        # the transcript: one effect, one record message.
        assert len(records) == 1
        assert _user_texts(loop) == [records[0].text]
        assert controller.completed_turns[-1].status is PublicTurnStatus.COMPLETED
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_a_full_queue_retains_the_record_until_the_next_admission() -> None:
    """R2-F2: retention survives a full queue instead of dropping the record."""
    backend = _GatedFailureBackend()
    controller, loop = _controller(backend)
    controller._turn_queue = TurnQueue(max_items=1)
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _wait_for(lambda: controller.active_turn is not None)

        # The record turn is running, so its queue place is free: the user's
        # next prompt takes the one place the queue has, and the queue is paused
        # -- nothing may promote over the failure until the user acts.
        controller.enqueue(_params("prompt", "typed prompt"))
        assert controller._turn_queue.pause()
        backend.release.set()
        await _wait_for(
            lambda: any(
                turn.status is PublicTurnStatus.FAILED
                for turn in controller.completed_turns
            )
        )

        # The effect ran once and its record message reached the transcript, but
        # there is no room to re-queue it: it is retained, never dropped.
        assert len(records) == 1
        assert list(controller._shell_retained) == ["op-1"]
        assert _queued_user_texts(controller) == ["typed prompt"]
        assert _user_texts(loop) == [records[0].text]

        # The next admission offers the retained record the place a removal
        # freed. It is held there so the same pass cannot repeat the failed turn,
        # and the message it already committed is not committed again.
        removed, after_removal = controller.remove_queued_turn(
            controller.queue_state.items[0].id
        )
        assert removed and after_removal is not None
        after_removal()
        await _wait_for(lambda: _queued_user_texts(controller) == [records[0].text])
        assert controller._shell_retained == {}
        assert _user_texts(loop) == [records[0].text]

        backend.failing = False
        after_resume = controller.resume_queue()
        assert after_resume is not None
        after_resume()
        await _settle(controller)

        assert len(records) == 1
        assert _user_texts(loop) == [records[0].text]
        assert not controller.has_queued_turns
        assert controller.completed_turns[-1].status is PublicTurnStatus.COMPLETED
        record_message = next(
            message for message in loop.messages if message.role is Role.user
        )
        assert record_message.manual_shell is not None
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_failed_command_record_becomes_the_user_turn() -> None:
    controller, loop = _controller()
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="fails",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _settle(controller)

        assert [record.context.command for record in records] == ["fails"]
        assert _user_texts(loop) == [records[0].text]
        record_message = next(
            message for message in loop.messages if message.role is Role.user
        )
        shell = record_message.manual_shell
        assert shell is not None
        assert shell.status == "failed"
        assert shell.exit_code == 1
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_turn_enqueue_refuses_a_shell_entry() -> None:
    controller, loop = _controller()
    try:
        with pytest.raises(TurnConflictError):
            controller.enqueue(_shell_params())
        assert not controller.has_queued_turns
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_a_full_queue_names_the_retained_record_as_pending_work() -> None:
    """R3-A1: a record the queue could not take is public pending work.

    The record is the outcome of a command that already ran, so a full queue
    can only delay it. The public queue state names it while it waits, which is
    the user-visible pending state B.2a promises.
    """
    backend = _InterleavedFailureBackend()
    controller, loop = _controller(backend)
    controller._turn_queue = TurnQueue(max_items=1)
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _wait_for(lambda: controller.active_turn is not None)

        # The record turn is running, so its queue place is free: the user's
        # prompt takes the one place, and the recorded failure then has nowhere
        # to put the record.
        controller.enqueue(_params("typed", "typed prompt"))
        backend.release_failure.set()
        await _wait_for(lambda: bool(controller._shell_retained))

        assert list(controller._shell_retained) == ["op-1"]
        retained = controller.queue_state.retained
        assert [item.id for item in retained] == ["op-1"]
        retained_entry = retained[0].entries[0]
        assert isinstance(retained_entry, TurnUserInputEntry)
        assert retained_entry.content == [SessionTextContentBlock(text=records[0].text)]
        # The full queue already handed the user's prompt to its turn, and the
        # record still waits beside it instead of being lost.
        assert controller.queue_state.items == []
        # Only the committed record message is on the transcript: the retained
        # record has not run a second time, it is still pending.
        assert _user_texts(loop) == [records[0].text, "typed prompt"]
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_a_retained_record_drains_onto_its_own_text() -> None:
    """R3-A1 + R3-A2: the drain admits the record and it runs on its text.

    The prompt queued beside the failure runs first; the retained record is
    then admitted and must become a turn on the record text itself, not a turn
    that runs on the later transcript's tail.
    """
    backend = _InterleavedFailureBackend()
    controller, loop = _controller(backend)
    controller._turn_queue = TurnQueue(max_items=1)
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _wait_for(lambda: controller.active_turn is not None)
        controller.enqueue(_params("typed", "typed prompt"))
        backend.release_failure.set()
        await _wait_for(lambda: bool(controller._shell_retained))
        await _wait_for(lambda: not controller.has_queued_turns)

        # The intervening prompt is the turn in flight; releasing it drains the
        # queue and the terminal pass admits the retained record.
        backend.release_turns.set()
        await _settle(controller)

        assert records and len(records) == 1
        assert controller._shell_retained == {}
        assert controller.queue_state.retained == []
        assert controller.completed_turns[-1].status is PublicTurnStatus.COMPLETED
        assert controller.completed_turns[-1].queue_item_id is not None
        # The record message committed once, the intervening prompt ran, and the
        # retried record turn carried the record text as its own user input.
        assert _user_texts(loop) == [records[0].text, "typed prompt", records[0].text]
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_a_failed_intervening_turn_re_offers_the_retained_record() -> None:
    """C3: a failed ordinary turn is an admission for an earlier record.

    The record turn fails while the queue is full, so its record waits beside
    the FIFO. The prompt that took the last place fails too; its terminal must
    offer the record the place it freed, or the drained queue would leave the
    record stranded in the controller's private state.
    """
    backend = _TwoFailureBackend()
    controller, loop = _controller(backend)
    controller._turn_queue = TurnQueue(max_items=1)
    records = _bind_executor(controller)
    try:
        shell_future = controller.enqueue_shell_input(
            session_id=loop.session_id,
            operation_id="op-1",
            command="echo hi",
            cwd=str(loop.cwd),
            timeout_seconds=5.0,
        )
        await asyncio.wait_for(shell_future, timeout=5.0)
        await _wait_for(lambda: controller.active_turn is not None)
        # The queued prompt takes the one place the record's item left behind;
        # no queue mutation runs after it, so the record's failure is the next
        # admission that can offer the record a place.
        controller.enqueue(_params("typed", "typed prompt"))

        backend.release_record_failure.set()
        await _wait_for(lambda: bool(controller._shell_retained))
        # The prompt's place is the one the queue has, so the record waits beside
        # it instead of taking it.
        assert list(controller._shell_retained) == ["op-1"]
        assert [item.id for item in controller.queue_state.retained] == ["op-1"]

        backend.release_prompt_failure.set()
        await _settle(controller)

        # The prompt failed and its terminal admitted the record: the effect ran
        # once, the record ran as the retried turn's own input, and no record is
        # left waiting in private state.
        assert len(records) == 1
        assert controller._shell_retained == {}
        assert controller.queue_state.retained == []
        assert not controller.has_queued_turns
        assert controller.completed_turns[-1].status is PublicTurnStatus.COMPLETED
        assert _user_texts(loop) == [records[0].text, "typed prompt", records[0].text]
    finally:
        await controller.close()
        await loop.aclose()


@pytest.mark.asyncio
async def test_an_empty_command_record_is_retained_when_the_queue_is_full() -> None:
    """R3-G2: the refusal record takes the same full-safe path as an executed one.

    A bare `!` executes nothing, so its record is ready immediately, but a full
    queue must still not drop it: it is retained, named publicly, and admitted
    by the next admission.
    """
    controller, loop = _controller()
    controller._turn_queue = TurnQueue(max_items=1)
    try:
        queued, _ = controller.enqueue(_params("typed", "typed prompt"))
        record = build_manual_shell_record(
            ManualShellOutcome(
                operation_id="op-empty",
                command="",
                cwd=str(loop.cwd),
                status="refused",
                created_at=1,
                error="Shell command cannot be empty",
            ),
            max_output_bytes=16_000,
        )
        controller.enqueue_shell_record(
            session_id=loop.session_id, operation_id="op-empty", record=record
        )
        await _wait_for(lambda: bool(controller._shell_retained))

        assert list(controller._shell_retained) == ["op-empty"]
        assert [item.id for item in controller.queue_state.retained] == ["op-empty"]

        removed, after_removal = controller.remove_queued_turn(
            queued.record.queued_turn.id
        )
        assert removed and after_removal is not None
        after_removal()
        await _settle(controller)
        assert controller._shell_retained == {}
        assert controller.queue_state.retained == []
        assert _user_texts(loop) == [record.text]
        record_message = next(
            message for message in loop.messages if message.role is Role.user
        )
        assert record_message.manual_shell is not None
        assert record_message.manual_shell.status == "refused"
    finally:
        await controller.close()
        await loop.aclose()
