"""Tool-response messages append in tool-call order, not completion order.

Providers require ``tool_response`` messages to match the order of the
``tool_calls`` they answer, while streamed events must stay completion-ordered
for the UI.  A mixed success/failure batch exercises both guarantees at once;
the tests below also pin the window's edge cases: an aborted call that never
produces a response, a cancelled batch, and a response from outside the active
batch.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import contextlib
import json
from pathlib import Path
from typing import Any, ClassVar, cast

from pydantic import BaseModel
import pytest

from chartreux.core.events import BaseEvent, ToolResultEvent, ToolStreamEvent
from chartreux.core.hooks.manager import HooksManager
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class _BatchArgs(BaseModel):
    pass


class _BatchResult(BaseModel):
    message: str


class _FailingTool(BaseTool[_BatchArgs, _BatchResult, BaseToolConfig, BaseToolState]):
    @classmethod
    def get_name(cls) -> str:
        return "batch_fail"

    async def run(
        self, args: _BatchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _BatchResult, None]:
        raise RuntimeError("boom")
        yield  # pragma: no cover


class _SlowTool(BaseTool[_BatchArgs, _BatchResult, BaseToolConfig, BaseToolState]):
    @classmethod
    def get_name(cls) -> str:
        return "batch_slow"

    async def run(
        self, args: _BatchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _BatchResult, None]:
        await asyncio.sleep(0.05)
        yield _BatchResult(message="slow done")


class _FastTool(BaseTool[_BatchArgs, _BatchResult, BaseToolConfig, BaseToolState]):
    @classmethod
    def get_name(cls) -> str:
        return "batch_fast"

    async def run(
        self, args: _BatchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _BatchResult, None]:
        yield _BatchResult(message="fast done")


@pytest.mark.asyncio
async def test_mixed_batch_messages_follow_call_order_events_follow_completion(
    tmp_path: Path,
) -> None:
    tool_calls = [
        ToolCall(
            id=f"call_{index}",
            index=index,
            function=FunctionCall(name=name, arguments=json.dumps({})),
        )
        for index, name in enumerate(["batch_fail", "batch_slow", "batch_fast"])
    ]
    backend = FakeBackend([
        [mock_llm_chunk(content="Run batch.", tool_calls=tool_calls)],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["batch_fail", "batch_slow", "batch_fast"]
        ),
        cwd=tmp_path,
        backend=backend,
    )
    loop.tool_manager._all_tools["batch_fail"] = _FailingTool
    loop.tool_manager._all_tools["batch_slow"] = _SlowTool
    loop.tool_manager._all_tools["batch_fast"] = _FastTool
    try:
        events = [event async for event in loop.act("go")]
    finally:
        await loop.aclose()

    # Events stream in completion order: the failing and fast calls finish
    # before the slow one, so the event sequence is not call-index order.
    completion_order = [
        event.tool_call_id for event in events if isinstance(event, ToolResultEvent)
    ]
    assert completion_order.index("call_1") > completion_order.index("call_0")
    assert completion_order.index("call_1") > completion_order.index("call_2")

    # The appended tool-response messages follow tool-call index order.
    tool_messages = [m for m in loop.messages if m.role.value == "tool"]
    assert [m.tool_call_id for m in tool_messages] == ["call_0", "call_1", "call_2"]
    # Mixed outcomes survived: call_0 failed, the others succeeded.
    assert "batch_fail" in (tool_messages[0].content or "")
    assert "boom" in (tool_messages[0].content or "")
    assert "slow done" in (tool_messages[1].content or "")
    assert "fast done" in (tool_messages[2].content or "")

    # The provider-visible request for the follow-up turn keeps the order.
    follow_up_tools = [
        m for m in backend.requests_messages[1] if m.role.value == "tool"
    ]
    assert [m.tool_call_id for m in follow_up_tools] == ["call_0", "call_1", "call_2"]


class _BlockingTool(BaseTool[_BatchArgs, _BatchResult, BaseToolConfig, BaseToolState]):
    """Never completes on its own; signals the test when it starts running."""

    started: ClassVar[asyncio.Event]

    @classmethod
    def get_name(cls) -> str:
        return "batch_block"

    async def run(
        self, args: _BatchArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _BatchResult, None]:
        type(self).started.set()
        await asyncio.sleep(3600)
        yield _BatchResult(message="unreachable")  # pragma: no cover


class _CancellingPreToolHooks:
    """Pre-tool hook that aborts one call before it can produce a response."""

    def __init__(self, cancel_call_id: str) -> None:
        self._cancel_call_id = cancel_call_id

    def reset_retry_count(self) -> None:
        return

    async def run(self, invocation: object) -> AsyncGenerator[object, None]:
        if (
            getattr(invocation, "hook_event_name", "") == "pre_tool"
            and getattr(invocation, "tool_call_id", None) == self._cancel_call_id
        ):
            raise asyncio.CancelledError()
        return
        yield  # pragma: no cover


def _batch_tool_call(index: int, name: str, call_id: str | None = None) -> ToolCall:
    return ToolCall(
        id=call_id or f"call_{index}",
        index=index,
        function=FunctionCall(name=name, arguments=json.dumps({})),
    )


def _tool_messages(messages: list[Any]) -> list[Any]:
    return [m for m in messages if m.role.value == "tool"]


@pytest.mark.asyncio
async def test_aborted_call_gap_flushes_later_responses_and_backfills(
    tmp_path: Path,
) -> None:
    """A call aborted before producing any response leaves a gap.

    The pre-tool hook cancels call_0 before it runs, so it never appends a
    response: the window must still close cleanly, the later calls' responses
    must flush at batch end, and the next act() must backfill the gap before
    its completion.
    """
    tool_calls = [
        _batch_tool_call(0, "batch_fast"),
        _batch_tool_call(1, "batch_slow"),
        _batch_tool_call(2, "batch_fast"),
    ]
    backend = FakeBackend([
        [mock_llm_chunk(content="Run batch.", tool_calls=tool_calls)],
        [mock_llm_chunk(content="done")],
        [mock_llm_chunk(content="done again")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["batch_fail", "batch_slow", "batch_fast"]
        ),
        cwd=tmp_path,
        backend=backend,
    )
    loop.tool_manager._all_tools["batch_slow"] = _SlowTool
    loop.tool_manager._all_tools["batch_fast"] = _FastTool
    loop._hooks_manager = cast(HooksManager, _CancellingPreToolHooks("call_0"))
    try:
        events = [event async for event in loop.act("go")]

        # call_0 died before producing a response: no result event for it,
        # while the sibling calls completed normally (call_2 before call_1).
        result_ids = [
            event.tool_call_id for event in events if isinstance(event, ToolResultEvent)
        ]
        assert "call_0" not in result_ids
        assert result_ids.index("call_2") < result_ids.index("call_1")
        # The ordering window closed cleanly with no staged residue.
        assert loop._tool_response_order is None
        assert loop._staged_tool_responses == {}
        # The later responses flushed at batch end in call order; the gap for
        # call_0 is still open at this point.
        assert [m.tool_call_id for m in _tool_messages(list(loop.messages))] == [
            "call_1",
            "call_2",
        ]

        [event async for event in loop.act("again")]
    finally:
        await loop.aclose()

    # The next act() backfilled the gap before its completion: all three
    # responses are in the message list and the provider-visible request.
    tool_messages = _tool_messages(list(loop.messages))
    assert [m.tool_call_id for m in tool_messages] == ["call_1", "call_2", "call_0"]
    assert tool_messages[2].content == (
        "<user_cancellation>Tool execution interrupted - no response "
        "available</user_cancellation>"
    )
    follow_up_tools = [
        m for m in backend.requests_messages[-1] if m.role.value == "tool"
    ]
    assert {m.tool_call_id for m in follow_up_tools} == {"call_0", "call_1", "call_2"}


@pytest.mark.asyncio
async def test_cancelled_batch_leaks_no_state_into_next_batch(tmp_path: Path) -> None:
    """A batch cancelled mid-flight unwinds completely.

    Cancelling the turn while tools are blocked must close the ordering
    window and leave no staging state behind, so the next batch still
    appends its responses in tool-call order.
    """
    first_calls = [
        _batch_tool_call(0, "batch_block"),
        _batch_tool_call(1, "batch_block"),
    ]
    second_calls = [
        _batch_tool_call(0, "batch_slow", call_id="call_a"),
        _batch_tool_call(1, "batch_fast", call_id="call_b"),
    ]
    backend = FakeBackend([
        [mock_llm_chunk(content="Run first batch.", tool_calls=first_calls)],
        [mock_llm_chunk(content="Run second batch.", tool_calls=second_calls)],
        [mock_llm_chunk(content="done")],
    ])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["batch_block", "batch_slow", "batch_fast"]
        ),
        cwd=tmp_path,
        backend=backend,
    )
    loop.tool_manager._all_tools["batch_block"] = _BlockingTool
    loop.tool_manager._all_tools["batch_slow"] = _SlowTool
    loop.tool_manager._all_tools["batch_fast"] = _FastTool
    _BlockingTool.started = asyncio.Event()

    collected: list[BaseEvent] = []

    async def collect() -> None:
        async for event in loop.act("first"):
            collected.append(event)

    act_task = asyncio.create_task(collect())
    await asyncio.wait_for(_BlockingTool.started.wait(), 5)
    act_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await act_task

    # The cancelled batch unwound completely: the ordering window closed and
    # neither staged responses nor the tool-event queue leaked.
    assert loop._tool_response_order is None
    assert loop._staged_tool_responses == {}
    assert loop._tool_event_queue is None
    # Both cancelled calls recorded their interrupted responses, in call order.
    assert [m.tool_call_id for m in _tool_messages(list(loop.messages))] == [
        "call_0",
        "call_1",
    ]

    try:
        [event async for event in loop.act("second")]
    finally:
        await loop.aclose()

    # The fresh batch after the cancelled one still orders deterministically:
    # call_b completes first, but its response appends after call_a's.
    tool_messages = _tool_messages(list(loop.messages))
    assert [m.tool_call_id for m in tool_messages] == [
        "call_0",
        "call_1",
        "call_a",
        "call_b",
    ]
    assert "slow done" in (tool_messages[2].content or "")
    assert "fast done" in (tool_messages[3].content or "")
    follow_up_tools = [
        m for m in backend.requests_messages[-1] if m.role.value == "tool"
    ]
    assert [m.tool_call_id for m in follow_up_tools] == [
        "call_0",
        "call_1",
        "call_a",
        "call_b",
    ]


@pytest.mark.asyncio
async def test_out_of_batch_response_appends_directly(tmp_path: Path) -> None:
    """A response whose call is not in an active batch appends directly.

    Mentioned-file injection runs read_file through the direct
    ``_process_one_tool_call`` path, outside any batch window: its response
    must append immediately instead of staging behind an index that will
    never arrive.
    """
    (tmp_path / "notes.txt").write_text("mentioned file body")
    backend = FakeBackend([[mock_llm_chunk(content="done reading")]])
    loop = build_test_agent_loop(
        config=build_test_vibe_config(enabled_tools=["read_file"]),
        cwd=tmp_path,
        backend=backend,
    )
    try:
        events = [event async for event in loop.act("read @notes.txt")]
    finally:
        await loop.aclose()

    # The injected call ran with no ordering window active.
    assert loop._tool_response_order is None
    result_events = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(result_events) == 1
    injected_id = result_events[0].tool_call_id

    # The response appended directly: it sits immediately after its injected
    # assistant call, ahead of the model's own turn.
    messages = list(loop.messages)
    assistant_index = next(
        index
        for index, message in enumerate(messages)
        if message.role.value == "assistant"
        and message.tool_calls
        and message.tool_calls[0].id == injected_id
    )
    tool_message = messages[assistant_index + 1]
    assert tool_message.role.value == "tool"
    assert tool_message.tool_call_id == injected_id
    assert "mentioned file body" in (tool_message.content or "")

    # The provider saw the response in the same direct position.
    request_tools = [
        message
        for message in backend.requests_messages[0]
        if message.role.value == "tool"
    ]
    assert [message.tool_call_id for message in request_tools] == [injected_id]
