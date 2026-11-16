"""W3 acceptance: live context counter.

The statusbar context counter must move mid-turn, once per LLM call, via the
existing value-delta hook in ``TurnController._run_turn_phase`` — never per
streamed chunk or per projected event. The dedupe compares the full projected
stats snapshot (``project_stats``), so cached-token and tool-call fields also
surface mid-turn, and the turn-end / interrupt emits are preserved.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from chartreux.app_server.events import StatsUpdated
from chartreux.app_server.models import PublicTurnStatus
from chartreux.core.llm_models import FunctionCall, ToolCall
from tests.app_server._session_runtime_support import (  # noqa: F401 - autouse fixture
    _isolate_recovery_registry,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import create_test_app_server_session
from tests.stubs.fake_backend import FakeBackend

# Two tool calls, so three LLM calls: tool, tool, final answer.
LLM_CALLS = 3
TOOL_CALLS = 2
# Turn-boundary emits: turn start and turn end.
BOUNDARY_EMITS = 2


def _todo_call(index: int) -> ToolCall:
    return ToolCall(
        id=f"todo-{index}",
        index=0,
        function=FunctionCall(name="todo", arguments=json.dumps({"action": "read"})),
    )


def _three_call_backend() -> FakeBackend:
    """call1: tool (100+10, cached 0); call2: tool (150+15, cached 40);
    call3: final answer (200+20, cached 50), streamed in several chunks.
    """
    return FakeBackend([
        [
            mock_llm_chunk(
                content="",
                tool_calls=[_todo_call(1)],
                prompt_tokens=100,
                completion_tokens=10,
                cached_tokens=0,
            )
        ],
        [
            mock_llm_chunk(
                content="",
                tool_calls=[_todo_call(2)],
                prompt_tokens=150,
                completion_tokens=15,
                cached_tokens=40,
            )
        ],
        [
            mock_llm_chunk(content="Done ", prompt_tokens=0, completion_tokens=0),
            mock_llm_chunk(content="now", prompt_tokens=0, completion_tokens=0),
            mock_llm_chunk(
                content="!", prompt_tokens=200, completion_tokens=20, cached_tokens=50
            ),
        ],
    ])


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_tool_loop_turn_emits_stats_updated_mid_turn_per_llm_call(
    streaming: bool,
) -> None:
    config = build_test_vibe_config(enabled_tools=["todo"])
    agent_loop = build_test_agent_loop(
        config=config, backend=_three_call_backend(), enable_streaming=streaming
    )
    session = await create_test_app_server_session(agent_loop)
    stats_events: list[StatsUpdated] = []
    try:
        async for event in session.act("do things"):
            if isinstance(event, StatsUpdated):
                stats_events.append(event)
    finally:
        await session.close()
        await agent_loop.aclose()
    ctx_values = [ev.params.stats.context_tokens for ev in stats_events]
    # One mid-turn update per LLM call: after call 1 (ctx=110) and after
    # call 2 (ctx=165) the counter moves before the turn completes.
    assert 110 in ctx_values
    assert 165 in ctx_values
    # Turn boundaries still emit: first event is the turn-start snapshot,
    # the last carries the final context (220) at turn end.
    assert ctx_values[0] == 0
    assert ctx_values[-1] == 220


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_stats_updated_count_bounded_and_deduped(streaming: bool) -> None:
    config = build_test_vibe_config(enabled_tools=["todo"])
    agent_loop = build_test_agent_loop(
        config=config, backend=_three_call_backend(), enable_streaming=streaming
    )
    session = await create_test_app_server_session(agent_loop)
    stats_events: list[StatsUpdated] = []
    all_events: list[object] = []
    try:
        async for event in session.act("do things"):
            all_events.append(event)
            if isinstance(event, StatsUpdated):
                stats_events.append(event)
    finally:
        await session.close()
        await agent_loop.aclose()
    # Storm regression: emits are value-delta deduped on the full snapshot, so
    # consecutive updates never repeat and the count stays bounded by the LLM
    # calls plus turn boundaries plus one emit per tool stat transition —
    # never per streamed chunk (call 3 streams 3 chunks) or per event.
    snapshots = [ev.params.stats.model_dump() for ev in stats_events]
    assert all(a != b for a, b in zip(snapshots, snapshots[1:], strict=False))
    assert len(stats_events) <= LLM_CALLS + BOUNDARY_EMITS + 2 * TOOL_CALLS + 1
    assert len(stats_events) < len(all_events)


@pytest.mark.asyncio
async def test_mid_turn_stats_update_includes_cached_token_fields() -> None:
    config = build_test_vibe_config(enabled_tools=["todo"])
    agent_loop = build_test_agent_loop(
        config=config, backend=_three_call_backend(), enable_streaming=False
    )
    session = await create_test_app_server_session(agent_loop)
    stats_events: list[StatsUpdated] = []
    try:
        async for event in session.act("do things"):
            if isinstance(event, StatsUpdated):
                stats_events.append(event)
    finally:
        await session.close()
        await agent_loop.aclose()

    # The mid-turn update after LLM call 2 carries the cached-token fields,
    # before the turn completes (the final update is the turn-end one).
    mid_turn = stats_events[:-1]
    cached_mid_turn = [
        ev for ev in mid_turn if ev.params.stats.session_cached_tokens > 0
    ]
    assert cached_mid_turn, "expected a mid-turn StatsUpdated with cached tokens"
    assert cached_mid_turn[0].params.stats.last_turn_cached_tokens == 40
    assert cached_mid_turn[0].params.stats.context_tokens == 165


class _BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__([])
        self.started = asyncio.Event()

    async def complete_streaming(self, **kwargs):
        self.started.set()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk
        await asyncio.Event().wait()  # block until the interrupt cancels us


@pytest.mark.asyncio
async def test_interrupt_still_yields_final_stats_update() -> None:
    backend = _BlockingBackend()
    agent_loop = build_test_agent_loop(backend=backend, enable_streaming=True)
    session = await create_test_app_server_session(agent_loop)
    stats_events: list[StatsUpdated] = []
    stream = session.act("block forever")

    async def drain() -> None:
        async for event in stream:
            if isinstance(event, StatsUpdated):
                stats_events.append(event)

    drain_task = asyncio.create_task(drain())
    try:
        await asyncio.wait_for(backend.started.wait(), timeout=2)
        await session.interrupt()
        await asyncio.wait_for(drain_task, timeout=2)
    finally:
        await stream.aclose()
        await session.close()
        await agent_loop.aclose()
    assert session.last_turn_status is not None
    assert session.last_turn_status[1] is PublicTurnStatus.INTERRUPTED
    # The interrupt/final path still emits a terminal stats update after the
    # turn-start one; the interrupted call produced no usage, but the final
    # snapshot reflects the post-interrupt step count.
    assert len(stats_events) >= 2
    assert stats_events[0].params.stats.steps == 0
    assert stats_events[-1].params.stats.steps == 1
