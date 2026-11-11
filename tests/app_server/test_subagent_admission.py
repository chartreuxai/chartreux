"""WP5 [5B] admission: capacity/depth denials, atomic reservation, the
incarnation-stamped event feed, foreground/background terminal completion
frames, and retention semantics (design.md:448-456).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server.agent_wire import (
    AGENT_TURN_ENDED_METHOD,
    AGENT_TURN_EVENT_METHOD,
    TURN_OUTCOME_COMPLETED,
    TURN_OUTCOME_FAILED,
    TURN_OUTCOME_INTERRUPTED,
)
from chartreux.app_server.events import RELAY_INCARNATION_FIELD
from chartreux.app_server.subagent_ports import (
    AdmittingSubagentSpawn,
    SpawnedSubagent,
    SubagentAdmission,
    SubagentAdmissionDeniedError,
    SubagentCapacityDeniedError,
    SubagentCompletion,
    SubagentCompletionSource,
    SubagentDepthDeniedError,
    SubagentEventFeed,
    SubagentFrame,
    SubagentSpawnSpec,
)
from chartreux.core.agent_loop._execution_port import ExecutionTurnRequest
from chartreux.core.agent_loop._loop_types import AgentTurnOptions
from chartreux.core.events import AssistantEvent, UserMessageEvent
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _foreground_result,
    _idle_record,
    _ManualClock,
    _retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def _turn_request() -> ExecutionTurnRequest:
    return ExecutionTurnRequest(
        message="hello", turn_options=AgentTurnOptions(turn_id="turn-1")
    )


def _spec(agent_id: str) -> SubagentSpawnSpec:
    return SubagentSpawnSpec(
        agent_id=agent_id,
        parent_session_id="root-session-1",
        session_dir=f"sessions/root-session-1/agents/{agent_id}",
        payload={"task_summary": "do the thing"},
    )


def _default_frames() -> list[SubagentFrame]:
    return [
        UserMessageEvent(content="hello", message_id="m1"),
        AssistantEvent(content="done"),
        SubagentCompletion(outcome=TURN_OUTCOME_COMPLETED),
    ]


class _ScriptedSource:
    """A completion source yielding one fixed frame script."""

    def __init__(self, agent_id: str, frames: list[SubagentFrame]) -> None:
        self._agent_id = agent_id
        self._frames = frames

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def stream(
        self, request: ExecutionTurnRequest
    ) -> AsyncGenerator[SubagentFrame, None]:
        return self._stream()

    async def _stream(self) -> AsyncGenerator[SubagentFrame, None]:
        for frame in self._frames:
            yield frame


class _FakeSpawned:
    def __init__(self, agent_id: str, frames: list[SubagentFrame]) -> None:
        self._agent_id = agent_id
        self.completion_source: SubagentCompletionSource = _ScriptedSource(
            agent_id, frames
        )
        self.closed = False

    @property
    def agent_id(self) -> str:
        return self._agent_id

    async def close(self) -> None:
        self.closed = True


class _FakePort:
    """A spawn port that yields once before producing the fake handle."""

    def __init__(
        self, frames: list[SubagentFrame] | None = None, *, fail: bool = False
    ) -> None:
        self.spawn_calls = 0
        self._frames = frames
        self._fail = fail

    async def spawn(self, spec: SubagentSpawnSpec) -> SpawnedSubagent:
        self.spawn_calls += 1
        await asyncio.sleep(0)  # a real suspension between reserve and use
        if self._fail:
            raise RuntimeError("spawn boom")
        return _FakeSpawned(spec.agent_id, self._frames or _default_frames())


class _FrameLog:
    """The port-level feed sink: records every relayed frame (frame-only)."""

    def __init__(self) -> None:
        self.frames: list[dict[str, Any]] = []

    async def __call__(self, frame: Any) -> None:
        self.frames.append(dict(frame))

    def for_agent(self, agent_id: str) -> list[dict[str, Any]]:
        return [f for f in self.frames if f["params"].get("agent_id") == agent_id]


# --- typed denials -----------------------------------------------------------


def test_capacity_denial_is_typed_and_never_queues() -> None:
    admission = SubagentAdmission()
    assert issubclass(SubagentCapacityDeniedError, SubagentAdmissionDeniedError)
    assert issubclass(SubagentCapacityDeniedError, RuntimeError)
    admission.reserve(cap=2)
    admission.reserve(cap=2)
    with pytest.raises(SubagentCapacityDeniedError) as exc_info:
        admission.reserve(cap=2)
    exc = exc_info.value
    assert exc.cap == 2 and exc.running == 2
    assert "cap 2" in str(exc)
    assert len(admission.slots) == 2  # the denied request held no slot
    # Released capacity is reusable; nothing was silently queued.
    slot = admission.slots.pop()
    admission.release(slot)
    admission.reserve(cap=2)
    assert len(admission.slots) == 2


def test_depth_denial_is_typed() -> None:
    admission = SubagentAdmission()  # max depth 1
    with pytest.raises(SubagentDepthDeniedError, match="depth limit of 1"):
        admission.reserve(cap=4, depth=1)
    assert not admission.slots
    assert isinstance(admission.reserve(cap=4, depth=0), object)


# --- atomic reservation around the spawn port --------------------------------


@pytest.mark.asyncio
async def test_reservation_is_atomic_under_concurrent_spawns() -> None:
    admission = SubagentAdmission()
    log = _FrameLog()
    fake = _FakePort()
    port = AdmittingSubagentSpawn(fake, admission, sink=log)
    results = await asyncio.gather(
        *(port.spawn(_spec(f"agent-{i}"), cap=3) for i in range(5)),
        return_exceptions=True,
    )
    admitted = [r for r in results if not isinstance(r, BaseException)]
    denied = [r for r in results if isinstance(r, SubagentCapacityDeniedError)]
    assert len(admitted) == 3 and len(denied) == 2
    # The reservation was visible to every concurrent request: exactly the
    # cap's worth of physical spawns happened, no silent queueing.
    assert len(admission.slots) == 3
    assert fake.spawn_calls == 3
    for denied_error in denied:
        assert denied_error.running == 3 and denied_error.cap == 3


@pytest.mark.asyncio
async def test_spawn_failure_releases_the_reservation() -> None:
    admission = SubagentAdmission()
    log = _FrameLog()
    failing = AdmittingSubagentSpawn(_FakePort(fail=True), admission, sink=log)
    with pytest.raises(RuntimeError, match="spawn boom"):
        await failing.spawn(_spec("agent-1"), cap=1)
    assert not admission.slots  # released on spawn failure
    # The freed capacity admits the next spawn.
    ok = AdmittingSubagentSpawn(_FakePort(), admission, sink=log)
    admitted = await ok.spawn(_spec("agent-2"), cap=1)
    assert admitted.agent_id == "agent-2"
    assert len(admission.slots) == 1


# --- event feed: incarnation stamps + terminal completion frame --------------


@pytest.mark.asyncio
async def test_completion_releases_slot_and_relays_incarnation_stamped_frames() -> None:
    admission = SubagentAdmission()
    log = _FrameLog()
    port = AdmittingSubagentSpawn(_FakePort(), admission, sink=log)
    admitted = await port.spawn(_spec("agent-1"), cap=1)
    assert len(admission.slots) == 1
    terminal = await admitted.run(_turn_request())
    assert terminal.outcome == TURN_OUTCOME_COMPLETED
    assert not admission.slots  # released on completion

    frames = log.for_agent("agent-1")
    assert frames, "the feed relayed frames"
    assert all(
        frame[RELAY_INCARNATION_FIELD] == admitted.incarnation for frame in frames
    )
    event_frames = [f for f in frames if f["method"] == AGENT_TURN_EVENT_METHOD]
    terminal_frames = [f for f in frames if f["method"] == AGENT_TURN_ENDED_METHOD]
    assert len(event_frames) == 2
    assert all(
        "event" in f["params"] and "type" in f["params"]["event"] for f in event_frames
    )
    # Exactly one terminal completion frame, and it is the stream's last frame.
    assert len(terminal_frames) == 1
    assert frames[-1]["method"] == AGENT_TURN_ENDED_METHOD
    assert terminal_frames[0]["params"]["outcome"] == TURN_OUTCOME_COMPLETED
    assert terminal_frames[0]["params"]["agent_id"] == "agent-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "error"),
    [(TURN_OUTCOME_FAILED, "RuntimeError: boom"), (TURN_OUTCOME_INTERRUPTED, None)],
)
async def test_feed_delivers_terminal_frame_for_failed_and_interrupted_runs(
    outcome: str, error: str | None
) -> None:
    frames: list[SubagentFrame] = [
        UserMessageEvent(content="hello", message_id="m1"),
        SubagentCompletion(outcome=outcome, error=error),
    ]
    log = _FrameLog()
    feed = SubagentEventFeed("agent-9", log, incarnation=7)
    source = _ScriptedSource("agent-9", frames)
    terminal = await feed.pump(source, _turn_request())
    assert terminal.outcome == outcome and terminal.error == error
    relayed = log.for_agent("agent-9")
    assert [f["method"] for f in relayed] == [
        AGENT_TURN_EVENT_METHOD,
        AGENT_TURN_ENDED_METHOD,
    ]
    assert all(f[RELAY_INCARNATION_FIELD] == 7 for f in relayed)
    assert relayed[-1]["params"]["outcome"] == outcome
    assert relayed[-1]["params"]["error"] == error


@pytest.mark.asyncio
async def test_feed_synthesizes_a_terminal_frame_when_the_source_omits_it() -> None:
    log = _FrameLog()
    feed = SubagentEventFeed("agent-1", log, incarnation=0)
    source = _ScriptedSource(
        "agent-1", [UserMessageEvent(content="hi", message_id="m1")]
    )
    terminal = await feed.pump(source, _turn_request())
    assert terminal.outcome == TURN_OUTCOME_FAILED
    relayed = log.for_agent("agent-1")
    assert relayed[-1]["method"] == AGENT_TURN_ENDED_METHOD
    assert relayed[-1]["params"]["outcome"] == TURN_OUTCOME_FAILED
    assert len([f for f in relayed if f["method"] == AGENT_TURN_ENDED_METHOD]) == 1


@pytest.mark.asyncio
async def test_each_spawn_gets_a_fresh_incarnation() -> None:
    admission = SubagentAdmission()
    log = _FrameLog()
    port = AdmittingSubagentSpawn(_FakePort(), admission, sink=log)
    first = await port.spawn(_spec("agent-1"), cap=2)
    second = await port.spawn(_spec("agent-2"), cap=2)
    assert first.incarnation != second.incarnation
    await first.run(_turn_request())
    await second.run(_turn_request())
    assert all(
        f[RELAY_INCARNATION_FIELD] == first.incarnation
        for f in log.for_agent("agent-1")
    )
    assert all(
        f[RELAY_INCARNATION_FIELD] == second.incarnation
        for f in log.for_agent("agent-2")
    )


# --- retention semantics (design.md:448-456) ---------------------------------


@pytest.mark.asyncio
async def test_retained_idle_counts_against_cap_until_explicit_close() -> None:
    admission = SubagentAdmission()
    log = _FrameLog()
    port = AdmittingSubagentSpawn(_FakePort(), admission, sink=log)
    retained = await port.spawn(_spec("agent-1"), cap=1, retain=True)
    await retained.run(_turn_request())
    # Completed and retained: the active slot is gone, but the retained idle
    # agent still counts against the cap.
    assert not admission.slots
    assert admission.retained_idle == {"agent-1"}
    assert admission.occupied == 1
    with pytest.raises(SubagentCapacityDeniedError, match="cap 1"):
        await port.spawn(_spec("agent-2"), cap=1)
    # No automatic eviction: the denial left the retained agent in place.
    assert admission.retained_idle == {"agent-1"}
    # Reusing the retained agent converts its occupancy into the new slot.
    reused = await port.spawn(_spec("agent-1"), cap=1, retain=True)
    assert not admission.retained_idle
    assert len(admission.slots) == 1
    # Explicit closure is the only closure kind and frees the budget.
    await reused.close()
    assert not admission.slots and not admission.retained_idle
    admitted = await port.spawn(_spec("agent-3"), cap=1)
    assert admitted.agent_id == "agent-3"


# --- registry-level admission ------------------------------------------------


def test_registry_over_cap_raises_typed_denial_and_holds_no_slot() -> None:
    registry = _retention_registry()
    registry._max_running_subagents = 1
    registry._admission.reserve(cap=1)  # one active run occupies the cap
    args = TaskArgs(task="work", identity="coder", background=True)
    ctx = InvokeContext(tool_call_id="t", session_id="root")
    with pytest.raises(SubagentCapacityDeniedError, match="cap 1"):
        registry._reserve_admission(args, ctx)
    assert len(registry._active_work_slots) == 1  # denied request held nothing


def test_registry_over_depth_raises_typed_denial() -> None:
    registry = _retention_registry()
    args = TaskArgs(task="work", identity="coder", background=True)
    ctx = InvokeContext(tool_call_id="t", session_id="root", is_subagent=True)
    with pytest.raises(SubagentDepthDeniedError, match="depth limit of 1"):
        registry._reserve_admission(args, ctx)
    assert not registry._active_work_slots


def test_registry_retained_idle_records_count_against_the_cap() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    registry._max_running_subagents = 1
    registry._agent_records["agent-1"] = _idle_record("agent-1", clock=clock)
    args = TaskArgs(task="work", identity="coder", background=True)
    ctx = InvokeContext(tool_call_id="t", session_id="root")
    # A fresh launch is denied while the retained idle agent holds the budget.
    with pytest.raises(SubagentCapacityDeniedError, match="cap 1"):
        registry._reserve_admission(args, ctx)
    # Reusing that idle agent is admitted (no double-count).
    slot = registry._reserve_admission(
        args.model_copy(update={"agent_id": "agent-1"}), ctx
    )
    assert slot in registry._active_work_slots
    registry._admission.release(slot)
    assert not registry._active_work_slots


# --- registry-level event feed: foreground + background terminal frames -------


async def _wait_for(predicate, timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


@pytest.mark.asyncio
async def test_foreground_and_background_runs_deliver_terminal_completion_frames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    relayed: list[tuple[str, dict[str, Any]]] = []

    async def event_sink(agent_id: str, frame: Any) -> None:
        relayed.append((agent_id, dict(frame)))

    child_backend = FakeBackend([
        [mock_llm_chunk(content="foreground done")],
        [mock_llm_chunk(content="background done")],
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [mock_llm_chunk(content="Notification received")],
            [mock_llm_chunk(content="Notification received")],
        ]),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    registry._subagent_event_sink = event_sink
    try:
        ctx = InvokeContext(tool_call_id="run", session_id=parent.session_id)
        foreground = await _foreground_result(registry, parent, "foreground")
        assert foreground.completed
        background = await _background_result(
            registry,
            TaskArgs(task="bg", identity="coder", background=True, retain=True),
            ctx,
        )
        assert background.agent_id is not None and background.run_id is not None
        await registry.wait_for_agent(background.agent_id, background.run_id)
        await _wait_for(
            lambda: (
                len([f for _k, f in relayed if f["method"] == AGENT_TURN_ENDED_METHOD])
                == 2
            )
        )
    finally:
        await registry.close()
        await parent.aclose()

    frames = [f for _k, f in relayed]
    terminal = [f for f in frames if f["method"] == AGENT_TURN_ENDED_METHOD]
    assert len(terminal) == 2
    # Both runs delivered exactly one terminal completion frame, outcome
    # completed, each stamped with its own fresh incarnation.
    assert all(f["params"]["outcome"] == TURN_OUTCOME_COMPLETED for f in terminal)
    incarnations = {f[RELAY_INCARNATION_FIELD] for f in terminal}
    assert len(incarnations) == 2
    assert all(
        type(f[RELAY_INCARNATION_FIELD]) is int and f[RELAY_INCARNATION_FIELD] >= 0
        for f in frames
    )
    # Child events streamed as incarnation-stamped relay frames on both runs.
    event_frames = [f for f in frames if f["method"] == AGENT_TURN_EVENT_METHOD]
    assert event_frames, "child events relayed into the coordinator event feed"
    assert all("event" in f["params"] for f in event_frames)
    # Each run's terminal frame is the last frame of its own incarnation.
    for incarnation in incarnations:
        run_frames = [f for f in frames if f[RELAY_INCARNATION_FIELD] == incarnation]
        assert run_frames[-1]["method"] == AGENT_TURN_ENDED_METHOD
        assert all(f["method"] == AGENT_TURN_EVENT_METHOD for f in run_frames[:-1])
