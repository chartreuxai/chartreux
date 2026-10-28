"""WP2 — safe explicit warm resume: suspension, admission, successor semantics.

Covers the resume half of the WP2 acceptance: a recoverably failed retained
agent SUSPENDS warm (handoff §1.2) instead of closing; ``resume_agent`` is the
ONE explicit admission operation (no auto-scheduling) with typed
dispositions; concurrent same-key and different-key calls admit exactly one
successor (single-flight, run-pinned) under a NEW run_id on the SAME agent
identity; the old failed result and waits pinned to it stay immutable; ONE
resume marker + injected continuation mirroring /retry never sets
``user_initiated_retry`` (F6); context, tool state, and cumulative budgets are
preserved (handoff §4); only the failed backend is retired (F3); minimal warm
bounds (``same_agent_resumes_max``) apply; closed/evicted agents remain
non-reusable.

The fault injected here is the WP1 typed EMPTY_RESPONSE transient (an empty
assistant response at the final-chunk boundary): it is recovery-eligible
without arming an availability cooldown (``failover_eligible`` is False), so
resume admission is deterministic. A settled tool call before the fault makes
the envelope's tool effects SETTLED (ELIGIBLE); no tool calls makes them
UNKNOWN (RECONCILE_REQUIRED — covered in ``test_reconciliation.py``).

Harness: the full app-server surface (``start_test_app_server`` +
``attach_test_app_server_session``) drives the parent's ``task`` tool, so the
children execute tool calls through the real registry. Child backends come
from a per-loop scripted factory behind the patched
``chartreux.core.agent_loop._loop.create_backend`` facade: ``create_backend``
runs once per child loop at construction and at each resume-time backend
refresh (no ``retry_budget`` kwarg — throwaway backends that are never
called) and once per logical call (with ``retry_budget`` — a real attempt
backend drawing the next scripted outcome). The parent backend owns one
script per launch turn (a retained task call + final response, keyed by the
act text) and answers the monitor's background-completion notification turns
— and notification steps drained into a launch turn — with a default
response, so the scripts stay aligned regardless of notification timing.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from typing import Any, cast

import pytest

from chartreux.app_server._recovery_controller import RecoveryController, _Availability
from chartreux.app_server._sessions import SessionRuntimeRegistry, _AgentState
from chartreux.app_server.protocol import TurnStartParams
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerSession
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.config.models import RecoveryConfig, RecoverySectionConfig
from chartreux.core.llm_models import FunctionCall, LLMChunk, LLMMessage, Role, ToolCall
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.subagents import (
    AgentAvailability,
    ReleaseAgentOutcome,
    ResumeOutcome,
    RunStatus,
    TaskResult,
    WaitStatus,
)
from tests.app_server._subagent_test_support import (
    _config,
    _consume,
    _evicted_tombstone,
    _task_call,
    _todo_call,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


class _Gated:
    """A chunks outcome held in flight until ``gate`` is set.

    ``started`` is set when a call reaches the gate, so a test can wait for a
    sibling to be mid-request before failing the other child (F3 sibling
    survival under suspension).
    """

    def __init__(self, chunks: list[LLMChunk], gate: asyncio.Event) -> None:
        self.chunks = chunks
        self.gate = gate
        self.started = asyncio.Event()


# One scripted outcome per logical call: the chunks of one response, an
# exception raised on every call (the gateway replays an empty response once,
# so a fault must keep raising), or a gated response held in flight.
_Outcome = list[LLMChunk] | Exception | _Gated


def _empty_fault() -> EmptyLLMResponseError:
    """The WP1 typed EMPTY_RESPONSE transient (see module docstring)."""
    return EmptyLLMResponseError("test-provider", "test-model")


def _todo_write_call(call_id: str, content: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="todo",
            arguments=json.dumps({
                "action": "write",
                "todos": [{"id": "1", "content": content, "status": "pending"}],
            }),
        ),
    )


def _write_outcome(call_id: str, content: str) -> list[LLMChunk]:
    """One logical call answering with a todo write (settled tool effect)."""
    return [mock_llm_chunk(content="", tool_calls=[_todo_write_call(call_id, content)])]


def _read_outcome(call_id: str) -> list[LLMChunk]:
    """One logical call answering with a todo read (observes preserved state)."""
    return [mock_llm_chunk(content="", tool_calls=[_todo_call(call_id)])]


def _final_outcome(text: str) -> list[LLMChunk]:
    """One logical call answering with terminal assistant content."""
    return [mock_llm_chunk(content=text)]


class _ScriptedBackend(FakeBackend):
    """One scripted outcome per logical call; counts ``__aexit__`` closes.

    An outcome is either the chunks of one logical call's response, an
    Exception raised on every call, or a ``_Gated`` response held in flight
    until its event is set. ``None`` marks a throwaway backend
    (construction / resume-time refresh) that is never called.
    """

    def __init__(self, outcome: _Outcome | None) -> None:
        self._gated = outcome if isinstance(outcome, _Gated) else None
        chunks: list[LLMChunk]
        if self._gated is not None:
            chunks = self._gated.chunks
        elif isinstance(outcome, list):
            chunks = outcome
        else:
            chunks = []
        super().__init__(chunks)
        self._fault = outcome if isinstance(outcome, Exception) else None
        self.calls = 0
        self.aexit_calls = 0

    async def _await_gate(self) -> None:
        """Hold an in-flight request until the test releases it."""
        if self._gated is not None:
            self._gated.started.set()
            await self._gated.gate.wait()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.aexit_calls += 1
        return None

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.calls += 1
        if self._fault is not None:
            raise self._fault
        await self._await_gate()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        self.calls += 1
        if self._fault is not None:
            raise self._fault
        await self._await_gate()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


class _ParentBackend(FakeBackend):
    """Parent backend: per-launch-turn scripts, default for the rest.

    The monitor starts a parent turn for every settled background child (a
    "Background agent ... failed/completed" notification) whenever the parent
    is idle, and a notification that arrives while a turn is active is
    drained after that turn's next step — which can displace the act marker
    as the request's last user message. Each launch turn therefore owns its
    script keyed by its act marker: the task call is served while the marker
    is the request's last user message, the final response while the last
    message is that task call's tool result, and everything else (the
    monitor's notification turns and drained-notification steps) gets a
    default response. A skipped final response stays unserved instead of
    shifting the next launch's script.
    """

    def __init__(self, launches: list[str]) -> None:
        super().__init__([])
        self._task_ids = {
            marker: f"task-{index + 1}" for index, marker in enumerate(launches)
        }
        self._scripts = {
            marker: (
                mock_llm_chunk(
                    content="",
                    tool_calls=[
                        _task_call(
                            background=True,
                            retain=True,
                            tool_call_id=self._task_ids[marker],
                        )
                    ],
                ),
                mock_llm_chunk(content=f"parent done {index + 1}"),
            )
            for index, marker in enumerate(launches)
        }
        self._stage = {marker: 0 for marker in launches}

    def _scripted(self, messages: list[LLMMessage]) -> list[LLMChunk] | None:
        last_user: str | None = None
        for message in reversed(messages):
            if message.role is Role.user:
                last_user = str(message.content)
                break
        if last_user is not None:
            for marker, stage in self._stage.items():
                if stage == 0 and marker in last_user:
                    self._stage[marker] = 1
                    return [self._scripts[marker][0]]
        if messages and messages[-1].role is Role.tool:
            tool_call_id = messages[-1].tool_call_id
            for marker, stage in self._stage.items():
                if stage == 1 and tool_call_id == self._task_ids[marker]:
                    self._stage[marker] = 2
                    return [self._scripts[marker][1]]
        return None

    async def complete(self, **kwargs: Any) -> LLMChunk:
        stream = self._scripted(kwargs["messages"])
        if stream is None:
            return mock_llm_chunk(content="parent noted")
        chunk_agg = LLMChunk(message=LLMMessage(role=Role.assistant))
        for chunk in stream:
            chunk_agg += chunk
        return chunk_agg

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        stream = self._scripted(kwargs["messages"])
        if stream is None:
            yield mock_llm_chunk(content="parent noted")
            return
        for chunk in stream:
            yield chunk


class _ChildBackendFactory:
    """Per-child-loop scripted outcomes behind the patched create_backend.

    Scripts bind to loops in launch order — the first call from an unbound
    loop is that loop's construction — and every created backend is recorded
    per loop for retirement assertions.
    """

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, scripts: list[list[_Outcome]]
    ) -> None:
        self._pending = [list(script) for script in scripts]
        self._bound: dict[int, list[_Outcome]] = {}
        self.created: dict[int, list[_ScriptedBackend]] = {}
        monkeypatch.setattr(
            "chartreux.core.agent_loop._loop.create_backend", self._create_backend
        )

    def _create_backend(self, **kwargs: Any) -> Any:
        on_retry = kwargs.get("on_retry")
        loop = getattr(on_retry, "__self__", None)
        key = id(loop) if loop is not None else -1
        if key not in self._bound:
            if not self._pending:
                raise AssertionError("create_backend called by an unscripted loop")
            self._bound[key] = self._pending.pop(0)
        if "retry_budget" in kwargs:
            script = self._bound[key]
            if not script:
                raise AssertionError("child backend script exhausted")
            backend = _ScriptedBackend(script.pop(0))
        else:
            backend = _ScriptedBackend(None)
        self.created.setdefault(key, []).append(backend)
        return backend


class _Harness:
    """Full app-server harness: scripted children behind a scripted parent."""

    def __init__(
        self,
        registry: SessionRuntimeRegistry,
        parent: AgentLoop,
        session: AppServerSession,
        factory: _ChildBackendFactory,
    ) -> None:
        self.registry = registry
        self.parent = parent
        self.session = session
        self.factory = factory

    @property
    def requester_session_id(self) -> str:
        """The owning parent's session id — the only resume/reconcile authority."""
        return self.parent.session_id


async def _start_harness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    child_scripts: list[list[_Outcome]],
    launches: list[str],
    recovery: RecoveryConfig | None = None,
) -> _Harness:
    """Build the parent loop, start the app server, and attach a session.

    ``launches`` are the act texts used to launch each child through the
    parent's ``task`` tool, in launch order. ``recovery`` overrides the
    ``[recovery.subagents]`` policy (the WP4 controller is enabled by
    default since v0.5.6; these tests drive the explicit admission well
    inside the first scheduled cooldown, so the automatic timer never
    wins the single-flight race).
    """
    factory = _ChildBackendFactory(monkeypatch, child_scripts)
    parent_backend = _ParentBackend(launches)
    config = _config(enabled_tools=["task", "todo"])
    if recovery is not None:
        config = config.model_copy(
            update={"recovery": RecoverySectionConfig(subagents=recovery)}
        )
    parent = build_test_agent_loop(
        config=config, backend=parent_backend, enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    return _Harness(registry, parent, session, factory)


async def _until(predicate: Any, timeout: float = 10.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


async def _launch_child(harness: _Harness, marker: str, index: int) -> tuple[str, str]:
    """Launch child ``index`` through the parent's task tool.

    ``latest_run_id`` is the launch signal: it is set when the run is prepared
    and never cleared, so a child that settles before the first poll (an idle
    child completes in milliseconds) is still observed.
    """
    await _consume(harness.session.act(marker))
    agent_id = f"agent-{index}"

    def launched() -> bool:
        record = harness.registry._agent_records.get(agent_id)
        return record is not None and record.latest_run_id is not None

    await _until(launched)
    record = harness.registry._agent_records[agent_id]
    run_id = record.latest_run_id
    assert run_id is not None
    return agent_id, run_id


def _parent_idle(harness: _Harness) -> bool:
    """The monitor's notification condition: the parent has no active turn."""
    root = harness.registry._root
    if root is None:
        return True
    turns = root.turns
    return (
        turns.active_turn is None
        and turns._active_task is None
        and not turns.queue_state.items
    )


async def _settle_child(harness: _Harness, agent_id: str, run_id: str) -> TaskResult:
    """Wait for a child run to settle and the parent to return to idle.

    Joining the run's monitor task lands the completion transition and the
    monitor's parent notification turn (started whenever the parent was idle
    at settlement); waiting out that turn — on the server AND in the client
    projection, which the receive loop updates asynchronously — leaves the
    parent idle for the next ``session.act``: a launch racing the
    notification turn is rejected with "A turn is already running". A
    notification deferred while the parent was busy leaves the parent already
    idle; it rides inside the next act turn.
    """
    result = await _wait_result(harness.registry, agent_id, run_id)
    record = harness.registry._agent_records[agent_id]
    run = next(
        (
            item
            for item in [record.current_run, *record.run_history]
            if item is not None and item.run_id == run_id
        ),
        None,
    )
    assert run is not None
    await run.completion_task
    await _until(lambda: _parent_idle(harness) and not harness.session.turn_active)
    return result


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_recoverable_failure_suspends_and_explicit_resume_admits_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main WP2 flow: suspend → explicit resume → one completing successor.

    Covers: suspension truthfulness in summaries, identity preservation with a
    NEW run_id, ONE resume marker + injected continuation that never sets
    ``user_initiated_retry`` (F6), context/tool-state/cumulative-budget
    preservation (handoff §4), and immutability of the failed run's result and
    pinned waits.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "wp2-resume-item"),
                _empty_fault(),
                _read_outcome("todo-2"),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        assert failed.run_status is RunStatus.FAILED
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE

        # Suspension: warm, resident, truthfully summarized (not idle/closed).
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        assert record.availability is AgentAvailability.SUSPENDED
        assert record.runtime._closed is not True
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.availability is AgentAvailability.SUSPENDED
        assert summary.recovery_chain_id == envelope.recovery_chain_id
        assert summary.same_agent_resumes_used == 0
        assert summary.recovery_eligibility is RecoveryEligibility.ELIGIBLE
        assert summary.latest_run_id == run_id
        assert summary.last_failure is not None
        assert summary.last_failure.event_id == envelope.event_id

        # The failed run's result is stored (immutable) before the resume.
        stored = await registry.get_agent_result(agent_id, run_id)
        assert stored is not None and stored.response == failed.response

        # Capture the successor turn's start params for the F6 assertion.
        captured: list[TurnStartParams] = []
        original_start = record.runtime.turns.start

        def capture_start(params: TurnStartParams) -> Any:
            captured.append(params)
            return original_start(params)

        monkeypatch.setattr(record.runtime.turns, "start", capture_start)

        # Explicit resume: exactly one successor, NEW run_id, same identity.
        resume = await registry.resume_agent(
            agent_id,
            run_id,
            f"{agent_id}:{run_id}",
            requester_session_id=harness.requester_session_id,
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        assert resume.successor_run_id != run_id
        assert resume.recovery_operation_id is not None
        assert resume.same_agent_resumes_used == 1

        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        # Identity preserved: same record, session, runtime, and agent id.
        assert registry._agent_records[agent_id] is record
        successor_run = record.run_history[-1]
        assert successor_run.run_id == resume.successor_run_id
        assert successor_run.resumed_from_run_id == run_id
        assert successor_run.recovery_chain_id == record.recovery_chain_id
        assert record.recovery_chain_id == envelope.recovery_chain_id
        assert record.same_agent_resumes_used == 1

        # ONE resume marker + injected continuation, never user-initiated.
        assert len(captured) == 1
        assert captured[0].injected is True
        assert captured[0].user_initiated_retry is False
        messages = record.runtime.agent_loop.messages
        markers = [
            message
            for message in messages
            if message.role is Role.user
            and "This is a resumed run" in str(message.content)
        ]
        assert len(markers) == 1
        assert markers[0].injected is True
        assert f"resume of run {run_id}" in str(markers[0].content)
        assert "reconcile them rather than repeat them blindly" in str(
            markers[0].content
        )

        # Context preserved as-is (handoff §4): no compaction — the original
        # brief, the run-1 tool call, and its result survive; the successor's
        # run measures from the preserved conversation (cumulative budgets).
        roles = [message.role for message in messages]
        assert roles == [
            Role.system,
            Role.user,
            Role.assistant,
            Role.tool,
            Role.user,
            Role.assistant,
            Role.tool,
            Role.assistant,
        ]
        assert successor_run.start_count == 1
        assert successor.turns_used == 2

        # Tool state preserved: the item written in run 1 is read in run 2.
        tool_results = [message for message in messages if message.role is Role.tool]
        assert "wp2-resume-item" in str(tool_results[-1].content)

        # The failed run's result and pinned waits stay immutable.
        after = await registry.get_agent_result(agent_id, run_id)
        assert after is not None
        assert after.response == stored.response
        assert after.run_status is RunStatus.FAILED
        assert after.failure is not None
        assert after.failure.event_id == envelope.event_id
        pinned = await registry.wait_for_agent(agent_id, run_id)
        assert pinned.status is WaitStatus.COMPLETED
        assert pinned.result is not None
        assert pinned.result.run_id == run_id
        assert not pinned.result.completed
        latest = await registry.get_agent_result(agent_id)
        assert latest is not None
        assert latest.run_id == resume.successor_run_id

        # Post-completion summary: resident idle, successor linkage visible.
        await _until(lambda: record.state is _AgentState.IDLE)
        final_summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert final_summary.availability is AgentAvailability.IDLE
        assert final_summary.resumed_from_run_id == run_id
        assert final_summary.same_agent_resumes_used == 1
        assert final_summary.recovery_chain_id == envelope.recovery_chain_id
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_single_flight_admits_exactly_one_successor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent same-key and different-key resume calls admit exactly one
    successor (single-flight, run-pinned); late calls deduplicate.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    # Defined before the try so the finally can always release it.
    gate = asyncio.Event()
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        await _settle_child(harness, agent_id, run_id)
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # Hold the admission's prepare phase so the concurrent calls land
        # while the resume operation is in flight.
        original_prepare = registry._execution.prepare_background

        async def gated_prepare(*args: Any, **kwargs: Any) -> Any:
            await gate.wait()
            return await original_prepare(*args, **kwargs)

        monkeypatch.setattr(registry._execution, "prepare_background", gated_prepare)

        key = f"{agent_id}:{run_id}"
        first = asyncio.create_task(
            registry.resume_agent(
                agent_id, run_id, key, requester_session_id=harness.requester_session_id
            )
        )
        await _until(lambda: record.pending_resume is not None)
        pending = record.pending_resume
        assert pending is not None

        # Same key + same run: deduplicated onto the in-flight admission.
        same_key = await registry.resume_agent(
            agent_id, run_id, key, requester_session_id=harness.requester_session_id
        )
        assert same_key.outcome is ResumeOutcome.ALREADY_SCHEDULED
        assert same_key.successor_run_id == pending.successor_run_id
        assert same_key.recovery_operation_id == pending.operation_id
        # Different key: refused — a resume admission is already in flight.
        other_key = await registry.resume_agent(
            agent_id,
            run_id,
            "different-key",
            requester_session_id=harness.requester_session_id,
        )
        assert other_key.outcome is ResumeOutcome.ALREADY_RESUMED
        assert other_key.successor_run_id == pending.successor_run_id

        gate.set()
        scheduled = await first
        assert scheduled.outcome is ResumeOutcome.SCHEDULED
        assert scheduled.successor_run_id == pending.successor_run_id
        assert scheduled.successor_run_id is not None

        # The admitted successor completes on the SAME identity.
        successor = await _settle_child(harness, agent_id, scheduled.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        # Exactly one successor run exists for the pinned failed run.
        successors = [
            item for item in record.run_history if item.resumed_from_run_id == run_id
        ]
        assert len(successors) == 1
        assert successors[0].run_id == scheduled.successor_run_id

        # A late call after admission completes deduplicates via run history.
        late = await registry.resume_agent(
            agent_id, run_id, key, requester_session_id=harness.requester_session_id
        )
        assert late.outcome is ResumeOutcome.ALREADY_RESUMED
        assert late.successor_run_id == scheduled.successor_run_id
    finally:
        gate.set()
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_dispositions_matrix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Typed resume dispositions: not_resumable (unknown/idle/stale),
    forbidden (wrong requester), expired (advisory window), and
    reconcile_required (unknown tool outcomes block replay).
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [_final_outcome("idle done")],
            [_write_outcome("todo-1", "item"), _empty_fault()],
            [_empty_fault()],
        ],
        launches=[
            "launch the idle child",
            "launch the eligible child",
            "launch the barrier child",
        ],
    )
    registry = harness.registry
    try:
        # Unknown agent → NOT_RESUMABLE.
        unknown = await registry.resume_agent(
            "agent-999", "run-x", "k", requester_session_id=harness.requester_session_id
        )
        assert unknown.outcome is ResumeOutcome.NOT_RESUMABLE
        assert unknown.reason_codes == ["not_suspended"]

        # A completed retained agent is idle — not suspended → NOT_RESUMABLE.
        idle_id, idle_run = await _launch_child(harness, "launch the idle child", 1)
        idle_done = await _settle_child(harness, idle_id, idle_run)
        assert idle_done.completed
        idle_record = registry._agent_records[idle_id]
        await _until(lambda: idle_record.state is _AgentState.IDLE)
        not_suspended = await registry.resume_agent(
            idle_id, idle_run, "k", requester_session_id=harness.requester_session_id
        )
        assert not_suspended.outcome is ResumeOutcome.NOT_RESUMABLE
        assert not_suspended.reason_codes == ["not_suspended"]

        # An eligible suspension: wrong requester → FORBIDDEN; a stale
        # expected_run_id → NOT_RESUMABLE; an aged envelope → EXPIRED.
        fail_id, fail_run = await _launch_child(harness, "launch the eligible child", 2)
        await _settle_child(harness, fail_id, fail_run)
        fail_record = registry._agent_records[fail_id]
        await _until(lambda: fail_record.state is _AgentState.SUSPENDED)

        forbidden = await registry.resume_agent(
            fail_id, fail_run, "k", requester_session_id="other-session"
        )
        assert forbidden.outcome is ResumeOutcome.FORBIDDEN
        assert forbidden.reason_codes == ["forbidden_requester"]

        stale = await registry.resume_agent(
            fail_id,
            "run-not-the-failed-one",
            "k",
            requester_session_id=harness.requester_session_id,
        )
        assert stale.outcome is ResumeOutcome.NOT_RESUMABLE
        assert stale.reason_codes == ["stale_expected_run_id"]

        failed_run = next(
            item
            for item in [fail_record.current_run, *fail_record.run_history]
            if item is not None and item.run_id == fail_run
        )
        envelope = failed_run.failure
        assert envelope is not None
        failed_run.failure = envelope.model_copy(
            update={"observed_at": envelope.observed_at - 10_000}
        )
        expired = await registry.resume_agent(
            fail_id, fail_run, "k", requester_session_id=harness.requester_session_id
        )
        assert expired.outcome is ResumeOutcome.EXPIRED
        assert expired.reason_codes == ["episode_deadline_passed"]
        failed_run.failure = envelope

        # A reconcile barrier (unknown tool outcomes) blocks replay.
        bar_id, bar_run = await _launch_child(harness, "launch the barrier child", 3)
        bar_failed = await _settle_child(harness, bar_id, bar_run)
        assert bar_failed.failure is not None
        assert (
            bar_failed.failure.recovery.eligibility
            is RecoveryEligibility.RECONCILE_REQUIRED
        )
        bar_record = registry._agent_records[bar_id]
        await _until(lambda: bar_record.state is _AgentState.SUSPENDED)
        blocked = await registry.resume_agent(
            bar_id, bar_run, "k", requester_session_id=harness.requester_session_id
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED
        assert blocked.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_budget_exhausted_stops_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Minimal warm bounds: after ``same_agent_resumes_max`` (2) resumes, a
    further recoverable failure no longer suspends — the retained record
    returns to idle and resume is rejected.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item-1"),
                _empty_fault(),
                _write_outcome("todo-2", "item-2"),
                _empty_fault(),
                _write_outcome("todo-3", "item-3"),
                _empty_fault(),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    try:
        agent_id, failed_run_id = await _launch_child(harness, "launch the child", 1)
        record = registry._agent_records[agent_id]
        for expected_uses in (1, 2):
            failed = await _settle_child(harness, agent_id, failed_run_id)
            assert not failed.completed
            await _until(lambda: record.state is _AgentState.SUSPENDED)
            assert record.same_agent_resumes_used == expected_uses - 1
            resume = await registry.resume_agent(
                agent_id,
                failed_run_id,
                f"key-{expected_uses}",
                requester_session_id=harness.requester_session_id,
            )
            assert resume.outcome is ResumeOutcome.SCHEDULED
            assert resume.same_agent_resumes_used == expected_uses
            failed_run_id = resume.successor_run_id
            assert failed_run_id is not None

        # The third failure exhausts the same-agent resume budget (max 2):
        # no suspension — the retained record returns to idle.
        failed = await _settle_child(harness, agent_id, failed_run_id)
        assert not failed.completed
        await _until(lambda: record.state is _AgentState.IDLE)
        assert record.same_agent_resumes_used == 2
        rejected = await registry.resume_agent(
            agent_id,
            failed_run_id,
            "key-3",
            requester_session_id=harness.requester_session_id,
        )
        assert rejected.outcome is ResumeOutcome.NOT_RESUMABLE
        assert rejected.reason_codes == ["not_suspended"]
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.availability is AgentAvailability.IDLE
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_released_and_terminal_agents_are_not_resumable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Closed/evicted agents remain non-reusable: releasing a suspended agent
    through the normal ownership path, or an evicted tombstone, makes resume
    NOT_RESUMABLE.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_write_outcome("todo-1", "item"), _empty_fault()]],
        launches=["launch the child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        await _settle_child(harness, agent_id, run_id)
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # Release the suspended agent; the record is gone from the registry.
        outcome = await registry.release_agent(agent_id)
        assert outcome is ReleaseAgentOutcome.RELEASED
        assert agent_id not in registry._agent_records
        released = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert released.outcome is ResumeOutcome.NOT_RESUMABLE

        # An evicted tombstone is terminal metadata — also NOT_RESUMABLE.
        registry._evicted_agents["agent-77"] = _evicted_tombstone(
            "agent-77", run_id=None, status=RunStatus.FAILED
        )
        evicted = await registry.resume_agent(
            "agent-77", "run-1", "k", requester_session_id=harness.requester_session_id
        )
        assert evicted.outcome is ResumeOutcome.NOT_RESUMABLE
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_retires_only_the_failed_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F3 borrowed-backend semantics: suspension keeps the failed backend
    warm; the resume retires ONLY that loop's backend, while the sibling
    agent's backend is untouched and keeps serving.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item-a"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ],
            [_final_outcome("sibling done")],
        ],
        launches=["launch child a", "launch child b"],
    )
    registry = harness.registry
    try:
        agent_a, run_a = await _launch_child(harness, "launch child a", 1)
        await _settle_child(harness, agent_a, run_a)
        record_a = registry._agent_records[agent_a]
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        loop_a = record_a.runtime.agent_loop
        # Suspension is warm: the failed backend is NOT retired yet.
        failed_backend = cast(_ScriptedBackend, loop_a._backend_lifetime.active)
        assert failed_backend.aexit_calls == 0

        # The sibling launches on its own backend and completes.
        agent_b, run_b = await _launch_child(harness, "launch child b", 2)
        sibling = await _settle_child(harness, agent_b, run_b)
        assert sibling.completed
        assert sibling.response == "sibling done"
        record_b = registry._agent_records[agent_b]
        loop_b = record_b.runtime.agent_loop
        sibling_backend = cast(_ScriptedBackend, loop_b._backend_lifetime.active)
        assert sibling_backend.aexit_calls == 0

        # Resume A: the refresh retires A's failed backend and installs a new
        # one; the successor completes on it.
        resume = await registry.resume_agent(
            agent_a, run_a, "k", requester_session_id=harness.requester_session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        successor = await _settle_child(harness, agent_a, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        # Only the failed loop's pre-resume backends were retired (closed
        # exactly once each, through its ownership-aware lifetime); the
        # successor's serving backend is the sole survivor.
        backends_a = harness.factory.created[id(loop_a)]
        await _until(lambda: failed_backend.aexit_calls >= 1)
        assert failed_backend.aexit_calls == 1
        await _until(lambda: all(b.aexit_calls >= 1 for b in backends_a[:-1]))
        assert loop_a._backend_lifetime.active is backends_a[-1]
        assert backends_a[-1].aexit_calls == 0
        # The sibling's loop kept its own backend instance, untouched.
        backends_b = harness.factory.created[id(loop_b)]
        assert backends_b[-1] is sibling_backend
        assert loop_b._backend_lifetime.active is sibling_backend
        assert sibling_backend.aexit_calls == 0
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_suspension_preserves_an_in_flight_sibling_and_drain_closes_the_warm_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F3 sibling survival under WP2 suspension (handoff §1.2).

    A recoverably failed child suspends WARM — its backend is not closed at
    failure time — while a sibling is mid-request. The sibling's await is not
    cancelled and its run completes on its own backend. The drain later
    closes the suspended runtime (and its backend) through the normal
    ownership path, and the drained agent is no longer resumable. This pins
    the failure-time half of the F3 borrowed-backend characterization for the
    WP2 semantics: ``tests/recovery/test_foundation.py`` records the
    pre-suspension teardown characterization (E4 amendment requested).
    """
    sibling_gate = asyncio.Event()
    sibling_in_flight = _Gated([mock_llm_chunk(content="sibling done")], sibling_gate)
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            # Launch order: the sibling (agent-1) first, the failing child
            # (agent-2) second — the factory binds scripts in launch order.
            [sibling_in_flight],
            [_write_outcome("todo-1", "item-a"), _empty_fault()],
        ],
        launches=["launch child b", "launch child a"],
    )
    registry = harness.registry
    try:
        agent_b, run_b = await _launch_child(harness, "launch child b", 1)
        # The sibling is mid-request before the other child is launched.
        await asyncio.wait_for(sibling_in_flight.started.wait(), timeout=5)
        agent_a, run_a = await _launch_child(harness, "launch child a", 2)
        result_a = await _settle_child(harness, agent_a, run_a)
        assert not result_a.completed
        record_a = registry._agent_records[agent_a]
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        loop_a = record_a.runtime.agent_loop
        warm_backend = cast(_ScriptedBackend, loop_a._backend_lifetime.active)
        # Suspension is warm: the failed loop's backend is NOT closed.
        assert warm_backend.aexit_calls == 0
        # The sibling's in-flight request was not cancelled by the failure.
        record_b = registry._agent_records[agent_b]
        assert record_b.current_run is not None
        assert record_b.current_run.run_id == run_b
        assert record_b.current_run.status is RunStatus.RUNNING
        assert not sibling_gate.is_set()

        sibling_gate.set()
        result_b = await _settle_child(harness, agent_b, run_b)
        assert result_b.completed
        assert result_b.response == "sibling done"
        loop_b = record_b.runtime.agent_loop
        sibling_backend = cast(_ScriptedBackend, loop_b._backend_lifetime.active)
        assert sibling_backend.aexit_calls == 0

        # The drain closes the suspended runtime through the normal ownership
        # path — the warm backend is retired exactly once — and the drained
        # agent is no longer resumable.
        await registry.drain_children()
        await _until(lambda: warm_backend.aexit_calls >= 1)
        assert warm_backend.aexit_calls == 1
        drained = await registry.resume_agent(
            agent_a, run_a, "k", requester_session_id=harness.requester_session_id
        )
        assert drained.outcome is ResumeOutcome.NOT_RESUMABLE
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_rejected_while_recovery_concurrency_is_occupied(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R8: ``recovery_concurrency`` bounds CONCURRENT recovery
    admissions — while another agent's resumed successor is still active,
    a second resume is rejected (RuntimeError, exactly like the running-cap
    rejection) and admits no successor.
    """
    gate = asyncio.Event()
    gated = _Gated(_final_outcome("a recovered"), gate)
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            # A: settles a tool effect, fails (suspends), and its resumed
            # successor stays in flight (gated) — the concurrency cap (1)
            # stays occupied.
            [_write_outcome("todo-1", "a item"), _empty_fault(), gated],
            # B: settles a tool effect and fails (suspends).
            [
                _write_outcome("todo-1", "b item"),
                _empty_fault(),
                _final_outcome("b done"),
            ],
        ],
        launches=["launch child a", "launch child b"],
        recovery=RecoveryConfig(recovery_concurrency=1),
    )
    registry = harness.registry
    try:
        agent_a, run_a = await _launch_child(harness, "launch child a", 1)
        failed_a = await _settle_child(harness, agent_a, run_a)
        assert not failed_a.completed
        agent_b, run_b = await _launch_child(harness, "launch child b", 2)
        failed_b = await _settle_child(harness, agent_b, run_b)
        assert not failed_b.completed
        record_a = registry._agent_records[agent_a]
        record_b = registry._agent_records[agent_b]
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        await _until(lambda: record_b.state is _AgentState.SUSPENDED)

        # A's explicit resume admits a successor that stays active (gated).
        resumed = await registry.resume_agent(
            agent_a, run_a, "k-a", requester_session_id=harness.requester_session_id
        )
        assert resumed.outcome is ResumeOutcome.SCHEDULED
        await asyncio.wait_for(gated.started.wait(), timeout=5)

        # B's resume is rejected: the recovery concurrency cap (1) is
        # occupied by A's active successor — no successor is admitted.
        with pytest.raises(RuntimeError, match="recovery"):
            await registry.resume_agent(
                agent_b, run_b, "k-b", requester_session_id=harness.requester_session_id
            )
        assert record_b.latest_run_id == run_b
        assert record_b.state is _AgentState.SUSPENDED
    finally:
        gate.set()
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_capacity_slot_is_reserved_at_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R8: the successor's capacity slot is RESERVED in the locked
    admission — a concurrent resume cannot pass the same running-cap check
    while another admission is still joining prior tasks (the old
    check-then-reserve race admitted both).
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "a item"),
                _empty_fault(),
                _final_outcome("a done"),
            ],
            [
                _write_outcome("todo-1", "b item"),
                _empty_fault(),
                _final_outcome("b done"),
            ],
        ],
        launches=["launch child a", "launch child b"],
    )
    registry = harness.registry
    # One running slot makes the reservation race observable.
    registry._max_running_subagents = 1
    try:
        agent_a, run_a = await _launch_child(harness, "launch child a", 1)
        failed_a = await _settle_child(harness, agent_a, run_a)
        assert not failed_a.completed
        agent_b, run_b = await _launch_child(harness, "launch child b", 2)
        failed_b = await _settle_child(harness, agent_b, run_b)
        assert not failed_b.completed
        record_a = registry._agent_records[agent_a]
        record_b = registry._agent_records[agent_b]
        await _until(lambda: record_a.state is _AgentState.SUSPENDED)
        await _until(lambda: record_b.state is _AgentState.SUSPENDED)

        # Freeze A's admission INSIDE the window between its locked cap
        # check and its slot creation: the failed run's completion task is
        # replaced with a joinable that only completes when released.
        failed_run_a = next(
            item
            for item in [record_a.current_run, *record_a.run_history]
            if item is not None and item.run_id == run_a
        )
        release = asyncio.Event()

        async def hold() -> None:
            await release.wait()

        failed_run_a.completion_task = asyncio.ensure_future(hold())
        resume_a = asyncio.ensure_future(
            registry.resume_agent(
                agent_a, run_a, "k-a", requester_session_id=harness.requester_session_id
            )
        )
        # A passed the cap check and is mid-admission (RESTORING) — its slot
        # is reserved (post-fix) or not yet created (pre-fix).
        await _until(lambda: record_a.state is _AgentState.RESTORING)

        # B's concurrent resume sees the reserved slot: rejected at the cap.
        with pytest.raises(RuntimeError, match="already running"):
            await registry.resume_agent(
                agent_b, run_b, "k-b", requester_session_id=harness.requester_session_id
            )
        assert record_b.latest_run_id == run_b
        assert record_b.state is _AgentState.SUSPENDED

        release.set()
        resumed = await resume_a
        assert resumed.outcome is ResumeOutcome.SCHEDULED
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resumed_run_failure_envelope_keeps_recovery_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R10: a RESUMED run's failure envelope keeps the record's
    recovery identity — the episode chain seeded by the first run's envelope
    and the actual same-agent resume count — never a fresh chain derived per
    run or a reset counter.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item-1"),
                _empty_fault(),
                _write_outcome("todo-2", "item-2"),
                _empty_fault(),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        first = await _settle_child(harness, agent_id, run_id)
        assert not first.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        first_envelope = first.failure
        assert first_envelope is not None
        # The first run derives the episode chain and reports zero resumes.
        assert first_envelope.recovery_chain_id == f"recovery-chain:{agent_id}:{run_id}"
        assert first_envelope.recovery.same_agent_resumes_used == 0
        assert record.recovery_chain_id == first_envelope.recovery_chain_id

        # The resumed run fails again: its envelope stays on the SAME chain
        # and reports the actual resume count — not a fresh per-run chain
        # or a reset counter.
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        second = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert not second.completed
        second_envelope = second.failure
        assert second_envelope is not None
        assert second_envelope.run_id == resume.successor_run_id
        assert second_envelope.recovery_chain_id == first_envelope.recovery_chain_id
        assert second_envelope.recovery_chain_id != (
            f"recovery-chain:{agent_id}:{resume.successor_run_id}"
        )
        assert second_envelope.recovery.same_agent_resumes_used == 1
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_resume_rejected_when_unavailable_preserves_timer_budget_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6a: an unavailable admission is rejected BEFORE consuming a resume
    or publishing RESTORING — the same read-only availability snapshot the
    automatic controller schedules against. A rejected admission consumes no
    budget, publishes no RESTORING, reserves no slot, and leaves the pending
    automatic timer armed; a subsequent available admission succeeds.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        # The automatic controller armed its pending schedule at suspension.
        controller = registry._recovery_controller
        episode = controller._episodes[agent_id]
        assert episode.active is True
        assert episode.timer is not None
        assert episode.scheduled_failed_run_id == run_id

        original_availability = RecoveryController._availability

        def _unavailable(_self: Any, _record: Any) -> _Availability:
            return _Availability(
                available=False,
                permanent=False,
                next_check_delay=12.0,
                endpoint_blocked=False,
            )

        def _permanent(_self: Any, _record: Any) -> _Availability:
            return _Availability(
                available=False,
                permanent=True,
                next_check_delay=0.0,
                endpoint_blocked=False,
            )

        async def assert_preserved() -> None:
            """The rejected admission consumed nothing and preserved state."""
            assert record.state is _AgentState.SUSPENDED
            assert record.same_agent_resumes_used == 0
            assert record.pending_resume is None
            assert record.latest_run_id == run_id
            assert not record.reserved_for_replacement
            successors = [
                item
                for item in record.run_history
                if item.resumed_from_run_id == run_id
            ]
            assert successors == []
            # The pending automatic timer stays armed for a rejected admission.
            assert episode.active is True
            assert episode.timer is not None
            assert episode.scheduled_failed_run_id == run_id

        # Temporary unavailability (cooldown): rejected without consuming.
        monkeypatch.setattr(RecoveryController, "_availability", _unavailable)
        rejected = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert rejected.outcome is ResumeOutcome.NOT_RESUMABLE
        assert rejected.reason_codes == ["model_unavailable"]
        await assert_preserved()

        # Permanent incompatibility: rejected with its own code, same
        # preservation guarantees.
        monkeypatch.setattr(RecoveryController, "_availability", _permanent)
        blocked = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert blocked.outcome is ResumeOutcome.NOT_RESUMABLE
        assert blocked.reason_codes == ["permanent_incompatibility"]
        await assert_preserved()

        # Availability restored: the admission succeeds and consumes exactly
        # one resume — the rejections consumed none.
        monkeypatch.setattr(RecoveryController, "_availability", original_availability)
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.same_agent_resumes_used == 1
        assert resume.successor_run_id is not None
        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_committed_resume_stands_down_timer_and_arms_progress_watch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6a: a COMMITTED explicit admission stands down the pending automatic
    timer (cancelled at the locked commit, the episode kept active for the
    successor's completion accounting) and installs the C5b resumed-progress
    context on the successor run's loop; the successor's completion closes
    the episode.
    """
    prepare_gate = asyncio.Event()
    call_gate = asyncio.Event()
    gated = _Gated(_final_outcome("resumed done"), call_gate)
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_write_outcome("todo-1", "item"), _empty_fault(), gated]],
        launches=["launch the child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        controller = registry._recovery_controller
        episode = controller._episodes[agent_id]
        assert episode.timer is not None

        # Hold the admission's prepare phase so the test observes the locked
        # commit before the successor run is prepared.
        original_prepare = registry._execution.prepare_background

        async def gated_prepare(*args: Any, **kwargs: Any) -> Any:
            await prepare_gate.wait()
            return await original_prepare(*args, **kwargs)

        monkeypatch.setattr(registry._execution, "prepare_background", gated_prepare)

        resume_task = asyncio.ensure_future(
            registry.resume_agent(
                agent_id, run_id, "k", requester_session_id=harness.requester_session_id
            )
        )
        await _until(lambda: record.pending_resume is not None)

        # The commit stood down the pending automatic timer atomically (the
        # stand-down shares the commit's critical section) while keeping the
        # episode active for the successor's completion accounting.
        assert episode.timer is None
        assert episode.scheduled_failed_run_id is None
        assert episode.active is True

        prepare_gate.set()
        resume = await resume_task
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None

        # The C5b resumed-progress context is installed on the successor
        # run's loop (armed by prepare_background with the resume admission).
        await asyncio.wait_for(gated.started.wait(), timeout=5)
        watch = record.runtime.agent_loop._completion_coordinator._resume_progress_watch
        assert watch is not None
        assert watch.disarmed is False
        assert watch.expired is False

        call_gate.set()
        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        # The successor's completion closed the kept-active episode.
        await _until(lambda: agent_id not in controller._episodes)
    finally:
        prepare_gate.set()
        call_gate.set()
        await harness.session.close()
