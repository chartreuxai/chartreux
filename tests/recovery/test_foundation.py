"""WP0 F2/F3/F4 — characterization with fault injection (plan gates F2-F4).

F2 (authoritative measurements): identifies the actual evidence sources for
each envelope field and fault-injects the four planned scenarios — zero
requests, three real attempts, partial output, finalization failure. The
evidence-to-source matrix is documented below; a field the current evidence
cannot supply is asserted as ``None`` (null = unavailable, never zero).

F3 (cancellation/client ownership): controlled blocked SDK requests,
cancellation during streaming and during backoff, cleanup, sibling isolation
on a borrowed backend, delayed cancellation acknowledgement, the proposed
cooldown windows on a fake clock, and the explicit proof that the current
retry budget cannot interrupt a blocked await.

F4 (state/effects/boundaries): the tool reconciliation matrix
(settled/unresolved/unknown/prohibited), the metadata-lag boundary
(messages.jsonl authoritative, meta.json may lag), resume-boundary malformed
tails (truncated final record, missing tool results, compaction envelope,
launch metadata), the run-pinned write-ownership token race, and Q9's
provider-switch history preservation (positive and prohibited).

Honest labeling: every fault here is a SYNTHETIC fixture on a controlled
transport — none of it is a live-provider reproduction. The observed production
incidents behind the scenarios (the Oct 13 hang-mode deaths, the Oct 17 fast
429 exhaustion, the L1/L2 zero-request launch deaths) are recorded in the
v0.5.6 recovery plan's F1 findings, not replayed here.

F2 evidence-to-source matrix (verified by the tests below):

| Envelope field | Authoritative source | Unavailable |
|---|---|---|
| transport-start (attempt count) | ``backend.requests_messages`` (one entry per real transport attempt) | n/a (always measurable) |
| logical-call count | assistant messages in ``agent_loop.messages`` | n/a |
| request-ID | ``BackendError.headers["x-request-id"|"request-id"]`` → ``FailureObservation.request_id`` | ``None`` on transport faults without headers (timeouts) |
| attempted deployment | committed identity (``base_model``/``provider``/``wire_name``) + coordinator's attempted candidate | n/a |
| attempted endpoint | provider ``api_base`` from the catalog snapshot; ``BackendError.endpoint`` when the fault carries it | ``None`` when the fault carries no endpoint |
| phase | ``FailureObservation.phase`` ("turn" / "turn_terminal" / "monitor") | n/a |
| semantic output | assistant message content in the transcript (published before the failure) | empty transcript |
| exhaustion | ``AllDeploymentsUnavailableError.exclusions`` (reason per deployment) | n/a |
| terminal result | ``TaskResult`` via ``wait_for_agent`` / ``PublicTurn`` via the turn controller | n/a |
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server._sessions import _AgentState
from chartreux.app_server._subagent_results import RunResults
from chartreux.app_server.models import PublicTurnStatus, TurnErrorCode
from chartreux.app_server.server import AppServer
from chartreux.app_server.session import AppServerTurnError
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm.failures import FailureCategory, FailureInfo, RequestRetryBudget
from chartreux.core.llm_models import (
    FunctionCall,
    LLMMessage,
    PersistedToolResult,
    Role,
    ToolCall,
)
from chartreux.core.model_catalog.availability import (
    AllDeploymentsUnavailableError,
    AvailabilityRegistry,
    ExclusionReason,
    eligible_deployments,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_observation import turn_observation
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import (
    AgentStats,
    CommittedModelIdentity,
    SessionMetadata,
)
from chartreux.core.subagents import (
    AgentAvailability,
    CancelOutcome,
    ReleaseAgentOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.utils.retry import async_retry
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _consume,
    _wait_result,
    _zero_retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import FakeClock, RetryableFault
from tests.session.persistence._session_persistence_support import (
    mock_tool_manager as mock_tool_manager,
    mock_vibe_config as mock_vibe_config,
    session_config as session_config,
    temp_session_dir as temp_session_dir,
)
from tests.stubs.app_server import (
    attach_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)


def _backend_error(
    status: int = 503,
    *,
    request_id: str = "req-wp0",
    endpoint: str = "https://api.mistral.ai/v1/chat/completions",
    code: str = "overloaded_error",
) -> BackendError:
    """A typed provider fault carrying the request-ID/endpoint evidence."""
    return BackendError(
        provider="mistral",
        endpoint=endpoint,
        status=status,
        reason="service unavailable",
        headers={"x-request-id": request_id},
        body_text=json.dumps({"error": {"code": code}}),
        parsed_error=json.dumps({"error": {"code": code}}),
        model="zai-glm-5-3",
        payload_summary=PayloadSummary(
            model="zai-glm-5-3",
            message_count=1,
            approx_chars=10,
            temperature=0,
            has_tools=False,
            tool_choice=None,
        ),
    )


def _assistant_count(loop: Any) -> int:
    return sum(1 for message in loop.messages if message.role is Role.assistant)


async def _poll(predicate: Any, timeout: float = 2.0) -> bool:
    """Poll a predicate without a pilot (event barriers, no fixed sleeps)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


# ---------------------------------------------------------------------------
# F2 — authoritative measurements (fault injection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f2_zero_requests_cooldown_blocked_turn(
    controlled_transport, clean_observation_store
) -> None:
    """Zero transport requests: the turn is blocked at admission (L1/L2 shape).

    The shared availability cooldown excludes every deployment of the
    committed base, so the turn fails before any transport start: zero new
    requests, zero new turns, an ``all_deployments_unavailable`` terminal
    result whose exclusions name the cooldown, and a typed observation
    (phase ``turn``) preserved before the message-only projection.
    """
    backend = controlled_transport([[mock_llm_chunk(content="first answer")]])
    loop = build_test_agent_loop(backend=backend)
    client = start_test_app_server(loop)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    committed = loop.committed_model
    assert committed is not None
    registry = loop.config_orchestrator.availability_registry
    try:
        async for _event in session.act("hello"):
            pass
        assert len(backend.requests_messages) == 1
        assert _assistant_count(loop) == 1

        registry.record_failure(committed.base_model, committed.provider)
        with pytest.raises(AppServerTurnError) as exc_info:
            async for _event in session.act("second"):
                pass

        # Terminal-result source: the dedicated exhaustion projection.
        assert exc_info.value.error.code == TurnErrorCode.ALL_DEPLOYMENTS_UNAVAILABLE
        assert "cooldown" in exc_info.value.error.message
        # Transport-start source: zero new requests; logical-call source: zero
        # new turns.
        assert len(backend.requests_messages) == 1
        assert _assistant_count(loop) == 1
        completed = legacy_backend(server).session.turns._completed_turns[-1]
        assert completed.status is PublicTurnStatus.FAILED
        # Typed evidence preserved before the message-only projection (WP0 F5).
        observation = turn_observation(completed.id)
        assert observation is not None
        assert observation.phase == "turn"
        assert observation.exception_type == "AllDeploymentsUnavailableError"
        assert observation.turn_id == completed.id
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_f2_three_real_attempts_one_logical_call(controlled_transport) -> None:
    """Three real transport attempts, one logical call, retry telemetry.

    The scripted transport fails twice with a typed 503 and succeeds on the
    third attempt, mirroring the production backend wrapper's internal
    ``async_retry`` (the gateway itself retries only empty replays). The
    measurement contract: the transport-start source counts three attempts,
    the logical-call source counts one assistant message, and the retry
    observer saw two notices.
    """
    transport = controlled_transport([
        RetryableFault(_backend_error(503, request_id="req-attempt-1")),
        RetryableFault(_backend_error(503, request_id="req-attempt-2")),
        [mock_llm_chunk(content="recovered after retries")],
    ])
    reasons: list[Any] = []

    async def observe(reason: Any) -> None:
        reasons.append(reason)

    transport.on_retry = observe
    loop = build_test_agent_loop(backend=transport)

    async for _event in loop.act("hi"):
        pass

    # Transport-start source: three real attempts.
    assert len(transport.requests_messages) == 3
    # Retry telemetry: two notices between the three attempts.
    assert len(reasons) == 2
    # Logical-call source: exactly one assistant message.
    assistant = [m for m in loop.messages if m.role is Role.assistant]
    assert len(assistant) == 1
    assert assistant[0].content == "recovered after retries"
    # Attempted-deployment source: the committed identity; attempted-endpoint
    # source: the provider's api_base from the catalog snapshot.
    committed = loop.committed_model
    assert committed is not None
    assert (committed.base_model, committed.provider, committed.wire_name) == (
        "glm-5-3",
        "mistral",
        "zai-glm-5-3",
    )
    snapshot = loop.config.catalog_snapshot
    assert snapshot is not None
    provider = snapshot.catalog.providers[committed.provider]
    assert provider.api_base == "https://api.mistral.ai/v1"


@pytest.mark.asyncio
async def test_f2_partial_output_published_no_transparent_replay(
    controlled_transport,
) -> None:
    """Partial output: semantic output published, no transparent replay.

    The stream yields content and then faults. The gateway appends the
    partial assistant message to the transcript (semantic-output source), and
    the completion coordinator refuses a transparent replay once semantic
    output exists: exactly one transport attempt, and the fault surfaces with
    its request-ID evidence intact in the cause chain.
    """
    transport = controlled_transport([
        [
            mock_llm_chunk(content="partial answer"),
            _backend_error(503, request_id="req-partial"),
        ]
    ])
    loop = build_test_agent_loop(backend=transport, enable_streaming=True)

    with pytest.raises(RuntimeError) as exc_info:
        async for _event in loop.act("hi"):
            pass

    # No transparent replay: one transport attempt only.
    assert len(transport.requests_messages) == 1
    # Semantic-output source: the partial content is published to the
    # transcript before the failure.
    assistant = [m for m in loop.messages if m.role is Role.assistant]
    assert [m.content for m in assistant] == ["partial answer"]
    # Request-ID source survives the message-only mapping in the cause chain.
    cause = exc_info.value.__cause__
    assert isinstance(cause, BackendError)
    assert cause.headers["x-request-id"] == "req-partial"
    assert cause.endpoint == "https://api.mistral.ai/v1/chat/completions"
    assert cause.status == 503


@pytest.mark.asyncio
async def test_f2_finalization_failure_captured_before_projection(
    monkeypatch: pytest.MonkeyPatch, controlled_transport, clean_observation_store
) -> None:
    """Finalization failure: typed evidence captured at turn terminalization.

    The model call succeeds but the turn finalization faults. The exception
    escapes the turn phase, so the terminalization path records the typed
    observation (phase ``turn_terminal``) before the message-only
    ``PublicError`` projection, and the terminal result is FAILED even though
    the assistant output remains in the transcript.
    """
    backend = controlled_transport([[mock_llm_chunk(content="ok")]])
    loop = build_test_agent_loop(backend=backend)
    client = start_test_app_server(loop)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    turns = legacy_backend(server).session.turns

    async def faulting_finalize(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("finalization fault")

    monkeypatch.setattr(turns, "_finalize_turn", faulting_finalize)
    try:
        with pytest.raises(AppServerTurnError) as exc_info:
            async for _event in session.act("hi"):
                pass

        # The public projection is message-only (WP0 F5 preserves the typed
        # evidence privately; the envelope is WP1's scope).
        assert exc_info.value.error.code == TurnErrorCode.INTERNAL_ERROR
        assert "finalization fault" in exc_info.value.error.message
        completed = turns._completed_turns[-1]
        assert completed.status is PublicTurnStatus.FAILED
        observation = turn_observation(completed.id)
        assert observation is not None
        assert observation.phase == "turn_terminal"
        assert observation.exception_type == "RuntimeError"
        assert observation.message == "finalization fault"
        # The successful model call's output stays in the transcript.
        assert _assistant_count(loop) == 1
    finally:
        await session.close()


# ---------------------------------------------------------------------------
# F3 — cancellation and client ownership
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_f3_cancel_blocked_request_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation during a blocked SDK request: cleanup runs, run settles."""
    child = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, child)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child.started.wait(), timeout=2)
        outcome = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        # Cleanup: the blocked await is cancelled and the backend observes it.
        await asyncio.wait_for(child.stopped.wait(), timeout=2)
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )
        assert not result.completed
        assert result.stop_reason is RunStopReason.USER_CANCELLED
    finally:
        child.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_f3_delayed_cancellation_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STOP_REQUESTED precedes the actual stop (delayed acknowledgement).

    The stop request is acknowledged before the blocked request finishes
    stopping: the backend's cleanup is gated, so the outcome is returned while
    the backend is still inside its request.
    """

    class SlowCancelBackend(BlockingBackend):
        def __init__(self) -> None:
            super().__init__()
            self.cancel_gate = asyncio.Event()

        async def complete(self, **kwargs: Any) -> Any:
            self.started.set()
            try:
                await self.release.wait()
                return mock_llm_chunk(content="done")
            finally:
                # Delay the cleanup so the acknowledgement observably
                # precedes the stop.
                await self.cancel_gate.wait()
                self.stopped.set()

    child = SlowCancelBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, child)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child.started.wait(), timeout=2)
        outcome = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        # The stop is requested but has not landed yet.
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        assert not child.stopped.is_set()
        child.cancel_gate.set()
        await asyncio.wait_for(child.stopped.wait(), timeout=2)
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )
        assert not result.completed
        assert result.stop_reason is RunStopReason.USER_CANCELLED
    finally:
        child.release.set()
        child.cancel_gate.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_f3_cancel_during_streaming_closes_stream(controlled_transport) -> None:
    """Cancellation during streaming: the client interrupt closes the stream.

    The root turn streams one chunk and then blocks mid-stream (an in-stream
    gate). The client interrupts its own turn; cleanup closes the in-flight
    stream (the transport's generator finally runs) and the terminal turn is
    INTERRUPTED — the client owns the interruption.
    """
    in_stream_gate = asyncio.Event()
    transport = controlled_transport([
        [mock_llm_chunk(content="streamed chunk"), in_stream_gate]
    ])
    loop = build_test_agent_loop(backend=transport, enable_streaming=True)
    client = start_test_app_server(loop)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    try:
        events = session.act("hi")
        task = asyncio.create_task(_consume(events))
        await asyncio.wait_for(transport.stream_started.wait(), timeout=2)
        assert await _poll(lambda: len(transport.requests_messages) == 1)

        await session.interrupt()
        # Cleanup closes the in-flight stream (the generator's finally runs).
        await asyncio.wait_for(transport.stream_closed.wait(), timeout=2)
        await asyncio.wait_for(task, timeout=5)
        completed = legacy_backend(server).session.turns._completed_turns[-1]
        assert completed.status is PublicTurnStatus.INTERRUPTED
    finally:
        in_stream_gate.set()
        await session.close()


@pytest.mark.asyncio
async def test_f3_cancel_during_backoff_stops_before_next_attempt(
    monkeypatch: pytest.MonkeyPatch, controlled_transport
) -> None:
    """Cancellation during backoff: the retry never reaches the transport.

    The scripted transport fails once and then backs off before its second
    attempt (mirroring the production wrapper's backoff sleep). Cancelling
    during the backoff stops the run before the next attempt: exactly one
    transport start is recorded.
    """
    transport = controlled_transport([
        RetryableFault(_backend_error(503, request_id="req-backoff")),
        [mock_llm_chunk(content="never reached")],
    ])
    transport.retry_delay = 30.0
    noticed = asyncio.Event()

    async def observe(reason: Any) -> None:
        noticed.set()

    transport.on_retry = observe
    registry, parent, _ = await _zero_retention_registry(monkeypatch, transport)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        # Wait until the first attempt failed and the backoff started (the
        # retry notice fires before the backoff sleep).
        await asyncio.wait_for(noticed.wait(), timeout=2)
        assert len(transport.requests_messages) == 1
        outcome = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )
        assert not result.completed
        # The backoff was interrupted: no second transport attempt.
        assert len(transport.requests_messages) == 1
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_f3_borrowed_backend_cleanup_and_sibling_survival(
    monkeypatch: pytest.MonkeyPatch, controlled_transport
) -> None:
    """A recoverably failed child suspends warm; a sibling's in-flight request survives.

    Two background children share one borrowed backend instance (the test
    factory hands the same transport to every child). Child A (ephemeral)
    fails with a transient 503 while child B (retained) is still inside its
    request. Post-WP2 semantics (handoff §1.2): the tool-effects-settled
    failure is recovery-eligible, so child A SUSPENDS warm — its runtime
    stays resident and the borrowed instance is NOT closed at failure time.
    The sibling's await is not cancelled and its run completes successfully.
    Borrow protection stays per-owner, not per-instance (recorded honestly
    via ``aexit_calls``): releasing B closes B's borrow exactly once, and
    the drain then closes the suspended runtime's borrow exactly once
    through the normal ownership path — the shared instance ends at one
    close per owner.
    """
    shared = controlled_transport()
    gate_a = shared.gate_attempt(0)
    gate_b = shared.gate_attempt(1)
    shared.add_outcome(_backend_error(503, request_id="req-sibling-a"))
    shared.add_outcome([mock_llm_chunk(content="sibling B done")])
    registry, parent, _ = await _zero_retention_registry(monkeypatch, shared)
    try:
        launch_a = await _background_result(
            registry,
            TaskArgs(task="child A", identity="coder", background=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch_a.agent_id is not None and launch_a.run_id is not None
        # Attempt indexes are transport-global: A's request must be the
        # transport's first attempt (index 0), so wait for it to start
        # before launching B — then B's request is deterministically index 1.
        await asyncio.wait_for(gate_a[0].wait(), timeout=2)
        launch_b = await _background_result(
            registry,
            TaskArgs(task="child B", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="t2", session_id=parent.session_id),
        )
        assert launch_b.agent_id is not None and launch_b.run_id is not None
        await asyncio.wait_for(gate_b[0].wait(), timeout=2)
        assert len(shared.requests_messages) == 2

        # Fail A while B is still inside its request.
        gate_a[1].set()
        result_a = await asyncio.wait_for(
            _wait_result(registry, launch_a.agent_id, launch_a.run_id), timeout=5
        )
        assert not result_a.completed
        record_a = registry._agent_records[launch_a.agent_id]
        record_b = registry._agent_records[launch_b.agent_id]
        # Post-WP2 (handoff §1.2): the eligible failure suspends A warm —
        # the record is SUSPENDED and the borrowed backend is NOT closed
        # at failure time.
        assert await _poll(lambda: record_a.state is _AgentState.SUSPENDED)
        assert record_a.availability is AgentAvailability.SUSPENDED
        assert shared.aexit_calls == 0
        # Sibling survival: B's request was not cancelled by A's failure
        # and is still in flight.
        assert record_b.current_run is not None
        assert record_b.current_run.run_id == launch_b.run_id
        assert record_b.current_run.status is RunStatus.RUNNING
        assert not gate_b[1].is_set()
        gate_b[1].set()
        result_b = await asyncio.wait_for(
            _wait_result(registry, launch_b.agent_id, launch_b.run_id), timeout=5
        )
        assert result_b.completed
        assert "sibling B done" in result_b.response
        assert len(shared.requests_messages) == 2
        # Completion does not close the borrowed instance either: B's
        # runtime stays resident (idle, retained) with its borrow open.
        assert record_b.state is _AgentState.IDLE
        assert shared.aexit_calls == 0

        # Per-owner borrow protection on the close path: releasing B
        # closes B's borrow of the shared instance exactly once.
        released = await registry.release_agent(launch_b.agent_id)
        assert released is ReleaseAgentOutcome.RELEASED
        assert await _poll(lambda: shared.aexit_calls >= 1)
        assert shared.aexit_calls == 1
        # The drain closes the suspended runtime through the normal
        # ownership path — its backend closes exactly once more, so the
        # shared instance ends at one close per owner (per-owner, not
        # per-instance).
        await registry.drain_children()
        assert await _poll(lambda: shared.aexit_calls >= 2)
        assert shared.aexit_calls == 2
    finally:
        gate_a[1].set()
        gate_b[1].set()
        await registry.drain_children()
        await parent.aclose()


def test_f3_cooldown_windows_on_fake_clock(fake_clock: FakeClock) -> None:
    """The proposed cooldown windows, exercised on a fake clock.

    Q7 starting proposals (30s initial cooldown, longer Retry-After, single-
    flight probes) validated deterministically: no real sleeps, no flakiness.
    """
    registry = AvailabilityRegistry(clock=fake_clock)
    assert registry.initial_cooldown == 30.0
    assert registry.is_available("base", "p1")

    # A failure starts the 30s initial cooldown.
    registry.record_failure("base", "p1")
    assert not registry.is_available("base", "p1")
    assert registry.cooldown_until("base", "p1") == 30.0
    fake_clock.advance(29.9)
    assert not registry.is_available("base", "p1")
    fake_clock.advance(0.2)
    assert registry.is_available("base", "p1")

    # A success clears the cooldown.
    registry.record_failure("base", "p1")
    registry.record_success("base", "p1")
    assert registry.is_available("base", "p1")
    assert registry.cooldown_until("base", "p1") is None

    # A longer Retry-After extends the window past the 30s floor (120s
    # proposal): until = now + max(30, retry_after).
    now = fake_clock.value
    registry.record_failure(
        "base",
        "p1",
        FailureInfo(category=FailureCategory.RATE_LIMIT, retry_after=120.0),
    )
    assert registry.cooldown_until("base", "p1") == now + 120.0
    fake_clock.advance(119.9)
    assert not registry.is_available("base", "p1")
    fake_clock.advance(0.2)
    assert registry.is_available("base", "p1")

    # reset_base clears every provider of the base (the user-retry reset).
    registry.record_failure("base", "p1")
    registry.record_failure("base", "p2")
    registry.reset_base("base")
    assert registry.is_available("base", "p1")
    assert registry.is_available("base", "p2")
    assert registry.cooldown_until("base", "p1") is None

    # Admission after expiry claims a single-flight probe; release is
    # token-checked.
    registry.record_failure("base", "p1")
    fake_clock.advance(30.0)
    admitted, probe, token = registry.admission("base", "p1")
    assert admitted is True and probe is True and token is not None
    assert registry.admission("base", "p1")[:2] == (False, False)
    assert registry.is_available("base", "p1") is False
    registry.release_probe("base", "p1", token + 999)
    assert registry.admission("base", "p1")[:2] == (False, False)
    registry.release_probe("base", "p1", token)
    assert registry.admission("base", "p1")[:2] == (True, True)


@pytest.mark.asyncio
async def test_f3_retry_budget_cannot_interrupt_a_blocked_await() -> None:
    """The current retry budget cannot interrupt a blocked await.

    WP0 acceptance item, proven explicitly: ``async_retry`` consults the
    budget only BETWEEN attempts. While the wrapped call is blocked inside
    its await, the budget expiring changes nothing — the call stays blocked,
    no retry starts, and the call returns normally once released. A negative
    Q2 result: there is no hard in-flight timeout today, so any admission
    deadline must be advisory, never advertised as a hard bound.
    """
    release = asyncio.Event()
    calls = 0

    async def blocked_call() -> str:
        nonlocal calls
        calls += 1
        await release.wait()
        return "completed after release"

    budget = RequestRetryBudget(0.05)
    call = async_retry(tries=None, budget=budget)(blocked_call)

    task = asyncio.ensure_future(call())
    # Let the budget expire well past its deadline while the call is blocked.
    await asyncio.sleep(0.25)
    assert budget.exhausted
    assert not task.done()
    assert calls == 1

    release.set()
    assert await asyncio.wait_for(task, timeout=2) == "completed after release"
    assert calls == 1


# ---------------------------------------------------------------------------
# F4 — state, effects, and boundaries
# ---------------------------------------------------------------------------


def _tool_call(call_id: str, name: str = "bash") -> ToolCall:
    return ToolCall(
        id=call_id, index=0, function=FunctionCall(name=name, arguments="{}")
    )


def _is_denial_result(result: LLMMessage) -> bool:
    """Transcript-level denial signal (heuristic; WP1 owns the typed version)."""
    content = result.content
    if isinstance(content, str):
        lowered = content.lower()
        return "denied" in lowered or "permission" in lowered
    return False


def _reconcile_tool_dispositions(transcript: list[LLMMessage]) -> dict[str, str]:
    """Characterization classifier: the reconciliation dispositions the
    transcript evidence supports (settled | unresolved | unknown | prohibited).

    The production reconcile tool is WP2's scope; this classifier documents
    what the persisted transcript can and cannot prove. The transcript
    boundary is the evidence limit: a tool call beyond the boundary is
    invisible here and must be treated as ``unknown`` by any reconciler —
    zero observed tool results is NOT proof of no effects.
    """
    results = {
        message.tool_call_id: message
        for message in transcript
        if message.role is Role.tool and message.tool_call_id
    }
    calls = [
        call
        for message in transcript
        if message.role is Role.assistant
        for call in (message.tool_calls or [])
    ]
    dispositions: dict[str, str] = {}
    for call in calls:
        call_id = call.id
        if call_id is None:
            # An unidentifiable call cannot be keyed for reconciliation.
            continue
        result = results.get(call_id)
        if result is None:
            # Admitted (the call is in the transcript) but no result: the
            # outcome is unknown — the body may have executed.
            dispositions[call_id] = "unresolved"
        elif _is_denial_result(result):
            # The action did not execute and must never be replayed.
            dispositions[call_id] = "prohibited"
        else:
            dispositions[call_id] = "settled"
    for call_id in results:
        if call_id not in {call.id for call in calls}:
            # A result whose call is not in the transcript: the admission
            # evidence is missing (boundary cut) — unknown, not settled.
            dispositions[call_id] = "unknown"
    return dispositions


def test_f4_tool_reconciliation_matrix() -> None:
    """The tool reconciliation matrix over transcript evidence."""
    transcript = [
        LLMMessage(role=Role.user, content="run three commands"),
        # settled: call admitted and a normal result recorded.
        LLMMessage(
            role=Role.assistant, content="", tool_calls=[_tool_call("call-settled")]
        ),
        LLMMessage(
            role=Role.tool,
            tool_call_id="call-settled",
            content="command output",
            tool_result=PersistedToolResult(output={"text": "command output"}),
        ),
        # unresolved: call admitted, no result (missing tool result).
        LLMMessage(
            role=Role.assistant, content="", tool_calls=[_tool_call("call-unresolved")]
        ),
        # prohibited: call admitted, the result records a denial.
        LLMMessage(
            role=Role.assistant, content="", tool_calls=[_tool_call("call-denied")]
        ),
        LLMMessage(
            role=Role.tool,
            tool_call_id="call-denied",
            content="Error: permission denied",
            tool_result=PersistedToolResult(output={"error": "denied"}),
        ),
        # unknown: a result whose call is beyond the transcript boundary.
        LLMMessage(role=Role.tool, tool_call_id="call-orphan", content="late result"),
    ]
    dispositions = _reconcile_tool_dispositions(transcript)
    assert dispositions == {
        "call-settled": "settled",
        "call-unresolved": "unresolved",
        "call-denied": "prohibited",
        "call-orphan": "unknown",
    }


@pytest.mark.asyncio
async def test_f4_metadata_lag_boundary(
    session_config: SessionLoggingConfig,
    mock_vibe_config: ChartreuxConfigSchema,
    mock_tool_manager: Any,
) -> None:
    """Metadata lag: messages.jsonl is authoritative; meta.json may lag.

    An interrupted save can leave meta.json behind the transcript. The next
    save must not duplicate transcript records (the transcript prevents
    duplicate appends) and repairs the stale metadata.
    """
    logger = SessionLogger(session_config, "lag-session")
    messages = [
        LLMMessage(role=Role.user, content="Hello"),
        LLMMessage(role=Role.assistant, content="Hi"),
    ]
    await logger.save_interaction(
        messages=messages,
        stats=AgentStats(steps=1),
        config=mock_vibe_config,
        tool_manager=mock_tool_manager,
    )
    assert logger.session_dir is not None
    messages_path = logger.session_dir / "messages.jsonl"
    meta_path = logger.session_dir / "meta.json"
    assert len(messages_path.read_text().splitlines()) == 2
    assert json.loads(meta_path.read_text())["total_messages"] == 2

    # Simulate the interrupted-save lag: metadata behind the transcript.
    metadata = json.loads(meta_path.read_text())
    metadata["total_messages"] = 1
    meta_path.write_text(json.dumps(metadata))

    # A later save appends without duplicating and repairs the metadata.
    await logger.save_interaction(
        messages=[*messages, LLMMessage(role=Role.user, content="Again")],
        stats=AgentStats(steps=2),
        config=mock_vibe_config,
        tool_manager=mock_tool_manager,
    )
    transcript_lines = messages_path.read_text().splitlines()
    assert len(transcript_lines) == 3
    assert json.loads(meta_path.read_text())["total_messages"] == 3


@pytest.mark.asyncio
async def test_f4_resume_boundary_malformed_tails(tmp_path: Path) -> None:
    """resume_child's load boundary: malformed tails and resume metadata.

    ``SessionLoader.load_session`` is the resume_child entry point
    (``chartreux/app_server/_runtime.py:resume_child``). Characterized:
    a truncated final record recovers its valid prefix with a warning;
    interior corruption and an empty log with mismatched metadata are hard
    errors; missing tool results do NOT block loading (pairing is a
    reconciliation concern, not a load error); the compaction envelope and
    launch metadata survive the round trip.
    """
    session_dir = tmp_path / "session_resume"
    session_dir.mkdir()
    messages_path = session_dir / "messages.jsonl"
    meta_path = session_dir / "meta.json"

    def write(messages_text: str, metadata: dict[str, Any]) -> None:
        messages_path.write_text(messages_text)
        meta_path.write_text(json.dumps(metadata))

    # (a) Truncated final record (interrupted write): the valid prefix is
    # recovered and a RuntimeWarning is emitted.
    write(
        '{"role": "user", "content": "ok"}\n{"role": "assist',
        {"session_id": "s", "total_messages": 1},
    )
    with pytest.warns(RuntimeWarning, match="malformed final"):
        loaded, _metadata = SessionLoader.load_session(session_dir)
    assert [m.content for m in loaded] == ["ok"]

    # (b) Interior invalid JSON: hard error (no recovery).
    write(
        '{"role": "user", "content": "ok"}\n{invalid}\n{"role": "user", "content": "x"}',
        {"session_id": "s", "total_messages": 3},
    )
    with pytest.raises(ValueError, match="interior record"):
        SessionLoader.load_session(session_dir)

    # (c) Empty log with mismatched metadata: hard error.
    write("", {"session_id": "s", "total_messages": 2})
    with pytest.raises(ValueError, match="empty"):
        SessionLoader.load_session(session_dir)

    # (d) Missing tool results: loads fine — the loader does not pair calls
    # with results; that is the reconciliation matrix's concern (above).
    write(
        json.dumps({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "index": 0,
                    "function": {"name": "bash", "arguments": "{}"},
                }
            ],
        })
        + "\n",
        {"session_id": "s", "total_messages": 1},
    )
    loaded, _metadata = SessionLoader.load_session(session_dir)
    assert loaded[0].tool_calls is not None
    assert loaded[0].tool_calls[0].id == "call-1"

    # (e) Compaction envelope: the boundary marker survives the round trip.
    write(
        json.dumps({
            "role": "user",
            "content": "summary",
            "context_boundary": "compaction",
        })
        + "\n",
        {"session_id": "s", "total_messages": 1},
    )
    loaded, _metadata = SessionLoader.load_session(session_dir)
    assert loaded[0].context_boundary == "compaction"

    # (f) Launch metadata: a stored legacy V1 envelope still parses as
    # session metadata (resume rejects it later with new-session guidance).
    write(
        '{"role": "user", "content": "ok"}\n',
        {
            "session_id": "s",
            "total_messages": 1,
            "start_time": "2024-01-01T12:00:00Z",
            "end_time": "2024-01-01T12:05:00Z",
            "git_commit": None,
            "git_branch": None,
            "username": "test-user",
            "environment": {"working_directory": str(tmp_path)},
            "launch_config": {
                "version": 1,
                "profile": "worker",
                "overrides": {},
                "persona": {"system_prompt_id": "tests", "instructions": ""},
            },
        },
    )
    _loaded, raw_metadata = SessionLoader.load_session(session_dir)
    metadata = SessionMetadata.model_validate(raw_metadata)
    envelope = metadata.launch_config
    assert envelope is not None
    assert envelope.version == 1
    assert envelope.profile == "worker"


class _StubResultsView:
    """Minimal ``ResultsStateView`` for the write-ownership token protocol."""

    def __init__(self) -> None:
        self._agent_records: dict[str, Any] = {}
        self._result_store: dict[tuple[str, str], Any] = {}
        self._expired_results: set[tuple[str, str]] = set()
        self._latest_run_ids: dict[str, str] = {}
        self._wait_leases: dict[tuple[str, str], int] = {}
        self._pending_notifications: set[tuple[str, str]] = set()
        self._result_write_tokens: dict[tuple[str, str], object] = {}
        self._evicted_agents: dict[str, Any] = {}
        self._children: dict[str, Any] = {}
        self._admission_closed = False
        self._draining_children = False
        self._suppressed_notifications: set[str] = set()
        self._registry_lock = asyncio.Lock()
        self._clock = lambda: 0.0

    def _agent_turns_used(self, record: Any) -> int | None:
        return None

    def _generation_is_current(self, generation: int) -> bool:
        return True


def _publication_context(
    agent_id: str, run_id: str, *, record: Any, runtime: Any, token: object
) -> Any:
    """Build the monitor's publication context bound to one exact run record."""
    run_record = SimpleNamespace(run_id=run_id)
    record.current_run = run_record
    return SimpleNamespace(
        agent_id=agent_id,
        run_id=run_id,
        record=record,
        runtime=runtime,
        run_record=run_record,
        publication_token=token,
    )


def _agent_record(agent_id: str) -> Any:
    return SimpleNamespace(
        session_id=f"session-{agent_id}",
        current_run=None,
        root_generation=0,
        work_package_id=None,
    )


def test_f4_run_pinned_write_ownership_token_race() -> None:
    """Run-pinned write ownership: exact-token check/pop, replacement handoff.

    ``register_writer_locked`` installs the exact write token for one accepted
    run; ``publish_locked`` accepts only that exact token and pops it
    (exactly-once publication). A replacement handoff registers a NEW run's
    token: the old run's publication can never spend the new run's authority,
    and stale residency never forbids storing a result.
    """
    view = _StubResultsView()
    results = RunResults(view)
    agent_id = "agent-1"
    runtime = object()
    old_token = object()
    new_token = object()

    # The exact token publishes exactly once (check/pop).
    record1 = _agent_record(agent_id)
    view._agent_records[agent_id] = record1
    view._children[record1.session_id] = runtime
    context = _publication_context(
        agent_id, "run-1", record=record1, runtime=runtime, token=old_token
    )
    results.register_writer_locked(agent_id, "run-1", old_token)
    decision = results.publish_locked(context)
    assert decision.resident is True
    assert decision.write_permitted is True
    assert decision.notify_permitted is True
    # The token is consumed: a second publication is refused.
    assert results.publish_locked(context).write_permitted is False

    # Replacement handoff: two runs, two tokens, per-key exact checks. The
    # new run's authority never authorizes the old run's publication.
    results.register_writer_locked(agent_id, "run-1", old_token)
    results.register_writer_locked(agent_id, "run-2", new_token)
    record2 = _agent_record(agent_id)
    view._agent_records[agent_id] = record2
    view._children[record2.session_id] = runtime
    old_run_new_token = _publication_context(
        agent_id, "run-1", record=record1, runtime=runtime, token=new_token
    )
    assert results.publish_locked(old_run_new_token).write_permitted is False
    fresh = _publication_context(
        agent_id, "run-2", record=record2, runtime=runtime, token=new_token
    )
    decision = results.publish_locked(fresh)
    assert decision.write_permitted is True
    assert decision.resident is True

    # Stale residency never forbids storing a result: the agent record was
    # replaced (this run is no longer resident), but the exact token still
    # permits the write; only the notification is suppressed.
    record3 = _agent_record(agent_id)
    results.register_writer_locked(agent_id, "run-3", old_token)
    view._agent_records[agent_id] = _agent_record(agent_id)
    stale_residency = _publication_context(
        agent_id, "run-3", record=record3, runtime=runtime, token=old_token
    )
    decision = results.publish_locked(stale_residency)
    assert decision.resident is False
    assert decision.write_permitted is True
    assert decision.notify_permitted is False


def _q9_catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": "https://first.test"},
                "test-second": {"api_base": "https://second.test"},
            },
            "models": {
                "base": {
                    "deployments": [
                        {
                            "provider": "test-first",
                            "name": "first",
                            "supported_thinking_levels": ["off"],
                        },
                        {
                            "provider": "test-second",
                            "name": "second",
                            "supported_thinking_levels": ["off"],
                        },
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )


def _q9_config(snapshot: CatalogSnapshot) -> ChartreuxConfigSchema:
    return ChartreuxConfigSchema(active_model="base").attach_catalog_snapshot(snapshot)


def _q9_committed() -> CommittedModelIdentity:
    return CommittedModelIdentity(
        base_model="base",
        provider="test-first",
        wire_name="first",
        catalog_revision="test",
    )


def test_f4_q9_provider_switch_preserves_history_positive() -> None:
    """Q9 positive: an eligible same-base provider switch preserves history.

    With the committed provider in cooldown, the same-base switch to the
    second provider is admitted, and the retained history (reasoning payloads
    and content) passes through unchanged — the switch is history-preserving.
    """
    snapshot = _q9_catalog()
    config = _q9_config(snapshot)
    history = [
        LLMMessage(role=Role.user, content="hi"),
        LLMMessage(
            role=Role.assistant,
            content="answer",
            reasoning_content="thinking",
            reasoning_payloads=[{"type": "opaque"}],
        ),
    ]
    before = [message.model_dump(mode="json") for message in history]
    registry = AvailabilityRegistry()
    registry.record_failure("base", "test-first")

    result = eligible_deployments(
        snapshot=snapshot,
        committed=_q9_committed(),
        registry=registry,
        config=config,
        history=history,
        thinking="off",
    )
    assert len(result.candidates) == 1
    assert result.candidates[0].resolved.identity.provider == "test-second"
    assert result.candidates[0].resolved.identity.wire_name == "second"
    # The retained history is preserved, not stripped or rewritten.
    assert [message.model_dump(mode="json") for message in history] == before


def test_f4_q9_provider_switch_prohibited_preserves_history() -> None:
    """Q9 prohibited: an incompatible switch is excluded, history intact.

    Retained reasoning the target provider cannot serve excludes every
    deployment with ``reasoning-unsupported``; the history itself is never
    stripped — the prohibition is a property of the (model, history) pair.
    """
    snapshot = _q9_catalog()
    config = _q9_config(snapshot)
    history = [LLMMessage(role=Role.assistant, reasoning_payloads=[{"type": "opaque"}])]
    before = [message.model_dump(mode="json") for message in history]

    with pytest.raises(AllDeploymentsUnavailableError) as exc_info:
        eligible_deployments(
            snapshot=snapshot,
            committed=_q9_committed(),
            registry=AvailabilityRegistry(),
            config=config,
            history=history,
            thinking="off",
        )
    assert ExclusionReason.REASONING_UNSUPPORTED in {
        exclusion.reason for exclusion in exc_info.value.exclusions
    }
    assert [message.model_dump(mode="json") for message in history] == before
