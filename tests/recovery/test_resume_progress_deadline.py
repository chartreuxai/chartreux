"""C5a/C5b — the resume progress deadline: classification and producer.

Covers the C5a acceptance surface (classification):

- The measured timeout reason vocabulary: ``resume_progress_deadline`` is a
  SPECIFIC reason riding the failure block's evidence codes — never a new
  ``FailureClass`` member (P11's wire mirror is untouched), so the failure
  class stays the existing timeout operational signature and projects onto
  the public wire unchanged.
- The classification: a deadline-marked observation classifies as a MEASURED
  timeout (``provider_stall`` class, MEASURED confidence, TIMEOUT cause
  category, DEADLINE exhaustion, harness provenance) — while an ordinary
  provider timeout keeps its SIGNATURE classification and carries no reason.
- The automatic-continuation prohibition: the recovery controller NEVER
  schedules the automatic same-payload continuation for an envelope carrying
  the reason — even when every other gate (and a cleared reconcile barrier)
  would pass — while an ordinary settled transient still arms its schedule.
- The failure-time replacement policy: an explicit replacement claim over a
  deadline failure stays available (only the AUTOMATIC continuation is
  prohibited) and the milestone notice names the measured reason.

C5b extends the file with the PRODUCER itself: the
``[recovery.subagents].resume_progress_deadline_seconds`` config field (the
explicit exception to the advisory-deadlines policy), the run-scoped
:class:`ResumeProgressWatch` (armed at the resumed run's first conversation
model call, disarmed at the OBSERVATION of the first semantic delta —
content/reasoning/tool-call deltas disarm; heartbeat/usage-only chunks do
not), the gateway's cancellation-to-typed-failure translation (cleanup joins
before the typed failure propagates), the resume admission's arm/clear
marking, and the end-to-end composition: a resumed successor run stalled on
its first model call is killed at the deadline and its envelope carries the
C5a measured reason.

All faults are synthetic classification fixtures and offline stub backends;
no unmocked provider call can occur (fail-closed egress).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing, suppress
import logging
from typing import Any, cast

import httpx
from pydantic import ValidationError
import pytest

from chartreux.app_server._recovery_controller import RecoveryController
from chartreux.app_server._sessions import _AgentState
from chartreux.app_server.models import PublicFailureClass, PublicFailureEnvelope
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
)
from chartreux.core.agent_loop.resume_progress import (
    ResumeProgressDeadlineError,
    ResumeProgressWatch,
    arm_resume_progress_deadline,
    clear_resume_progress_deadline,
    semantic_progress,
)
from chartreux.core.config.models import RecoveryConfig
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm.failures import FailureCategory, classify_failure
from chartreux.core.llm_models import FunctionCall, LLMChunk, LLMMessage, Role, ToolCall
from chartreux.core.recovery_models import (
    CallEvidence,
    ExhaustionReason,
    FailureClass,
    FailureConfidence,
    FailureEnvelope,
    FailurePhase,
    MeasuredTimeoutReason,
    PackageTier,
    RecoveryEligibility,
    attach_call_evidence,
    build_failure_envelope,
    measured_timeout_reason_of,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.recovery_policy import (
    ReplacementCandidate,
    validate_replacement_selection,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.subagents import RunStatus
from chartreux.core.usage import UsagePurpose
from tests.app_server._subagent_test_support import _ManualClock, _record, _run
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_explicit_resume import (
    _empty_fault,
    _Gated,
    _launch_child,
    _settle_child,
    _start_harness,
    _write_outcome,
)
from tests.stubs.fake_backend import FakeBackend

FAILED_ENDPOINT = "https://api.alpha.test/v1"
ALT_ENDPOINT = "https://api.beta.test/v1"


@pytest.fixture(autouse=True)
def _controller_events_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The controller's milestone lines are INFO on the ``vibe`` logger."""
    caplog.set_level(logging.INFO, logger="vibe")


class _ResumeProgressDeadlineFault(RuntimeError):
    """The C5b producer's deadline-failure shape (the C5a producer contract).

    The harness-raised deadline failure carries the measured timeout reason
    value as its ``error_code``; the WP0 observation carrier extracts
    ``error_code`` from the exception chain.
    """

    error_code = "resume_progress_deadline"
    endpoint = FAILED_ENDPOINT


class _OtherCodeFault(RuntimeError):
    """An ordinary fault carrying an unrelated structured code."""

    error_code = "provider_meltdown"
    endpoint = FAILED_ENDPOINT


class _EndpointTimeout(httpx.ReadTimeout):
    """An ordinary provider read timeout on the same endpoint."""

    endpoint = FAILED_ENDPOINT


class _EmptyResponseFault(EmptyLLMResponseError):
    """An ordinary typed empty-response transient on the same endpoint."""

    endpoint = FAILED_ENDPOINT


def _deadline_observation() -> Any:
    """A mid-call deadline kill: the marked fault with measured call evidence."""
    fault = _ResumeProgressDeadlineFault(
        "no semantic progress within the resume progress deadline"
    )
    attach_call_evidence(
        fault,
        CallEvidence(
            logical_call_id="call-1",
            started_at=0.0,
            transport_attempts=1,
            elapsed_seconds=180.0,
        ),
    )
    return observe_exception(fault, phase="turn")


def _deadline_envelope(**kwargs: Any) -> FailureEnvelope:
    return build_failure_envelope(
        observation=_deadline_observation(),
        agent_id="agent-1",
        run_id="run-1",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Classification — a SPECIFIC measured timeout reason, not a failure class
# ---------------------------------------------------------------------------


def test_resume_progress_deadline_classifies_as_measured_timeout() -> None:
    """The deadline reason is a MEASURED timeout inside the existing class.

    The class stays ``provider_stall`` (the timeout operational signature —
    never a new ``FailureClass`` member), the confidence is MEASURED (the
    harness measured the deadline directly, not a signature inference), the
    cause category is the measured TIMEOUT, and the exhaustion is DEADLINE.
    """
    envelope = _deadline_envelope()
    failure = envelope.failure
    assert failure.failure_class is FailureClass.PROVIDER_STALL
    assert failure.confidence is FailureConfidence.MEASURED
    assert failure.cause_category is FailureCategory.TIMEOUT
    assert failure.phase is FailurePhase.LLM_CALL
    assert "resume_progress_deadline" in failure.evidence_codes
    assert "exhaustion_deadline" in failure.evidence_codes
    assert envelope.request.exhaustion is ExhaustionReason.DEADLINE
    assert (
        "measured harness deadline resume_progress_deadline"
        in failure.classification_provenance
    )


def test_resume_progress_deadline_is_a_reason_never_a_failure_class() -> None:
    """The reason vocabulary never grows the failure-class surface.

    P11's wire mirror is pinned by set equality against ``FailureClass``:
    the reason must ride the evidence codes (which mirror losslessly), so
    the public projection carries the specific reason with NO wire change.
    """
    assert MeasuredTimeoutReason.RESUME_PROGRESS_DEADLINE.value == (
        "resume_progress_deadline"
    )
    assert "resume_progress_deadline" not in {cls.value for cls in FailureClass}
    envelope = _deadline_envelope()
    assert (
        measured_timeout_reason_of(envelope)
        is MeasuredTimeoutReason.RESUME_PROGRESS_DEADLINE
    )
    public = PublicFailureEnvelope.model_validate(
        envelope.model_dump(mode="json", by_alias=True)
    )
    assert public.failure.failure_class is PublicFailureClass.PROVIDER_STALL
    assert "resume_progress_deadline" in public.failure.evidence_codes


def test_ordinary_provider_timeout_keeps_signature_and_carries_no_reason() -> None:
    """Recognition does not over-fire: a plain provider timeout is unchanged.

    The generic ladder's SIGNATURE stall classification and its evidence are
    untouched, and no measured timeout reason is derived from it.
    """
    fault = _EndpointTimeout("read timed out")
    attach_call_evidence(
        fault,
        CallEvidence(logical_call_id="call-2", started_at=0.0, transport_attempts=1),
    )
    envelope = build_failure_envelope(
        observation=observe_exception(fault, phase="turn"),
        agent_id="agent-1",
        run_id="run-2",
    )
    assert envelope.failure.failure_class is FailureClass.PROVIDER_STALL
    assert envelope.failure.confidence is FailureConfidence.SIGNATURE
    assert measured_timeout_reason_of(envelope) is None


def test_unrelated_error_code_stays_on_the_generic_ladder() -> None:
    """Only the measured-timeout vocabulary is recognized — never a guess."""
    envelope = build_failure_envelope(
        observation=observe_exception(_OtherCodeFault("boom"), phase="turn"),
        agent_id="agent-1",
        run_id="run-3",
    )
    assert measured_timeout_reason_of(envelope) is None
    assert "resume_progress_deadline" not in envelope.failure.evidence_codes


def test_deadline_kill_with_cancellation_cause_classifies_cancellation_in_loop() -> (
    None
):
    """The gateway translates the watch's cancellation into the typed
    deadline failure raised ``from`` the CancelledError; the in-loop
    classifier reads the cause chain's protected CANCELLATION member, so a
    deadline kill is never retried and never failed over — and never
    recorded as a measured endpoint failure by the C1 episode wiring.
    """
    cancel = asyncio.CancelledError()
    error = ResumeProgressDeadlineError(180.0)
    error.__cause__ = cancel
    info = classify_failure(error)
    assert info.category is FailureCategory.CANCELLATION
    assert not info.retry_eligible
    assert not info.failover_eligible


def test_deadline_kill_envelope_is_measured_timeout_with_reason() -> None:
    """The WP0 envelope builder classifies the deadline kill as a MEASURED
    timeout carrying the SPECIFIC resume-progress-deadline reason — the
    automatic-continuation prohibition keys off that reason.
    """
    cancel = asyncio.CancelledError()
    error = ResumeProgressDeadlineError(180.0)
    error.__cause__ = cancel
    envelope = build_failure_envelope(
        agent_id="agent-1", run_id="run-1", observation=error
    )
    assert envelope.failure.failure_class is FailureClass.PROVIDER_STALL
    assert envelope.failure.confidence is FailureConfidence.MEASURED
    assert (
        measured_timeout_reason_of(envelope)
        is MeasuredTimeoutReason.RESUME_PROGRESS_DEADLINE
    )
    assert "resume_progress_deadline" in envelope.failure.evidence_codes


def test_unwrapped_deadline_error_classifies_unknown_in_loop() -> None:
    """Without the CancelledError cause (e.g. the between-calls enter_call
    raise), the in-loop classifier reads the bare typed error: no status,
    no declared category, no httpx shape — UNKNOWN (report-and-park: still
    never retried, never failed over, never a measured endpoint failure).
    """
    info = classify_failure(ResumeProgressDeadlineError(180.0))
    assert info.category is FailureCategory.UNKNOWN
    assert not info.retry_eligible
    assert not info.failover_eligible


# ---------------------------------------------------------------------------
# The automatic-continuation prohibition (the controller's gate)
# ---------------------------------------------------------------------------


class _StubView:
    """A minimal controller state-view double for the gate tests.

    ``_root`` is None, so the controller reads the default
    ``[recovery.subagents]`` policy (enabled, every eligible class);
    ``_wakeup`` blocks forever so an armed schedule stays pending and
    deterministic; ``resume_agent`` fails the test if ever reached.
    """

    def __init__(self, records: dict[str, Any]) -> None:
        self._agent_records = records
        self._registry_lock = asyncio.Lock()
        self._clock = _ManualClock(1000.0)
        self._wakeup_calls: list[float] = []
        self._root = None
        self._draining_children = False
        self._admission_closed = False
        self.resume_calls: list[tuple[str, str, str]] = []

    def _generation_is_current(self, generation: int) -> bool:
        return True

    async def _wakeup(self, delay: float) -> None:
        self._wakeup_calls.append(delay)
        await asyncio.Event().wait()

    async def resume_agent(
        self,
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> Any:
        self.resume_calls.append((agent_id, expected_run_id, idempotency_key))
        raise AssertionError("the automatic prohibition must never admit")


def _suspended_record(view: _StubView, agent_id: str) -> Any:
    clock = _ManualClock(900.0)
    record = _record(agent_id, clock=clock)
    # The read-only availability snapshot short-circuits on a missing
    # catalog snapshot: the stub view has no availability registry.
    cast(Any, record.runtime.agent_loop.config).catalog_snapshot = None
    record.state = _AgentState.SUSPENDED
    record.recovery_chain_id = "recovery-chain:agent-1:run-1"
    view._agent_records[agent_id] = record
    return record


def _failed_run(record: Any, envelope: FailureEnvelope) -> Any:
    run = _run(record, "run-1", started_at=900.0, status=RunStatus.FAILED)
    run.failure = envelope
    record.latest_run_id = "run-1"
    record.current_run = run
    return run


@pytest.mark.asyncio
async def test_controller_never_schedules_the_automatic_same_payload_continuation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A deadline envelope stops the episode with the specific reason.

    The gate fires BEFORE the eligibility ladder: no schedule is armed and
    no admission is attempted. A fresh undriven episode stays silent
    (exactly Gate 1's semantics); a DRIVEN episode's stop is announced by
    the exhaustion milestone naming the specific prohibition (the next test).
    """
    view = _StubView({})
    controller = RecoveryController(view)  # type: ignore[arg-type]
    record = _suspended_record(view, "agent-1")
    run = _failed_run(record, _deadline_envelope())
    controller.observe_run_locked(record, run)
    assert controller._episodes == {}
    assert view.resume_calls == []
    assert not any("event=exhaustion" in message for message in caplog.messages)


@pytest.mark.asyncio
async def test_driven_episode_announces_the_deadline_prohibition(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A driven episode's deadline stop emits the specific exhaustion reason.

    An ordinary settled transient arms its schedule (the episode is driven);
    the successor's failure carries the measured deadline reason, and the
    stop milestone names the specific prohibition — not a generic class
    refusal.
    """
    ordinary = _EmptyResponseFault(model="wire-model", provider="prov")
    attach_call_evidence(
        ordinary,
        CallEvidence(logical_call_id="call-3", started_at=0.0, transport_attempts=1),
    )
    ordinary_envelope = build_failure_envelope(
        observation=observe_exception(ordinary, phase="turn"),
        agent_id="agent-1",
        run_id="run-1",
        tool_calls_present=True,
        tool_calls_pending=False,
    )

    view = _StubView({})
    controller = RecoveryController(view)  # type: ignore[arg-type]
    record = _suspended_record(view, "agent-1")
    run = _failed_run(record, ordinary_envelope)
    controller.observe_run_locked(record, run)
    episode = controller._episodes.get("agent-1")
    assert episode is not None and episode.timer is not None

    run.failure = _deadline_envelope()
    controller.observe_run_locked(record, run)
    assert controller._episodes == {}
    assert view.resume_calls == []
    assert any(
        "event=exhaustion" in message
        and "resume_progress_deadline_no_auto_recovery" in message
        for message in caplog.messages
    )


@pytest.mark.asyncio
async def test_controller_prohibition_holds_even_with_a_cleared_reconcile_barrier() -> (
    None
):
    """The prohibition is absolute for the automatic layer.

    A cleared reconciliation makes the envelope's EFFECTIVE eligibility
    ELIGIBLE; the measured timeout reason still stops the episode — the
    automatic same-payload continuation never re-arms over a no-progress
    deadline.
    """
    view = _StubView({})
    controller = RecoveryController(view)  # type: ignore[arg-type]
    record = _suspended_record(view, "agent-1")
    record.reconcile_outcome = "cleared"
    run = _failed_run(record, _deadline_envelope())
    controller.observe_run_locked(record, run)
    assert controller._episodes == {}
    assert view.resume_calls == []


@pytest.mark.asyncio
async def test_controller_still_arms_ordinary_settled_transients() -> None:
    """The prohibition is specific: an ordinary settled transient schedules.

    A typed EMPTY_RESPONSE transient with measured call evidence, settled
    tool effects, and a suspended record passes every gate and arms the
    cancellable backoff timer — proving the deadline gate did not become a
    blanket suspension of automatic recovery.
    """
    fault = _EmptyResponseFault(model="wire-model", provider="prov")
    attach_call_evidence(
        fault,
        CallEvidence(logical_call_id="call-4", started_at=0.0, transport_attempts=1),
    )
    envelope = build_failure_envelope(
        observation=observe_exception(fault, phase="turn"),
        agent_id="agent-1",
        run_id="run-1",
        tool_calls_present=True,
        tool_calls_pending=False,
    )
    assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE

    view = _StubView({})
    controller = RecoveryController(view)  # type: ignore[arg-type]
    record = _suspended_record(view, "agent-1")
    run = _failed_run(record, envelope)
    controller.observe_run_locked(record, run)
    episode = controller._episodes.get("agent-1")
    assert episode is not None
    assert episode.timer is not None
    episode.timer.cancel()
    assert view.resume_calls == []


# ---------------------------------------------------------------------------
# The failure-time replacement policy — claims stay available, honestly named
# ---------------------------------------------------------------------------


def _candidate(
    base_model: str = "alt-model", endpoint: str = ALT_ENDPOINT
) -> ReplacementCandidate:
    return ReplacementCandidate(
        base_model=base_model,
        provider="beta",
        endpoint=endpoint,
        output_price_per_million=1.0,
    )


def test_replacement_claim_over_deadline_failure_names_the_measured_reason() -> None:
    """An explicit replacement claim stays available and names the reason.

    Only the AUTOMATIC same-payload continuation is prohibited (the
    controller's gate); the orchestrator's failure-time escalation remains
    a valid route, and its milestone notice records that the claim crossed
    a measured no-progress deadline.
    """
    decision = validate_replacement_selection(
        envelope=_deadline_envelope(),
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_candidate(),
        candidates=[_candidate()],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "selected"
    assert "measured reason resume_progress_deadline" in decision.notice


def test_ordinary_stall_notice_carries_no_measured_reason() -> None:
    """The notice augmentation is specific to the measured-timeout reasons."""
    fault = _EndpointTimeout("read timed out")
    attach_call_evidence(
        fault,
        CallEvidence(logical_call_id="call-5", started_at=0.0, transport_attempts=1),
    )
    envelope = build_failure_envelope(
        observation=observe_exception(fault, phase="turn"),
        agent_id="agent-1",
        run_id="run-2",
        tool_calls_present=True,
        tool_calls_pending=False,
    )
    decision = validate_replacement_selection(
        envelope=envelope,
        package_tier=PackageTier.SUBSTANTIVE,
        claimed=_candidate(),
        candidates=[_candidate()],
        config=RecoveryConfig(),
    )
    assert decision.disposition == "selected"
    assert "measured reason" not in decision.notice


# ---------------------------------------------------------------------------
# C5b — the config field: the explicit exception to the advisory deadlines
# ---------------------------------------------------------------------------


def test_resume_progress_deadline_config_defaults_and_rejects_bad_shapes() -> None:
    """180s default; booleans and nonpositive/infinitary values are rejected."""
    config = RecoveryConfig()
    assert config.resume_progress_deadline_seconds == 180.0
    with pytest.raises(ValidationError, match="not booleans"):
        RecoveryConfig(resume_progress_deadline_seconds=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        RecoveryConfig(resume_progress_deadline_seconds=0.0)
    with pytest.raises(ValidationError):
        RecoveryConfig(resume_progress_deadline_seconds=float("inf"))


# ---------------------------------------------------------------------------
# C5b — the run-scoped watch: arm, disarm-at-observation, expiry, one-shot
# ---------------------------------------------------------------------------


def _tool_call_delta(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id, index=0, function=FunctionCall(name="todo", arguments="{}")
    )


@pytest.mark.asyncio
async def test_semantic_deltas_disarm_and_heartbeats_do_not() -> None:
    """The predicate: content/reasoning/tool-call disarm; empty chunks never."""
    assert not semantic_progress(LLMMessage(role=Role.assistant))
    assert semantic_progress(LLMMessage(role=Role.assistant, content="delta"))
    assert semantic_progress(
        LLMMessage(role=Role.assistant, reasoning_content="thinking")
    )
    assert semantic_progress(
        LLMMessage(role=Role.assistant, tool_calls=[_tool_call_delta("call-1")])
    )

    watch = ResumeProgressWatch(0.05)
    watch.enter_call()
    # Heartbeat/usage-only chunks do not disarm.
    watch.observe(LLMMessage(role=Role.assistant, content=""))
    assert not watch.disarmed
    # The OBSERVATION of the first semantic delta disarms terminally.
    watch.observe(LLMMessage(role=Role.assistant, content="delta"))
    assert watch.disarmed
    await asyncio.sleep(0.08)
    assert not watch.expired
    # Terminal: further observation and entries are harmless no-ops.
    watch.observe(LLMMessage(role=Role.assistant, content="more"))
    watch.enter_call()


@pytest.mark.asyncio
async def test_expiry_kills_the_registered_inflight_call() -> None:
    """Expiry cancels the in-flight call task; the typed failure is C5a's."""
    watch = ResumeProgressWatch(0.02)
    killed = asyncio.Event()

    async def stalled_call() -> None:
        watch.enter_call()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            killed.set()
            raise

    task = asyncio.create_task(stalled_call())
    await asyncio.wait_for(killed.wait(), 5.0)
    assert watch.expired
    assert not watch.disarmed
    with suppress(asyncio.CancelledError):
        await task
    # The producer contract: the typed failure carries the C5a reason.
    failure = watch.deadline_failure()
    assert failure.error_code == "resume_progress_deadline"
    assert failure.non_retryable is True


@pytest.mark.asyncio
async def test_expiry_between_calls_fails_the_next_call() -> None:
    """An expiry with no in-flight task arms the next-call kill (fail closed)."""
    watch = ResumeProgressWatch(0.01)
    watch.enter_call()
    watch.exit_call()
    await asyncio.sleep(0.03)
    assert watch.expired
    with pytest.raises(ResumeProgressDeadlineError):
        watch.enter_call()


# ---------------------------------------------------------------------------
# C5b — the gateway translation: the kill surfaces as the typed failure
# ---------------------------------------------------------------------------


class _HangingTransportBackend(FakeBackend):
    """Notifies the real transport start, then stalls with no bytes at all."""

    def __init__(self) -> None:
        super().__init__()
        # The attribute name the measured-evidence counter reads.
        self.transport_attempts = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.transport_attempts += 1
        notify_request_started()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        self.transport_attempts += 1
        notify_request_started()
        await asyncio.Event().wait()
        yield mock_llm_chunk(content="")  # pragma: no cover - unreachable


class _HeartbeatThenHangBackend(FakeBackend):
    """One usage-only (non-semantic) chunk, then a stall with no bytes."""

    def __init__(self) -> None:
        super().__init__()
        self.transport_attempts = 0

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        self.transport_attempts += 1
        notify_request_started()
        yield mock_llm_chunk(content="", prompt_tokens=1, completion_tokens=0)
        await asyncio.Event().wait()
        yield mock_llm_chunk(content="never")  # pragma: no cover - unreachable


class _DeltaThenStallBackend(FakeBackend):
    """One semantic delta, then a stall — the long-first-tool-call shape."""

    def __init__(self) -> None:
        super().__init__()
        self.gate = asyncio.Event()

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        yield mock_llm_chunk(content="progress")
        await self.gate.wait()
        yield mock_llm_chunk(content="done")  # pragma: no cover - unreachable


def _gateway_model() -> Any:
    from chartreux.core.config.models import ModelConfig

    return ModelConfig(
        name="test", provider="test", alias="test", input_price=1, output_price=2
    )


def _gateway_inputs(**updates: Any) -> CompletionInputs:
    values: dict[str, Any] = dict(
        model=_gateway_model(),
        provider_name="test",
        emits_finish_reason=True,
        messages=(LLMMessage(role=Role.user, content="hello"),),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=42,
    )
    values.update(updates)
    return CompletionInputs(**values)


def _gateway_resources(
    backend: FakeBackend, watch: ResumeProgressWatch | None
) -> CallResources:
    return CallResources(
        backend=backend,
        stats=AgentStats(),
        process_message=lambda message: message,
        resume_progress=watch,
    )


async def _drain_stream(stream: AsyncGenerator[LLMChunk]) -> list[LLMChunk]:
    chunks: list[LLMChunk] = []
    async with aclosing(stream) as closed:
        async for chunk in closed:
            chunks.append(chunk)
    return chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_deadline_kill_surfaces_as_the_typed_c5a_contract_failure(
    streaming: bool,
) -> None:
    """A stalled first call is killed and the failure carries the reason.

    The measured evidence rides the typed error (one real transport start),
    and the observation composed from it classifies exactly as C5a's
    synthetic fixtures do — the producer contract holds end to end.
    """
    watch = ResumeProgressWatch(0.05)
    backend = _HangingTransportBackend()
    gateway = LLMGateway()
    transcript: list[Any] = []

    with pytest.raises(ResumeProgressDeadlineError) as exc_info:
        if streaming:
            await _drain_stream(
                gateway.chat_streaming(
                    _gateway_inputs(),
                    _gateway_resources(backend, watch),
                    transcript=transcript.append,
                )
            )
        else:
            await gateway.chat(
                _gateway_inputs(),
                _gateway_resources(backend, watch),
                transcript=transcript.append,
            )
    error = exc_info.value
    assert error.error_code == "resume_progress_deadline"
    assert "no semantic progress" in str(error)
    # Measured evidence: exactly one transport start, elapsed measured.
    evidence = getattr(error, "call_evidence", None)
    assert evidence is not None
    assert evidence.transport_attempts == 1
    assert evidence.elapsed_seconds is not None
    # The C5a classification composes with the REAL producer's failure.
    envelope = build_failure_envelope(
        observation=observe_exception(error, phase="turn"),
        agent_id="agent-1",
        run_id="run-9",
    )
    assert (
        measured_timeout_reason_of(envelope)
        is MeasuredTimeoutReason.RESUME_PROGRESS_DEADLINE
    )


@pytest.mark.asyncio
async def test_heartbeat_chunks_do_not_disarm_the_streaming_deadline() -> None:
    """A usage-only chunk never disarms: the stall after it still gets killed."""
    watch = ResumeProgressWatch(0.05)
    backend = _HeartbeatThenHangBackend()
    gateway = LLMGateway()

    with pytest.raises(ResumeProgressDeadlineError) as exc_info:
        await _drain_stream(
            gateway.chat_streaming(
                _gateway_inputs(),
                _gateway_resources(backend, watch),
                transcript=lambda outcome: None,
            )
        )
    assert exc_info.value.error_code == "resume_progress_deadline"
    assert watch.expired and not watch.disarmed


@pytest.mark.asyncio
async def test_observed_semantic_delta_disarms_before_a_long_stall() -> None:
    """The disarm boundary is the delta's OBSERVATION, at emission.

    After the semantic delta is observed the deadline is spent: the run's
    long stall afterward (the executing first tool call) is not killed, and
    an ordinary user cancellation stays an ordinary cancellation — never
    translated into the typed deadline failure.
    """
    watch = ResumeProgressWatch(0.05)
    backend = _DeltaThenStallBackend()
    gateway = LLMGateway()

    task = asyncio.create_task(
        _drain_stream(
            gateway.chat_streaming(
                _gateway_inputs(),
                _gateway_resources(backend, watch),
                transcript=lambda outcome: None,
            )
        )
    )
    for _ in range(1000):
        if watch.disarmed:
            break
        await asyncio.sleep(0.005)
    assert watch.disarmed and not watch.expired
    # The first tool call may now execute far longer than the deadline.
    await asyncio.sleep(0.08)
    assert not watch.expired
    # A user cancellation of the disarmed call is NOT a deadline failure.
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not watch.expired


@pytest.mark.asyncio
async def test_sideband_calls_neither_arm_nor_disarm_the_deadline() -> None:
    """Only conversation calls interact with the watch (C5b scope)."""
    watch = ResumeProgressWatch(30.0)
    backend = FakeBackend([mock_llm_chunk(content="summary")])
    gateway = LLMGateway()

    result = await gateway.chat(
        _gateway_inputs(purpose=UsagePurpose.TITLE),
        _gateway_resources(backend, watch),
        transcript=lambda outcome: None,
    )
    assert result.message.content == "summary"
    # The sideband call never armed the watch, and its content never
    # disarmed it: the deadline is still live for the run's first
    # conversation call.
    assert not watch.disarmed
    assert not watch.expired


# ---------------------------------------------------------------------------
# C5b — the marking: only an explicitly resumed context carries a watch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_arming_binds_the_watch_into_every_call_resource() -> None:
    """The coordinator slot feeds CallResources; clearing removes it."""
    loop = build_test_agent_loop()
    coordinator = loop._completion_coordinator
    # An ordinary (unmarked) loop binds None — ordinary runs are unaffected.
    assert coordinator._resume_progress_watch is None
    assert (
        coordinator.build_call_resources(
            FakeBackend([mock_llm_chunk()])
        ).resume_progress
        is None
    )

    arm_resume_progress_deadline(loop)
    watch = coordinator._resume_progress_watch
    assert watch is not None
    assert watch._deadline_seconds == (
        loop.config.recovery.subagents.resume_progress_deadline_seconds
    )
    assert (
        coordinator.build_call_resources(
            FakeBackend([mock_llm_chunk()])
        ).resume_progress
        is watch
    )

    clear_resume_progress_deadline(loop)
    assert coordinator._resume_progress_watch is None
    assert (
        coordinator.build_call_resources(
            FakeBackend([mock_llm_chunk()])
        ).resume_progress
        is None
    )


# ---------------------------------------------------------------------------
# C5b — end to end: the resumed successor run is killed at the deadline
# ---------------------------------------------------------------------------


@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_resumed_successor_stalled_on_first_call_is_killed_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full producer path: resume admission -> stall -> typed deadline.

    The first (ordinary, unmarked) run completes its tool call and fails
    with the empty-response transient — never deadline-killed. The explicit
    resume arms the watch on the successor run; its first model call stalls
    with no semantic delta and the harness kills it at the deadline; the
    published envelope carries the C5a measured reason (which the automatic
    continuation prohibition keys off).
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("c5b-1", "deadline-item"),
                _empty_fault(),
                _Gated([], asyncio.Event()),
            ]
        ],
        launches=["launch the c5b child"],
        recovery=RecoveryConfig(resume_progress_deadline_seconds=0.2),
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the c5b child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert failed.run_status is RunStatus.FAILED
        assert failed.failure is not None
        # The ordinary (unmarked) run was never deadline-killed.
        assert measured_timeout_reason_of(failed.failure) is None

        record = registry._agent_records[agent_id]
        await _wait_suspended(record)

        # The explicit resume marks the successor run as a resumed context.
        key = f"{agent_id}:{run_id}"
        scheduled = await registry.resume_agent(
            agent_id, run_id, key, requester_session_id=harness.requester_session_id
        )
        assert scheduled.outcome.value == "scheduled"
        successor_run_id = scheduled.successor_run_id
        assert successor_run_id is not None

        successor = await _settle_child(harness, agent_id, successor_run_id)
        assert not successor.completed
        assert successor.run_status is RunStatus.FAILED
        envelope = successor.failure
        assert envelope is not None
        assert (
            measured_timeout_reason_of(envelope)
            is MeasuredTimeoutReason.RESUME_PROGRESS_DEADLINE
        )
        assert "resume_progress_deadline" in envelope.failure.evidence_codes
        assert envelope.failure.confidence is FailureConfidence.MEASURED
    finally:
        await harness.session.close()


async def _wait_suspended(record: Any) -> None:
    for _ in range(2000):
        if record.state is _AgentState.SUSPENDED:
            return
        await asyncio.sleep(0.005)
    raise AssertionError("record never suspended")
