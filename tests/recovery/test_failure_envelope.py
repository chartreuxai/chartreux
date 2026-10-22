"""WP1a — the public typed failure envelope (handoff §1.7, schema v1).

Covers the classifier defects D1-D3 (with their before/after behavior), the
typed transient classifications (incomplete_stream / empty_response), the
full envelope schema, the deduplicatable exhausted-logical-call identity,
the admission-evidence requirement for launch_failure, null-vs-zero
semantics, bounded/redacted public messages, eligibility derivation, the
gateway's measured CallEvidence, and the background-monitor integration:
a failed run's stored TaskResult carries the envelope through the
wait/get/summary round trips, and legacy results stay immutable.

Honest labeling: every fault here is a SYNTHETIC fixture on a controlled
fake backend — none of it is a live-provider reproduction. All transports
are fakes; no unmocked provider call can occur (fail-closed egress).
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    IncompleteLLMResponseError,
    LLMGateway,
)
from chartreux.core.config import ModelConfig
from chartreux.core.errors import ContextTooLongError, RefusalError
from chartreux.core.llm.exceptions import (
    BackendError,
    BackendErrorBuilder,
    IncompleteStreamError,
    PayloadSummary,
)
from chartreux.core.llm.failures import FailureCategory, FailureInfo, classify_failure
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role, StopInfo
from chartreux.core.model_catalog.availability import AllDeploymentsUnavailableError
from chartreux.core.recovery_models import (
    CLASSIFICATION_VERSION,
    CallEvidence,
    ExhaustionReason,
    FailureClass,
    FailureConfidence,
    FailurePhase,
    ProgressState,
    RecoveryEligibility,
    RecoveryReference,
    ToolEffectsDisposition,
    attach_call_evidence,
    backend_transport_attempts,
    build_failure_envelope,
    canonical_endpoint_identity,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.session_types import AgentStats
from chartreux.core.subagents import RetentionPolicy, RunStatus, TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermissionError
from chartreux.core.tools.secret_redaction import ScrubPolicy, bind_policy
from tests.app_server._subagent_test_support import (
    ImmediateFailureBackend,
    _background_result,
    _wait_result,
    _zero_retention_registry,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

# ---------------------------------------------------------------------------
# D1 — status=None must no longer auto-become CONNECTION
# ---------------------------------------------------------------------------


class _UnmappedStreamFault(RuntimeError):
    """An unmapped stream error: exposes ``status=None``, no transport cause."""

    status = None


class _PermanentLocalFault(RuntimeError):
    """A permanent local fault (TLS/certificate, proxy, DNS-config)."""

    status = None


def test_d1_status_none_is_no_longer_relabeled_connection() -> None:
    """D1: only a real transport error is a connection failure.

    Before: any exception exposing ``status=None`` classified CONNECTION and
    was retried — unmapped stream errors and permanent local faults included.
    After: both stay UNKNOWN (report-and-park, no recovery authority), while
    a genuine httpx transport error remains CONNECTION.
    """
    for fault in (_UnmappedStreamFault("truncated frame"), _PermanentLocalFault("tls")):
        info = classify_failure(fault)
        assert info.category is FailureCategory.UNKNOWN
        assert info.retry_eligible is False
        assert info.failover_eligible is False
    assert (
        classify_failure(httpx.ConnectError("connection refused")).category
        is FailureCategory.CONNECTION
    )


# ---------------------------------------------------------------------------
# D2 — a wrapped READ TIMEOUT keeps its timeout classification
# ---------------------------------------------------------------------------


def _payload_summary() -> PayloadSummary:
    return PayloadSummary(
        model="wire",
        message_count=1,
        approx_chars=2,
        temperature=0.0,
        has_tools=False,
        tool_choice=None,
    )


def _backend_error(**overrides: Any) -> BackendError:
    kwargs: dict[str, Any] = {
        "provider": "prov",
        "endpoint": "https://api.example.com/v1",
        "status": None,
        "reason": "request failed",
        "headers": None,
        "body_text": None,
        "parsed_error": None,
        "model": "wire",
        "payload_summary": _payload_summary(),
    }
    kwargs.update(overrides)
    return BackendError(**kwargs)


def test_d2_wrapped_read_timeout_classifies_as_timeout() -> None:
    """D2: the cached construction-time classification no longer wins.

    Before: BackendError cached ``failure_info`` at construction with
    ``status=None`` BEFORE the caller's ``raise ... from`` attached the HTTPX
    cause, so a wrapped READ TIMEOUT was recorded as a plain CONNECTION and
    lost its timeout subtype and retry semantics. After: the lazy property
    resolves through the cause chain — the wrapped timeout classifies TIMEOUT.
    """
    cause = httpx.ReadTimeout("read timed out")
    try:
        raise _backend_error(reason="timeout", parsed_error="Network error") from cause
    except BackendError as error:
        info = classify_failure(error)
        assert info.category is FailureCategory.TIMEOUT
        assert info.retry_eligible is True
        assert info.failover_eligible is True


def test_d2_build_request_error_preserves_transport_subtype() -> None:
    """D2: the builder classifies the HTTPX cause at construction."""
    error = BackendErrorBuilder.build_request_error(
        provider="prov",
        endpoint="https://api.example.com/v1",
        error=httpx.ReadTimeout("read timed out"),
        model="wire",
        messages=(LLMMessage(role=Role.user, content="hi"),),
        temperature=0.0,
        has_tools=False,
        tool_choice=None,
    )
    assert error.failure_info.category is FailureCategory.TIMEOUT
    connect = BackendErrorBuilder.build_request_error(
        provider="prov",
        endpoint="https://api.example.com/v1",
        error=httpx.ConnectError("connection refused"),
        model="wire",
        messages=(LLMMessage(role=Role.user, content="hi"),),
        temperature=0.0,
        has_tools=False,
        tool_choice=None,
    )
    assert connect.failure_info.category is FailureCategory.CONNECTION


# ---------------------------------------------------------------------------
# D3 — cycle-safe, group-reducing cause traversal
# ---------------------------------------------------------------------------


def test_d3_cyclic_cause_does_not_crash_classification() -> None:
    """D3: a cyclic cause graph terminates and is not relabeled."""
    error = RuntimeError("cycle")
    error.__cause__ = error
    assert classify_failure(error).category is FailureCategory.UNKNOWN


def test_d3_grouped_failure_does_not_mask_protected_member() -> None:
    """D3: a grouped cancellation is never masked by a transient sibling."""
    group = BaseExceptionGroup(
        "mixed", [httpx.ReadTimeout("slow"), asyncio.CancelledError()]
    )
    assert classify_failure(group).category is FailureCategory.CANCELLATION
    inner = BaseExceptionGroup("inner", [asyncio.CancelledError()])
    outer = BaseExceptionGroup("outer", [httpx.ReadTimeout("slow"), inner])
    assert classify_failure(outer).category is FailureCategory.CANCELLATION


# ---------------------------------------------------------------------------
# Review R1 — the bounded traversal preserves protected members and
# uncertainty (truncation is never a safe classification)
# ---------------------------------------------------------------------------


class _ProtectedStatusFault(Exception):
    """A fault exposing a protected HTTP status (authorization), no code."""

    status = 401


def _timeout_faults(count: int) -> list[Exception]:
    return [httpx.ReadTimeout(f"timeout-{index}") for index in range(count)]


def test_r1_truncated_group_walk_reports_uncertainty() -> None:
    """R1: the depth-bounded walk must not let a transient sibling stand in
    for a protected member hiding beyond the bound.

    Before: a group of seven distinct timeouts followed by an authorization
    failure classified TIMEOUT — the protected member was never visited, so
    the transient sibling carried retry/failover authority it should not
    have. After: the truncation is reported and the classification degrades
    to UNKNOWN (report-and-park), never a retryable transient.
    """
    group = BaseExceptionGroup(
        "mixed", [*_timeout_faults(7), _ProtectedStatusFault("denied")]
    )
    info = classify_failure(group)
    assert info.category is FailureCategory.UNKNOWN
    assert info.retry_eligible is False
    assert info.failover_eligible is False


def test_r1_truncated_all_transient_group_reports_uncertainty() -> None:
    """R1: even with no protected member, a truncated walk is uncertainty —
    the bound cannot prove the unvisited members are transient too.
    """
    group = BaseExceptionGroup("many", _timeout_faults(8))
    info = classify_failure(group)
    assert info.category is FailureCategory.UNKNOWN
    assert info.retry_eligible is False
    assert info.failover_eligible is False


def test_r1_truncated_cause_chain_reports_uncertainty() -> None:
    """R1 (chained form): the same invariant through a ``__cause__`` chain —
    eight visited timeouts with the authorization failure beyond the bound.
    """
    current: BaseException = _ProtectedStatusFault("denied")
    for index in reversed(range(8)):
        timeout = httpx.ReadTimeout(f"timeout-{index}")
        timeout.__cause__ = current
        current = timeout
    info = classify_failure(current)
    assert info.category is FailureCategory.UNKNOWN
    assert info.retry_eligible is False
    assert info.failover_eligible is False


def test_r1_protected_member_within_the_bound_still_dominates() -> None:
    """R1 control: a protected member the walk DOES reach still dominates."""
    group = BaseExceptionGroup(
        "mixed", [httpx.ReadTimeout("slow"), _ProtectedStatusFault("denied")]
    )
    assert classify_failure(group).category is FailureCategory.AUTH
    chained = httpx.ReadTimeout("slow")
    chained.__cause__ = _ProtectedStatusFault("denied")
    assert classify_failure(chained).category is FailureCategory.AUTH


def test_r1_local_hard_stops_are_protected_not_unknown() -> None:
    """R1: local ToolPermissionError / ContextTooLongError carry no provider
    status or structured code; unrecognized they read UNKNOWN and a
    transient sibling in a group masks them. Recognized, they are protected
    hard stops: never masked, never retried, never failed over.
    """
    permission = classify_failure(ToolPermissionError("Task tool is disabled"))
    assert permission.category is FailureCategory.AUTH
    assert permission.retry_eligible is False
    assert permission.failover_eligible is False

    context = classify_failure(ContextTooLongError("prov", "wire"))
    assert context.category is FailureCategory.CONTEXT_SIZE
    assert context.retry_eligible is False
    assert context.failover_eligible is False

    permission_group = BaseExceptionGroup(
        "mixed", [httpx.ReadTimeout("slow"), ToolPermissionError("denied")]
    )
    assert classify_failure(permission_group).category is FailureCategory.AUTH
    context_group = BaseExceptionGroup(
        "mixed", [httpx.ReadTimeout("slow"), ContextTooLongError("prov", "wire")]
    )
    assert classify_failure(context_group).category is FailureCategory.CONTEXT_SIZE


def test_r1_model_refusal_dominates_transient_sibling() -> None:
    """R1: a model refusal is a measured content stop. It stays UNKNOWN
    (never retried, never failed over — report-and-park) and dominates group
    reduction so a transient sibling can never mask it.
    """
    refusal = RefusalError("prov", "wire", "safety", "declined")
    info = classify_failure(refusal)
    assert info.category is FailureCategory.UNKNOWN
    assert info.retry_eligible is False
    assert info.failover_eligible is False

    group = BaseExceptionGroup("mixed", [httpx.ReadTimeout("slow"), refusal])
    grouped = classify_failure(group)
    assert grouped.category is FailureCategory.UNKNOWN
    assert grouped.retry_eligible is False
    assert grouped.failover_eligible is False


# ---------------------------------------------------------------------------
# Typed transient classifications (WP1) + the deliberate set decision
# ---------------------------------------------------------------------------


def test_typed_transient_categories_carry_measured_provenance() -> None:
    """IncompleteStreamError/IncompleteLLMResponseError/EmptyLLMResponseError
    gain typed transient classifications with measured provenance, so the
    recovery layer's eligible_classes can accept the legacy paths.
    """
    assert (
        classify_failure(IncompleteStreamError("prov", "wire")).category
        is FailureCategory.INCOMPLETE_STREAM
    )
    assert (
        classify_failure(IncompleteLLMResponseError("marked incomplete")).category
        is FailureCategory.INCOMPLETE_STREAM
    )
    assert (
        classify_failure(EmptyLLMResponseError("prov", "wire")).category
        is FailureCategory.EMPTY_RESPONSE
    )


def test_typed_transients_stay_out_of_low_level_retry_failover_sets() -> None:
    """Deliberate WP1 decision: the new typed categories are consumed by the
    recovery layer's ``eligible_classes`` config; they do NOT join the
    low-level in-call retry or same-base failover sets. Joining those sets
    would silently grant retry/failover authority to model-content
    dispositions — a separate decision, never automatic.
    """
    for category in (FailureCategory.INCOMPLETE_STREAM, FailureCategory.EMPTY_RESPONSE):
        info = FailureInfo(category)
        assert info.retry_eligible is False
        assert info.failover_eligible is False


# ---------------------------------------------------------------------------
# Envelope schema v1 (handoff §1.7) — IN FULL
# ---------------------------------------------------------------------------


def _rate_limit_error() -> BackendError:
    return _backend_error(
        status=429,
        reason="rate limited",
        headers={"x-request-id": "req-123"},
        parsed_error="rate_limit",
        body_text='{"error": {"code": "rate_limit", "type": "rate_limit"}}',
        endpoint="https://user:secret@API.Example.com/v1",
    )


def _measured_evidence() -> CallEvidence:
    return CallEvidence(
        logical_call_id="call-1",
        started_at=0.0,
        budget_seconds=900.0,
        response_started=True,
        semantic_output_published=True,
        transport_attempts=2,
        elapsed_seconds=1.5,
        budget_exhausted=True,
    )


def test_envelope_schema_v1_in_full() -> None:
    """Every §1.7 field is present and correctly derived."""
    error = _rate_limit_error()
    attach_call_evidence(error, _measured_evidence())
    observation = observe_exception(error, phase="turn", turn_id="turn-1")
    envelope = build_failure_envelope(
        observation=observation,
        agent_id="agent-1",
        run_id="run-1",
        turn_id="turn-1",
        turns_used=3,
        attempted_deployment=("base", "prov", "wire"),
        configured_model="model",
        stop_reason=None,
        exclusion_reasons=None,
        tool_calls_present=True,
        tool_calls_pending=False,
        endpoint_correlated=True,
    )

    # Identity + lineage.
    assert envelope.event_id == "failure-event:agent-1:run-1:turn-1"
    assert envelope.agent_id == "agent-1"
    assert envelope.run_id == "run-1"
    assert envelope.work_package_id is None
    assert envelope.recovery_chain_id == "recovery-chain:agent-1:run-1"
    assert envelope.observed_at == observation.observed_at

    # failure{} block.
    failure = envelope.failure
    assert failure.failure_class is FailureClass.PROVIDER_RATE_LIMIT
    assert failure.cause_category is FailureCategory.RATE_LIMIT
    assert failure.phase is FailurePhase.LLM_CALL
    assert failure.exception_type == "BackendError"
    assert failure.message is not None
    assert failure.confidence is FailureConfidence.MEASURED
    assert "provider_rate_limit_code" in failure.evidence_codes
    assert "endpoint_correlated" in failure.evidence_codes
    assert "semantic_output_published_no_replay" in failure.evidence_codes
    assert failure.fingerprint is not None
    assert failure.classification_version == CLASSIFICATION_VERSION
    assert failure.classification_provenance
    assert failure.diagnostics_state == "captured"

    # deployment{} block: attempted deployment + canonical endpoint identity.
    deployment = envelope.deployment
    assert deployment.configured_model == "model"
    assert deployment.base_model == "base"
    assert deployment.provider == "prov"
    assert deployment.wire_name == "wire"
    assert deployment.endpoint == "api.example.com"

    # request{} block: one measured logical call.
    request = envelope.request
    assert request.logical_call_id == "call-1"
    assert request.transport_attempts == 2
    assert request.elapsed_seconds == 1.5
    assert request.retry_budget_seconds == 900.0
    assert request.exhaustion is ExhaustionReason.DEADLINE
    assert request.request_id == "req-123"
    assert request.request_state == "failed"
    assert request.retry_after_seconds is None
    assert request.response_started is True
    assert request.semantic_output_published is True

    # progress{} block (complete §1.7 field set).
    progress = envelope.progress
    assert progress.turns_used == 3
    assert progress.state is ProgressState.UNCHECKPOINTED
    assert progress.checkpoint_ref is None
    assert progress.transcript_boundary == "turn-1"
    assert progress.touched_paths_hint == []
    assert progress.artifact_refs == []
    assert progress.tool_effects is ToolEffectsDisposition.SETTLED
    assert progress.last_progress_at == observation.observed_at

    # recovery{} block.
    recovery = envelope.recovery
    assert recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert "transient_cause" in recovery.reason_codes
    assert recovery.same_agent_resumes_used == 0
    assert recovery.expires_at is None

    # The "class" wire alias + JSON serializability of the whole envelope.
    dumped = envelope.model_dump(mode="json", by_alias=True)
    assert dumped["failure"]["class"] == "provider_rate_limit"
    json.dumps(dumped)


def test_recovery_reference_is_a_compact_linkage() -> None:
    envelope = build_failure_envelope(observation=None, agent_id="a", run_id="r")
    reference = RecoveryReference.from_envelope(envelope)
    assert reference.recovery_chain_id == envelope.recovery_chain_id
    assert reference.event_id == envelope.event_id
    assert reference.work_package_id is None
    assert reference.eligibility is envelope.recovery.eligibility
    assert reference.same_agent_resumes_used == 0
    assert reference.expires_at is None


def test_task_result_carries_envelope_and_legacy_payloads_stay_valid() -> None:
    """The envelope rides TaskResult; legacy results (no new fields) are
    unchanged and still validate — old pinned results stay immutable.
    """
    envelope = build_failure_envelope(observation=None, agent_id="a", run_id="r")
    result = TaskResult(
        response="failed",
        turns_used=1,
        completed=False,
        agent_id="a",
        run_id="r",
        run_status=RunStatus.FAILED,
        failure=envelope,
        recovery=RecoveryReference.from_envelope(envelope),
    )
    dumped = result.model_dump(mode="json")
    assert dumped["failure"]["failure"]["failure_class"] == "unknown"
    assert dumped["run_status"] == "failed"
    assert dumped["recovery"]["recovery_chain_id"] == "recovery-chain:a:r"
    assert TaskResult.model_validate(dumped) == result

    legacy = TaskResult.model_validate({
        "response": "done",
        "turns_used": 1,
        "completed": True,
    })
    assert legacy.failure is None
    assert legacy.recovery is None
    assert legacy.run_status is None
    legacy_dump = legacy.model_dump(mode="json")
    assert "failure" not in legacy_dump
    assert "recovery" not in legacy_dump
    assert "run_status" not in legacy_dump


# ---------------------------------------------------------------------------
# Dedup identity + launch_failure admission evidence + null-vs-zero
# ---------------------------------------------------------------------------


def test_event_id_is_deduplicatable_per_exhausted_logical_call() -> None:
    """ONE deduplicatable event per exhausted logical call — not per retry."""
    first = build_failure_envelope(
        observation=None, agent_id="a", run_id="r", turn_id="t"
    )
    second = build_failure_envelope(
        observation=None, agent_id="a", run_id="r", turn_id="t"
    )
    assert first.event_id == second.event_id
    other_turn = build_failure_envelope(
        observation=None, agent_id="a", run_id="r", turn_id="t2"
    )
    assert other_turn.event_id != first.event_id
    other_run = build_failure_envelope(
        observation=None, agent_id="a", run_id="r2", turn_id="t"
    )
    assert other_run.event_id != first.event_id


def test_launch_failure_requires_admission_evidence() -> None:
    """launch_failure requires a MEASURED zero transport start (or the
    admission-phase exhaustion error with one); an unmeasured failure is
    never relabeled.
    """
    admission_error = AllDeploymentsUnavailableError("base", ())

    # Unmeasured: null measurement — the failure stays UNKNOWN (report-and-park).
    unmeasured = observe_exception(admission_error, phase="turn", turn_id="t")
    envelope = build_failure_envelope(observation=unmeasured, agent_id="a", run_id="r")
    assert envelope.failure.failure_class is FailureClass.UNKNOWN
    assert envelope.failure.phase is FailurePhase.LLM_CALL
    assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED

    # Measured zero transport starts: admission-phase evidence -> launch_failure.
    attach_call_evidence(admission_error, CallEvidence.admission_phase())
    measured = observe_exception(admission_error, phase="turn", turn_id="t")
    envelope = build_failure_envelope(
        observation=measured, agent_id="a", run_id="r", exclusion_reasons=["cooldown"]
    )
    assert envelope.failure.failure_class is FailureClass.LAUNCH_FAILURE
    assert envelope.failure.phase is FailurePhase.ADMISSION
    assert envelope.request.transport_attempts == 0
    assert envelope.request.request_state == "no_transport_start"
    assert envelope.progress.tool_effects is ToolEffectsDisposition.SETTLED
    assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE

    # A turn-phase failure with measured zero transport starts is admission too.
    runtime = RuntimeError("no backend was created")
    attach_call_evidence(runtime, CallEvidence.admission_phase())
    turn_measured = observe_exception(runtime, phase="turn", turn_id="t")
    envelope = build_failure_envelope(
        observation=turn_measured,
        agent_id="a",
        run_id="r",
        exclusion_reasons=["cooldown"],
    )
    assert envelope.failure.failure_class is FailureClass.LAUNCH_FAILURE
    assert envelope.failure.phase is FailurePhase.ADMISSION


def test_null_is_unavailable_never_zero() -> None:
    """Unmeasured fields are null; a measured zero is a distinct fact."""
    envelope = build_failure_envelope(observation=None, agent_id="a", run_id="r")
    assert envelope.request.transport_attempts is None
    assert envelope.request.elapsed_seconds is None
    assert envelope.request.logical_call_id is None
    assert envelope.request.request_id is None
    assert envelope.request.exhaustion is None
    assert envelope.deployment.endpoint is None
    assert envelope.deployment.base_model is None
    assert envelope.deployment.provider is None
    assert envelope.deployment.wire_name is None
    assert envelope.failure.cause_category is None
    assert envelope.failure.message is None
    assert envelope.failure.fingerprint is None
    assert envelope.failure.diagnostics_state == "unavailable"
    assert envelope.progress.last_progress_at is None

    error = RuntimeError("never started")
    attach_call_evidence(error, CallEvidence.admission_phase())
    measured = build_failure_envelope(
        observation=observe_exception(error, phase="turn", turn_id="t"),
        agent_id="a",
        run_id="r",
    )
    assert measured.request.transport_attempts == 0
    assert measured.request.request_state == "no_transport_start"


def test_canonical_endpoint_identity_strips_userinfo_and_case() -> None:
    assert canonical_endpoint_identity("https://user:pw@API.Example.com/v1") == (
        "api.example.com"
    )
    assert (
        canonical_endpoint_identity("https://api.example.com/v1") == "api.example.com"
    )
    assert canonical_endpoint_identity(None) is None
    assert canonical_endpoint_identity("") is None
    assert canonical_endpoint_identity("not a url") is None


# ---------------------------------------------------------------------------
# Bounded/redacted public messages + fingerprint hygiene
# ---------------------------------------------------------------------------


def test_public_message_is_bounded_and_redacted() -> None:
    secret = "sk-envelope-test-0123456789abcdef"
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_KEY", secret),))):
        error = RuntimeError(f"call failed with api key {secret} attached")
        envelope = build_failure_envelope(
            observation=observe_exception(error, phase="turn", turn_id="t"),
            agent_id="a",
            run_id="r",
        )
        message = envelope.failure.message
        assert message is not None
        assert len(message) <= 500
        assert secret not in message
        assert "REDACTED" in message

    long_error = RuntimeError("x" * 2000)
    bounded = build_failure_envelope(
        observation=observe_exception(long_error, phase="turn", turn_id="t"),
        agent_id="a",
        run_id="r",
    )
    assert bounded.failure.message is not None
    assert len(bounded.failure.message) <= 500


def test_fingerprint_carries_no_raw_message_text() -> None:
    class _MarkedFault(RuntimeError):
        status = 503
        error_code = "overloaded"

    marker = "unique-marker-text-98765"
    error = _MarkedFault(marker)
    envelope = build_failure_envelope(
        observation=observe_exception(error, phase="turn", turn_id="t"),
        agent_id="a",
        run_id="r",
        attempted_deployment=("base", "prov", "wire"),
    )
    fingerprint = envelope.failure.fingerprint
    assert fingerprint is not None
    assert marker not in fingerprint
    assert "origin=llm_call" in fingerprint
    assert "type=_MarkedFault" in fingerprint
    assert "status=503" in fingerprint
    assert "adapter=prov" in fingerprint


# ---------------------------------------------------------------------------
# Review R4 — public-envelope redaction is fail-closed, never fail-open
# ---------------------------------------------------------------------------


def test_r4_secret_straddling_the_message_bound_is_redacted() -> None:
    """R4: redaction runs BEFORE truncation.

    Before: the message was truncated first, so a secret straddling the
    500-char cut survived as an unredactable fragment prefix (the full
    value was gone, so no later redaction pass could match it). After:
    the full text is redacted before bounding, so no fragment survives.
    """
    secret = "sk-envelope-straddle-0123456789abcdef"
    # The secret starts at char 491 and runs past the 500-char cut.
    error = RuntimeError(f"{'x' * 490} {secret} {'y' * 2000}")
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_KEY", secret),))):
        envelope = build_failure_envelope(
            observation=observe_exception(error, phase="turn", turn_id="t"),
            agent_id="a",
            run_id="r",
        )
    message = envelope.failure.message
    assert message is not None
    assert len(message) <= 500
    assert secret not in message
    # The fragment truncation alone would have left behind is gone too.
    assert secret[:5] not in message
    assert "REDACTED" in message


def test_r4_redaction_failure_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """R4: a redaction failure never carries the original text publicly —
    both the observation capture and the public envelope degrade to a
    fixed marker, never the bounded original.
    """
    import chartreux.core.tools.secret_redaction as secret_redaction

    def explode(text: str) -> str:
        raise RuntimeError("redaction backend down")

    monkeypatch.setattr(secret_redaction, "redact", explode)

    observation = observe_exception(
        RuntimeError("sensitive payload"), phase="turn", turn_id="t"
    )
    assert observation.message == "[REDACTED]"

    envelope = build_failure_envelope(observation=observation, agent_id="a", run_id="r")
    message = envelope.failure.message
    assert message == "[REDACTED]"
    assert "sensitive payload" not in message


def test_r4_fingerprint_sanitizes_provider_error_code() -> None:
    """R4: the fingerprint's structured code is provider-controlled text —
    it is reduced to a bounded identifier charset, never interpolated raw.
    """

    class _MarkedFault(RuntimeError):
        status = 503
        error_code = (
            "overloaded_error; DROP TABLE users; <script>alert(1)</script> " + "z" * 200
        )

    envelope = build_failure_envelope(
        observation=observe_exception(_MarkedFault("boom"), phase="turn", turn_id="t"),
        agent_id="a",
        run_id="r",
        attempted_deployment=("base", "prov", "wire"),
    )
    fingerprint = envelope.failure.fingerprint
    assert fingerprint is not None
    assert "DROP TABLE" not in fingerprint
    assert "<script>" not in fingerprint
    assert "alert(1)" not in fingerprint
    assert "overloaded_error" in fingerprint
    code_segment = fingerprint.split("code=")[1].split(";")[0]
    assert len(code_segment) <= 64
    assert " " not in code_segment


def test_r4_public_request_id_is_bounded_and_redacted() -> None:
    """R4: the request id is provider-controlled text copied onto the
    public envelope — bounded and redacted, never raw, never unbounded.
    """
    secret = "sk-envelope-reqid-0123456789abcdef"
    with bind_policy(ScrubPolicy(redaction_credentials=(("TEST_KEY", secret),))):
        error = _backend_error(
            status=429,
            reason="rate limited",
            headers={"x-request-id": f"req-{secret}-{'y' * 500}"},
            parsed_error="rate_limit",
            body_text='{"error": {"code": "rate_limit", "type": "rate_limit"}}',
        )
        envelope = build_failure_envelope(
            observation=observe_exception(error, phase="turn", turn_id="t"),
            agent_id="a",
            run_id="r",
        )
    request_id = envelope.request.request_id
    assert request_id is not None
    assert len(request_id) <= 128
    assert secret not in request_id
    assert "REDACTED" in request_id


# ---------------------------------------------------------------------------
# Eligibility derivation (represented disposition, never acted on)
# ---------------------------------------------------------------------------


def _envelope_for(error: BaseException, *, phase: str = "turn", **kwargs: Any):
    observation = observe_exception(error, phase=phase, turn_id="t")
    return build_failure_envelope(
        observation=observation, agent_id="a", run_id="r", **kwargs
    )


def test_eligibility_protected_and_unknown_classes_are_prohibited() -> None:
    auth = _envelope_for(_backend_error(status=401))
    assert auth.failure.failure_class is FailureClass.AUTHORIZATION
    assert auth.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "protected_authorization" in auth.failure.evidence_codes

    cancelled = _envelope_for(asyncio.CancelledError())
    assert cancelled.failure.failure_class is FailureClass.CANCELLATION
    assert cancelled.recovery.eligibility is RecoveryEligibility.PROHIBITED

    unknown = _envelope_for(RuntimeError("unfamiliar"))
    assert unknown.failure.failure_class is FailureClass.UNKNOWN
    assert unknown.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "unknown_cause_report_and_park" in unknown.recovery.reason_codes

    monitor = _envelope_for(RuntimeError("harness"), phase="monitor")
    assert monitor.failure.failure_class is FailureClass.INTERNAL
    assert monitor.failure.phase is FailurePhase.MONITOR
    assert monitor.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "harness_internal_no_automatic_recovery" in monitor.recovery.reason_codes

    budget = _envelope_for(RuntimeError("limit"), stop_reason="budget_exceeded")
    assert budget.failure.failure_class is FailureClass.BUDGET_STOP
    assert budget.recovery.eligibility is RecoveryEligibility.PROHIBITED


def test_eligibility_transient_classes_need_settled_tool_effects() -> None:
    # Measured transients (the gateway attaches CallEvidence to every error
    # it raises): the tool-effects dimension decides the disposition.
    rate_limited = _rate_limit_error()
    attach_call_evidence(rate_limited, _measured_evidence())
    settled = _envelope_for(rate_limited, tool_calls_present=True)
    assert settled.recovery.eligibility is RecoveryEligibility.ELIGIBLE
    assert "tool_effects_settled" in settled.recovery.reason_codes

    pending_error = _rate_limit_error()
    attach_call_evidence(pending_error, _measured_evidence())
    pending = _envelope_for(
        pending_error, tool_calls_present=True, tool_calls_pending=True
    )
    assert pending.progress.tool_effects is ToolEffectsDisposition.UNRESOLVED
    assert pending.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
    assert "tool_effects_unresolved_reconcile_first" in pending.recovery.reason_codes

    # No transcript-visible tool calls: effects are unknown, never settled (F4).
    incomplete = IncompleteStreamError("prov", "wire")
    attach_call_evidence(incomplete, _measured_evidence())
    no_tools = _envelope_for(incomplete)
    assert no_tools.failure.failure_class is FailureClass.INCOMPLETE_STREAM
    assert no_tools.progress.tool_effects is ToolEffectsDisposition.UNKNOWN
    assert no_tools.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
    assert "tool_effects_unknown_reconcile_first" in no_tools.recovery.reason_codes

    empty_error = EmptyLLMResponseError("prov", "wire")
    attach_call_evidence(empty_error, _measured_evidence())
    empty = _envelope_for(empty_error, tool_calls_present=True)
    assert empty.failure.failure_class is FailureClass.EMPTY_RESPONSE
    assert empty.recovery.eligibility is RecoveryEligibility.ELIGIBLE


def test_r2_unmeasured_transient_never_authorizes_recovery() -> None:
    """R2: the child recovery path requires the root's measured-attempt
    evidence gate.

    Before: any turn observation with a timeout category and settled
    transcript tool calls became ELIGIBLE — an event/projector/local timeout
    could authorize another model call. After: a transient-classed cause
    without measured logical-call evidence in a model-call phase is
    PROHIBITED (report-and-park), for BOTH the eligible form and the
    reconcile barrier (a barrier a reconcile submission can clear must only
    exist for a measured transient).
    """
    unmeasured = _envelope_for(_rate_limit_error(), tool_calls_present=True)
    assert unmeasured.failure.failure_class is FailureClass.PROVIDER_RATE_LIMIT
    assert unmeasured.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "transient_cause_unmeasured_no_recovery" in (
        unmeasured.recovery.reason_codes
    )
    assert "transient_cause_unmeasured" in unmeasured.failure.evidence_codes

    # The barrier form is gated too: unmeasured + pending tool calls must
    # not even create a reconcile-clearable barrier.
    unmeasured_pending = _envelope_for(
        _rate_limit_error(), tool_calls_present=True, tool_calls_pending=True
    )
    assert unmeasured_pending.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "transient_cause_unmeasured_no_recovery" in (
        unmeasured_pending.recovery.reason_codes
    )

    # Measured evidence in a NON-model-call phase is still not a confirmed
    # provider transient: a finalization timeout is not a provider timeout.
    finalization_error = _rate_limit_error()
    attach_call_evidence(finalization_error, _measured_evidence())
    finalization = _envelope_for(finalization_error, phase="turn_terminal")
    assert finalization.failure.phase is FailurePhase.FINALIZE
    assert finalization.failure.failure_class is FailureClass.PROVIDER_RATE_LIMIT
    assert finalization.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "transient_cause_unmeasured_no_recovery" in (
        finalization.recovery.reason_codes
    )


def test_eligibility_launch_failure_exclusion_matrix() -> None:
    admission_error = AllDeploymentsUnavailableError("base", ())
    attach_call_evidence(admission_error, CallEvidence.admission_phase())
    observation = observe_exception(admission_error, phase="turn", turn_id="t")

    cooldown_only = build_failure_envelope(
        observation=observation,
        agent_id="a",
        run_id="r",
        exclusion_reasons=["cooldown", "cooldown"],
    )
    assert cooldown_only.failure.failure_class is FailureClass.LAUNCH_FAILURE
    assert cooldown_only.recovery.eligibility is RecoveryEligibility.ELIGIBLE

    permanent = build_failure_envelope(
        observation=observation,
        agent_id="a",
        run_id="r",
        exclusion_reasons=["disabled"],
    )
    assert permanent.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "permanent_launch_exclusion" in permanent.recovery.reason_codes
    assert "permanent_exclusion" in permanent.failure.evidence_codes

    unavailable = build_failure_envelope(
        observation=observation, agent_id="a", run_id="r", exclusion_reasons=None
    )
    assert unavailable.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert "launch_failure_exclusions_unavailable" in (
        unavailable.recovery.reason_codes
    )


# ---------------------------------------------------------------------------
# Gateway CallEvidence measurement
# ---------------------------------------------------------------------------


def _inputs(*, emits_finish_reason: bool = True) -> CompletionInputs:
    return CompletionInputs(
        model=ModelConfig(
            name="wire",
            alias="model",
            provider="provider",
            input_price=1,
            output_price=2,
            cached_input_price=0.5,
        ),
        provider_name="provider",
        emits_finish_reason=emits_finish_reason,
        messages=(LLMMessage(role=Role.user, content="hello"),),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=None,
    )


def _resources(backend: Any) -> CallResources:
    return CallResources(
        backend=backend, stats=AgentStats(), process_message=lambda message: message
    )


class _CountingBackend(FakeBackend):
    """A fake backend exposing the production transport-attempt counter."""

    def __init__(self, chunks: Any = None) -> None:
        super().__init__(chunks)
        self._transport_attempts = 0

    @property
    def transport_attempts(self) -> int:
        return self._transport_attempts

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self._transport_attempts += 1
        return await super().complete(**kwargs)


class _CountingFailureBackend(FakeBackend):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception
        self._transport_attempts = 0

    @property
    def transport_attempts(self) -> int:
        return self._transport_attempts

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self._transport_attempts += 1
        raise self.exception


def test_backend_transport_attempts_reads_counter_or_none() -> None:
    class _Stub:
        pass

    assert backend_transport_attempts(_Stub()) is None

    class _Counted:
        transport_attempts = 3

    assert backend_transport_attempts(_Counted()) == 3

    class _Bool:
        transport_attempts = True

    assert backend_transport_attempts(_Bool()) is None


@pytest.mark.asyncio
async def test_gateway_measures_transport_attempts_on_raised_error() -> None:
    backend = _CountingFailureBackend(RuntimeError("backend exploded"))
    with pytest.raises(RuntimeError, match="backend exploded"):
        await LLMGateway().complete(_inputs(), _resources(backend))
    # The propagated error carries the measured evidence for the logical call.
    with pytest.raises(RuntimeError) as exc_info:
        await LLMGateway().complete(_inputs(), _resources(backend))
    evidence = getattr(exc_info.value, "call_evidence", None)
    assert isinstance(evidence, CallEvidence)
    assert evidence.transport_attempts == 1
    assert evidence.logical_call_id
    assert evidence.elapsed_seconds is not None and evidence.elapsed_seconds >= 0
    assert evidence.response_started is False
    assert evidence.semantic_output_published is False


@pytest.mark.asyncio
async def test_empty_replay_shares_one_logical_call_evidence() -> None:
    """ONE deduplicatable exhausted-logical-call event, not one per retry:
    the bounded in-call empty replay shares a single CallEvidence, so both
    transport starts measure against ONE logical call identity.
    """
    empty = LLMChunk(
        message=LLMMessage(role=Role.assistant, content=" \n"),
        usage=mock_llm_chunk(content="x").usage,
        stop=StopInfo(reason="stop"),
    )
    backend = _CountingBackend([[empty], [empty]])
    with pytest.raises(EmptyLLMResponseError) as exc_info:
        await LLMGateway().complete(_inputs(), _resources(backend))
    evidence = getattr(exc_info.value, "call_evidence", None)
    assert isinstance(evidence, CallEvidence)
    # Two transport starts, ONE logical call identity: the exhausted logical
    # call is a single event (deduplicatable by event_id), never per-retry.
    assert evidence.transport_attempts == 2
    assert len(backend.requests_messages) == 2
    assert evidence.response_started is True


@pytest.mark.asyncio
async def test_unmeasured_backend_reports_unavailable_not_zero() -> None:
    backend = ImmediateFailureBackend(RuntimeError("boom"))
    with pytest.raises(RuntimeError) as exc_info:
        await LLMGateway().complete(_inputs(), _resources(backend))
    evidence = getattr(exc_info.value, "call_evidence", None)
    assert isinstance(evidence, CallEvidence)
    assert evidence.transport_attempts is None


@pytest.mark.asyncio
async def test_streaming_marks_the_semantic_output_boundary() -> None:
    fault = RuntimeError("stream broke mid-flight")

    class _StreamFaultBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self._transport_attempts = 0

        @property
        def transport_attempts(self) -> int:
            return self._transport_attempts

        async def complete_streaming(self, **kwargs: Any):
            self._transport_attempts += 1
            yield mock_llm_chunk(content="partial")
            raise fault

    published: list[LLMChunk] = []
    with pytest.raises(RuntimeError, match="stream broke mid-flight"):
        async for part in LLMGateway().chat_streaming(
            _inputs(), _resources(_StreamFaultBackend()), transcript=lambda _: None
        ):
            published.append(part)
    assert [part.message.content for part in published] == ["partial"]
    with pytest.raises(RuntimeError) as exc_info:
        async for _part in LLMGateway().chat_streaming(
            _inputs(), _resources(_StreamFaultBackend()), transcript=lambda _: None
        ):
            pass
    evidence = getattr(exc_info.value, "call_evidence", None)
    assert isinstance(evidence, CallEvidence)
    assert evidence.response_started is True
    assert evidence.semantic_output_published is True
    assert evidence.transport_attempts == 1


# ---------------------------------------------------------------------------
# Monitor integration: a failed background run carries the envelope
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_background_run_carries_envelope_round_trip(
    monkeypatch: pytest.MonkeyPatch, clean_observation_store: None
) -> None:
    """The monitor builds the envelope for a failed background run; the
    stored TaskResult carries failure/run_status/recovery; the wait and get
    round trips preserve the typed evidence; the summary carries last_failure.
    """
    child = ImmediateFailureBackend(RuntimeError("child backend failed immediately"))
    registry, parent, _notifications = await _zero_retention_registry(
        monkeypatch, child
    )
    # Keep the failed agent resident so the summary projection (agents/update)
    # and the run record stay inspectable after the run finalizes.
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=1000, max_idle_agents=10, retain_agents=True
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="fail", identity="coder", background=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )
        assert not failed.completed
        assert "child backend failed immediately" in failed.response

        # The typed failure evidence rides the result.
        assert failed.run_status is RunStatus.FAILED
        envelope = failed.failure
        assert envelope is not None
        assert envelope.agent_id == launch.agent_id
        assert envelope.run_id == launch.run_id
        assert envelope.event_id.startswith(
            f"failure-event:{launch.agent_id}:{launch.run_id}"
        )
        assert envelope.recovery_chain_id == (
            f"recovery-chain:{launch.agent_id}:{launch.run_id}"
        )
        # The synthetic RuntimeError is unfamiliar: unknown stays UNKNOWN and
        # reports-and-parks — it never acquires recovery authority by label.
        assert envelope.failure.failure_class is FailureClass.UNKNOWN
        assert envelope.failure.phase is FailurePhase.LLM_CALL
        assert envelope.failure.diagnostics_state == "captured"
        assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
        assert "unknown_cause_report_and_park" in envelope.recovery.reason_codes
        # The stub backend has no transport counter: unavailable, never zero.
        assert envelope.request.transport_attempts is None

        # The compact recovery reference matches the envelope.
        reference = failed.recovery
        assert reference is not None
        assert reference.event_id == envelope.event_id
        assert reference.recovery_chain_id == envelope.recovery_chain_id
        assert reference.eligibility is envelope.recovery.eligibility

        # The get round trip preserves the typed evidence.
        fetched = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert fetched is not None
        assert fetched.failure is not None
        assert fetched.failure.event_id == envelope.event_id
        assert fetched.run_status is RunStatus.FAILED

        # The run record is the canonical store; the summary carries the
        # latest failure (the agents/update projection source).
        record = registry._agent_records[launch.agent_id]
        latest = record.current_run or record.run_history[-1]
        assert latest.failure is not None
        assert latest.failure.event_id == envelope.event_id
        summaries = await registry.check_agents()
        summary = next(s for s in summaries if s.agent_id == launch.agent_id)
        assert summary.last_failure is not None
        assert summary.last_failure.event_id == envelope.event_id
        assert summary.last_run_status is RunStatus.FAILED
    finally:
        await registry.drain_children()
        await parent.aclose()
