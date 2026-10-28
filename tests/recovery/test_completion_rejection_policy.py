"""P11 — typed rejection classification + replacement policy + wire mirror.

Covers the P11 acceptance surface:

- The bounded constructor/classification: the completion-rejection failure
  class is built directly from P1's typed rejection vocabulary, bounded and
  redacted throughout, with the typed rejection value carried as an evidence
  identifier and the recovery block PROHIBITED with its own distinct reason
  code.
- The protected vocabulary is preserved EXACTLY: ``_PROTECTED_CLASSES``
  contains PREFLIGHT/CANCELLATION/AUTHORIZATION/PERMANENT_CONFIG/BUDGET_STOP/
  TASK — it does NOT contain UNKNOWN or INTERNAL; those are gated separately
  (both PROHIBITED, distinct reason codes).
- The replacement policy: explicit replacement authorization for the new
  class only; no same-agent auto-retry; the general eligibility invariants
  that do not depend on a failed endpoint still apply (tier capability, the
  separate candidate capability axis, the allowlist, the price cap); the
  endpoint-independence guards stay in force for every other class.
- The wire layer: the new class mirrors into ``PublicFailureClass`` so the
  set-equality pin in ``tests/app_server/test_protocol_models.py`` stays
  green (extended there with the lossless-projection case).

All transports are mocked (pure policy unit tests — zero network egress).
"""

from __future__ import annotations

from typing import Any

import httpx

from chartreux.core.completion_contract import CompletionRejectionReason
from chartreux.core.config.models import RecoveryConfig
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.recovery_models import (
    _PROTECTED_CLASSES,
    CLASSIFICATION_VERSION,
    FailureClass,
    FailureConfidence,
    FailureEnvelope,
    FailurePhase,
    RecoveryEligibility,
    build_completion_rejection_envelope,
    build_failure_envelope,
)
from chartreux.core.recovery_observation import observe_exception
from chartreux.core.recovery_policy import (
    PackageTier,
    ReplacementCandidate,
    ReplacementCapability,
    is_infrastructure_failure,
    validate_replacement_selection,
)

# ---------------------------------------------------------------------------
# Fixtures: the typed rejection vocabulary and the eligible-set candidates
# ---------------------------------------------------------------------------


def _payload_summary() -> PayloadSummary:
    return PayloadSummary(
        model="main-wire",
        message_count=1,
        approx_chars=2,
        temperature=0.0,
        has_tools=False,
        tool_choice=None,
    )


def _candidate(
    base_model: str = "alt-model",
    provider: str = "gamma",
    endpoint: str | None = "https://gamma.test/v1",
    *,
    tier: PackageTier = PackageTier.SUBSTANTIVE,
    price: float | None = 5.0,
    capability: Any = None,
) -> ReplacementCandidate:
    return ReplacementCandidate(
        base_model=base_model,
        provider=provider,
        wire_name=f"{base_model}-wire",
        endpoint=endpoint,
        tier=tier,
        capability=capability,
        output_price_per_million=price,
    )


def _rejection_envelope(
    *,
    agent_id: str = "agent-1",
    run_id: str = "run-1",
    rejection: CompletionRejectionReason = CompletionRejectionReason.CLAIM_CONTRADICTED,
    turn_id: str | None = "turn-1",
    work_package_id: str | None = "wp-1",
    predecessor_agent_id: str | None = None,
    message: str | None = None,
) -> FailureEnvelope:
    return build_completion_rejection_envelope(
        agent_id=agent_id,
        run_id=run_id,
        rejection=rejection,
        turn_id=turn_id,
        work_package_id=work_package_id,
        predecessor_agent_id=predecessor_agent_id,
        message=message,
    )


def _validate(
    envelope: Any,
    candidate: Any,
    *,
    package_tier: PackageTier = PackageTier.SUBSTANTIVE,
    config: RecoveryConfig | None = None,
):
    return validate_replacement_selection(
        envelope=envelope,
        package_tier=package_tier,
        claimed=candidate,
        candidates=[candidate],
        config=config or RecoveryConfig(),
    )


# ---------------------------------------------------------------------------
# The bounded constructor/classification
# ---------------------------------------------------------------------------


def test_bounded_constructor_classifies_the_typed_rejection() -> None:
    """The bounded constructor builds the failure envelope directly from P1's
    typed rejection vocabulary: the class is the completion rejection, the
    phase is finalize, the confidence is measured, the typed rejection value
    rides the evidence codes as an identifier, and the free text is redacted
    then bounded.
    """
    envelope = build_completion_rejection_envelope(
        agent_id="agent-1",
        run_id="run-1",
        rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
        turn_id="turn-3",
        work_package_id="wp-1",
        predecessor_agent_id="agent-0",
        message="claimed check_pass without an executed check; git push --force x" * 40,
        clock=lambda: 1_700_000_000.0,
    )
    failure = envelope.failure
    assert failure.failure_class is FailureClass.COMPLETION_REJECTION
    assert failure.phase is FailurePhase.FINALIZE
    assert failure.confidence is FailureConfidence.MEASURED
    assert failure.exception_type is None
    assert failure.cause_category is None
    assert failure.classification_version == CLASSIFICATION_VERSION
    assert "completion_rejection_claim_contradicted" in failure.evidence_codes
    assert "completion_audit_rejection" in failure.evidence_codes
    # Bounded: the free text is redacted then bounded, never carried raw.
    assert len(failure.message or "") <= 500


def test_the_rejection_envelope_is_prohibited_with_a_distinct_reason_code() -> None:
    """The rejection envelope's recovery block is PROHIBITED with its own
    distinct reason code — a typed audit rejection authorizes a cross-agent
    replacement, never a warm same-agent auto-resume.
    """
    envelope = build_completion_rejection_envelope(
        agent_id="agent-1",
        run_id="run-1",
        rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
    )
    assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert envelope.recovery.reason_codes == ["completion_rejection_no_auto_recovery"]
    # Distinct from the three existing codes on the mapping.
    assert envelope.recovery.reason_codes != ["protected_class_no_automatic_recovery"]
    assert envelope.recovery.reason_codes != ["harness_internal_no_automatic_recovery"]
    assert envelope.recovery.reason_codes != ["unknown_cause_report_and_park"]


def test_protected_set_stays_exactly_the_six_protected_classes() -> None:
    """The protected vocabulary is preserved EXACTLY: _PROTECTED_CLASSES
    contains PREFLIGHT/CANCELLATION/AUTHORIZATION/PERMANENT_CONFIG/BUDGET_STOP/
    TASK — it does NOT contain UNKNOWN or INTERNAL; the new class is gated
    separately (both PROHIBITED, distinct reason codes).
    """
    assert _PROTECTED_CLASSES == frozenset({
        FailureClass.PREFLIGHT,
        FailureClass.CANCELLATION,
        FailureClass.AUTHORIZATION,
        FailureClass.PERMANENT_CONFIG,
        FailureClass.BUDGET_STOP,
        FailureClass.TASK,
    })
    assert FailureClass.COMPLETION_REJECTION not in _PROTECTED_CLASSES
    assert FailureClass.UNKNOWN not in _PROTECTED_CLASSES
    assert FailureClass.INTERNAL not in _PROTECTED_CLASSES


def test_unknown_and_internal_are_gated_separately_with_distinct_codes() -> None:
    """UNKNOWN and INTERNAL are gated separately (both PROHIBITED, distinct
    reason codes) — the round-9 citation correction's three-line prohibition
    below the protected gate stays exactly as it was.
    """
    unknown = build_failure_envelope(
        observation=None, agent_id="agent-1", run_id="run-1"
    )
    assert unknown.failure.failure_class is FailureClass.UNKNOWN
    assert unknown.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert unknown.recovery.reason_codes == ["unknown_cause_report_and_park"]
    stall = BackendError(
        provider="beta",
        endpoint="https://beta.test/v1",
        status=None,
        reason="read timed out",
        headers=None,
        body_text=None,
        parsed_error="Network error",
        model="main-wire",
        payload_summary=_payload_summary(),
    )
    stall.__cause__ = httpx.ReadTimeout("read timed out")
    internal = build_failure_envelope(
        observation=observe_exception(
            stall, phase="monitor", agent_id="agent-1", run_id="run-1"
        ),
        agent_id="agent-1",
        run_id="run-1",
    )
    assert internal.failure.failure_class is FailureClass.INTERNAL
    assert internal.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert internal.recovery.reason_codes == ["harness_internal_no_automatic_recovery"]


# ---------------------------------------------------------------------------
# The replacement policy: explicit authorization for the new class only
# ---------------------------------------------------------------------------


def test_replacement_is_authorized_for_the_new_class_only() -> None:
    """A typed completion-audit rejection authorizes a replacement — the only
    class the policy authorizes on the audit path; every other class still
    follows the unchanged guard matrix.
    """
    rejection_envelope = build_completion_rejection_envelope(
        agent_id="agent-1",
        run_id="run-1",
        rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
        work_package_id="wp-1",
    )
    candidate = _candidate("alt-model", "gamma", "https://gamma.test/v1", price=5.0)
    decision = _validate(rejection_envelope, candidate)
    assert decision.disposition == "selected"
    assert "completion_rejection_replacement_authorized" in decision.reason_codes
    # No same-agent auto-retry: the replacement is a NEW agent with the
    # predecessor/package lineage.
    assert decision.predecessor_agent_id == "agent-1"
    assert "new_agent_lineage_no_same_agent_retry" in decision.reason_codes

    # The authorization is for the new class only: the unchanged guard matrix
    # still rejects every other class.
    unknown_envelope = build_failure_envelope(
        observation=None, agent_id="agent-1", run_id="run-1"
    )
    unknown_decision = _validate(unknown_envelope, _candidate())
    assert unknown_decision.disposition == "rejected"
    assert unknown_decision.reason_codes == ["unknown_cause_report_and_park"]


def test_the_general_eligibility_invariants_still_apply() -> None:
    """The general eligibility invariants that do not depend on a failed
    endpoint stay in force on the authorization path.
    """
    rejection_envelope = build_completion_rejection_envelope(
        agent_id="agent-1",
        run_id="run-1",
        rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
    )
    luna = _candidate(
        "luna-model", "alpha", "https://alpha.test/v1", tier=PackageTier.MECHANICAL
    )
    assert (
        _validate(rejection_envelope, luna, config=RecoveryConfig()).disposition
        == "rejected"
    )
    assert (
        _validate(
            rejection_envelope,
            ReplacementCandidate(
                base_model="alt-model",
                provider="gamma",
                wire_name="alt-model-wire",
                endpoint="https://gamma.test/v1",
                tier=PackageTier.MECHANICAL,
                capability=ReplacementCapability.NONE,
            ),
            package_tier=PackageTier.MECHANICAL,
        ).disposition
        == "rejected"
    )
    assert _validate(
        rejection_envelope,
        _candidate("alt-model", "gamma", "https://gamma.test/v1"),
        config=RecoveryConfig(fallback_allowed_base_models=["other-model"]),
    ).reason_codes == ["base_model_not_in_fallback_allowlist"]
    assert _validate(
        rejection_envelope,
        _candidate("alt-model", "gamma", "https://gamma.test/v1", price=5.0),
        config=RecoveryConfig(fallback_max_output_price_per_million=1.0),
    ).reason_codes == ["fallback_price_cap_exceeded"]
    assert _validate(
        rejection_envelope,
        _candidate("alt-model", "gamma", "https://gamma.test/v1", price=None),
        config=RecoveryConfig(fallback_max_output_price_per_million=1.0),
    ).reason_codes == ["fallback_price_unavailable_cap_unverifiable"]


def test_the_unattended_resilience_rule_never_parks_a_rejection() -> None:
    """A completion rejection is never an infrastructure failure under the
    unattended-resilience rule: the park-and-continue vocabulary is reserved
    for harness/endpoint failures; a rejection is escalated, never parked.
    """
    rejection_envelope = build_completion_rejection_envelope(
        agent_id="agent-1",
        run_id="run-1",
        rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
    )
    assert not is_infrastructure_failure(rejection_envelope)
