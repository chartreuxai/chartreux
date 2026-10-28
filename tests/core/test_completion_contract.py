"""P1: the typed completion contract — vocabularies, grammar, truth table.

Covers the envelope/receipt grammar (marker key, requirement ids, bounds,
duplicate/unknown-field rejection), the [completion] config section
registration, the frozen identity/launch contract fields, the TaskResult
audit-stamp dump pin, and the truth table's fail-open, never-escalate, and
never-misattribute rules.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.core.agents.identity_launch import TaskIdentityArgs
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    MAX_ENVELOPE_REQUIREMENTS,
    MAX_REQUIREMENT_ID_LENGTH,
    MAX_SATISFIED_REQUIREMENTS,
    CompletionArtifactKind,
    CompletionAuditClassification,
    CompletionCheckTemplate,
    CompletionClaimFidelity,
    CompletionDeclaredOutcome,
    CompletionDisclosure,
    CompletionEnvelope,
    CompletionEvidence,
    CompletionMandateSatisfaction,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRequirement,
    CompletionRuntimeFailure,
    classify_completion,
)
from chartreux.core.config.models import RecoveryEligibleClass
from chartreux.core.subagents import TaskArgs, TaskResult


def _requirement(requirement_id: str = "req-1", **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "requirement_id": requirement_id,
        "artifact": CompletionArtifactKind.COMMIT,
    }
    payload.update(overrides)
    return payload


def _envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "requirements": [_requirement()],
    }
    payload.update(overrides)
    return payload


def _receipt(
    envelope: CompletionEnvelope | None = None, **overrides: Any
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
    }
    if envelope is not None:
        payload["envelope"] = envelope
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# The envelope grammar
# ---------------------------------------------------------------------------


def test_envelope_round_trips() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())

    assert envelope.marker == COMPLETION_ENVELOPE_MARKER
    assert envelope.schema_version == COMPLETION_CONTRACT_SCHEMA_VERSION
    assert len(envelope.requirements) == 1
    assert CompletionEnvelope.model_validate(envelope.model_dump()) == envelope


def test_envelope_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CompletionEnvelope.model_validate({**_envelope(), "surprise": True})


def test_envelope_rejects_duplicate_requirement_ids() -> None:
    payload = _envelope(requirements=[_requirement("req-a"), _requirement("req-a")])
    with pytest.raises(ValidationError):
        CompletionEnvelope.model_validate(payload)


def test_envelope_rejects_oversized_requirement_lists() -> None:
    payload = _envelope(
        requirements=[
            _requirement(f"req-{index}")
            for index in range(MAX_ENVELOPE_REQUIREMENTS + 1)
        ]
    )
    with pytest.raises(ValidationError):
        CompletionEnvelope.model_validate(payload)


def test_envelope_rejects_empty_requirement_lists() -> None:
    with pytest.raises(ValidationError):
        CompletionEnvelope.model_validate(_envelope(requirements=[]))


def test_envelope_rejects_oversized_requirement_ids() -> None:
    payload = _envelope(
        requirements=[_requirement("req-" + "a" * MAX_REQUIREMENT_ID_LENGTH)]
    )
    with pytest.raises(ValidationError):
        CompletionEnvelope.model_validate(payload)


def test_envelope_rejects_empty_requirement_ids() -> None:
    payload = _envelope(requirements=[_requirement("")])
    with pytest.raises(ValidationError):
        CompletionEnvelope.model_validate(payload)


def test_requirement_binds_exactly_one_vocabulary() -> None:
    with pytest.raises(ValidationError):
        CompletionRequirement.model_validate({"requirement_id": "req-1"})


def test_receipt_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        CompletionReceipt.model_validate({
            "marker": COMPLETION_ENVELOPE_MARKER,
            "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
            "satisfied_requirement_ids": ["req-commit"],
            "nonsense": True,
        })


def test_receipt_rejects_oversized_satisfied_id_lists() -> None:
    envelope = CompletionEnvelope.model_validate(
        _envelope(
            requirements=[
                _requirement(f"req-{index}")
                for index in range(MAX_ENVELOPE_REQUIREMENTS)
            ]
        )
    )
    with pytest.raises(ValidationError):
        CompletionReceipt.model_validate({
            "marker": COMPLETION_ENVELOPE_MARKER,
            "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
            "envelope": envelope,
            "satisfied_requirement_ids": [
                f"req-{index}" for index in range(MAX_SATISFIED_REQUIREMENTS + 1)
            ],
        })


def test_receipt_rejects_duplicate_satisfied_ids() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    with pytest.raises(ValidationError):
        CompletionReceipt.model_validate(
            _receipt(envelope, satisfied_requirement_ids=["req-1", "req-1"])
        )


def test_receipt_without_envelope_accepts_bounded_ids() -> None:
    receipt = CompletionReceipt.model_validate({
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "satisfied_requirement_ids": ["req-a", "req-b"],
    })

    assert receipt.satisfied_requirement_ids == ("req-a", "req-b")


# ---------------------------------------------------------------------------
# The [completion] config section
# ---------------------------------------------------------------------------


def test_completion_section_registers_as_a_field() -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    assert "completion" in ChartreuxConfigSchema.model_fields
    config = ChartreuxConfigSchema.model_validate({"completion": {"mode": "enforce"}})
    assert config.completion.mode == "enforce"


def test_completion_section_defaults_to_enforce() -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    # The released default is enforce (the recorded enforce-activation
    # decision); an explicit stamp still validates.
    config = ChartreuxConfigSchema.model_validate({})
    assert config.completion.mode == "enforce"
    explicit = ChartreuxConfigSchema.model_validate({"completion": {"mode": "stamp"}})
    assert explicit.completion.mode == "stamp"


def test_completion_section_rejects_unknown_fields() -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({"completion": {"mode": "stamp", "x": 1}})


# ---------------------------------------------------------------------------
# The truth table
# ---------------------------------------------------------------------------


def test_no_mandate_disables_only_itself() -> None:
    classification = classify_completion(
        mandate=None,
        receipt=None,
        declared_completed=True,
        evidence=CompletionEvidence(),
    )

    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_DISABLED
    )
    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.declared_outcome is (
        CompletionDeclaredOutcome.OUTCOME_COMPLETED
    )
    assert classification.primary_runtime_failure is (
        CompletionRuntimeFailure.RUNTIME_HEALTHY
    )
    assert classification.rejection is None


def test_missing_receipt_classifies_disclosure_incomplete() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=None,
        declared_completed=True,
        evidence=CompletionEvidence(),
    )

    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_UNADDRESSED
    )
    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.rejection is (CompletionRejectionReason.RECEIPT_UNVERIFIABLE)


def test_verifier_infrastructure_failure_is_never_a_worker_fault() -> None:
    classification = classify_completion(
        mandate=None,
        receipt=None,
        declared_completed=False,
        evidence=CompletionEvidence(runtime_failure="unavailable"),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.primary_runtime_failure is (
        CompletionRuntimeFailure.RUNTIME_UNAVAILABLE
    )
    assert classification.rejection is None


def test_matched_claim_against_observed_requirements() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=CompletionReceipt.model_validate(
            _receipt(envelope, satisfied_requirement_ids=["req-1"])
        ),
        declared_completed=True,
        evidence=CompletionEvidence(requirement_ids_observed=("req-1",)),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_MATCHED
    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_COMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_SATISFIED
    )


def test_claim_against_unobserved_requirements_is_unverified() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    receipt = CompletionReceipt.model_validate(
        _receipt(satisfied_requirement_ids=["req-commit"])
    )
    classification = classify_completion(
        mandate=envelope,
        receipt=receipt,
        declared_completed=True,
        evidence=CompletionEvidence(requirement_ids_observed=()),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_UNADDRESSED
    )


def test_verifier_infrastructure_failure_with_a_receipt_never_contradicts() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    receipt = CompletionReceipt.model_validate(
        _receipt(envelope, satisfied_requirement_ids=["req-1"])
    )
    classification = classify_completion(
        mandate=envelope,
        receipt=receipt,
        declared_completed=True,
        evidence=CompletionEvidence(runtime_failure="timeout"),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.primary_runtime_failure is (
        CompletionRuntimeFailure.RUNTIME_TIMEOUT
    )
    assert classification.rejection is None


# ---------------------------------------------------------------------------
# The closed vocabularies and the pinned grammar version
# ---------------------------------------------------------------------------


def test_check_template_vocabulary_is_closed() -> None:
    assert {member.value for member in CompletionCheckTemplate} == {
        "pytest",
        "ruff_check",
        "pyright",
        "git_head",
        "git_status",
        "git_diff_stat",
    }


def test_artifact_kind_vocabulary_is_closed() -> None:
    assert {member.value for member in CompletionArtifactKind} == {
        "commit",
        "touched_path",
        "check_pass",
        "envelope",
    }


def test_grammar_version_and_marker_are_pinned() -> None:
    assert COMPLETION_CONTRACT_SCHEMA_VERSION == 1
    assert COMPLETION_ENVELOPE_MARKER == "completion_envelope_v1"


def test_classification_is_the_versioned_audit_stamp() -> None:
    classification = classify_completion(
        mandate=None, receipt=None, declared_completed=True
    )

    assert classification.schema_version == COMPLETION_CONTRACT_SCHEMA_VERSION
    dumped = classification.model_dump()
    assert dumped["schema_version"] == COMPLETION_CONTRACT_SCHEMA_VERSION
    assert CompletionAuditClassification.model_validate(dumped) == classification


# ---------------------------------------------------------------------------
# The [completion] config section (registration-only; root stays open)
# ---------------------------------------------------------------------------


def test_completion_section_rejects_unknown_modes() -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    with pytest.raises(ValidationError):
        ChartreuxConfigSchema.model_validate({"completion": {"mode": "block"}})


def test_completion_section_model_is_extra_forbid() -> None:
    from chartreux.core.config.models import CompletionSectionConfig

    with pytest.raises(ValidationError):
        CompletionSectionConfig.model_validate({"mode": "stamp", "extra": 1})


def test_root_schema_stays_open_for_unknown_sections() -> None:
    # Amendment 8: the ROOT schema is frozen but NOT extra="forbid" — the
    # [completion] section takes effect only by its registration as a field,
    # and unknown top-level sections stay ignored.
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema

    config = ChartreuxConfigSchema.model_validate({"not_a_section": {"x": 1}})

    assert config.completion.mode == "enforce"


# ---------------------------------------------------------------------------
# The frozen identity/launch contract
# ---------------------------------------------------------------------------


def test_task_identity_args_declares_the_completion_mandates() -> None:
    args = TaskIdentityArgs.model_validate({
        "task": "do the thing",
        "completion_check": ["pytest", "git_head"],
        "required_artifacts": ["commit", "envelope"],
    })

    assert args.completion_check == [
        CompletionCheckTemplate.PYTEST,
        CompletionCheckTemplate.GIT_HEAD,
    ]
    assert args.required_artifacts == [
        CompletionArtifactKind.COMMIT,
        CompletionArtifactKind.ENVELOPE,
    ]


def test_completion_mandates_are_optional_and_fail_open() -> None:
    args = TaskIdentityArgs.model_validate({"task": "do the thing"})

    assert args.completion_check is None
    assert args.required_artifacts is None
    # Unset mandates stay out of the supplied-field set: None disables only
    # itself.
    assert "completion_check" not in args.model_fields_set
    assert "required_artifacts" not in args.model_fields_set


def test_completion_mandates_reject_off_vocabulary_values() -> None:
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({
            "task": "do the thing",
            "completion_check": ["nmap"],
        })
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({
            "task": "do the thing",
            "required_artifacts": ["tea_leaves"],
        })


def test_task_identity_args_stays_extra_forbid() -> None:
    with pytest.raises(ValidationError):
        TaskIdentityArgs.model_validate({"task": "do the thing", "bogus": 1})


def test_task_args_inherits_the_completion_mandates() -> None:
    args = TaskArgs.model_validate({
        "task": "do the thing",
        "completion_check": ["ruff_check"],
    })

    assert args.completion_check == [CompletionCheckTemplate.RUFF_CHECK]
    assert args.required_artifacts is None


# ---------------------------------------------------------------------------
# The TaskResult audit-stamp dump pin (extends tests/core/test_subagents.py)
# ---------------------------------------------------------------------------


def test_completion_audit_is_omitted_from_dumps_when_none() -> None:
    result = TaskResult(response="done", turns_used=1, completed=True)

    dumped = result.model_dump()
    assert "completion_audit" not in dumped
    assert "completion_audit" not in result.model_dump_json()
    assert TaskResult.model_validate(dumped).completion_audit is None


def test_completion_audit_stamp_round_trips_through_the_wire() -> None:
    classification = classify_completion(
        mandate=None, receipt=None, declared_completed=True
    )
    result = TaskResult(
        response="done", turns_used=1, completed=True, completion_audit=classification
    )

    dumped = result.model_dump(mode="json")
    assert dumped["completion_audit"]["schema_version"] == (
        COMPLETION_CONTRACT_SCHEMA_VERSION
    )
    restored = TaskResult.model_validate(dumped)
    assert restored.completion_audit == classification


# ---------------------------------------------------------------------------
# The truth table over raw worker claims
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {**_envelope(), "marker": "not_the_marker"},
            CompletionRejectionReason.ENVELOPE_MALFORMED,
        ),
        (
            {**_envelope(), "surprise": True},
            CompletionRejectionReason.ENVELOPE_UNKNOWN_FIELD,
        ),
        (
            _envelope(requirements=[_requirement("req-a"), _requirement("req-a")]),
            CompletionRejectionReason.REQUIREMENT_ID_DUPLICATE,
        ),
        (
            _envelope(
                requirements=[
                    _requirement(f"req-{index}")
                    for index in range(MAX_ENVELOPE_REQUIREMENTS + 1)
                ]
            ),
            CompletionRejectionReason.ENVELOPE_OVERSIZED,
        ),
        ("not-a-mapping", CompletionRejectionReason.ENVELOPE_MALFORMED),
    ],
)
def test_grammar_violations_are_disclosure_failures_never_escalation(
    payload: Any, expected: CompletionRejectionReason
) -> None:
    classification = classify_completion(
        mandate=payload, receipt=None, declared_completed=True
    )

    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_UNADDRESSED
    )
    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.rejection is expected
    # Never escalation: the only worker-side disposition is a contradiction.
    assert classification.rejection is not CompletionRejectionReason.CLAIM_CONTRADICTED
    assert (
        classification.claim_fidelity is not CompletionClaimFidelity.CLAIM_CONTRADICTED
    )


def test_malformed_receipt_classifies_as_disclosure_incomplete() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt={**_receipt(), "marker": "not_the_marker"},
        declared_completed=True,
    )

    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.rejection is CompletionRejectionReason.ENVELOPE_MALFORMED
    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED


def test_receipt_with_unknown_satisfied_ids_classifies_as_disclosure_incomplete() -> (
    None
):
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(satisfied_requirement_ids=["req-other"]),
        declared_completed=True,
    )

    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.rejection is CompletionRejectionReason.REQUIREMENT_ID_UNKNOWN


def test_partial_receipt_is_disclosure_incomplete() -> None:
    envelope = CompletionEnvelope.model_validate(
        _envelope(requirements=[_requirement("req-a"), _requirement("req-b")])
    )
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(satisfied_requirement_ids=["req-a"]),
        declared_completed=True,
        evidence=CompletionEvidence(requirement_ids_observed=("req-a",)),
    )

    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_UNADDRESSED
    )
    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.rejection is None


def test_disclosure_and_mandate_axes_are_separate_from_claim_fidelity() -> None:
    # The worker disclosed and addressed every mandate, but the runtime
    # observed no satisfying evidence: the claim stays unverified.
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(satisfied_requirement_ids=["req-1"]),
        declared_completed=True,
        evidence=CompletionEvidence(),
    )

    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_COMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_SATISFIED
    )
    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.rejection is None


def test_claimed_requirement_contradicted_by_direct_evidence() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(satisfied_requirement_ids=["req-1"]),
        declared_completed=True,
        evidence=CompletionEvidence(contradicted_requirement_ids=("req-1",)),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_CONTRADICTED
    assert classification.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_COMPLETE
    )
    assert classification.mandate_satisfaction is (
        CompletionMandateSatisfaction.MANDATE_SATISFIED
    )


def test_contradiction_requires_the_worker_to_have_claimed_it() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(),
        declared_completed=True,
        evidence=CompletionEvidence(contradicted_requirement_ids=("req-1",)),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.rejection is None


def test_infrastructure_failure_suppresses_contradiction() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(satisfied_requirement_ids=["req-1"]),
        declared_completed=True,
        evidence=CompletionEvidence(
            contradicted_requirement_ids=("req-1",), runtime_failure="timeout"
        ),
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
    assert classification.rejection is None
    assert classification.primary_runtime_failure is (
        CompletionRuntimeFailure.RUNTIME_TIMEOUT
    )


def test_infrastructure_failure_suppresses_grammar_rejection() -> None:
    classification = classify_completion(
        mandate={**_envelope(), "marker": "not_the_marker"},
        receipt=None,
        declared_completed=True,
        evidence=CompletionEvidence(runtime_failure="unavailable"),
    )

    assert classification.rejection is None
    assert classification.disclosure_completeness is (
        CompletionDisclosure.DISCLOSURE_INCOMPLETE
    )
    assert classification.primary_runtime_failure is (
        CompletionRuntimeFailure.RUNTIME_UNAVAILABLE
    )


def test_evidence_accepts_a_raw_mapping() -> None:
    envelope = CompletionEnvelope.model_validate(_envelope())
    classification = classify_completion(
        mandate=envelope,
        receipt=_receipt(satisfied_requirement_ids=["req-1"]),
        declared_completed=True,
        evidence={"requirement_ids_observed": ["req-1"]},
    )

    assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_MATCHED


# ---------------------------------------------------------------------------
# The recovery-eligibility protection
# ---------------------------------------------------------------------------


def test_recovery_eligible_class_does_not_gain_the_rejection_class() -> None:
    # P1 pins the protection: the completion rejection vocabulary is NOT a
    # recovery-eligible failure class (recovery_models.py maps protected
    # classes to prohibited already).
    eligible = {member.value for member in RecoveryEligibleClass}
    assert eligible == {
        "connection",
        "timeout",
        "server_error",
        "overload",
        "rate_limit",
        "launch_failure",
        "incomplete_stream",
        "empty_response",
    }
    rejected = {member.value for member in CompletionRejectionReason}
    assert not eligible & rejected
