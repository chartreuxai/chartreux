"""The receipt's bounded claimed-value surface (the P1/P7/C1 grammar extension).

The completion receipt carries, alongside ``satisfied_requirement_ids``, a
bounded ``claimed_values`` object keyed by requirement id: the claimed
commit hash for an ``artifact:commit`` requirement and the claimed artifact
path for an ``artifact:touched_path`` requirement. The surface is strictly
bounded (count, key, and value caps) and strictly fail-open: a claim without
values is well-formed and resolves unverified — never a fault — and a
grammar violation classifies as a disclosure fact, never a worker fault.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError
import pytest

from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    MAX_CLAIMED_VALUE_LENGTH,
    MAX_CLAIMED_VALUES,
    MAX_REQUIREMENT_ID_LENGTH,
    CompletionArtifactKind,
    CompletionClaimFidelity,
    CompletionDisclosure,
    CompletionEnvelope,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRequirement,
    claimed_values_from_receipt,
    classify_completion,
)

FULL_HASH = "abc1234" + "0" * 33


def _envelope(*requirements: CompletionRequirement) -> CompletionEnvelope:
    return CompletionEnvelope(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        requirements=tuple(requirements),
    )


def _commit_envelope() -> CompletionEnvelope:
    return _envelope(
        CompletionRequirement(
            requirement_id="artifact:commit", artifact=CompletionArtifactKind.COMMIT
        )
    )


def _touched_path_envelope() -> CompletionEnvelope:
    return _envelope(
        CompletionRequirement(
            requirement_id="artifact:touched_path",
            artifact=CompletionArtifactKind.TOUCHED_PATH,
        )
    )


def _receipt(
    envelope: CompletionEnvelope,
    *satisfied: str,
    claimed_values: dict[str, str] | None = None,
) -> CompletionReceipt:
    return CompletionReceipt(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        envelope=envelope,
        satisfied_requirement_ids=tuple(satisfied),
        claimed_values=claimed_values or {},
    )


def _payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "envelope": _commit_envelope().model_dump(mode="json"),
        "satisfied_requirement_ids": ["artifact:commit"],
    }
    payload.update(overrides)
    return payload


class TestClaimedValuesGrammar:
    """The typed receipt carries the bounded claimed-value surface."""

    def test_receipt_carries_claimed_values(self) -> None:
        receipt = _receipt(
            _commit_envelope(),
            "artifact:commit",
            claimed_values={"artifact:commit": FULL_HASH},
        )
        assert receipt.claimed_values == {"artifact:commit": FULL_HASH}
        assert claimed_values_from_receipt(receipt) == {"artifact:commit": FULL_HASH}

    def test_claim_without_values_is_well_formed(self) -> None:
        receipt = _receipt(_commit_envelope(), "artifact:commit")
        assert receipt.claimed_values == {}
        assert claimed_values_from_receipt(receipt) == {}

    def test_claimed_values_round_trip_through_the_dump(self) -> None:
        # The extraction path dumps the nested envelope; the claimed values
        # ride the receipt's own payload unchanged.
        receipt = _receipt(
            _touched_path_envelope(),
            "artifact:touched_path",
            claimed_values={"artifact:touched_path": "/workspace/out.txt"},
        )
        dumped = receipt.model_dump(mode="json")
        assert dumped["claimed_values"] == {
            "artifact:touched_path": "/workspace/out.txt"
        }
        assert CompletionReceipt.model_validate(dumped) == receipt

    def test_unknown_claimed_value_key_is_a_grammar_violation(self) -> None:
        with pytest.raises(ValidationError):
            _receipt(
                _commit_envelope(),
                "artifact:commit",
                claimed_values={"artifact:other": FULL_HASH},
            )

    def test_too_many_claimed_values_is_a_grammar_violation(self) -> None:
        with pytest.raises(ValidationError):
            _receipt(
                _commit_envelope(),
                "artifact:commit",
                claimed_values={
                    f"req-{index}": "v" for index in range(MAX_CLAIMED_VALUES + 1)
                },
            )

    def test_oversized_claimed_value_is_a_grammar_violation(self) -> None:
        with pytest.raises(ValidationError):
            _receipt(
                _commit_envelope(),
                "artifact:commit",
                claimed_values={
                    "artifact:commit": "x" * (MAX_CLAIMED_VALUE_LENGTH + 1)
                },
            )

    def test_overlong_claimed_value_key_is_a_grammar_violation(self) -> None:
        with pytest.raises(ValidationError):
            _receipt(
                _commit_envelope(),
                "artifact:commit",
                claimed_values={"x" * (MAX_REQUIREMENT_ID_LENGTH + 1): "v"},
            )

    def test_non_string_claimed_value_is_a_grammar_violation(self) -> None:
        with pytest.raises(ValidationError):
            CompletionReceipt.model_validate(
                _payload(claimed_values={"artifact:commit": 12})
            )


class TestClaimedValuesRawPayloadClassification:
    """A raw payload's claimed-value violations classify as disclosure facts."""

    def test_valid_claimed_values_classify_normally(self) -> None:
        classification = classify_completion(
            mandate=_commit_envelope(),
            receipt=_payload(claimed_values={"artifact:commit": FULL_HASH}),
            declared_completed=True,
        )
        assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
        assert classification.disclosure_completeness is (
            CompletionDisclosure.DISCLOSURE_COMPLETE
        )
        assert classification.rejection is None

    def test_unknown_claimed_value_key_is_requirement_id_unknown(self) -> None:
        classification = classify_completion(
            mandate=_commit_envelope(),
            receipt=_payload(claimed_values={"artifact:other": FULL_HASH}),
            declared_completed=True,
        )
        assert (
            classification.rejection is CompletionRejectionReason.REQUIREMENT_ID_UNKNOWN
        )
        assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
        assert classification.disclosure_completeness is (
            CompletionDisclosure.DISCLOSURE_INCOMPLETE
        )

    def test_oversized_claimed_values_is_envelope_oversized(self) -> None:
        classification = classify_completion(
            mandate=_commit_envelope(),
            receipt=_payload(
                claimed_values={f"req-{index}": "v" for index in range(17)}
            ),
            declared_completed=True,
        )
        assert classification.rejection is CompletionRejectionReason.ENVELOPE_OVERSIZED
        assert classification.disclosure_completeness is (
            CompletionDisclosure.DISCLOSURE_INCOMPLETE
        )

    def test_oversized_claimed_value_is_envelope_oversized(self) -> None:
        classification = classify_completion(
            mandate=_commit_envelope(),
            receipt=_payload(
                claimed_values={"artifact:commit": "x" * (MAX_CLAIMED_VALUE_LENGTH + 1)}
            ),
            declared_completed=True,
        )
        assert classification.rejection is CompletionRejectionReason.ENVELOPE_OVERSIZED

    def test_non_mapping_claimed_values_is_envelope_malformed(self) -> None:
        classification = classify_completion(
            mandate=_commit_envelope(),
            receipt=_payload(claimed_values=["artifact:commit"]),
            declared_completed=True,
        )
        assert classification.rejection is CompletionRejectionReason.ENVELOPE_MALFORMED

    def test_non_string_claimed_value_is_envelope_malformed(self) -> None:
        classification = classify_completion(
            mandate=_commit_envelope(),
            receipt=_payload(claimed_values={"artifact:commit": 12}),
            declared_completed=True,
        )
        assert classification.rejection is CompletionRejectionReason.ENVELOPE_MALFORMED

    def test_claim_without_values_stays_unverified_never_a_fault(self) -> None:
        # Fail-open: the artifact audit resolves "no commit hash to check"
        # (check-context) — the claim is unverified, never contradicted, and
        # no rejection is attributed.
        classification = classify_completion(
            mandate=_commit_envelope(), receipt=_payload(), declared_completed=True
        )
        assert classification.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
        assert classification.rejection is None
        assert classification.disclosure_completeness is (
            CompletionDisclosure.DISCLOSURE_COMPLETE
        )


class TestClaimedValuesFromReceipt:
    """The derivation surface the composition consumes."""

    def test_none_claim_yields_the_empty_surface(self) -> None:
        assert claimed_values_from_receipt(None) == {}

    def test_raw_payload_contributes_only_well_formed_pairs(self) -> None:
        raw = _payload(
            claimed_values={"artifact:commit": FULL_HASH, "bad": 12, 13: "x"}
        )
        assert claimed_values_from_receipt(raw) == {"artifact:commit": FULL_HASH}

    def test_raw_payload_without_values_yields_the_empty_surface(self) -> None:
        assert claimed_values_from_receipt(_payload()) == {}

    def test_non_mapping_claimed_values_yields_the_empty_surface(self) -> None:
        assert claimed_values_from_receipt(_payload(claimed_values="nope")) == {}

    def test_unrelated_object_yields_the_empty_surface(self) -> None:
        assert claimed_values_from_receipt({"marker": "other"}) == {}
