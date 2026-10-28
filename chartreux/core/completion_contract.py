"""The typed completion contract (P1).

This module owns the closed vocabularies, the versioned envelope/receipt
grammar, and the audit truth table for the completion-contract work:

- :class:`CompletionCheckTemplate` — the closed vocabulary of acceptance-check
  templates a contract may mandate;
- :class:`CompletionArtifactKind` — the closed vocabulary of artifact kinds a
  contract requirement may demand;
- :class:`CompletionEnvelope` / :class:`CompletionReceipt` — the strict,
  versioned grammar binding a worker's claim to a contract's mandates;
- :class:`CompletionAuditClassification` — the truth table's output: five
  separate axes (claim fidelity, disclosure completeness, mandate
  satisfaction, declared outcome, primary runtime failure), classified by
  :func:`classify_completion`.

Classification rules pinned here (P1): malformed/duplicate/oversized
envelopes are disclosure failures, never escalation; verifier-infrastructure
failures are never worker faults (they force the claim fidelity to
unverified, never contradicted); ``None`` mandates disable only themselves
(fail-open) and never produce a rejection. :func:`classify_completion`
accepts both the typed grammar models and the raw claim payloads a worker
emits, so a grammar violation is classified by the truth table itself
instead of escaping as an exception.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

__all__ = [
    "COMPLETION_CONTRACT_SCHEMA_VERSION",
    "COMPLETION_ENVELOPE_MARKER",
    "CompletionArtifactKind",
    "CompletionAuditClassification",
    "CompletionCheckTemplate",
    "CompletionEnvelope",
    "CompletionEvidence",
    "CompletionReceipt",
    "CompletionRejectionReason",
    "CompletionRequirement",
    "classify_completion",
]

#: The schema version stamped on every envelope and receipt this module
#: accepts. Bumping it invalidates all previously issued contracts.
COMPLETION_CONTRACT_SCHEMA_VERSION = 1

#: The marker key value every envelope and receipt of this grammar carries.
COMPLETION_ENVELOPE_MARKER = "completion_envelope_v1"

#: Upper bounds of the envelope grammar. A requirement payload exceeding any
#: of them is a grammar violation, not a worker fault.
MAX_ENVELOPE_REQUIREMENTS = 16
MAX_SATISFIED_REQUIREMENTS = 16
MAX_REQUIREMENT_ID_LENGTH = 64


class CompletionCheckTemplate(StrEnum):
    """The closed vocabulary of acceptance-check templates."""

    PYTEST = "pytest"
    RUFF_CHECK = "ruff_check"
    PYRIGHT = "pyright"
    GIT_HEAD = "git_head"
    GIT_STATUS = "git_status"
    GIT_DIFF_STAT = "git_diff_stat"


class CompletionArtifactKind(StrEnum):
    """The closed vocabulary of artifact kinds a requirement may demand."""

    COMMIT = "commit"
    TOUCHED_PATH = "touched_path"
    CHECK_PASS = "check_pass"
    ENVELOPE = "envelope"


class CompletionRequirement(BaseModel):
    """One contract requirement: a check template or an artifact kind."""

    model_config = ConfigDict(extra="forbid")

    requirement_id: str = Field(min_length=1, max_length=MAX_REQUIREMENT_ID_LENGTH)
    template: CompletionCheckTemplate | None = None
    artifact: CompletionArtifactKind | None = None

    @model_validator(mode="after")
    def _require_exactly_one_binding(self) -> CompletionRequirement:
        if (self.template is None) == (self.artifact is None):
            raise ValueError(
                "a completion requirement binds exactly one of a check "
                "template or an artifact kind"
            )
        return self


class CompletionEnvelope(BaseModel):
    """The mandate side of the grammar: the requirements the contract issues.

    Unknown fields are rejected (``extra="forbid"``) and duplicate
    requirement ids are a grammar violation — both at construction time.
    """

    model_config = ConfigDict(extra="forbid")

    marker: Literal["completion_envelope_v1"]
    schema_version: int = Field(ge=1)
    requirements: tuple[CompletionRequirement, ...] = Field(
        min_length=1, max_length=MAX_ENVELOPE_REQUIREMENTS
    )

    @model_validator(mode="after")
    def _reject_duplicate_requirement_ids(self) -> CompletionEnvelope:
        ids = [requirement.requirement_id for requirement in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate requirement id in the completion envelope")
        return self


class CompletionReceipt(BaseModel):
    """The claim side of the grammar: the requirement ids the worker
    reports satisfied against one issued envelope.

    Unknown fields are rejected (``extra="forbid"``) and duplicate
    requirement ids are a grammar violation — both at construction time.
    """

    model_config = ConfigDict(extra="forbid")

    marker: Literal["completion_envelope_v1"]
    schema_version: int = Field(ge=1)
    envelope: CompletionEnvelope | None = None
    satisfied_requirement_ids: tuple[str, ...] = Field(
        default=(), max_length=MAX_SATISFIED_REQUIREMENTS
    )

    @model_validator(mode="after")
    def _satisfied_ids_resolve_into_the_envelope(self) -> CompletionReceipt:
        if self.envelope is None:
            return self
        known = {
            requirement.requirement_id for requirement in self.envelope.requirements
        }
        unknown = [
            requirement_id
            for requirement_id in self.satisfied_requirement_ids
            if requirement_id not in known
        ]
        if unknown:
            raise ValueError(
                "satisfied requirement ids unknown to the bound envelope: "
                f"{sorted(unknown)}"
            )
        return self

    @model_validator(mode="after")
    def _reject_duplicate_satisfied_ids(self) -> CompletionReceipt:
        if len(self.satisfied_requirement_ids) != len(
            set(self.satisfied_requirement_ids)
        ):
            raise ValueError("duplicate satisfied requirement id in the receipt")
        return self


class CompletionEvidence(BaseModel):
    """The observed runtime signals the truth table classifies.

    ``requirement_ids_observed`` names the requirement ids for which the
    runtime observed direct satisfying evidence (an executed check, a
    produced artifact); ``contradicted_requirement_ids`` names the ids for
    which the runtime observed DIRECT contradictory evidence (a contract-
    bound check that executed and failed, a paired same-command conflict) —
    final-state disagreement alone never lands here.
    ``runtime_failure`` is the primary runtime failure the verifier itself
    observed, or ``None`` when the runtime stayed healthy.
    """

    model_config = ConfigDict(extra="forbid")

    requirement_ids_observed: tuple[str, ...] = ()
    contradicted_requirement_ids: tuple[str, ...] = ()
    runtime_failure: (
        Literal["denied", "timeout", "unavailable", "unrunnable"] | None
    ) = None


class CompletionClaimFidelity(StrEnum):
    """The claim-fidelity axis: what the claim's evidence actually shows."""

    CLAIM_MATCHED = "claim_matched"
    CLAIM_UNVERIFIED = "claim_unverified"
    CLAIM_CONTRADICTED = "claim_contradicted"


class CompletionDisclosure(StrEnum):
    """The disclosure-completeness axis."""

    DISCLOSURE_COMPLETE = "disclosure_complete"
    DISCLOSURE_INCOMPLETE = "disclosure_incomplete"


class CompletionMandateSatisfaction(StrEnum):
    """The mandate-satisfaction axis."""

    MANDATE_SATISFIED = "mandate_satisfied"
    MANDATE_UNADDRESSED = "mandate_unaddressed"
    MANDATE_DISABLED = "mandate_disabled"


class CompletionDeclaredOutcome(StrEnum):
    """The declared-outcome axis: what the worker itself declared."""

    OUTCOME_COMPLETED = "outcome_completed"
    OUTCOME_INCOMPLETE = "outcome_incomplete"


class CompletionRuntimeFailure(StrEnum):
    """The primary-runtime-failure axis: the verifier's own health."""

    RUNTIME_HEALTHY = "runtime_healthy"
    RUNTIME_DENIED = "runtime_denied"
    RUNTIME_TIMEOUT = "runtime_timeout"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    RUNTIME_UNRUNNABLE = "runtime_unrunnable"


class CompletionRejectionReason(StrEnum):
    """The typed rejection vocabulary the P11 failure class consumes.

    Grammar violations (``envelope_malformed``, ``envelope_duplicate``,
    ``envelope_oversized``, ``requirement_id_unknown``,
    ``requirement_id_duplicate``) are grammar facts, never worker faults;
    only ``claim_contradicted`` is a worker-side disposition.
    """

    ENVELOPE_MALFORMED = "envelope_malformed"
    ENVELOPE_DUPLICATE = "envelope_duplicate"
    ENVELOPE_UNKNOWN_FIELD = "envelope_unknown_field"
    ENVELOPE_OVERSIZED = "envelope_oversized"
    REQUIREMENT_ID_UNKNOWN = "requirement_id_unknown"
    REQUIREMENT_ID_DUPLICATE = "requirement_id_duplicate"
    RECEIPT_UNVERIFIABLE = "receipt_unverifiable"
    CLAIM_CONTRADICTED = "claim_contradicted"


class CompletionAuditClassification(BaseModel):
    """The truth table's output: one typed value per audit axis.

    The classification is the versioned audit stamp: ``schema_version`` pins
    the contract grammar the classification was produced under, and
    ``rejection`` carries the typed rejection vocabulary the P11 failure
    class consumes. Only ``claim_contradicted`` is a worker-side
    disposition; every grammar rejection is a disclosure fact.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(default=COMPLETION_CONTRACT_SCHEMA_VERSION, ge=1)
    claim_fidelity: CompletionClaimFidelity
    disclosure_completeness: CompletionDisclosure
    mandate_satisfaction: CompletionMandateSatisfaction
    declared_outcome: CompletionDeclaredOutcome
    primary_runtime_failure: CompletionRuntimeFailure
    rejection: CompletionRejectionReason | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


#: The evidence field's vocabulary maps onto the axis enum by raw value:
#: the evidence surface carries the verifier's own failure words, the
#: classification axis carries the ``runtime_*``-prefixed members.
_EVIDENCE_FAILURE_TO_AXIS = {
    "denied": CompletionRuntimeFailure.RUNTIME_DENIED,
    "timeout": CompletionRuntimeFailure.RUNTIME_TIMEOUT,
    "unavailable": CompletionRuntimeFailure.RUNTIME_UNAVAILABLE,
    "unrunnable": CompletionRuntimeFailure.RUNTIME_UNRUNNABLE,
}

_TEMPLATE_VALUES = frozenset(member.value for member in CompletionCheckTemplate)
_ARTIFACT_VALUES = frozenset(member.value for member in CompletionArtifactKind)


def _schema_version_is_valid(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _envelope_grammar_rejection(  # noqa: PLR0911 - one early exit per guard
    payload: Any,
) -> CompletionRejectionReason | None:
    """Name the grammar violation in one raw envelope payload.

    Returns the specific rejection reason, or ``None`` when the payload
    satisfies the grammar. The pydantic models remain the strict grammar of
    record; this pre-check only names the violation so the truth table can
    classify a malformed claim instead of raising.
    """
    if not isinstance(payload, Mapping):
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if set(payload) - set(CompletionEnvelope.model_fields):
        return CompletionRejectionReason.ENVELOPE_UNKNOWN_FIELD
    if payload.get("marker") != COMPLETION_ENVELOPE_MARKER:
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if not _schema_version_is_valid(payload.get("schema_version")):
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    requirements = payload.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if len(requirements) > MAX_ENVELOPE_REQUIREMENTS:
        return CompletionRejectionReason.ENVELOPE_OVERSIZED
    requirement_fields = set(CompletionRequirement.model_fields)
    requirement_ids: list[str] = []
    for requirement in requirements:
        if not isinstance(requirement, Mapping):
            return CompletionRejectionReason.ENVELOPE_MALFORMED
        if set(requirement) - requirement_fields:
            return CompletionRejectionReason.ENVELOPE_UNKNOWN_FIELD
        requirement_id = requirement.get("requirement_id")
        if not isinstance(requirement_id, str) or not requirement_id:
            return CompletionRejectionReason.ENVELOPE_MALFORMED
        if len(requirement_id) > MAX_REQUIREMENT_ID_LENGTH:
            return CompletionRejectionReason.ENVELOPE_OVERSIZED
        template = requirement.get("template")
        artifact = requirement.get("artifact")
        if (template is None) == (artifact is None):
            return CompletionRejectionReason.ENVELOPE_MALFORMED
        if template is not None and template not in _TEMPLATE_VALUES:
            return CompletionRejectionReason.ENVELOPE_MALFORMED
        if artifact is not None and artifact not in _ARTIFACT_VALUES:
            return CompletionRejectionReason.ENVELOPE_MALFORMED
        requirement_ids.append(requirement_id)
    if len(requirement_ids) != len(set(requirement_ids)):
        return CompletionRejectionReason.REQUIREMENT_ID_DUPLICATE
    return None


def _receipt_grammar_rejection(  # noqa: PLR0911 - one early exit per guard
    payload: Any,
) -> CompletionRejectionReason | None:
    """Name the grammar violation in one raw receipt payload.

    See :func:`_envelope_grammar_rejection`.
    """
    if not isinstance(payload, Mapping):
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if set(payload) - set(CompletionReceipt.model_fields):
        return CompletionRejectionReason.ENVELOPE_UNKNOWN_FIELD
    if payload.get("marker") != COMPLETION_ENVELOPE_MARKER:
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if not _schema_version_is_valid(payload.get("schema_version")):
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    envelope = payload.get("envelope")
    if envelope is not None:
        reason = _envelope_grammar_rejection(envelope)
        if reason is not None:
            return reason
    satisfied = payload.get("satisfied_requirement_ids", ())
    if not isinstance(satisfied, (list, tuple)):
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if len(satisfied) > MAX_SATISFIED_REQUIREMENTS:
        return CompletionRejectionReason.ENVELOPE_OVERSIZED
    if any(not isinstance(item, str) or not item for item in satisfied):
        return CompletionRejectionReason.ENVELOPE_MALFORMED
    if len(satisfied) != len(set(satisfied)):
        return CompletionRejectionReason.REQUIREMENT_ID_DUPLICATE
    if envelope is not None:
        try:
            bound = CompletionEnvelope.model_validate(envelope)
        except ValidationError:
            return CompletionRejectionReason.ENVELOPE_MALFORMED
        known = {requirement.requirement_id for requirement in bound.requirements}
        if any(satisfied_id not in known for satisfied_id in satisfied):
            return CompletionRejectionReason.REQUIREMENT_ID_UNKNOWN
    return None


def _normalize_evidence(
    evidence: CompletionEvidence | Mapping[str, Any] | None,
) -> CompletionEvidence:
    if evidence is None:
        return CompletionEvidence()
    if isinstance(evidence, CompletionEvidence):
        return evidence
    try:
        return CompletionEvidence.model_validate(evidence)
    except ValidationError:
        # The evidence surface is harness-produced; a malformed payload is a
        # harness-side defect, never a worker fault.
        return CompletionEvidence()


def classify_completion(  # noqa: PLR0911 - one classification per truth-table branch
    *,
    mandate: CompletionEnvelope | Mapping[str, Any] | None,
    receipt: CompletionReceipt | Mapping[str, Any] | None,
    declared_completed: bool,
    evidence: CompletionEvidence | Mapping[str, Any] | None = None,
) -> CompletionAuditClassification:
    """Classify one completion against its contract on the five axes.

    ``mandate`` and ``receipt`` accept the typed grammar models or the raw
    payloads a worker emits; a grammar violation classifies as a disclosure
    failure (never an escalation) instead of raising. The classification
    never escalates on its own: a worker-side contradiction is recorded as
    a typed rejection the P11 policy consumes, and a verifier-
    infrastructure failure forces the claim fidelity to unverified rather
    than attributing the failure to the worker.
    """
    observed_evidence = _normalize_evidence(evidence)
    primary_runtime_failure = (
        CompletionRuntimeFailure.RUNTIME_HEALTHY
        if observed_evidence.runtime_failure is None
        else _EVIDENCE_FAILURE_TO_AXIS[observed_evidence.runtime_failure]
    )
    infrastructure_failure = primary_runtime_failure is not (
        CompletionRuntimeFailure.RUNTIME_HEALTHY
    )
    declared_outcome = (
        CompletionDeclaredOutcome.OUTCOME_COMPLETED
        if declared_completed
        else CompletionDeclaredOutcome.OUTCOME_INCOMPLETE
    )
    # A rejection is only ever attributed while the verifier infrastructure
    # is healthy: harness-side failures are informational stamps, never
    # worker misconduct.
    attribute = not infrastructure_failure

    def _classify(
        *,
        claim_fidelity: CompletionClaimFidelity,
        disclosure_completeness: CompletionDisclosure,
        mandate_satisfaction: CompletionMandateSatisfaction,
        rejection: CompletionRejectionReason | None,
    ) -> CompletionAuditClassification:
        return CompletionAuditClassification(
            claim_fidelity=claim_fidelity,
            disclosure_completeness=disclosure_completeness,
            mandate_satisfaction=mandate_satisfaction,
            declared_outcome=declared_outcome,
            primary_runtime_failure=primary_runtime_failure,
            rejection=rejection if attribute else None,
        )

    if mandate is None:
        # Fail-open: a None mandate disables only itself and never produces
        # a rejection.
        return _classify(
            claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
            disclosure_completeness=CompletionDisclosure.DISCLOSURE_COMPLETE,
            mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_DISABLED,
            rejection=None,
        )

    envelope: CompletionEnvelope
    if isinstance(mandate, Mapping):
        reason = _envelope_grammar_rejection(mandate)
        if reason is not None:
            # A malformed/duplicate/oversized envelope is a disclosure
            # failure, never an escalation.
            return _classify(
                claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
                disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
                mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
                rejection=reason,
            )
        try:
            envelope = CompletionEnvelope.model_validate(mandate)
        except ValidationError:
            return _classify(
                claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
                disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
                mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
                rejection=CompletionRejectionReason.ENVELOPE_MALFORMED,
            )
    elif isinstance(mandate, CompletionEnvelope):
        envelope = mandate
    else:
        return _classify(
            claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
            disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
            mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
            rejection=CompletionRejectionReason.ENVELOPE_MALFORMED,
        )

    if receipt is None:
        return _classify(
            claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
            disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
            mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
            rejection=CompletionRejectionReason.RECEIPT_UNVERIFIABLE,
        )

    claim: CompletionReceipt
    if isinstance(receipt, Mapping):
        reason = _receipt_grammar_rejection(receipt)
        if reason is not None:
            return _classify(
                claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
                disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
                mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
                rejection=reason,
            )
        try:
            claim = CompletionReceipt.model_validate(receipt)
        except ValidationError:
            return _classify(
                claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
                disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
                mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
                rejection=CompletionRejectionReason.ENVELOPE_MALFORMED,
            )
    elif isinstance(receipt, CompletionReceipt):
        claim = receipt
    else:
        return _classify(
            claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
            disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
            mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
            rejection=CompletionRejectionReason.ENVELOPE_MALFORMED,
        )

    required = {requirement.requirement_id for requirement in envelope.requirements}
    satisfied = set(claim.satisfied_requirement_ids)
    unknown = sorted(satisfied - required)
    if unknown:
        return _classify(
            claim_fidelity=CompletionClaimFidelity.CLAIM_UNVERIFIED,
            disclosure_completeness=CompletionDisclosure.DISCLOSURE_INCOMPLETE,
            mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_UNADDRESSED,
            rejection=CompletionRejectionReason.REQUIREMENT_ID_UNKNOWN,
        )

    unaddressed = sorted(required - satisfied)
    observed = set(observed_evidence.requirement_ids_observed)
    # A contradiction is a DIRECT, paired conflict the runtime observed for
    # a requirement the worker claimed satisfied; final-state disagreement
    # alone never reaches ``contradicted_requirement_ids``. A verifier-
    # infrastructure failure never contradicts a worker.
    contradicted = (
        satisfied & set(observed_evidence.contradicted_requirement_ids)
        if attribute
        else set()
    )
    if contradicted:
        return _classify(
            claim_fidelity=CompletionClaimFidelity.CLAIM_CONTRADICTED,
            disclosure_completeness=(
                CompletionDisclosure.DISCLOSURE_COMPLETE
                if not unaddressed
                else CompletionDisclosure.DISCLOSURE_INCOMPLETE
            ),
            mandate_satisfaction=(
                CompletionMandateSatisfaction.MANDATE_SATISFIED
                if not unaddressed
                else CompletionMandateSatisfaction.MANDATE_UNADDRESSED
            ),
            rejection=CompletionRejectionReason.CLAIM_CONTRADICTED,
        )

    claim_fidelity = (
        CompletionClaimFidelity.CLAIM_MATCHED
        if required <= satisfied and required <= observed
        else CompletionClaimFidelity.CLAIM_UNVERIFIED
    )
    return _classify(
        claim_fidelity=claim_fidelity,
        disclosure_completeness=(
            CompletionDisclosure.DISCLOSURE_COMPLETE
            if not unaddressed
            else CompletionDisclosure.DISCLOSURE_INCOMPLETE
        ),
        mandate_satisfaction=(
            CompletionMandateSatisfaction.MANDATE_SATISFIED
            if not unaddressed
            else CompletionMandateSatisfaction.MANDATE_UNADDRESSED
        ),
        rejection=None,
    )
