"""The C4/C5 completion-check audit (P9).

This module owns the audit of acceptance-check claims: the C4 claim-
fidelity rule for check-pass claims against the frozen completion contract,
and the C5 evidence rule that classifies one executed check's observed
result against the per-run completion chronology.

C4 — claim fidelity (the admission rule). A worker's check-pass claim is
satisfiable only when BOTH bind:

- the claim names the CONTRACT-BOUND requirement ID — the requirement must
  be present in the contract's envelope, not merely a member of the closed
  template vocabulary (closed-vocabulary membership alone is insufficient);
- the claim's command is the requirement's CANONICAL command — the argv the
  acceptance runner derives for the requirement's bound template, not just
  a command the worker happened to run.

An unrelated check (a ``git_status`` invocation) can never satisfy a
required module check, however plausible its command reads.

C5 — evidence (the classification rule). One executed check's observed
result classifies against the chronology as follows:

- exit-zero evidence counts only AFTER the last observed mutation, and
  only for a check the chronology ties to the worker's own invocation: a
  final-state check (no collection-time witness tying it to the worker)
  never credits the worker — check-claim credit belongs to the C4
  receipt path (amendment 14);
- a failed check (any status other than PASSED) paired with a completed
  envelope classifies CONTRADICTED — unless the check observes only the
  FINAL STATE under a root shared with the parent, where an external
  write between the worker's evidence and the finalization is possible:
  the failed check is then check-context evidence (UNVERIFIED, the
  external mutation noted), never a contradiction (round-2 amendment 2,
  amendments 12 and 14);
- a failed or unavailable check with a partial or blocked run classifies
  INFORMATIONAL — never a contradiction (the run never claimed full
  completion, and the honesty-economics invariant keeps harness-side
  check failures informational);
- there are NO text ``"passed"`` heuristics: stdout text never credits a
  check, only the typed status and exit code do;
- a managed-job startup observation (a receipt without a completed result
  event) is not check evidence and never counts;
- a shell-less seat (one without the shell tool) is exempt from the check
  admission policy entirely — no check can be demanded of it, and its
  claims resolve exempt rather than denied.

Documented residual: the audit verifies execution and predicate, not
semantic relevance. A ``no_change`` completion whose named check is vacuous
stays UNVERIFIED — relevance is the parent's discriminating recheck before
commit/merge/publish, never this audit's.

Line anchors are re-derived from the on-disk bash tool: FailedShellCommand
is raised at chartreux/core/tools/builtins/bash.py:213 (inside
completed_shell_result, :195-231) and at bash.py:313 (inside
_build_timeout_error, :300-319); the CapturedShellResult exit_code field
lives at bash.py:109. The plan's bash.py:130-149/:107 numbers were stale
echoes of a superseded revision.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
import shlex
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from chartreux.core.completion_checks import (
    CompletionCheckOutcome,
    CompletionCheckRunner,
    CompletionCheckStatus,
    CompletionCheckTemplate,
)
from chartreux.core.completion_contract import CompletionEnvelope, CompletionRequirement
from chartreux.core.completion_evidence import PerRunCompletionChronology

__all__ = [
    "C4Claim",
    "C4Verdict",
    "CheckEvidenceClassification",
    "CheckEvidenceDisposition",
    "RunOutcome",
    "admit_check_claim",
    "classify_check_evidence",
    "last_mutation_sequence",
    "requirement_canonical_command",
]


class C4Verdict(StrEnum):
    """The C4 admission verdicts for one check-pass claim."""

    SATISFIED = "satisfied"
    REJECTED = "rejected"
    UNVERIFIED = "unverified"
    EXEMPT = "exempt"


class CheckEvidenceDisposition(StrEnum):
    """The C5 evidence dispositions for one executed check."""

    WORKER_CREDITED = "worker_credited"
    CONTRADICTED = "contradicted"
    INFORMATIONAL = "informational"
    UNAVAILABLE = "unavailable"
    UNVERIFIED = "unverified"


#: The declared run outcome a partial/blocked informational stamp pairs with.
RunOutcome = Literal["completed", "partial", "blocked"]


class C4Claim(BaseModel):
    """One worker check-pass claim under C4 adjudication.

    The claim names the requirement it satisfies and the command the worker
    executed for it. ``chronology_occurrence_id``, when present, ties the
    claim to the run-scoped occurrence ID the harness minted for the
    invocation the claim reports on.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str
    template: CompletionCheckTemplate | None = None
    claimed_command: str | None = None
    chronology_occurrence_id: str | None = None


class CheckEvidenceClassification(BaseModel):
    """The C5 classification of one executed check's observed result."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requirement_id: str | None = None
    disposition: CheckEvidenceDisposition
    reason: str
    status: CompletionCheckStatus | None = None
    exit_code: int | None = None
    evidence_sequence: int | None = None
    last_mutation_sequence: int | None = None


def bound_requirement(
    envelope: CompletionEnvelope | None, requirement_id: str
) -> CompletionRequirement | None:
    """The contract-bound requirement with one ID, or None when unbound.

    A None envelope binds nothing, so no requirement can resolve.
    """
    if envelope is None:
        return None
    for requirement in envelope.requirements:
        if requirement.requirement_id == requirement_id:
            return requirement
    return None


def requirement_canonical_command(
    envelope: CompletionEnvelope | None,
    requirement_id: str,
    *,
    argv: str | None = None,
    runner: CompletionCheckRunner | None = None,
) -> str | None:
    """C5: the requirement-ID -> canonical command mapping.

    The canonical command is the acceptance runner's argv for the
    requirement's bound check template: the caller-supplied ``argv`` when
    the harness hands one over, else the runner-derived argv, else ``None``
    when the requirement is unbound, artifact-only, or the runner cannot
    derive a command for it.
    """
    requirement = bound_requirement(envelope, requirement_id)
    if requirement is None or requirement.template is None:
        return None
    derived = (
        shlex.join(runner.check_argv(requirement.template))
        if runner is not None
        else None
    )
    if argv is not None:
        # The harness-supplied argv is the canonical command only when it
        # matches what the runner derives for the bound template; a
        # mismatch means the claimed command is not canonical.
        if derived is not None and argv != derived:
            return None
        return argv
    return derived


def admit_check_claim(
    *,
    claim: C4Claim,
    envelope: CompletionEnvelope | None,
    runner: CompletionCheckRunner | None = None,
    seat_has_shell: bool = True,
) -> C4Verdict:
    """C4: adjudicate one check-pass claim against the frozen contract.

    The claim satisfies the admission policy only when its requirement ID
    is contract-bound AND its claimed command is the requirement's
    canonical command. A shell-less seat is exempt from the policy
    entirely: no shell means no check obligation, so the claim is exempt
    rather than denied or verified.
    """
    if not seat_has_shell:
        return C4Verdict.EXEMPT
    requirement = bound_requirement(envelope, claim.requirement_id)
    if requirement is None:
        return C4Verdict.UNVERIFIED
    canonical = requirement_canonical_command(
        envelope, claim.requirement_id, argv=claim.claimed_command, runner=runner
    )
    if canonical is None or claim.claimed_command != canonical:
        return C4Verdict.REJECTED
    return C4Verdict.SATISFIED


def last_mutation_sequence(writes: Iterable[tuple[int, Any]]) -> int | None:
    """The chronology sequence of the last observed entry.

    Each item is a ``(sequence, payload)`` pair from the run's chronology;
    the result is the tail position of that chronology — the sequence the
    caller compares the check's evidence sequence against. Returns ``None``
    when the chronology is empty.
    """
    sequence: int | None = None
    for item_sequence, _payload in writes:
        sequence = item_sequence
    return sequence


def classify_check_evidence(
    *,
    outcome: CompletionCheckOutcome | None = None,
    requirement_id: str | None = None,
    envelope: CompletionEnvelope | None = None,
    chronology: PerRunCompletionChronology | None = None,
    occurrence_id: str | None = None,
    last_mutation_sequence: int | None = None,
    envelope_complete: bool | None = None,
    run_outcome: RunOutcome | None = None,
    seat_has_shell: bool = True,
    observes_shared_root: bool = False,
) -> CheckEvidenceClassification:
    """C5: classify one executed check against the per-run chronology.

    The classification order is the brief's truth table, most specific
    first:

    1. No completed outcome (a managed-job startup observation, a denied
       check that never spawned) is UNAVAILABLE — never evidence.
    2. Exit-zero evidence AFTER the last mutation, tied to the worker's
       own invocation, is WORKER_CREDITED.
    3. A failed check with a completed envelope is CONTRADICTED — unless
       the check observes only the final state under a root shared with
       the parent (``observes_shared_root``), where an external write is
       possible: UNVERIFIED (check-context, the external mutation noted),
       never a contradiction.
    4. A failed or unavailable check with a partial/blocked run is
       INFORMATIONAL.
    5. Everything else — including a passing final-state check with no
       collection-time witness — is UNVERIFIED: final-state checks never
       credit the worker (amendment 14).
    """
    evidence_sequence: int | None = None
    if chronology is not None and occurrence_id is not None:
        entry = chronology.entry(occurrence_id)
        if entry is not None:
            evidence_sequence = entry.sequence

    if outcome is None:
        # No completed outcome — a managed-job startup observation or a
        # check that never ran. Startup is not evidence: the disposition
        # is unavailable regardless of what the chronology collected.
        return CheckEvidenceClassification(
            requirement_id=requirement_id,
            disposition=CheckEvidenceDisposition.UNAVAILABLE,
            reason="no completed check result for this occurrence",
            evidence_sequence=evidence_sequence,
            last_mutation_sequence=last_mutation_sequence,
        )

    check_passed = (
        outcome.status is CompletionCheckStatus.PASSED and outcome.exit_code == 0
    )

    if (
        check_passed
        and evidence_sequence is not None
        and last_mutation_sequence is not None
        and evidence_sequence > last_mutation_sequence
    ):
        return CheckEvidenceClassification(
            requirement_id=requirement_id,
            disposition=CheckEvidenceDisposition.WORKER_CREDITED,
            reason="exit 0 after the last observed mutation",
            status=outcome.status,
            exit_code=outcome.exit_code,
            evidence_sequence=evidence_sequence,
            last_mutation_sequence=last_mutation_sequence,
        )

    if not check_passed and run_outcome in {"partial", "blocked"}:
        return CheckEvidenceClassification(
            requirement_id=requirement_id,
            disposition=CheckEvidenceDisposition.INFORMATIONAL,
            reason="check failed or was unavailable in a partial/blocked run",
            status=outcome.status,
            exit_code=outcome.exit_code,
            evidence_sequence=evidence_sequence,
            last_mutation_sequence=last_mutation_sequence,
        )
    if not check_passed and envelope_complete and observes_shared_root:
        # The check observes only the FINAL STATE under a root shared with
        # the parent, so an external write between the worker's evidence
        # and the finalization is possible (round-2 amendment 2,
        # amendments 12 and 14). The failed check is check-context
        # evidence — UNVERIFIED with the external mutation noted, never a
        # worker contradiction.
        return CheckEvidenceClassification(
            requirement_id=requirement_id,
            disposition=CheckEvidenceDisposition.UNVERIFIED,
            reason=(
                "check failed over final state in a root shared with the "
                "parent; check-context evidence with the external mutation "
                "noted, never a contradiction"
            ),
            status=outcome.status,
            exit_code=outcome.exit_code,
            evidence_sequence=evidence_sequence,
            last_mutation_sequence=last_mutation_sequence,
        )
    if not check_passed and envelope_complete:
        return CheckEvidenceClassification(
            requirement_id=requirement_id,
            disposition=CheckEvidenceDisposition.CONTRADICTED,
            reason="check failed while the envelope reported completion",
            status=outcome.status,
            exit_code=outcome.exit_code,
            evidence_sequence=evidence_sequence,
            last_mutation_sequence=last_mutation_sequence,
        )
    return CheckEvidenceClassification(
        requirement_id=requirement_id,
        disposition=CheckEvidenceDisposition.UNVERIFIED,
        reason="check evidence does not resolve to a typed disposition",
        status=outcome.status,
        exit_code=outcome.exit_code,
        evidence_sequence=evidence_sequence,
        last_mutation_sequence=last_mutation_sequence,
    )
