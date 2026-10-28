"""P10: compose the completion audit at ``build_result``.

This module is the composition layer between the landed audit packages and
the result publication boundary. It consumes the audits the earlier
packages own — never re-implements them — and composes their findings onto
the P1 truth table (``chartreux.core.completion_contract.classify_completion``)
to produce the typed, versioned :class:`CompletionAuditClassification`
stamped on ``TaskResult.completion_audit``.

Composition inputs, by package:

- C1/C3 (P7, ``completion_audit_artifacts``): ``audit_artifacts`` resolves
  the claimed artifact requirements against the collected evidence
  (``ArtifactEvidence`` built from the accumulator's touched paths and
  path-keyed collection-time snapshots, the mutation-roots footprint, the
  commit hashes the chronology's git-command results carry, and the
  finalization's HEAD snapshots).
- C2 (P8, ``completion_audit_receipts``): ``audit_completion_receipts``
  resolves the envelope-linked receipt against the run's chronology; its
  ``ReceiptAuditResult.evidence()`` is the base of the P1 evidence
  surface.
- C4/C5 (P9, ``completion_audit_checks``): ``admit_check_claim``
  adjudicates the run's check-pass claims (one per satisfied,
  template-bound requirement the receipt audit paired to its originating
  invocation); ``classify_check_evidence`` classifies each check the
  once-only finalization executed.

Composition rules (binding):

- The audits' dispositions compose onto the P1 truth table's axes via
  ``classify_completion``; the composition never overrides a typed
  disposition with a weaker one.
- Stamp mode (the default) leaves behavior unchanged: the composed
  classification rides on the result and never blocks publication.
- Enforce mode blocks the submission when the composed classification
  carries the worker-side disposition (``claim_contradicted``) or a
  rejected check claim: the result builds with ``completed=False`` and a
  bounded, actionable rejection message. The message names the worker-side
  dispositions, capped at ``BLOCK_MESSAGE_MAX_PATHS`` items — never a rule
  id, a matched secret line, or a chronology dump (confidentiality
  preserved).
- Unavailable evidence never becomes a contradiction: an ``unavailable``
  finding (a truncated or missing snapshot, an evicted chronology entry,
  a HEAD probe that found no repository) composes to the evidence
  surface's ``unavailable`` runtime failure — a harness-side availability
  fact, never worker misconduct, and never replacement-worthy.
- A failed finalization check never contradicts the worker when the
  check's observed root is shared with the parent: the once-only
  finalization's checks observe only the FINAL STATE of the run's tree,
  and no collection-time witness ties that state to the worker, so an
  external write between the worker's evidence and the finalization is
  possible. The ``shared_roots`` parameter carries the roots shared with
  the parent; a failed check observing state under one of them composes
  as check-context evidence (UNVERIFIED, the external mutation noted),
  never CONTRADICTED — and a passing final-state check never credits the
  worker either (amendment 14; round-2 amendment 2, amendments 12/14).
- Repeated builds run no checks: the once-only finalization (P6a) is
  resolved by the caller from the accumulator's run-keyed cache and handed
  to the composition; a cache hit supplies the finalization evidence
  without re-running any check.

No chronology dump: nothing from ``PerRunCompletionChronology`` — entries,
receipts, events, snapshots' sampled bytes — reaches the composed message
or the parent-facing result; the chronology is consumed only through its
typed resolution surface.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from chartreux.core.completion_audit_artifacts import (
    ArtifactAuditDisposition,
    ArtifactAuditFinding,
    ArtifactEvidence,
    audit_artifacts,
    git_result_hashes,
)
from chartreux.core.completion_audit_checks import (
    C4Claim,
    C4Verdict,
    CheckEvidenceClassification,
    CheckEvidenceDisposition,
    admit_check_claim,
    classify_check_evidence,
    last_mutation_sequence,
    requirement_canonical_command,
)
from chartreux.core.completion_audit_receipts import (
    ReceiptAuditResult,
    ReceiptEvidenceDisposition,
    audit_completion_receipts,
)
from chartreux.core.completion_contract import (
    CompletionAuditClassification,
    CompletionEnvelope,
    CompletionEvidence,
    CompletionRejectionReason,
    classify_completion,
)
from chartreux.core.completion_evidence import PerRunCompletionChronology
from chartreux.core.subagent_submission import (
    BLOCK_MESSAGE_MAX_PATHS,
    bash_write_paths_from_result_event,
    touched_path_from_result_event,
)

if TYPE_CHECKING:
    from chartreux.core.subagents import SubagentRunAccumulator

__all__ = ["AuditMode", "CompletionAuditComposition", "compose_completion_audit"]

#: The audit composition mode: ``stamp`` (the default) only stamps the
#: composed classification onto the result; ``enforce`` additionally blocks
#: the submission on a worker-side disposition.
AuditMode = Literal["stamp", "enforce"]

#: The verifier's own failure words (P6a honesty economics), accepted as
#: the evidence surface's primary runtime failure.
_RUNTIME_FAILURE_WORDS = frozenset({"denied", "timeout", "unavailable", "unrunnable"})


@dataclass(frozen=True, slots=True)
class CompletionAuditComposition:
    """The composed audit evidence of one ``build_result`` call.

    ``classification`` is the P1 truth-table stamp for
    ``TaskResult.completion_audit``; ``artifact_findings``,
    ``receipt_result``, ``check_verdicts``, and ``check_classifications``
    carry the C1/C3, C2, C4, and C5 findings the classification composes.
    ``blocks_publication`` is true only in enforce mode with a worker-side
    disposition present.
    """

    classification: CompletionAuditClassification
    artifact_findings: tuple[ArtifactAuditFinding, ...] = ()
    receipt_result: ReceiptAuditResult | None = None
    check_verdicts: tuple[tuple[str, C4Verdict], ...] = ()
    check_classifications: tuple[tuple[str, CheckEvidenceClassification], ...] = ()
    evidence: CompletionEvidence = field(default_factory=CompletionEvidence)
    blocks_publication: bool = False
    accumulator: SubagentRunAccumulator | None = None

    @property
    def has_surface(self) -> bool:
        """Whether the composition carries a real audit surface.

        A composition with any real input — findings, a receipt result,
        check verdicts or classifications, an evidence surface, or an
        accumulator that observed touched paths, mutation roots, artifact
        snapshots, or chronology entries — has a surface and stamps the
        result. An inert composition (no contract, no claim, no collected
        evidence, no finalization, and an accumulator that observed
        nothing) stamps nothing: the result's ``completion_audit`` stays
        ``None`` — the P1 dump shape for runs composed without an audit
        surface. This is the stamp-mode default's behavior preservation,
        not a suppression.
        """
        if self.artifact_findings or self.receipt_result is not None:
            return True
        if self.check_verdicts or self.check_classifications:
            return True
        if (
            self.evidence.requirement_ids_observed
            or self.evidence.contradicted_requirement_ids
            or self.evidence.runtime_failure is not None
        ):
            return True
        return self._accumulator_observed_anything()

    def _accumulator_observed_anything(self) -> bool:
        accumulator = self.accumulator
        if accumulator is None:
            return False
        if accumulator._touched_paths or accumulator._mutation_roots:
            return True
        if accumulator._artifact_snapshots:
            return True
        chronology = accumulator._chronology
        return bool(chronology is not None and chronology._entries)

    def rejection_message(self) -> str | None:
        """The bounded, actionable rejection message for enforce mode.

        Names the worker-side dispositions only — the contradicted
        requirement ids and the rejected check claims — capped at
        ``BLOCK_MESSAGE_MAX_PATHS`` items. Carries no rule id, no matched
        secret line, and no chronology content. ``None`` when the
        composition blocks nothing.
        """
        if not self.blocks_publication:
            return None
        named = [
            f"{requirement_id}: contradicted"
            for requirement_id in self.evidence.contradicted_requirement_ids
        ]
        named.extend(
            f"{requirement_id}: rejected check claim"
            for requirement_id, verdict in self.check_verdicts
            if verdict is C4Verdict.REJECTED
        )
        if not named:
            return None
        shown = named[:BLOCK_MESSAGE_MAX_PATHS]
        hidden = len(named) - len(shown)
        listing = "; ".join(shown)
        if hidden > 0:
            listing += f"; ... {hidden} more"
        return (
            "[Completion audit rejected: the run's completion claims do "
            f"not match the collected evidence ({listing}). The result is "
            "published as not completed; re-run the named checks or "
            "correct the claims and resubmit.]"
        )


def _artifact_evidence_from(
    accumulator: SubagentRunAccumulator,
    chronology: PerRunCompletionChronology | None,
    finalization: Any,
) -> ArtifactEvidence:
    """The ``ArtifactEvidence`` the C1/C3 audit consumes, from the run's
    collected surfaces.

    Touched paths and the path-keyed collection-time snapshots from the
    accumulator's collection, mutation roots from its footprint, commit
    hashes from the chronology's git-command results, and the HEAD
    snapshots from the once-only finalization.
    """
    observed_hashes: list[str] = []
    if chronology is not None:
        for entry in chronology._entries.values():
            event = entry.result_event
            if event is not None:
                observed_hashes.extend(git_result_hashes(event))
    return ArtifactEvidence(
        touched_paths=tuple(accumulator._touched_paths),
        snapshots=dict(accumulator._artifact_snapshots),
        mutation_roots=tuple(accumulator.mutation_roots),
        observed_commit_hashes=tuple(dict.fromkeys(observed_hashes)),
        admission_head=getattr(finalization, "admission_head", None),
        publication_head=getattr(finalization, "publication_head", None),
    )


def _admit_check_claims(
    requirements: Sequence[Any],
    receipt_result: ReceiptAuditResult | None,
    chronology: PerRunCompletionChronology | None,
    envelope: CompletionEnvelope | None,
    runner: Any,
    seat_has_shell: bool,
) -> tuple[tuple[str, C4Verdict], ...]:
    """C4 at the composition: adjudicate the run's check-pass claims.

    One claim per satisfied, template-bound requirement the receipt audit
    paired to its originating invocation: the claim names the requirement
    and the command that invocation executed. A satisfied requirement
    with no paired receipt carries no claim — the receipt audit's
    unverified finding already covers it (an availability fact, never a
    worker-side rejection).
    """
    if receipt_result is None or chronology is None:
        return ()
    claims: list[C4Claim] = []
    for finding in receipt_result.findings:
        if finding.occurrence_id is None:
            continue
        if finding.disposition not in {
            ReceiptEvidenceDisposition.OBSERVED,
            ReceiptEvidenceDisposition.CONTRADICTED,
        }:
            continue
        entry = chronology.entry(finding.occurrence_id)
        if entry is None or entry.receipt is None:
            continue
        executed_args = getattr(entry.receipt, "executed_args", None)
        command = (
            executed_args.get("command") if isinstance(executed_args, Mapping) else None
        )
        if not isinstance(command, str) or not command.strip():
            continue
        requirement = next(
            (
                candidate
                for candidate in requirements
                if candidate.requirement_id == finding.requirement_id
            ),
            None,
        )
        if requirement is None or requirement.template is None:
            continue
        claims.append(
            C4Claim(
                requirement_id=finding.requirement_id,
                claimed_command=command,
                chronology_occurrence_id=finding.occurrence_id,
            )
        )
    return tuple(
        (
            claim.requirement_id,
            admit_check_claim(
                claim=claim,
                envelope=envelope,
                runner=runner,
                seat_has_shell=seat_has_shell,
            ),
        )
        for claim in claims
    )


def _write_sequences(chronology: PerRunCompletionChronology | None) -> int | None:
    """The chronology's tail sequence over the write-classified entries."""
    if chronology is None:
        return None
    writes: list[tuple[int, Any]] = []
    for entry in chronology._entries.values():
        event = entry.result_event
        if event is None:
            continue
        if (
            touched_path_from_result_event(event) is not None
            or bash_write_paths_from_result_event(event)[0]
        ):
            writes.append((entry.sequence, event))
    return last_mutation_sequence(writes)


def _under_shared_root(path: str, shared_roots: Sequence[str]) -> bool:
    """Whether one observed path lives under a root shared with the parent.

    Mirrors the receipt audit's containment test
    (:func:`completion_audit_receipts._under_shared_root`): the exact root
    or any descendant binds.
    """
    return any(
        bool(root) and (path == root or path.startswith(root.rstrip("/") + "/"))
        for root in shared_roots
    )


def _check_observes_shared_root(
    contract: Any, accumulator: SubagentRunAccumulator, shared_roots: Sequence[str]
) -> bool:
    """Whether a finalization check's observed final state is in a shared root.

    The once-only finalization's checks run over the run's tree and
    observe only its FINAL STATE — they classify with no occurrence ID, so
    no collection-time witness ties that state to the worker (round-2
    amendment 2, amendments 12 and 14). The surfaces the check's observed
    state lives in are the run's collected touched paths, the
    mutation-roots footprint, and the frozen contract's own cwd and
    authorized roots. When any of them sits under a root shared with the
    parent, an external write between the worker's evidence and the
    finalization is possible, so the check's final state is check-context
    evidence: it may fail-close the gate but never contradicts — and
    never credits — the worker. With no shared roots declared, external
    writes are not established as possible and the classification keeps
    its full force.
    """
    roots = tuple(root for root in shared_roots if root)
    if not roots:
        return False
    observed: list[str] = [*accumulator._touched_paths, *accumulator.mutation_roots]
    cwd = getattr(contract, "cwd", None)
    if isinstance(cwd, str) and cwd:
        observed.append(cwd)
    contract_roots = getattr(contract, "roots", ())
    if isinstance(contract_roots, (list, tuple)):
        observed.extend(root for root in contract_roots if isinstance(root, str))
    return any(_under_shared_root(path, roots) for path in observed if path)


def _classify_finalization_checks(
    finalization: Any,
    requirements: Sequence[Any],
    chronology: PerRunCompletionChronology | None,
    declared_completed: bool,
    seat_has_shell: bool,
    observes_shared_root: bool = False,
) -> tuple[tuple[str, CheckEvidenceClassification], ...]:
    """C5 at the composition: classify each check the finalization ran.

    The once-only finalization (P6a) carries one outcome per contract-bound
    check template; each outcome classifies against every requirement
    bound to its template. A run without a finalization classifies no
    check — an absent verifier run is not fabricated evidence.
    ``observes_shared_root`` carries the composition's shared-root
    determination: a failed check observing final state under a root
    shared with the parent classifies check-context (UNVERIFIED), never
    CONTRADICTED.
    """
    if finalization is None:
        return ()
    outcomes: tuple[Any, ...] = tuple(getattr(finalization, "checks", ()) or ())
    if not outcomes:
        return ()
    mutation_sequence = _write_sequences(chronology)
    classifications: list[tuple[str, CheckEvidenceClassification]] = []
    for requirement in requirements:
        if requirement.template is None:
            continue
        outcome = next(
            (
                candidate
                for candidate in outcomes
                if getattr(candidate, "template", None) is requirement.template
            ),
            None,
        )
        if outcome is None:
            continue
        classifications.append((
            requirement.requirement_id,
            classify_check_evidence(
                outcome=outcome,
                requirement_id=requirement.requirement_id,
                chronology=chronology,
                occurrence_id=None,
                last_mutation_sequence=mutation_sequence,
                envelope_complete=declared_completed,
                run_outcome="completed" if declared_completed else "partial",
                seat_has_shell=seat_has_shell,
                observes_shared_root=observes_shared_root,
            ),
        ))
    return tuple(classifications)


def _compose_evidence(
    receipt_result: ReceiptAuditResult | None,
    artifact_findings: Iterable[ArtifactAuditFinding],
    check_classifications: Iterable[tuple[str, CheckEvidenceClassification]],
    finalization: Any,
) -> CompletionEvidence:
    """The P1 evidence surface the truth table classifies.

    Composed from the audits' own projections — never re-implemented: the
    receipt audit's evidence surface (observed and contradicted ids and
    the check-context availability failure), the artifact audit's
    worker-crediting findings, and the check audit's classifications. The
    finalization's runtime failure (the verifier's own health) is the
    primary runtime failure when present.
    """
    observed: list[str] = []
    contradicted: list[str] = []
    runtime_failure: str | None = None
    if receipt_result is not None:
        receipt_evidence = receipt_result.evidence()
        observed.extend(receipt_evidence.requirement_ids_observed)
        contradicted.extend(receipt_evidence.contradicted_requirement_ids)
        runtime_failure = receipt_evidence.runtime_failure
    for finding in artifact_findings:
        if finding.disposition is ArtifactAuditDisposition.WORKER_CREDITED:
            observed.append(finding.requirement_id)
    for requirement_id, classification in check_classifications:
        if classification.disposition is CheckEvidenceDisposition.WORKER_CREDITED:
            observed.append(requirement_id)
        elif classification.disposition is CheckEvidenceDisposition.CONTRADICTED:
            contradicted.append(requirement_id)
    if finalization is not None:
        finalization_failure = getattr(finalization, "runtime_failure", None)
        if finalization_failure in _RUNTIME_FAILURE_WORDS:
            runtime_failure = finalization_failure
    return CompletionEvidence(
        requirement_ids_observed=tuple(dict.fromkeys(observed)),
        contradicted_requirement_ids=tuple(dict.fromkeys(contradicted)),
        runtime_failure=runtime_failure,  # type: ignore[arg-type]
    )


def compose_completion_audit(
    *,
    accumulator: SubagentRunAccumulator,
    contract: Any,
    claim: Any = None,
    finalization: Any = None,
    claimed_values: Mapping[str, str] | None = None,
    shared_roots: Sequence[str] = (),
    declared_completed: bool = True,
    seat_has_shell: bool = True,
    runner: Any = None,
    mode: AuditMode = "stamp",
) -> CompletionAuditComposition:
    """Compose the run's completion audit at ``build_result`` (P10).

    Runs the landed audits over the run's collected evidence and composes
    their findings onto the P1 truth table. The once-only finalization
    (P6a) arrives resolved from the caller — the run-keyed cache — and no
    check runs here: a cached finalization supplies the HEAD snapshots and
    the check outcomes, and an uncached run composes from the chronology
    and the accumulator's own collection alone.

    In stamp mode the composition is observational: the composed
    classification is returned for stamping, publication is never
    blocked, and behavior is unchanged. In enforce mode a worker-side
    disposition (a contradicted claim, a rejected check claim) blocks the
    submission.

    ``shared_roots`` carries the root paths shared with the parent. It
    reaches the receipt audit (a claimed artifact under a shared root is
    check-context evidence) and the finalization-check classification (a
    failed check observing final state under a shared root is
    check-context, never a contradiction).
    """
    envelope = getattr(contract, "envelope", None)
    if not isinstance(envelope, CompletionEnvelope):
        envelope = None
    artifact_requirements = tuple(envelope.requirements) if envelope is not None else ()
    satisfied: tuple[str, ...] = ()
    if claim is not None:
        claim_satisfied = getattr(claim, "satisfied_requirement_ids", None)
        if isinstance(claim_satisfied, (list, tuple)):
            satisfied = tuple(claim_satisfied)
    # The run's own chronology and collected touched paths are the
    # composition's evidence surfaces.
    chronology = accumulator._chronology
    artifact_evidence = _artifact_evidence_from(accumulator, chronology, finalization)
    artifact_findings = (
        audit_artifacts(
            artifact_requirements, satisfied, claimed_values, artifact_evidence
        )
        if artifact_requirements
        else ()
    )
    receipt_result: ReceiptAuditResult | None = None
    if claim is not None:
        # The claim's bound canonical commands: the runner derives one for
        # each contract-bound requirement with a template; a requirement
        # without a template (artifact-only) binds no command.
        reference_commands: dict[str, str] = {}
        for requirement in artifact_requirements:
            command = requirement_canonical_command(
                envelope, requirement.requirement_id, runner=runner
            )
            if command is not None:
                reference_commands[requirement.requirement_id] = command
        receipt_result = audit_completion_receipts(
            claim=claim,
            chronology=chronology,
            reference_commands=reference_commands,
            touched_paths=tuple(accumulator._touched_paths),
            shared_roots=shared_roots,
        )
    check_verdicts = _admit_check_claims(
        artifact_requirements,
        receipt_result,
        chronology,
        envelope,
        runner,
        seat_has_shell,
    )
    check_classifications = _classify_finalization_checks(
        finalization,
        artifact_requirements,
        chronology,
        declared_completed,
        seat_has_shell,
        observes_shared_root=_check_observes_shared_root(
            contract, accumulator, shared_roots
        ),
    )
    evidence = _compose_evidence(
        receipt_result, artifact_findings, check_classifications, finalization
    )
    classification = classify_completion(
        mandate=envelope,
        receipt=claim,
        declared_completed=declared_completed,
        evidence=evidence,
    )
    blocks = mode == "enforce" and (
        classification.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
        or any(verdict is C4Verdict.REJECTED for _, verdict in check_verdicts)
    )
    return CompletionAuditComposition(
        classification=classification,
        artifact_findings=tuple(artifact_findings),
        receipt_result=receipt_result,
        check_verdicts=tuple(check_verdicts),
        check_classifications=tuple(check_classifications),
        evidence=evidence,
        blocks_publication=blocks,
        accumulator=accumulator,
    )
