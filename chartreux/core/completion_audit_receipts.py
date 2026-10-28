"""The C2 receipt audit (P8): retry-distinguishing receipt evidence.

The audit resolves the worker's envelope-linked completion receipt (the
strict P1 grammar, chartreux.core.completion_contract) against the run's
collected receipt chronology (P3, chartreux.core.completion_evidence),
keyed by the harness-minted, run-scoped receipt occurrence ID — never
tool_call_id alone (tool-call ids are reused across turns, so pairing by
tool_call_id alone can associate evidence with the wrong invocation). The
audit consumes the executor-minted ID unchanged; reminting is prohibited.

Envelope-linked strict receipts only; no prose extraction: the audit reads
only the strict receipt and the chronology's collected receipts — never
the worker's report prose.

Per-claimed-requirement dispositions:

- ``observed`` — worker-crediting: the claim resolves to its originating
  invocation's receipt, the receipt's outcome supports the claim, and the
  collection-time snapshot witness is intact.
- ``unverified`` — no matching command, or the claim is not envelope-linked.
- ``unavailable`` — the receipt's evidence is truncated, evicted, or
  ambiguous, or no collection-time mutation witness was recorded. An
  evicted entry resolves exactly like an unavailable one — an availability
  fact, never a contradiction, and never worker-crediting.
- ``contradicted`` — a paired same-command conflict: the claim's own
  referenced command's paired receipt directly conflicts with the claim.

Repeated commands with differing outputs (the red/green retry) resolve via
the occurrence-ID pairing: each receipt resolves ONLY to its originating
invocation, and a correctly referenced successful retry is never condemned
by the earlier failed attempt of the same command.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from chartreux.core.completion_contract import CompletionEvidence, CompletionReceipt
from chartreux.core.completion_evidence import (
    ContentSnapshot,
    EvidenceAvailability,
    PerRunCompletionChronology,
)

__all__ = [
    "ReceiptAuditFinding",
    "ReceiptAuditResult",
    "ReceiptEvidenceDisposition",
    "audit_completion_receipts",
]


class ReceiptEvidenceDisposition(StrEnum):
    """The disposition vocabulary of one claimed requirement's evidence."""

    OBSERVED = "observed"
    UNVERIFIED = "unverified"
    UNAVAILABLE = "unavailable"
    CONTRADICTED = "contradicted"


@dataclass(frozen=True, slots=True)
class ReceiptAuditFinding:
    """One claimed requirement's resolved receipt evidence.

    ``occurrence_id`` is the harness-minted, run-scoped occurrence ID of
    the receipt the claim resolved to — the executor-minted ID consumed
    unchanged, never reminted.
    """

    requirement_id: str
    disposition: ReceiptEvidenceDisposition
    occurrence_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ReceiptAuditResult:
    """The C2 receipt audit's output.

    ``findings`` carries one disposition per claimed requirement;
    :meth:`evidence` projects the findings onto the P1 evidence surface
    the truth table (and the P10 composition) consumes.
    """

    envelope_linked: bool
    findings: tuple[ReceiptAuditFinding, ...]

    @property
    def worker_credited(self) -> bool:
        """Whether every claimed requirement's evidence is worker-crediting.

        A single unavailable/contradicted/unverified requirement refuses
        the credit: worker crediting is all-or-nothing per claim.
        """
        return bool(self.findings) and all(
            finding.disposition is ReceiptEvidenceDisposition.OBSERVED
            for finding in self.findings
        )

    def evidence(self) -> CompletionEvidence:
        """The P1 evidence surface for the truth-table classification.

        ``observed`` findings land in ``requirement_ids_observed``;
        ``contradicted`` findings — the only worker-side dispositions — in
        ``contradicted_requirement_ids``. Unavailable evidence is a
        harness-side availability fact: it maps to the evidence surface's
        ``unavailable`` runtime failure (never a contradiction) unless a
        worker-side contradiction stands, which takes precedence.
        """
        observed = tuple(
            finding.requirement_id
            for finding in self.findings
            if finding.disposition is ReceiptEvidenceDisposition.OBSERVED
        )
        contradicted = tuple(
            finding.requirement_id
            for finding in self.findings
            if finding.disposition is ReceiptEvidenceDisposition.CONTRADICTED
        )
        unavailable = any(
            finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
            for finding in self.findings
        )
        return CompletionEvidence(
            requirement_ids_observed=observed,
            contradicted_requirement_ids=contradicted,
            runtime_failure=(
                "unavailable" if unavailable and not contradicted else None
            ),
        )


def _command_tokens(command: Any) -> tuple[str, ...] | None:
    """The whitespace tokens of one command line, or None when absent."""
    if not isinstance(command, str) or not command.strip():
        return None
    return tuple(command.split())


def _receipt_command_tokens(receipt: Any) -> tuple[str, ...] | None:
    """The command tokens of one collected side-channel receipt.

    The command operand is the shell tool's executed-args ``command``
    input; receipts of tools without a command operand carry no command
    line and match nothing.
    """
    executed_args = getattr(receipt, "executed_args", None)
    if not isinstance(executed_args, Mapping):
        return None
    return _command_tokens(executed_args.get("command"))


def _matches_reference(
    command_tokens: tuple[str, ...], reference_tokens: tuple[str, ...]
) -> bool:
    """Whether one receipt's command line starts with the reference command.

    A token-prefix match: the receipt's command may carry further operands
    (the check root, targets) after the canonical command's tokens, but
    the leading tokens must be exactly the reference's. No substring, no
    prose matching.
    """
    if len(command_tokens) < len(reference_tokens):
        return False
    return command_tokens[: len(reference_tokens)] == reference_tokens


def _under_shared_root(path: str, shared_roots: Sequence[str]) -> bool:
    """Whether one collected path lives under a root shared with the parent."""
    return any(
        bool(root) and (path == root or path.startswith(root.rstrip("/") + "/"))
        for root in shared_roots
    )


def _witness_intact(snapshot: ContentSnapshot | None) -> bool:
    """Whether one collection-time snapshot witnesses the artifact.

    The witness is the P3 collection-time capture (content hash + byte
    size + sampled head/tail). A truncated capture or a missing one is no
    witness: worker credit requires a recorded, intact mutation witness,
    and the absence is a check-context availability fact — never a
    contradiction.
    """
    return snapshot is not None and not snapshot.truncated


def _finding(
    requirement_id: str,
    disposition: ReceiptEvidenceDisposition,
    *,
    occurrence_id: str | None = None,
    detail: str = "",
) -> ReceiptAuditFinding:
    return ReceiptAuditFinding(
        requirement_id=requirement_id,
        disposition=disposition,
        occurrence_id=occurrence_id,
        detail=detail,
    )


def _snapshot_for(
    occurrence_id: str,
    chronology: PerRunCompletionChronology,
    snapshot_witness: Callable[[str], ContentSnapshot | None] | None,
) -> ContentSnapshot | None:
    """The collection-time snapshot witness for one occurrence ID.

    The witness is the P3 collection-time capture (content hash + byte
    size + sampled head/tail bytes). An injected provider overrides the
    chronology's own entry lookup; both are consumed as recorded — the
    audit never fabricates a witness.
    """
    if snapshot_witness is not None:
        return snapshot_witness(occurrence_id)
    entry = chronology.entry(occurrence_id)
    if entry is None:
        return None
    return entry.snapshot


def _resolve_matched(
    requirement_id: str,
    matched: list[tuple[str, Any]],
    chronology: PerRunCompletionChronology,
    snapshot_witness: Callable[[str], ContentSnapshot | None] | None,
) -> ReceiptAuditFinding:
    """Resolve the matched receipts of one claimed requirement.

    ``matched`` holds ``(occurrence_id, receipt)`` pairs in the
    chronology's monotonic collection order. Resolution rules (P8):

    - an evicted entry resolves exactly like unavailable — an availability
      fact, never contradictory;
    - a truncated or absent collection-time snapshot refuses worker credit
      (the recorded mutation witness is the crediting requirement);
    - a single matched receipt resolves to its outcome: success credits
      the claim, a failure is the paired same-command conflict, any other
      outcome is an availability fact;
    - repeated commands resolve via the occurrence-ID pairing to their
      LATEST invocation (the claim is evaluated at finalization): a latest
      success is the correctly referenced successful retry and is credited
      — the earlier failed attempt of the same command never condemns it;
      a latest failure with an earlier success is ambiguous and resolves
      unavailable, never condemned; a latest failure with no earlier
      success is the paired same-command conflict.
    """
    latest_occurrence, latest_receipt = matched[-1]
    if chronology.resolution(latest_occurrence) is EvidenceAvailability.EVICTED:
        return _finding(
            requirement_id,
            ReceiptEvidenceDisposition.UNAVAILABLE,
            occurrence_id=latest_occurrence,
            detail="the paired receipt was evicted from the bounded chronology",
        )
    if not _witness_intact(
        _snapshot_for(latest_occurrence, chronology, snapshot_witness)
    ):
        return _finding(
            requirement_id,
            ReceiptEvidenceDisposition.UNAVAILABLE,
            occurrence_id=latest_occurrence,
            detail=(
                "no intact collection-time snapshot witness; check-context "
                "evidence never credits the worker"
            ),
        )
    latest_outcome = getattr(latest_receipt, "status", None)
    earlier_outcomes = [getattr(receipt, "status", None) for _, receipt in matched[:-1]]
    if latest_outcome == "success":
        return _finding(
            requirement_id,
            ReceiptEvidenceDisposition.OBSERVED,
            occurrence_id=latest_occurrence,
        )
    if latest_outcome == "failure":
        if "success" in earlier_outcomes:
            return _finding(
                requirement_id,
                ReceiptEvidenceDisposition.UNAVAILABLE,
                occurrence_id=latest_occurrence,
                detail=(
                    "repeated commands with differing outputs; the conflict "
                    "is ambiguous, never a condemnation"
                ),
            )
        return _finding(
            requirement_id,
            ReceiptEvidenceDisposition.CONTRADICTED,
            occurrence_id=latest_occurrence,
            detail="the claim's own referenced command failed",
        )
    return _finding(
        requirement_id,
        ReceiptEvidenceDisposition.UNAVAILABLE,
        occurrence_id=latest_occurrence,
        detail=f"the paired receipt's outcome is {latest_outcome!r}",
    )


def _matched_receipts(
    chronology: PerRunCompletionChronology, reference_tokens: tuple[str, ...]
) -> list[tuple[str, Any]]:
    """The chronology's receipts whose command matches one reference."""
    matched: list[tuple[str, Any]] = []
    for entry in chronology._entries.values():
        receipt = entry.receipt
        if receipt is None:
            continue
        command_tokens = _receipt_command_tokens(receipt)
        if command_tokens is None:
            continue
        if _matches_reference(command_tokens, reference_tokens):
            matched.append((entry.occurrence_id, receipt))
    return matched


def audit_completion_receipts(
    *,
    claim: CompletionReceipt | Mapping[str, Any] | None,
    chronology: PerRunCompletionChronology,
    reference_commands: Mapping[str, str],
    touched_paths: Sequence[str] = (),
    shared_roots: Sequence[str] = (),
    snapshot_witness: Callable[[str], ContentSnapshot | None] | None = None,
) -> ReceiptAuditResult:
    """Audit the worker's envelope-linked receipt against the chronology.

    ``claim`` is the worker's strict-grammar completion receipt (the P1
    model or its raw payload); ``chronology`` is the run's collected
    receipt chronology (P3); ``reference_commands`` maps each claimed
    requirement id to the canonical command line the contract binds it
    to; ``touched_paths`` carries the run's collected touched paths and
    ``shared_roots`` the root paths shared with the parent — a claimed
    touched-path artifact under a shared root is check-context evidence
    that never credits the worker; ``snapshot_witness`` overrides where
    the collection-time mutation witness is read from.
    """
    if claim is None:
        return ReceiptAuditResult(envelope_linked=False, findings=())
    if isinstance(claim, CompletionReceipt):
        claim_model = claim
    else:
        try:
            claim_model = CompletionReceipt.model_validate(claim)
        except ValidationError:
            # A grammar violation is a grammar fact, never a worker fault:
            # the strict-grammar receipt the audit consumes does not exist.
            return ReceiptAuditResult(envelope_linked=False, findings=())
    if claim_model.envelope is None:
        # Envelope-linked strict receipts only: an unlinked claim carries
        # no mandate context to resolve against.
        return ReceiptAuditResult(
            envelope_linked=False,
            findings=tuple(
                _finding(
                    requirement_id,
                    ReceiptEvidenceDisposition.UNVERIFIED,
                    detail="the claim is not linked to its envelope",
                )
                for requirement_id in claim_model.satisfied_requirement_ids
            ),
        )

    shared_roots_normalized = tuple(root for root in shared_roots if root)
    touched_normalized = tuple(path for path in touched_paths if path)
    findings: list[ReceiptAuditFinding] = []
    for requirement in claim_model.envelope.requirements:
        requirement_id = requirement.requirement_id
        if requirement_id not in claim_model.satisfied_requirement_ids:
            continue
        if (
            requirement.artifact is not None
            and shared_roots_normalized
            and any(
                _under_shared_root(path, shared_roots_normalized)
                for path in touched_normalized
            )
        ):
            # A touched-path artifact under a root shared with the parent
            # is check-context evidence: it may fail-close the gate but
            # never credits the worker with honest completion.
            findings.append(
                _finding(
                    requirement_id,
                    ReceiptEvidenceDisposition.UNAVAILABLE,
                    detail=(
                        "the artifact lives under a root shared with the "
                        "parent; check-context evidence never credits the "
                        "worker"
                    ),
                )
            )
            continue
        reference = reference_commands.get(requirement_id)
        reference_tokens = _command_tokens(reference)
        if reference_tokens is None:
            findings.append(
                _finding(
                    requirement_id,
                    ReceiptEvidenceDisposition.UNVERIFIED,
                    detail="the requirement binds no canonical command",
                )
            )
            continue
        matched = _matched_receipts(chronology, reference_tokens)
        if not matched:
            findings.append(
                _finding(
                    requirement_id,
                    ReceiptEvidenceDisposition.UNVERIFIED,
                    detail="no collected receipt matches the bound command",
                )
            )
            continue
        findings.append(
            _resolve_matched(requirement_id, matched, chronology, snapshot_witness)
        )
    return ReceiptAuditResult(envelope_linked=True, findings=tuple(findings))
