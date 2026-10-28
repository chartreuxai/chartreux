"""P10: the completion-audit composition at ``build_result``.

Unit tests for :func:`chartreux.core.completion_audit.compose_completion_audit`
— the composition of the landed C1-C5 audits onto the P1 truth table, and
the stamp/enforce mode contract at the result-publication boundary. All
evidence is constructed in-process through the real collection path (the
accumulator's observe and its side-channel sink): no subprocess runs, no
check executes, no provider is contacted.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.completion_audit import compose_completion_audit
from chartreux.core.completion_audit_artifacts import (
    ArtifactAuditDisposition,
    ArtifactAuditFinding,
)
from chartreux.core.completion_audit_checks import C4Verdict, CheckEvidenceDisposition
from chartreux.core.completion_audit_receipts import ReceiptEvidenceDisposition
from chartreux.core.completion_checks import (
    CompletionCheckOutcome,
    CompletionCheckStatus,
    CompletionCheckTemplate,
)
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    CompletionArtifactKind,
    CompletionClaimFidelity,
    CompletionEnvelope,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRequirement,
    CompletionRuntimeFailure,
)
from chartreux.core.completion_evidence import EvidenceAvailability, snapshot_bytes
from chartreux.core.events import AssistantEvent, ToolResultEvent
from chartreux.core.hooks.models import ToolStatus
from chartreux.core.subagents import SubagentRunAccumulator
from chartreux.core.tools.builtins.bash import Bash, CapturedShellResult
from chartreux.core.tools.builtins.write_file import WriteFile, WriteFileResult

CANONICAL_COMMAND = "pytest tests/test_x.py"
RUNNER_ARGV = ["pytest", "tests/test_x.py"]


def _envelope(*requirement_ids: str) -> CompletionEnvelope:
    return CompletionEnvelope(
        requirements=tuple(
            CompletionRequirement(
                requirement_id=requirement_id, template=CompletionCheckTemplate.PYTEST
            )
            for requirement_id in requirement_ids
        ),
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        marker=COMPLETION_ENVELOPE_MARKER,
    )


def _touched_path_envelope(requirement_id: str) -> CompletionEnvelope:
    return CompletionEnvelope(
        requirements=(
            CompletionRequirement(
                requirement_id=requirement_id,
                artifact=CompletionArtifactKind.TOUCHED_PATH,
            ),
        ),
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        marker=COMPLETION_ENVELOPE_MARKER,
    )


def _receipt(envelope: CompletionEnvelope, *satisfied: str) -> CompletionReceipt:
    return CompletionReceipt(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        envelope=envelope,
        satisfied_requirement_ids=tuple(satisfied),
    )


def _runner() -> Any:
    return SimpleNamespace(check_argv=lambda template: list(RUNNER_ARGV))


def _contract(envelope: CompletionEnvelope) -> Any:
    return SimpleNamespace(envelope=envelope, audit_mode="stamp")


def _observe_write(
    accumulator: SubagentRunAccumulator,
    path: Path,
    occurrence_id: str,
    *,
    content: str = "payload\n",
) -> ToolResultEvent:
    """Observe one write_file result through the real collection path.

    The artifact exists on disk, so the accumulator's collection-time
    capture records its snapshot — the mutation witness the audits consume.
    """
    path.write_text(content, encoding="utf-8")
    event = ToolResultEvent(
        tool_name="write_file",
        tool_class=WriteFile,
        result=WriteFileResult(
            file_path=str(path), bytes_written=len(content.encode()), content=content
        ),
        tool_call_id="call-1",
        occurrence_id=occurrence_id,
    )
    accumulator.observe(AssistantEvent(content="done"), tool_call_id="task")
    accumulator.observe(event, tool_call_id="call-1")
    return event


def _emit_receipt(
    accumulator: SubagentRunAccumulator,
    occurrence_id: str,
    command: str,
    *,
    status: ToolStatus = "success",
    final_text: str = "ok",
) -> None:
    """Emit one side-channel receipt through the accumulator's sink."""
    accumulator.completion_evidence_sink()(
        CompletionEvidenceReceipt(
            occurrence_id=occurrence_id,
            tool_call_id="call-1",
            tool_name="write_file",
            status=status,
            final_text=final_text,
            executed_args={"command": command},
        )
    )


class TestCollectionTimeCapture:
    """The accumulator's collection-time capture (P3 round 5, producer)."""

    def test_observe_registers_the_event_and_captures_the_snapshot(
        self, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        event = _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        entry = accumulator._chronology.entry("occ-1")
        assert entry is not None
        assert entry.result_event is event
        assert entry.snapshot is not None
        assert entry.snapshot.content_hash == hashlib.sha256(b"payload\n").hexdigest()
        assert entry.snapshot.byte_size == len(b"payload\n")
        assert entry.snapshot.head_bytes == b"payload\n"
        assert entry.snapshot.truncated is False
        assert accumulator._artifact_snapshots[str(artifact)] is entry.snapshot

    def test_observe_without_occurrence_id_never_registers(
        self, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "out.txt"
        artifact.write_text("payload\n", encoding="utf-8")
        accumulator = SubagentRunAccumulator()
        accumulator.observe(
            ToolResultEvent(
                tool_name="write_file",
                tool_class=WriteFile,
                result=WriteFileResult(
                    file_path=str(artifact), bytes_written=8, content="payload\n"
                ),
                tool_call_id="call-1",
            ),
            tool_call_id="call-1",
        )
        # No minted key: the invocation cannot be paired — never reminted.
        assert accumulator._chronology._entries == {}
        # The path-keyed witness is still collected for the artifact audit.
        assert str(artifact) in accumulator._artifact_snapshots

    def test_observe_captures_bash_write_paths(self, tmp_path: Path) -> None:
        artifact = tmp_path / "shell-out.txt"
        artifact.write_text("shell payload\n", encoding="utf-8")
        accumulator = SubagentRunAccumulator()
        accumulator.observe(
            ToolResultEvent(
                tool_name="bash",
                tool_class=Bash,
                result=CapturedShellResult(
                    command="pytest tests/test_x.py", write_paths=[str(artifact)]
                ),
                tool_call_id="call-1",
                occurrence_id="occ-1",
            ),
            tool_call_id="call-1",
        )
        entry = accumulator._chronology.entry("occ-1")
        assert entry is not None
        assert entry.snapshot is not None
        assert entry.snapshot.head_bytes == b"shell payload\n"
        assert accumulator._artifact_snapshots[str(artifact)] is entry.snapshot
        # The write-classified path joins the submission scan's scope.
        assert str(artifact) in accumulator._touched_paths

    def test_missing_artifact_records_no_witness(self, tmp_path: Path) -> None:
        artifact = tmp_path / "gone.txt"  # never written
        accumulator = SubagentRunAccumulator()
        accumulator.observe(
            ToolResultEvent(
                tool_name="write_file",
                tool_class=WriteFile,
                result=WriteFileResult(
                    file_path=str(artifact), bytes_written=8, content="payload\n"
                ),
                tool_call_id="call-1",
                occurrence_id="occ-1",
            ),
            tool_call_id="call-1",
        )
        entry = accumulator._chronology.entry("occ-1")
        assert entry is not None
        assert entry.result_event is not None
        # An availability fact, never a fabricated witness.
        assert entry.snapshot is None
        assert accumulator._artifact_snapshots == {}


class TestStampModeLeavesBehaviorUnchanged:
    """Stamp mode (the default) is observational."""

    def test_stamp_mode_composes_and_never_blocks(self) -> None:
        accumulator = SubagentRunAccumulator()
        composition = compose_completion_audit(
            accumulator=accumulator, contract=None, mode="stamp"
        )
        assert composition.blocks_publication is False
        assert composition.rejection_message() is None
        classification = composition.classification
        assert classification.claim_fidelity.value == "claim_unverified"
        assert classification.mandate_satisfaction == "mandate_disabled"
        assert classification.rejection is None
        assert classification.disclosure_completeness == "disclosure_complete"

    def test_stamp_mode_with_a_satisfied_artifact_claim(self, tmp_path: Path) -> None:
        # The real capture path: the accumulator observes the write (the
        # collection-time snapshot), the side-channel sink registers the
        # receipt, and the composition credits the claim through the C2
        # audit's own witness read — no injected evidence.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(accumulator, "occ-1", CANONICAL_COMMAND)
        envelope = _envelope("req-1")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            runner=_runner(),
            mode="stamp",
        )
        assert composition.blocks_publication is False
        assert composition.rejection_message() is None
        assert composition.receipt_result is not None
        assert composition.receipt_result.worker_credited is True
        # C4: the claim's paired command is exactly canonical.
        assert composition.check_verdicts == (("req-1", C4Verdict.SATISFIED),)
        classification = composition.classification
        assert classification.claim_fidelity.value == "claim_matched"
        assert classification.mandate_satisfaction == "mandate_satisfied"
        assert classification.disclosure_completeness == "disclosure_complete"
        assert classification.rejection is None

    def test_stamp_mode_with_an_envelope_linked_receipt(self) -> None:
        accumulator = SubagentRunAccumulator()
        envelope = _envelope("req-1", "req-2")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=None,
            claim=_receipt(envelope, "req-1"),
            mode="stamp",
        )
        assert composition.blocks_publication is False
        assert composition.receipt_result is not None
        assert composition.receipt_result.envelope_linked is True
        # No frozen contract: the mandate axis is disabled, the claim
        # unverified — and the stamp mode never blocks.
        assert composition.classification.mandate_satisfaction == "mandate_disabled"
        assert composition.classification.claim_fidelity.value == "claim_unverified"


class TestEnforceModeBlocksWorkerSideDispositions:
    """Enforce mode blocks on worker-side dispositions only."""

    def test_contradicted_receipt_blocks(self, tmp_path: Path) -> None:
        # C2: a real side-channel receipt whose paired same-command result
        # conflicts with the claim — the paired same-command conflict.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(
            accumulator,
            "occ-1",
            CANONICAL_COMMAND,
            status="failure",
            final_text="1 failed",
        )
        envelope = _envelope("req-1")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            runner=_runner(),
            mode="enforce",
        )
        assert composition.receipt_result is not None
        finding = composition.receipt_result.findings[0]
        assert finding.disposition is ReceiptEvidenceDisposition.CONTRADICTED
        classification = composition.classification
        assert classification.claim_fidelity.value == "claim_contradicted"
        assert classification.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
        assert composition.blocks_publication is True
        message = composition.rejection_message()
        assert message is not None
        assert "req-1: contradicted" in message
        assert "rejected check claim" not in message

    def test_rejected_check_claim_blocks(self, tmp_path: Path) -> None:
        # C4: the claim's paired command prefix-matches the canonical
        # command (the C2 audit credits it) but is not exactly canonical —
        # the claim is rejected, a worker-side disposition.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(accumulator, "occ-1", f"{CANONICAL_COMMAND} --lf")
        envelope = _envelope("req-1")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            runner=_runner(),
            mode="enforce",
        )
        assert composition.check_verdicts == (("req-1", C4Verdict.REJECTED),)
        # The receipt evidence itself credits (witness intact, success)...
        assert composition.classification.claim_fidelity.value == "claim_matched"
        # ...but the rejected claim is worker-side: the submission blocks.
        assert composition.blocks_publication is True
        message = composition.rejection_message()
        assert message is not None
        assert "req-1: rejected check claim" in message
        assert "contradicted" not in message

    def test_failed_finalization_check_contradicts_in_enforce_mode(
        self, tmp_path: Path
    ) -> None:
        # C5: a failed harness-run check while the envelope reported
        # completion contradicts the claim — worker-side, blocks.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(
                CompletionCheckOutcome(
                    template=CompletionCheckTemplate.PYTEST,
                    status=CompletionCheckStatus.NONZERO,
                    argv=tuple(RUNNER_ARGV),
                    exit_code=1,
                ),
            ),
            admission_head=None,
            publication_head=None,
            runtime_failure=None,
        )
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            runner=_runner(),
            mode="enforce",
        )
        assert (
            composition.check_classifications[0][1].disposition
            is CheckEvidenceDisposition.CONTRADICTED
        )
        classification = composition.classification
        assert classification.claim_fidelity.value == "claim_contradicted"
        assert classification.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
        assert composition.blocks_publication is True
        message = composition.rejection_message()
        assert message is not None
        assert "req-1: contradicted" in message

    def test_harness_side_failures_never_block(self, tmp_path: Path) -> None:
        # Honesty economics: a denied/unavailable verifier is an
        # informational stamp, never a worker disposition.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        # The receipt's command matches nothing canonical: unverified.
        _emit_receipt(accumulator, "occ-1", "ruff check .")
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(),
            admission_head=None,
            publication_head=None,
            runtime_failure="timeout",
        )
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            runner=_runner(),
            mode="enforce",
        )
        assert composition.blocks_publication is False
        assert composition.rejection_message() is None
        classification = composition.classification
        assert (
            classification.primary_runtime_failure
            is CompletionRuntimeFailure.RUNTIME_TIMEOUT
        )
        assert classification.claim_fidelity.value == "claim_unverified"
        assert classification.rejection is None

    def test_failed_check_in_a_partial_run_is_informational(
        self, tmp_path: Path
    ) -> None:
        # C5: a failed harness-run check with a partial run is
        # informational — never a worker contradiction.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(
                CompletionCheckOutcome(
                    template=CompletionCheckTemplate.PYTEST,
                    status=CompletionCheckStatus.NONZERO,
                    argv=tuple(RUNNER_ARGV),
                    exit_code=1,
                ),
            ),
            admission_head=None,
            publication_head=None,
            runtime_failure=None,
        )
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            runner=_runner(),
            declared_completed=False,
            mode="enforce",
        )
        assert (
            composition.check_classifications[0][1].disposition
            is CheckEvidenceDisposition.INFORMATIONAL
        )
        assert composition.blocks_publication is False
        assert composition.classification.rejection is None


class TestSharedRootInterleaving:
    """Round-2 amendment 2 / amendments 12-14: the shared-root exclusion.

    A failed finalization check whose observed root is shared with the
    parent is check-context evidence — UNVERIFIED with the external
    mutation noted, never CONTRADICTED, never blocking. The honest
    worker's check claim stands on the C4 receipt path.
    """

    def test_failed_check_in_a_shared_root_is_unverified_not_blocked(
        self, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(accumulator, "occ-1", CANONICAL_COMMAND)
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(
                CompletionCheckOutcome(
                    template=CompletionCheckTemplate.PYTEST,
                    status=CompletionCheckStatus.NONZERO,
                    argv=tuple(RUNNER_ARGV),
                    exit_code=1,
                ),
            ),
            admission_head=None,
            publication_head=None,
            runtime_failure=None,
        )
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            shared_roots=(str(tmp_path),),
            runner=_runner(),
            mode="enforce",
        )
        classification = composition.check_classifications[0][1]
        assert classification.disposition is CheckEvidenceDisposition.UNVERIFIED
        assert "shared with the parent" in classification.reason
        assert composition.evidence.contradicted_requirement_ids == ()
        # The claim is not contradicted — the receipt path still credits
        # the worker's own witnessed check invocation.
        assert composition.classification.claim_fidelity is (
            CompletionClaimFidelity.CLAIM_MATCHED
        )
        assert composition.classification.rejection is None
        assert composition.blocks_publication is False
        assert composition.rejection_message() is None

    def test_failed_check_outside_the_shared_root_stays_contradicted(
        self, tmp_path: Path
    ) -> None:
        # The determination is about the observed root, not the mere
        # presence of shared roots: a declared root covering nothing the
        # run observed leaves external writes unestablished, and the
        # failed check keeps its full force.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(accumulator, "occ-1", CANONICAL_COMMAND)
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(
                CompletionCheckOutcome(
                    template=CompletionCheckTemplate.PYTEST,
                    status=CompletionCheckStatus.NONZERO,
                    argv=tuple(RUNNER_ARGV),
                    exit_code=1,
                ),
            ),
            admission_head=None,
            publication_head=None,
            runtime_failure=None,
        )
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            shared_roots=(str(tmp_path / "elsewhere"),),
            runner=_runner(),
            mode="enforce",
        )
        assert (
            composition.check_classifications[0][1].disposition
            is CheckEvidenceDisposition.CONTRADICTED
        )
        assert composition.classification.claim_fidelity is (
            CompletionClaimFidelity.CLAIM_CONTRADICTED
        )
        assert composition.blocks_publication is True

    def test_passing_final_state_check_never_credits_the_worker(
        self, tmp_path: Path
    ) -> None:
        # Amendment 14 / peer finding 5: the final-state crediting branch
        # is gone — an exit-0 check with no observed mutation and no
        # collection-time witness is check-context (UNVERIFIED), and the
        # claim stays unverified without receipt-path evidence.
        accumulator = SubagentRunAccumulator()
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(
                CompletionCheckOutcome(
                    template=CompletionCheckTemplate.PYTEST,
                    status=CompletionCheckStatus.PASSED,
                    argv=tuple(RUNNER_ARGV),
                    exit_code=0,
                ),
            ),
            admission_head=None,
            publication_head=None,
            runtime_failure=None,
        )
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            runner=_runner(),
            mode="stamp",
        )
        assert (
            composition.check_classifications[0][1].disposition
            is CheckEvidenceDisposition.UNVERIFIED
        )
        assert composition.evidence.requirement_ids_observed == ()
        assert composition.classification.claim_fidelity is (
            CompletionClaimFidelity.CLAIM_UNVERIFIED
        )
        assert composition.classification.rejection is None


class TestUnavailableEvidenceNeverBecomesContradiction:
    """Availability facts are never worker dispositions."""

    def test_missing_witness_is_unavailable_never_contradicted(
        self, tmp_path: Path
    ) -> None:
        # The artifact is gone at collection: no witness, so the C2 audit's
        # crediting requirement fails — check-context, never a
        # contradiction.
        artifact = tmp_path / "gone.txt"  # never written
        accumulator = SubagentRunAccumulator()
        accumulator.observe(
            ToolResultEvent(
                tool_name="write_file",
                tool_class=WriteFile,
                result=WriteFileResult(
                    file_path=str(artifact), bytes_written=8, content="payload\n"
                ),
                tool_call_id="call-1",
                occurrence_id="occ-1",
            ),
            tool_call_id="call-1",
        )
        _emit_receipt(accumulator, "occ-1", CANONICAL_COMMAND)
        envelope = _envelope("req-1")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            runner=_runner(),
            mode="enforce",
        )
        assert composition.receipt_result is not None
        finding = composition.receipt_result.findings[0]
        assert finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
        classification = composition.classification
        assert (
            classification.primary_runtime_failure
            is CompletionRuntimeFailure.RUNTIME_UNAVAILABLE
        )
        assert classification.claim_fidelity.value == "claim_unverified"
        assert classification.rejection is None
        assert composition.blocks_publication is False

    def test_evicted_receipt_evidence_never_contradicts(self, tmp_path: Path) -> None:
        # Force the paired entry past the chronology's cap: the evicted
        # receipt resolves as no-match — unverified, never contradicted.
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-0", content="payload\n")
        _emit_receipt(accumulator, "occ-0", CANONICAL_COMMAND)
        for index in range(1, 200):
            _observe_write(
                accumulator,
                tmp_path / f"out-{index}.txt",
                f"occ-{index}",
                content="x\n",
            )
        envelope = _envelope("req-1")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            runner=_runner(),
            mode="enforce",
        )
        assert (
            accumulator._chronology.resolution("occ-0") is EvidenceAvailability.EVICTED
        )
        assert composition.receipt_result is not None
        finding = composition.receipt_result.findings[0]
        assert finding.disposition is ReceiptEvidenceDisposition.UNVERIFIED
        assert composition.classification.claim_fidelity.value == "claim_unverified"
        assert composition.classification.rejection is None
        assert composition.blocks_publication is False

    def test_unavailable_finding_never_contradicts(self) -> None:
        finding = ArtifactAuditFinding(
            requirement_id="req-1",
            artifact=CompletionArtifactKind.COMMIT,
            disposition=ArtifactAuditDisposition.UNAVAILABLE,
            evidence="no commit-hash evidence was collected (evidence unavailable)",
        )
        # Unavailable is an availability fact about the harness's evidence:
        # it is not worker-crediting, and the vocabulary has no contradicted
        # member for it to collapse into.
        assert finding.disposition is ArtifactAuditDisposition.UNAVAILABLE
        assert finding.disposition is not ArtifactAuditDisposition.WORKER_CREDITED


class TestArtifactAuditComposition:
    """The C1/C3 audit consumes the composition's evidence surfaces."""

    def test_artifact_claim_credits_through_the_collected_evidence(
        self, tmp_path: Path
    ) -> None:
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        envelope = _touched_path_envelope("artifact:touched_path")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "artifact:touched_path"),
            claimed_values={"artifact:touched_path": str(artifact)},
            mode="stamp",
        )
        assert (
            composition.artifact_findings[0].disposition
            is ArtifactAuditDisposition.WORKER_CREDITED
        )
        assert composition.classification.claim_fidelity.value == "claim_matched"

    def test_snapshot_keyed_witness_reaches_the_artifact_audit(
        self, tmp_path: Path
    ) -> None:
        # The composition hands the artifact audit the PATH-KEYED
        # collection-time snapshots (P7's snapshot crediting path) — an
        # occurrence-keyed map would never match a claimed path.
        artifact = tmp_path / "out.txt"
        artifact.write_text("payload\n", encoding="utf-8")
        accumulator = SubagentRunAccumulator()
        accumulator._artifact_snapshots[str(artifact)] = snapshot_bytes(b"payload\n")
        envelope = _touched_path_envelope("artifact:touched_path")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "artifact:touched_path"),
            claimed_values={"artifact:touched_path": str(artifact)},
            mode="stamp",
        )
        assert (
            composition.artifact_findings[0].disposition
            is ArtifactAuditDisposition.WORKER_CREDITED
        )
        assert composition.artifact_findings[0].evidence == "collection-time snapshot"


class TestRepeatedBuildsRunNoChecks:
    """The once-only finalization: a repeated build consumes it unchanged."""

    def test_cached_finalization_supplies_the_evidence(self, tmp_path: Path) -> None:
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(accumulator, "occ-1", CANONICAL_COMMAND)
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(),
            admission_head="a" * 40,
            publication_head="b" * 40,
            runtime_failure=None,
        )
        first = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            runner=_runner(),
            mode="stamp",
        )
        second = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            finalization=finalization,
            runner=_runner(),
            mode="stamp",
        )
        # The cached finalization is consumed unchanged: the
        # classification is stable across builds, and no check re-runs.
        assert second.classification == first.classification
        assert second.blocks_publication is False

    def test_repeated_builds_are_stable(self, tmp_path: Path) -> None:
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(accumulator, artifact, "occ-1", content="payload\n")
        _emit_receipt(accumulator, "occ-1", CANONICAL_COMMAND)
        envelope = _envelope("req-1")
        finalization = SimpleNamespace(
            checks=(), admission_head=None, publication_head=None, runtime_failure=None
        )
        classifications = [
            compose_completion_audit(
                accumulator=accumulator,
                contract=_contract(envelope),
                claim=_receipt(envelope, "req-1"),
                finalization=finalization,
                runner=_runner(),
                mode="stamp",
            ).classification
            for _ in range(3)
        ]
        assert all(c == classifications[0] for c in classifications)


class TestNoChronologyDump:
    """The chronology never reaches the parent-facing surfaces.

    Pinned structurally: the composed rejection message is built ONLY from
    typed dispositions — never a chronology entry, a receipt's text, or a
    snapshot's sampled bytes.
    """

    def test_rejection_message_names_only_dispositions(self, tmp_path: Path) -> None:
        artifact = tmp_path / "out.txt"
        accumulator = SubagentRunAccumulator()
        _observe_write(
            accumulator, artifact, "occ-1", content="SNAPSHOT-MARKER payload\n"
        )
        _emit_receipt(
            accumulator,
            "occ-1",
            CANONICAL_COMMAND,
            status="failure",
            final_text="RECEIPT-TEXT-MARKER",
        )
        envelope = _envelope("req-1")
        composition = compose_completion_audit(
            accumulator=accumulator,
            contract=_contract(envelope),
            claim=_receipt(envelope, "req-1"),
            runner=_runner(),
            mode="enforce",
        )
        message = composition.rejection_message()
        assert message is not None
        assert "req-1: contradicted" in message
        assert "RECEIPT-TEXT-MARKER" not in message
        assert "SNAPSHOT-MARKER" not in message
        assert "out.txt" not in message
        assert "write_file" not in message
