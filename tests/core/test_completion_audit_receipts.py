"""P8: the C2 receipt audit — retry-distinguishing receipt evidence.

The audit resolves the worker's envelope-linked receipt claims against the
run's collected receipt chronology, keyed by the harness-minted, run-scoped
receipt occurrence ID — never tool_call_id alone (tool-call ids are reused
across turns). The round-12 regression drives the COMPLETE
dispatch-through-audit identity chain (identical occurrence IDs across
dispatch, callback, collection, and audit) through the real audit consumer;
the round-6 interleaving test pins the required non-crediting (check-context)
disposition for an intervened shared-root artifact.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
import json
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel
import pytest

from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.completion_audit_receipts import (
    ReceiptAuditFinding,
    ReceiptEvidenceDisposition,
    audit_completion_receipts,
)
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    CompletionArtifactKind,
    CompletionCheckTemplate,
    CompletionEnvelope,
    CompletionReceipt,
    CompletionRequirement,
)
from chartreux.core.completion_evidence import (
    ContentSnapshot,
    EvidenceAvailability,
    PerRunCompletionChronology,
    snapshot_bytes,
)
from chartreux.core.events import BaseEvent, ToolResultEvent, ToolStreamEvent
from chartreux.core.hooks.models import ToolStatus
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.subagents import SubagentRunAccumulator
from chartreux.core.tools import secret_redaction as sr
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolPermission,
)
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_tool import FakeToolResult


@pytest.fixture(autouse=True)
def _clean_redaction_state() -> Any:
    sr.set_env_passthrough(())
    sr.reset_cache()
    yield
    sr.set_env_passthrough(())
    sr.reset_cache()


def _envelope() -> CompletionEnvelope:
    return CompletionEnvelope(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        requirements=(
            CompletionRequirement(
                requirement_id="check:pytest", template=CompletionCheckTemplate.PYTEST
            ),
            CompletionRequirement(
                requirement_id="artifact:touched_path",
                artifact=CompletionArtifactKind.TOUCHED_PATH,
            ),
        ),
    )


def _linked_claim(*satisfied: str) -> CompletionReceipt:
    return CompletionReceipt(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        envelope=_envelope(),
        satisfied_requirement_ids=tuple(satisfied),
    )


def _unlinked_claim(*satisfied: str) -> CompletionReceipt:
    return CompletionReceipt(
        marker=COMPLETION_ENVELOPE_MARKER,
        schema_version=COMPLETION_CONTRACT_SCHEMA_VERSION,
        satisfied_requirement_ids=tuple(satisfied),
    )


def _check_receipt(
    occurrence_id: str,
    tool_call_id: str,
    *,
    status: ToolStatus = "success",
    command: str = "uv run pytest -q tests/x.py",
) -> CompletionEvidenceReceipt:
    return CompletionEvidenceReceipt(
        occurrence_id=occurrence_id,
        tool_call_id=tool_call_id,
        tool_name="stub_tool",
        status=status,
        final_text="ok",
        executed_args={"command": command},
    )


def _credited_result_event(occurrence_id: str, tool_call_id: str) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="stub_tool",
        tool_class=None,
        tool_call_id=tool_call_id,
        occurrence_id=occurrence_id,
    )


def _witness(content: bytes) -> ContentSnapshot:
    return snapshot_bytes(content)


def _by_id(
    findings: tuple[ReceiptAuditFinding, ...], requirement_id: str
) -> ReceiptAuditFinding:
    return next(f for f in findings if f.requirement_id == requirement_id)


# --- Envelope-linked strict receipts only; no prose extraction -----------


def test_unlinked_claim_resolves_unverified() -> None:
    claim = _unlinked_claim("check:pytest")
    result = audit_completion_receipts(
        claim=claim,
        chronology=PerRunCompletionChronology(),
        reference_commands={"check:pytest": "uv run pytest -q"},
    )

    assert result.envelope_linked is False
    assert result.worker_credited is False
    assert [finding.disposition for finding in result.findings] == [
        ReceiptEvidenceDisposition.UNVERIFIED
    ]


def test_malformed_claim_resolves_unverified() -> None:
    result = audit_completion_receipts(
        claim={"marker": "not-the-marker", "schema_version": 1},
        chronology=PerRunCompletionChronology(),
        reference_commands={"check:pytest": "uv run pytest -q"},
    )

    assert result.envelope_linked is False
    assert result.findings == ()
    assert result.worker_credited is False


def test_no_claim_yields_no_findings() -> None:
    result = audit_completion_receipts(
        claim=None,
        chronology=PerRunCompletionChronology(),
        reference_commands={"check:pytest": "uv run pytest -q"},
    )

    assert result.envelope_linked is False
    assert result.findings == ()
    assert result.worker_credited is False


# --- No matching command = unverified --------------------------------------


def test_no_matching_command_resolves_unverified() -> None:
    claim = _linked_claim("check:pytest")
    result = audit_completion_receipts(
        claim=claim,
        chronology=PerRunCompletionChronology(),
        reference_commands={"check:pytest": "uv run pytest -q"},
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNVERIFIED
    assert result.worker_credited is False


def test_unmatched_tool_operand_never_satisfies_the_check() -> None:
    """A different tool's invocation never satisfies a check requirement."""
    chronology = PerRunCompletionChronology()
    chronology.observe_receipt(
        _check_receipt("occ-other", "call-a", command="uv run ruff check .")
    )
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNVERIFIED
    assert result.worker_credited is False


def test_later_operands_never_break_the_command_match() -> None:
    """Further operands (targets, the check root) after the canonical
    command's tokens never break the match.
    """
    chronology = PerRunCompletionChronology()
    chronology.observe_receipt(
        _check_receipt(
            "occ-1",
            "call-a",
            command="uv run pytest -q tests/core/test_completion_audit_receipts.py",
        )
    )
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: snapshot_bytes(b"ok"),
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.OBSERVED


# --- Worker-crediting and the snapshot witness ------------------------------


def test_success_receipt_credits_with_the_snapshot_witness() -> None:
    """Round-5: the collection-time snapshot is the mutation witness the
    worker-crediting disposition consumes.
    """
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-a"))
    chronology.observe_receipt(
        _check_receipt("occ-1", "call-a", command="uv run pytest -q tests/x.py")
    )
    witness = snapshot_bytes(b"artifact content")
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.OBSERVED
    assert finding.occurrence_id == "occ-1"
    assert result.worker_credited is True


def test_missing_witness_refuses_worker_credit() -> None:
    """Round-5: worker credit requires a recorded collection-time witness."""
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-a"))
    chronology.observe_receipt(
        _check_receipt("occ-1", "call-a", command="uv run pytest -q tests/x.py")
    )
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    assert finding.occurrence_id == "occ-1"
    assert result.worker_credited is False
    evidence = result.evidence()
    assert evidence.runtime_failure == "unavailable"
    assert evidence.contradicted_requirement_ids == ()


def test_truncated_witness_refuses_worker_credit() -> None:
    witness = ContentSnapshot(
        content_hash="0" * 64,
        byte_size=4096,
        head_bytes=b"head",
        tail_bytes=b"tail",
        truncated=True,
    )
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-a"))
    chronology.observe_receipt(
        _check_receipt("occ-1", "call-a", command="uv run pytest -q tests/x.py")
    )
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    assert result.worker_credited is False


def test_evicted_receipt_resolves_unavailable() -> None:
    """An evicted entry resolves exactly like unavailable — an availability
    fact, never contradictory.
    """

    class _EvictingChronology(PerRunCompletionChronology):
        """A chronology whose matched entry resolves EVICTED.

        The real chronology drops evicted entries from its collected set
        (P3's own tests pin that); this double retains the entry metadata
        while reporting the eviction, pinning the audit's consumption rule:
        a matched receipt resolving EVICTED classifies unavailable.
        """

        def resolution(self, occurrence_id: str) -> Any:
            if occurrence_id == "occ-old":
                return EvidenceAvailability.EVICTED
            return super().resolution(occurrence_id)

    chronology = _EvictingChronology()
    chronology.observe_result_event(_credited_result_event("occ-old", "call-old"))
    chronology.observe_receipt(
        _check_receipt("occ-old", "call-old", command="uv run pytest -q tests/x.py")
    )
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: snapshot_bytes(b"ok"),
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    assert finding.occurrence_id == "occ-old"
    assert "evicted" in finding.detail
    evidence = result.evidence()
    assert evidence.runtime_failure == "unavailable"
    assert evidence.contradicted_requirement_ids == ()


# --- Worker-side conflicts ---------------------------------------------------


def test_single_failure_receipt_contradicts_the_claim() -> None:
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-a"))
    chronology.observe_receipt(
        _check_receipt(
            "occ-1", "call-a", status="failure", command="uv run pytest -q tests/x.py"
        )
    )
    witness = snapshot_bytes(b"ok")
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.CONTRADICTED
    assert finding.occurrence_id == "occ-1"
    evidence = result.evidence()
    assert evidence.runtime_failure is None
    assert evidence.contradicted_requirement_ids == ("check:pytest",)


def test_cancelled_receipt_resolves_unavailable() -> None:
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-a"))
    chronology.observe_receipt(
        _check_receipt(
            "occ-1", "call-a", status="cancelled", command="uv run pytest -q tests/x.py"
        )
    )
    witness = snapshot_bytes(b"ok")
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    assert result.worker_credited is False


# --- Retry-distinguishing (round 9) ------------------------------------------


def test_retry_credits_the_successful_retry_not_the_failed_attempt() -> None:
    """Round-9: two turns reusing one tool_call_id — a failed check followed
    by a successful retry — each receipt resolves ONLY to its originating
    invocation, and the correctly referenced successful retry is credited;
    the earlier failed attempt of the same command never condemns it.
    """
    chronology = PerRunCompletionChronology()
    # Callback-before-observation order: the receipts arrive first.
    chronology.observe_receipt(
        _check_receipt(
            "occ-failed",
            "reused",
            status="failure",
            command="uv run pytest -q tests/x.py",
        )
    )
    chronology.observe_receipt(
        _check_receipt("occ-retry", "reused", command="uv run pytest -q tests/x.py")
    )
    chronology.observe_result_event(_credited_result_event("occ-failed", "reused"))
    chronology.observe_result_event(_credited_result_event("occ-retry", "reused"))
    witness = snapshot_bytes(b"retry ok")
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.OBSERVED
    # The claim's evidence is the SUCCESSFUL retry's invocation, never the
    # earlier failed attempt sharing the tool_call_id.
    assert finding.occurrence_id == "occ-retry"
    failed_entry = chronology.entry("occ-failed")
    retry_entry = chronology.entry("occ-retry")
    assert failed_entry is not None and retry_entry is not None
    assert failed_entry.receipt is not None
    assert failed_entry.receipt.status == "failure"
    assert retry_entry.receipt is not None
    assert retry_entry.receipt.status == "success"


def test_regressed_check_resolves_unavailable_never_condemned() -> None:
    """A later same-command failure after an earlier success is ambiguous
    evidence: it resolves unavailable, never condemned.
    """
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-pass", "call-1"))
    chronology.observe_receipt(
        _check_receipt("occ-pass", "call-1", command="uv run pytest -q tests/x.py")
    )
    chronology.observe_result_event(_credited_result_event("occ-fail", "call-2"))
    chronology.observe_receipt(
        _check_receipt(
            "occ-fail",
            "call-2",
            status="failure",
            command="uv run pytest -q tests/x.py",
        )
    )
    witness = snapshot_bytes(b"ok")
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    evidence = result.evidence()
    assert evidence.runtime_failure == "unavailable"
    assert evidence.contradicted_requirement_ids == ()


def test_repeated_failures_contradict_the_claim() -> None:
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-1"))
    chronology.observe_receipt(
        _check_receipt(
            "occ-1", "call-1", status="failure", command="uv run pytest -q tests/x.py"
        )
    )
    chronology.observe_result_event(_credited_result_event("occ-2", "call-2"))
    chronology.observe_receipt(
        _check_receipt(
            "occ-2", "call-2", status="failure", command="uv run pytest -q tests/x.py"
        )
    )
    witness = snapshot_bytes(b"ok")
    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=lambda _occurrence_id: witness,
    )

    finding = result.findings[0]
    assert finding.disposition is ReceiptEvidenceDisposition.CONTRADICTED
    assert finding.occurrence_id == "occ-2"
    evidence = result.evidence()
    assert evidence.runtime_failure is None
    assert evidence.contradicted_requirement_ids == ("check:pytest",)


def test_shared_root_artifact_never_credits_the_worker() -> None:
    """Round-6 H2: even an intact collection-time witness never credits a
    shared-root artifact — check-context evidence never credits the worker.
    """
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_credited_result_event("occ-1", "call-a"))
    chronology.observe_receipt(
        _check_receipt("occ-1", "call-a", command="uv run pytest -q tests/x.py")
    )
    claim = _linked_claim("check:pytest", "artifact:touched_path")
    result = audit_completion_receipts(
        claim=claim,
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        touched_paths=["/tmp/chartreux-scratchpad/shared-artifact.txt"],
        shared_roots=["/tmp/chartreux-scratchpad"],
        snapshot_witness=lambda _occurrence_id: snapshot_bytes(b"artifact content"),
    )

    check_finding = _by_id(result.findings, "check:pytest")
    artifact_finding = _by_id(result.findings, "artifact:touched_path")
    assert check_finding.disposition is ReceiptEvidenceDisposition.OBSERVED
    assert artifact_finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    assert result.worker_credited is False
    evidence = result.evidence()
    assert evidence.runtime_failure == "unavailable"
    assert evidence.contradicted_requirement_ids == ()


# --- The complete dispatch-through-audit identity regression (round 12) ----


class _CheckToolArgs(BaseModel):
    command: str = ""


class _CheckTool(
    BaseTool[_CheckToolArgs, FakeToolResult, BaseToolConfig, BaseToolState]
):
    """A check tool: the first invocation fails, the retry succeeds.

    The successful retry writes its report artifact, which the test
    snapshots at collection time as the mutation witness.
    """

    attempts: ClassVar[int] = 0
    report_path: ClassVar[Path | None] = None

    @classmethod
    def get_name(cls) -> str:
        return "stub_tool"

    async def run(
        self, args: _CheckToolArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | FakeToolResult, None]:
        type(self).attempts += 1
        if type(self).attempts == 1:
            raise RuntimeError("first check failed")
        report_path = type(self).report_path
        if report_path is not None:
            report_path.write_text("retry ok\n", encoding="utf-8")
        yield FakeToolResult(message="retry ok")


def _stub_check_call(call_id: str, command: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="stub_tool", arguments=f'{{"command": "{command}"}}'
        ),
    )


@pytest.mark.asyncio
async def test_dispatch_through_audit_identity_regression(tmp_path: Path) -> None:
    """Round-12: the COMPLETE dispatch-through-audit identity regression.

    Identical occurrence IDs across dispatch, callback, collection, AND
    audit, exercised through the real audit consumer, with two turns
    reusing the same tool_call_id — a failed check followed by a
    successful retry. Each receipt resolves ONLY to its originating
    invocation: no double-counting, no fabricated distinction.
    """
    report = tmp_path / "check-report.txt"
    _CheckTool.attempts = 0
    _CheckTool.report_path = report
    accumulator = SubagentRunAccumulator()
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["stub_tool"],
            tools={"stub_tool": {"permission": ToolPermission.ALWAYS.value}},
        ),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Try.",
                    tool_calls=[
                        _stub_check_call("reused", "uv run pytest -q tests/x.py")
                    ],
                )
            ],
            [
                mock_llm_chunk(
                    content="Retry.",
                    tool_calls=[
                        _stub_check_call("reused", "uv run pytest -q tests/x.py")
                    ],
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ]),
    )
    loop.tool_manager._all_tools["stub_tool"] = _CheckTool
    chronology = accumulator._chronology
    # The real callback path, wrapped only to record the arrival timeline.
    real_sink = accumulator.completion_evidence_sink()
    timeline: list[tuple[str, str]] = []

    def recording_sink(receipt: CompletionEvidenceReceipt) -> None:
        timeline.append(("receipt", receipt.occurrence_id))
        real_sink(receipt)

    loop._completion_evidence_sink = recording_sink

    witnesses: dict[str, ContentSnapshot] = {}
    events: list[BaseEvent] = []
    async for event in loop.act("Go"):
        events.append(event)
        if isinstance(event, ToolResultEvent):
            accumulator.observe(event, tool_call_id="parent-call")
            chronology.observe_result_event(event)
            if event.occurrence_id:
                timeline.append(("observed", event.occurrence_id))
            if event.error is None and event.occurrence_id:
                # The collection-time mutation witness, captured at
                # collection — never at finalization.
                witnesses[event.occurrence_id] = snapshot_bytes(report.read_bytes())

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 2
    failed_event = next(e for e in results if e.error is not None)
    succeeded_event = next(e for e in results if e.error is None)
    assert failed_event.tool_call_id == succeeded_event.tool_call_id == "reused"
    # Dispatch: the executor minted a distinct run-scoped occurrence ID for
    # each invocation, before the invocation started.
    assert failed_event.occurrence_id
    assert succeeded_event.occurrence_id
    assert failed_event.occurrence_id != succeeded_event.occurrence_id

    # Collection: exactly one chronology entry per invocation — the reused
    # tool_call_id never double-counts.
    assert len(chronology._entries) == 2
    failed_entry = chronology.entry(failed_event.occurrence_id)
    retry_entry = chronology.entry(succeeded_event.occurrence_id)
    assert failed_entry is not None and retry_entry is not None
    assert failed_entry.receipt is not None
    assert failed_entry.result_event is not None
    assert retry_entry.receipt is not None
    assert retry_entry.result_event is not None
    # Callback-before-observation order: each side-channel receipt arrived
    # before the accumulator observed the queued result.
    for occurrence_id in (failed_event.occurrence_id, succeeded_event.occurrence_id):
        marks = [kind for kind, occ in timeline if occ == occurrence_id]
        assert marks == ["receipt", "observed"]
    # Callback payload identity: the receipts carry the executor-minted
    # occurrence IDs unchanged — never reminted.
    assert failed_entry.receipt.occurrence_id == failed_event.occurrence_id
    assert retry_entry.receipt.occurrence_id == succeeded_event.occurrence_id
    assert failed_entry.receipt.status == "failure"
    assert retry_entry.receipt.status == "success"

    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest"),
        chronology=chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        snapshot_witness=witnesses.get,
    )

    assert result.envelope_linked is True
    assert len(result.findings) == 1
    finding = result.findings[0]
    # Audit: the finding resolves to the SUCCESSFUL retry's invocation — the
    # identical occurrence ID across dispatch, callback, collection, and
    # audit. The correctly referenced successful retry is credited; the
    # earlier failed attempt of the same command never condemns it.
    assert finding.occurrence_id == succeeded_event.occurrence_id
    assert finding.occurrence_id != failed_event.occurrence_id
    assert finding.disposition is ReceiptEvidenceDisposition.OBSERVED
    assert result.worker_credited is True


# --- The round-5 end-to-end integration test --------------------------------


class _ArtifactToolArgs(BaseModel):
    command: str = ""
    file_path: str = ""
    content: str = ""


class _ArtifactToolResult(FakeToolResult):
    file_path: str = ""


class _ArtifactTool(
    BaseTool[_ArtifactToolArgs, _ArtifactToolResult, BaseToolConfig, BaseToolState]
):
    """A check that also writes its artifact under the shared root."""

    effect_kind = ToolEffectKind.FILE_WRITE

    @classmethod
    def get_name(cls) -> str:
        return "stub_tool"

    async def run(
        self, args: _ArtifactToolArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | _ArtifactToolResult, None]:
        Path(args.file_path).write_text(args.content, encoding="utf-8")
        yield _ArtifactToolResult(message="written", file_path=args.file_path)


@pytest.mark.asyncio
async def test_intervening_parent_edit_never_credits_the_worker(tmp_path: Path) -> None:
    """Round-5: the end-to-end integration test through the real audit.

    Real evidence collected through the accumulator — the side-channel
    receipts and the observed result events — then an intervening parent
    edit on the shared-root artifact. The audit must refuse worker credit
    for the interleaved artifact: a check-context availability fact, never
    a contradiction, even with an intact collection-time witness.
    """
    artifact = tmp_path / "shared-artifact.txt"
    accumulator = SubagentRunAccumulator()
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            enabled_tools=["stub_tool"],
            tools={"stub_tool": {"permission": ToolPermission.ALWAYS.value}},
        ),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="Checking.",
                    tool_calls=[
                        ToolCall(
                            id="call-write",
                            index=0,
                            function=FunctionCall(
                                name="stub_tool",
                                arguments=json.dumps({
                                    "command": "uv run pytest -q",
                                    "file_path": str(artifact),
                                    "content": "worker content\n",
                                }),
                            ),
                        )
                    ],
                )
            ],
            [mock_llm_chunk(content="Done.")],
        ]),
    )
    loop.tool_manager._all_tools["stub_tool"] = _ArtifactTool
    loop._completion_evidence_sink = accumulator.completion_evidence_sink()

    witnesses: dict[str, ContentSnapshot] = {}
    events: list[BaseEvent] = []
    async for event in loop.act("Go"):
        events.append(event)
        if isinstance(event, ToolResultEvent):
            accumulator.observe(event, tool_call_id="parent-call")
            accumulator._chronology.observe_result_event(event)
            if event.error is None and event.occurrence_id:
                # The collection-time mutation witness of the artifact the
                # run produced — captured at collection, before the parent's
                # intervening edit.
                witnesses[event.occurrence_id] = snapshot_bytes(artifact.read_bytes())

    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert len(results) == 1
    write_event = results[0]
    assert write_event.error is None
    assert write_event.occurrence_id
    # The accumulator collected the artifact as the run's touched path.
    touched = list(accumulator._touched_paths)
    assert touched == [str(artifact)]

    # The intervening parent edit on the shared-root artifact: the final
    # state no longer matches the collection-time witness.
    artifact.write_text("parent edit\n", encoding="utf-8")
    witness = witnesses[write_event.occurrence_id]
    assert witness.head_bytes == b"worker content\n"
    assert artifact.read_bytes() == b"parent edit\n"

    result = audit_completion_receipts(
        claim=_linked_claim("check:pytest", "artifact:touched_path"),
        chronology=accumulator._chronology,
        reference_commands={"check:pytest": "uv run pytest -q"},
        touched_paths=touched,
        shared_roots=[str(tmp_path)],
        snapshot_witness=witnesses.get,
    )

    check_finding = _by_id(result.findings, "check:pytest")
    artifact_finding = _by_id(result.findings, "artifact:touched_path")
    # The check the run executed is worker-credited through the real
    # receipt and its collection-time witness.
    assert check_finding.disposition is ReceiptEvidenceDisposition.OBSERVED
    assert check_finding.occurrence_id == write_event.occurrence_id
    # The interleaved shared-root artifact is check-context evidence: the
    # audit refuses worker credit — an availability fact, never a
    # contradiction, even with an intact collection-time witness.
    assert artifact_finding.disposition is ReceiptEvidenceDisposition.UNAVAILABLE
    assert "shared" in artifact_finding.detail
    assert result.worker_credited is False
    evidence = result.evidence()
    assert evidence.contradicted_requirement_ids == ()
    assert evidence.runtime_failure == "unavailable"
