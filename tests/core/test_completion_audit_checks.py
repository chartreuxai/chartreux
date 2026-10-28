"""P9 acceptance: the C4/C5 completion-check audit.

Unit tests for the audit functions in
:mod:`chartreux.core.completion_audit_checks` — C4 claim admission against
the frozen contract, and C5 evidence classification against the per-run
completion chronology. No subprocesses: the acceptance-check runner (P5)
and its spawn seam are not exercised here.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from chartreux.core.completion_audit_checks import (
    C4Claim,
    C4Verdict,
    CheckEvidenceDisposition,
    admit_check_claim,
    classify_check_evidence,
    last_mutation_sequence,
)
from chartreux.core.completion_checks import (
    CompletionCheckOutcome,
    CompletionCheckRunner,
    CompletionCheckStatus,
    CompletionCheckTemplate,
)
from chartreux.core.completion_contract import CompletionEnvelope, CompletionRequirement
from chartreux.core.completion_evidence import PerRunCompletionChronology
from chartreux.core.events import ToolResultEvent

pytestmark = pytest.mark.timeout(60)

_REQUIREMENT_ID = "tests.core.test_module"
_CANONICAL_PYTEST = "uv run pytest -q"
_NASTY_ROOT = "sub;$(touch pwned)`id`$((1+1))"


def _envelope(*requirement_ids: str) -> CompletionEnvelope:
    """A contract envelope binding the given requirement IDs to pytest."""
    return CompletionEnvelope(
        marker="completion_envelope_v1",
        schema_version=1,
        requirements=tuple(
            CompletionRequirement(
                requirement_id=requirement_id, template=CompletionCheckTemplate.PYTEST
            )
            for requirement_id in requirement_ids
        ),
    )


def _outcome(
    status: CompletionCheckStatus, *, exit_code: int | None = 0, stdout: str = ""
) -> CompletionCheckOutcome:
    return CompletionCheckOutcome(
        template=CompletionCheckTemplate.PYTEST,
        status=status,
        argv=("uv", "run", "pytest", "-q"),
        exit_code=exit_code,
        stdout=stdout,
    )


def _mutation_payload(sequence: int, **footprint: Any) -> Any:
    """A write-classified chronology payload: non-empty write surface."""
    return SimpleNamespace(
        write_paths=footprint.get("write_paths", ("src/app.py",)),
        mutation_roots=footprint.get("mutation_roots", ("src/",)),
    )


def _runner(tmp_path: Any) -> CompletionCheckRunner:
    return CompletionCheckRunner(cwd=tmp_path, workspace=None, path_authority=None)


# --- C4: the claim-fidelity admission rule ---------------------------------


def test_contract_bound_requirement_with_canonical_command_satisfies(
    tmp_path: Any,
) -> None:
    claim = C4Claim(
        requirement_id=_REQUIREMENT_ID,
        template=CompletionCheckTemplate.PYTEST,
        claimed_command=_CANONICAL_PYTEST,
    )
    assert (
        admit_check_claim(
            claim=claim, envelope=_envelope(_REQUIREMENT_ID), runner=_runner(tmp_path)
        )
        is C4Verdict.SATISFIED
    )


def test_unbound_requirement_id_is_unverified(tmp_path: Any) -> None:
    claim = C4Claim(
        requirement_id="tests.core.other_module",
        template=CompletionCheckTemplate.PYTEST,
        claimed_command=_CANONICAL_PYTEST,
    )
    assert (
        admit_check_claim(
            claim=claim, envelope=_envelope(_REQUIREMENT_ID), runner=_runner(tmp_path)
        )
        is C4Verdict.UNVERIFIED
    )


def test_unrelated_git_status_cannot_satisfy_required_module_check(
    tmp_path: Any,
) -> None:
    """A git_status claim cannot evidence a pytest module requirement."""
    claim = C4Claim(
        requirement_id=_REQUIREMENT_ID,
        template=CompletionCheckTemplate.GIT_STATUS,
        claimed_command="git status --porcelain",
    )
    assert (
        admit_check_claim(
            claim=claim, envelope=_envelope(_REQUIREMENT_ID), runner=_runner(tmp_path)
        )
        is C4Verdict.REJECTED
    )


def test_mismatched_claimed_command_is_rejected(tmp_path: Any) -> None:
    claim = C4Claim(
        requirement_id=_REQUIREMENT_ID,
        template=CompletionCheckTemplate.PYTEST,
        claimed_command="uv run pytest tests/core/other -q",
    )
    assert (
        admit_check_claim(
            claim=claim, envelope=_envelope(_REQUIREMENT_ID), runner=_runner(tmp_path)
        )
        is C4Verdict.REJECTED
    )


def test_shell_less_seat_is_exempt_from_admission_policy(tmp_path: Any) -> None:
    claim = C4Claim(
        requirement_id=_REQUIREMENT_ID,
        template=CompletionCheckTemplate.PYTEST,
        claimed_command=_CANONICAL_PYTEST,
    )
    assert (
        admit_check_claim(
            claim=claim,
            envelope=_envelope(_REQUIREMENT_ID),
            runner=_runner(tmp_path),
            seat_has_shell=False,
        )
        is C4Verdict.EXEMPT
    )


# --- C5: the evidence classification ---------------------------------------


def test_managed_job_startup_is_not_check_evidence(tmp_path: Any) -> None:
    """A start observation without a completed result never counts."""
    chronology = PerRunCompletionChronology()
    chronology.observe_receipt(
        CompletionEvidenceReceipt(
            occurrence_id="occ-start",
            tool_call_id="call-1",
            tool_name="bash",
            status="success",
            final_text="",
        )
    )
    classification = classify_check_evidence(
        outcome=None, chronology=chronology, occurrence_id="occ-start"
    )
    assert classification.disposition is CheckEvidenceDisposition.UNAVAILABLE


def test_honest_red_green_retry_sequence_accepted(tmp_path: Any) -> None:
    """A failing check followed by a passing retry credits the retry."""
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_result_event("occ-red", exit_code=1))
    chronology.observe_result_event(_result_event("occ-green", exit_code=0))
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.PASSED),
        chronology=chronology,
        occurrence_id="occ-green",
        last_mutation_sequence=1,
    )
    assert classification.disposition is CheckEvidenceDisposition.WORKER_CREDITED


def test_failed_check_with_completed_envelope_is_contradicted(tmp_path: Any) -> None:
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.NONZERO, exit_code=1),
        envelope=_envelope(_REQUIREMENT_ID),
        envelope_complete=True,
        run_outcome="completed",
    )
    assert classification.disposition is CheckEvidenceDisposition.CONTRADICTED


def test_failed_or_unavailable_with_partial_or_blocked_run_is_informational(
    tmp_path: Any,
) -> None:
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.DENIED, exit_code=None),
        envelope=_envelope(_REQUIREMENT_ID),
        envelope_complete=True,
        run_outcome="partial",
    )
    assert classification.disposition is CheckEvidenceDisposition.INFORMATIONAL


def test_no_text_passed_heuristics(tmp_path: Any) -> None:
    """Stdout text never credits a check; only the typed status does."""
    outcome = _outcome(
        CompletionCheckStatus.NONZERO, exit_code=1, stdout="all tests passed"
    )
    classification = classify_check_evidence(
        outcome=outcome,
        envelope=_envelope(_REQUIREMENT_ID),
        envelope_complete=True,
        run_outcome="completed",
    )
    assert classification.disposition is CheckEvidenceDisposition.CONTRADICTED


def test_exit_zero_after_last_mutation_is_worker_credited(tmp_path: Any) -> None:
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.PASSED, exit_code=0),
        chronology=_chronology_with_mutation_then_check(),
        occurrence_id="occ-check",
        last_mutation_sequence=1,
    )
    assert classification.disposition is CheckEvidenceDisposition.WORKER_CREDITED


def test_exit_zero_with_no_observed_mutation_is_check_context() -> None:
    """Amendment 14: a final-state check never credits the worker.

    The exit-0-with-no-observed-mutation crediting branch is gone: a
    passing check with no collection-time witness tying it to the worker
    is check-context evidence, and check-claim credit belongs to the C4
    receipt path.
    """
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.PASSED, exit_code=0),
        envelope=_envelope(_REQUIREMENT_ID),
        envelope_complete=True,
        run_outcome="completed",
    )
    assert classification.disposition is CheckEvidenceDisposition.UNVERIFIED


def test_failed_check_in_a_shared_root_is_unverified_never_contradicted() -> None:
    """Round-2 amendment 2 / amendments 12-14: external writes possible.

    A failed check observing only the final state under a root shared
    with the parent is check-context evidence — UNVERIFIED with the
    external mutation noted, never a contradiction.
    """
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.NONZERO, exit_code=1),
        envelope=_envelope(_REQUIREMENT_ID),
        envelope_complete=True,
        run_outcome="completed",
        observes_shared_root=True,
    )
    assert classification.disposition is CheckEvidenceDisposition.UNVERIFIED
    assert "shared with the parent" in classification.reason
    assert "external mutation" in classification.reason


def test_failed_check_outside_every_shared_root_stays_contradicted() -> None:
    """CONTRADICTED keeps its force where external writes are not possible."""
    classification = classify_check_evidence(
        outcome=_outcome(CompletionCheckStatus.NONZERO, exit_code=1),
        envelope=_envelope(_REQUIREMENT_ID),
        envelope_complete=True,
        run_outcome="completed",
        observes_shared_root=False,
    )
    assert classification.disposition is CheckEvidenceDisposition.CONTRADICTED


# --- helpers ---------------------------------------------------------------


def _result_event(occurrence_id: str, *, exit_code: int) -> ToolResultEvent:
    return ToolResultEvent(
        tool_name="bash",
        tool_class=None,
        occurrence_id=occurrence_id,
        tool_call_id=f"call-{occurrence_id}",
        result=None,
        error=None,
    )


from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt


def _chronology_with_mutation_then_check() -> PerRunCompletionChronology:
    chronology = PerRunCompletionChronology()
    chronology.observe_result_event(_result_event("occ-mutation", exit_code=0))
    chronology.observe_result_event(_result_event("occ-check", exit_code=0))
    return chronology


# --- last_mutation_sequence -------------------------------------------------


def test_last_mutation_sequence_none_without_writes() -> None:
    assert last_mutation_sequence([]) is None


def test_last_mutation_sequence_tracks_the_latest_write() -> None:
    writes = [
        (1, SimpleNamespace(write_paths=("a.py",))),
        (2, SimpleNamespace(write_paths=())),
    ]
    assert last_mutation_sequence(writes) == 2


def test_last_mutation_sequence_ignores_non_write_payloads() -> None:
    writes = [
        (1, SimpleNamespace(write_paths=())),
        (2, SimpleNamespace(write_paths=("b.py",))),
    ]
    assert last_mutation_sequence(writes) == 2
