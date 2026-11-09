"""v0.7.6 P2 chunk 2e: the worker-digest contract in ``core/subagents.py``.

Covers the shared definitions the fetch routes' digest rendering assembles:
the always-contains list, the decision-sufficient ``WorkerDigest`` model,
and ``build_worker_digest`` — the assembly's guarantees: the always-contains
fields are present in every assembled digest, truncation/missing indicators
are explicit, and worker claims stay distinguished from audited evidence.
"""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from chartreux.core.completion_contract import (
    CompletionAuditClassification,
    CompletionClaimFidelity,
    CompletionDeclaredOutcome,
    CompletionDisclosure,
    CompletionMandateSatisfaction,
    CompletionRuntimeFailure,
)
from chartreux.core.recovery_models import RecoveryEligibility, RecoveryReference
from chartreux.core.subagents import (
    WORKER_DIGEST_ALWAYS_CONTAINS,
    WORKER_DIGEST_SUMMARY_MAX_CHARS,
    RunStatus,
    RunStopReason,
    TaskResult,
    WorkerDigest,
    WorkerDigestClaims,
    WorkerDigestDisclosures,
    build_worker_digest,
)


def _audit_stamp() -> CompletionAuditClassification:
    return CompletionAuditClassification(
        claim_fidelity=CompletionClaimFidelity.CLAIM_MATCHED,
        disclosure_completeness=CompletionDisclosure.DISCLOSURE_COMPLETE,
        mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_SATISFIED,
        declared_outcome=CompletionDeclaredOutcome.OUTCOME_COMPLETED,
        primary_runtime_failure=CompletionRuntimeFailure.RUNTIME_HEALTHY,
    )


def _result(**overrides: object) -> TaskResult:
    base: dict[str, object] = {
        "response": "Worker report body",
        "turns_used": 3,
        "completed": True,
        "run_status": RunStatus.COMPLETED,
    }
    base.update(overrides)
    return TaskResult(**base)  # type: ignore[arg-type]


class TestAlwaysContainsContract:
    """The always-contains list is the pinned disclosure trio."""

    def test_the_list_is_exactly_the_disclosure_trio(self) -> None:
        assert WORKER_DIGEST_ALWAYS_CONTAINS == (
            "touched_paths",
            "denial_record",
            "check_status",
        )

    def test_the_list_is_structurally_present_on_the_claims_model(self) -> None:
        assert set(WORKER_DIGEST_ALWAYS_CONTAINS) <= set(
            WorkerDigestClaims.model_fields
        )

    @pytest.mark.parametrize(
        "disclosures",
        [
            None,
            WorkerDigestDisclosures(
                touched_paths=["a.py"], denial_record=[], check_status="pytest: 0"
            ),
            WorkerDigestDisclosures(
                touched_paths=[],
                denial_record=["denied: reason"],
                check_status="none run",
            ),
        ],
    )
    def test_every_assembled_digest_contains_the_always_contains_list(
        self, disclosures: WorkerDigestDisclosures | None
    ) -> None:
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(),
            disclosures=disclosures,
        )
        dump = digest.model_dump(mode="json")
        assert "claims" in dump
        claims_dump = dump["claims"]
        for key in WORKER_DIGEST_ALWAYS_CONTAINS:
            assert key in claims_dump
            assert f"{key}_missing" in claims_dump


class TestDigestAssembly:
    """build_worker_digest carries identity, disposition, and recovery state."""

    def test_identity_disposition_and_timing_carry_through(self) -> None:
        recovery = RecoveryReference(
            recovery_chain_id="chain-1",
            event_id="evt-1",
            eligibility=RecoveryEligibility.ELIGIBLE,
        )
        digest = build_worker_digest(
            agent_id="agent-7",
            run_id="run-9",
            result=_result(
                name="coder-model-7",
                stop_reason=RunStopReason.USER_CANCELLED,
                recovery=recovery,
                completed=False,
                run_status=RunStatus.CANCELLED,
            ),
            elapsed_seconds=12.5,
        )
        assert digest.agent_id == "agent-7"
        assert digest.run_id == "run-9"
        assert digest.name == "coder-model-7"
        assert digest.completed is False
        assert digest.run_status is RunStatus.CANCELLED
        assert digest.stop_reason is RunStopReason.USER_CANCELLED
        assert digest.turns_used == 3
        assert digest.elapsed_seconds == 12.5
        assert digest.recovery is recovery
        assert digest.response_available is True

    def test_name_falls_back_to_the_result_name(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result(name="coder-model-1")
        )
        assert digest.name == "coder-model-1"

    def test_run_record_facts_override_the_result_fields(self) -> None:
        # The registry's run record is the authoritative disposition source:
        # explicit run_status/stop_reason win over the result's own fields.
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(run_status=RunStatus.COMPLETED),
            run_status=RunStatus.FAILED,
            stop_reason=RunStopReason.BUDGET_EXCEEDED,
        )
        assert digest.run_status is RunStatus.FAILED
        assert digest.stop_reason is RunStopReason.BUDGET_EXCEEDED

    def test_summary_is_bounded_and_truncation_flagged(self) -> None:
        response = "x" * (WORKER_DIGEST_SUMMARY_MAX_CHARS + 500)
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result(response=response)
        )
        assert len(digest.claims.summary) == WORKER_DIGEST_SUMMARY_MAX_CHARS
        assert digest.claims.summary == response[:WORKER_DIGEST_SUMMARY_MAX_CHARS]
        assert digest.claims.summary_truncated is True

    def test_short_summary_is_carried_whole_and_not_flagged(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result(response="short report")
        )
        assert digest.claims.summary == "short report"
        assert digest.claims.summary_truncated is False

    def test_custom_summary_cap_is_honored(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(response="abcdefghij"),
            max_summary_chars=4,
        )
        assert digest.claims.summary == "abcd"
        assert digest.claims.summary_truncated is True

    def test_summary_cap_below_one_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_summary_chars"):
            build_worker_digest(
                agent_id="agent-1",
                run_id="run-1",
                result=_result(),
                max_summary_chars=0,
            )


class TestMissingAndTruncationIndicators:
    """Unestablished inputs become explicit indicators, never absent fields."""

    def test_missing_disclosure_inputs_set_every_indicator(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result()
        )
        claims = digest.claims
        assert claims.touched_paths == []
        assert claims.touched_paths_missing is True
        assert claims.denial_record == []
        assert claims.denial_record_missing is True
        assert claims.check_status == ""
        assert claims.check_status_missing is True

    def test_empty_response_is_an_explicit_missing_indicator(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(response="", completed=False),
        )
        assert digest.response_available is False
        assert digest.claims.summary == ""
        assert digest.claims.summary_truncated is False

    def test_reported_none_denials_distinguish_missing(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(),
            disclosures=WorkerDigestDisclosures(
                touched_paths=[], denial_record=[], check_status="none run"
            ),
        )
        claims = digest.claims
        assert claims.touched_paths == []
        assert claims.touched_paths_missing is False
        assert claims.denial_record == []
        assert claims.denial_record_missing is False
        assert claims.check_status == "none run"
        assert claims.check_status_missing is False

    def test_supplied_disclosures_populate_the_claims(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(),
            disclosures=WorkerDigestDisclosures(
                touched_paths=["chartreux/core/subagents.py", "tests/core/new.py"],
                denial_record=["git push: publication requires runtime approval"],
                check_status="pytest: exit 0; git_status: exit 0",
            ),
        )
        claims = digest.claims
        assert claims.touched_paths == [
            "chartreux/core/subagents.py",
            "tests/core/new.py",
        ]
        assert claims.touched_paths_missing is False
        assert claims.denial_record == [
            "git push: publication requires runtime approval"
        ]
        assert claims.denial_record_missing is False
        assert claims.check_status == "pytest: exit 0; git_status: exit 0"
        assert claims.check_status_missing is False


class TestClaimsVsAuditedEvidence:
    """Worker claims and audited evidence travel as separate provenance."""

    def test_audited_stamp_is_distinct_from_the_claims(self) -> None:
        stamp = _audit_stamp()
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(completion_audit=stamp),
            disclosures=WorkerDigestDisclosures(
                touched_paths=["a.py"], denial_record=[], check_status="pytest: 0"
            ),
        )
        assert digest.audited is stamp
        # The claims half carries exactly the summary plus the
        # always-contains list with its indicators — nothing audited.
        assert set(WorkerDigestClaims.model_fields) == {
            "summary",
            "summary_truncated",
            *WORKER_DIGEST_ALWAYS_CONTAINS,
            "touched_paths_missing",
            "denial_record_missing",
            "check_status_missing",
        }
        claims_dump = digest.claims.model_dump(mode="json")
        assert "audited" not in claims_dump
        assert "completion_audit" not in claims_dump

    def test_no_audit_stamp_means_no_audited_field_in_the_dump(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result()
        )
        assert digest.audited is None
        assert "audited" not in digest.model_dump(mode="json")


class TestDigestModelShape:
    """The digest is a strict, frozen, serializable contract."""

    def test_digest_is_frozen(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result()
        )
        with pytest.raises(ValidationError, match="frozen"):
            digest.turns_used = 99  # type: ignore[misc]

    def test_digest_rejects_unknown_fields(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1", run_id="run-1", result=_result()
        )
        with pytest.raises(ValidationError):
            WorkerDigest(**{**digest.model_dump(mode="json"), "surprise": 1})

    def test_digest_round_trips_through_serialization(self) -> None:
        digest = build_worker_digest(
            agent_id="agent-1",
            run_id="run-1",
            result=_result(completion_audit=_audit_stamp()),
            elapsed_seconds=1.5,
            disclosures=WorkerDigestDisclosures(
                touched_paths=["a.py"],
                denial_record=["cmd: reason"],
                check_status="pytest: 0",
            ),
        )
        payload = digest.model_dump(mode="json")
        assert WorkerDigest.model_validate(payload) == digest
        # The always-contains keys survive the wire serialization.
        for key in WORKER_DIGEST_ALWAYS_CONTAINS:
            assert key in payload["claims"]
