"""P12 — the publication-rejection lifecycle.

Covers the P12 acceptance: the production ``build_result`` call sites wire
the composition surface (the run key's finalization-cache read, the frozen
contract's mandate surface and audit mode, the worker's receipt claim, and
the acceptance-check runner) on both the background monitor and the
foreground path; an enforce-mode audit rejection finalizes the run with
completed=False + a failed run status + the measured P11 rejection envelope,
with the result's audit stamp, the run status, and the envelope agreeing;
primary runtime failures and cancellation reasons are preserved (never
replaced by the rejection envelope); honest abort/blocker outcomes are never
rejected; a rejected run never suspends and refuses a same-agent resume (the
suspension gates stay shut); the rejected predecessor's envelope authorizes
a cross-agent replacement through the existing lineage; and the audit stamp
survives metadata replacement, storage, wait, and retrieval.

Persisted-result compatibility: ``TaskResult.completion_audit`` is omitted
from dumps when ``None`` (the pre-audit dump shape is unchanged); mixed-version
consumption — an old reader consuming a new persisted result with a
populated audit — is unsupported.

No live-provider calls: the check spawn seam is monkeypatched; the child
backends are fakes.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import (
    SessionRuntimeRegistry,
    _AgentState,
    _public_failure_envelope,
)
from chartreux.app_server._subagent_execution import (
    BackgroundExecutionContext,
    extract_completion_claim,
    operative_completion_claim,
    rejection_envelope,
)
from chartreux.app_server._subagent_records import (
    CompletionCapability,
    FrozenCompletionContract,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server.models import (
    PublicFailureClass,
    PublicTurnStatus,
    PublicTurnStopReason,
)
from chartreux.app_server.protocol import TurnInterruptParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.completion_checks import CompletionCheckTemplate
from chartreux.core.completion_contract import (
    COMPLETION_CONTRACT_SCHEMA_VERSION,
    COMPLETION_ENVELOPE_MARKER,
    CompletionAuditClassification,
    CompletionClaimFidelity,
    CompletionDeclaredOutcome,
    CompletionDisclosure,
    CompletionMandateSatisfaction,
    CompletionReceipt,
    CompletionRejectionReason,
    CompletionRuntimeFailure,
)
from chartreux.core.config.models import (
    CompletionSectionConfig,
    RecoveryConfig,
    RecoverySectionConfig,
)
from chartreux.core.llm_models import LLMChunk
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.recovery_models import (
    FailureClass,
    FailureConfidence,
    FailurePhase,
    PackageTier,
    RecoveryEligibility,
)
from chartreux.core.subagents import (
    CancelOutcome,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    SubagentRunAccumulator,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    ImmediateFailureBackend,
    _background_result,
    _config,
    _GatedWakeup,
    _ManualClock,
    _retention_registry,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_replacement_policy import (
    _claim_config,
    _launch,
    _LoopBackends,
    _replacement_catalog,
)
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.timeout(60)

AuditMode = Literal["stamp", "enforce"]


# ----------------------------------------------------------------------
# Shared fixtures (the P6a/P6b harness surface, re-derived here)
# ----------------------------------------------------------------------


def _freeze(**overrides: Any) -> FrozenCompletionContract:
    """Freeze a contract with coder defaults plus the given overrides."""
    params: dict[str, Any] = {
        "identity": "coder",
        "identity_revision": "rev-1",
        "seat": None,
        "tier": PackageTier.SUBSTANTIVE,
        "cwd": "/workspace",
        "roots": ("/workspace",),
        "completion_check": None,
        "required_artifacts": None,
        "shell_less": False,
        "audit_mode": "stamp",
        "capability": CompletionCapability.AVAILABLE,
    }
    params.update(overrides)
    return FrozenCompletionContract.freeze(**params)


class _SpawnRecorder:
    """The monkeypatched check spawn seam: records commands, fakes processes."""

    def __init__(self, *, returncode: int = 0) -> None:
        self.commands: list[str] = []
        self.returncode = returncode

    async def __call__(self, command: str, **kwargs: Any) -> Any:
        self.commands.append(command)
        returncode = self.returncode

        class _Process:
            async def communicate(self) -> tuple[bytes, bytes]:
                return (b"ok\n", b"")

            @property
            def returncode(self) -> int:
                return returncode

        return _Process()


def _install_spawn(monkeypatch: pytest.MonkeyPatch, recorder: _SpawnRecorder) -> None:
    monkeypatch.setattr(
        "chartreux.core.completion_checks.spawn_shell_command", recorder
    )


class _NotedBackend(FakeBackend):
    """A parent backend that answers every turn with terminal content."""

    async def complete(self, **kwargs: Any) -> LLMChunk:
        return mock_llm_chunk(content="parent noted")

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        yield mock_llm_chunk(content="parent noted")


def _config_with_mode(mode: AuditMode) -> Any:
    """The parent config granting bash, with the completion audit mode."""
    config = _config(enabled_tools=["task", "bash"])
    return config.model_copy(update={"completion": CompletionSectionConfig(mode=mode)})


async def _registry_with_parent(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, *, mode: AuditMode = "stamp"
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    # The parent grants bash (a real orchestrator does), so its delegation
    # ceiling carries the shell tool and the harness verification capability
    # derives as available.
    parent = build_test_agent_loop(
        config=_config_with_mode(mode), backend=_NotedBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


async def _foreground_registry_with_shell(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, *, mode: AuditMode = "stamp"
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    """The foreground registry with the parent granting bash (the
    _registry_with_parent precedent) and the given audit mode.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(
        config=_config_with_mode(mode), backend=FakeBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


def _find_run(record: Any, run_id: str) -> Any:
    return next(
        item
        for item in [record.current_run, *record.run_history]
        if item is not None and item.run_id == run_id
    )


async def _wait_until(condition: Any, timeout: float = 5.0) -> None:
    """Poll a condition to truth, with a bounded deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not condition():
        if loop.time() > deadline:
            raise AssertionError("condition never became true")
        await asyncio.sleep(0.01)


def _claiming_report(*satisfied: str) -> str:
    """The worker's final report carrying its envelope-linked receipt."""
    return "Work complete.\n" + json.dumps({
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "envelope": {
            "marker": COMPLETION_ENVELOPE_MARKER,
            "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
            "requirements": [{"requirement_id": "check:pytest", "template": "pytest"}],
        },
        "satisfied_requirement_ids": list(satisfied),
    })


def _mandated_args(**overrides: Any) -> TaskArgs:
    params: dict[str, Any] = {
        "task": "Mandated work",
        "identity": "coder",
        "background": True,
        "retain": True,
        "completion_check": [CompletionCheckTemplate.PYTEST],
    }
    params.update(overrides)
    return TaskArgs(**params)


# ----------------------------------------------------------------------
# Unit level: the claim extraction
# ----------------------------------------------------------------------


def test_extract_completion_claim_reads_the_envelope_linked_receipt() -> None:
    claim = extract_completion_claim(_claiming_report("check:pytest"))
    assert isinstance(claim, CompletionReceipt)
    assert claim.satisfied_requirement_ids == ("check:pytest",)
    assert claim.envelope is not None
    assert [r.requirement_id for r in claim.envelope.requirements] == ["check:pytest"]


def test_extract_completion_claim_prefers_the_final_reports_receipt() -> None:
    earlier = json.dumps({
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "satisfied_requirement_ids": ["check:pytest"],
    })
    report = f"{earlier}\n{_claiming_report('check:pytest')}"
    claim = extract_completion_claim(report)
    assert isinstance(claim, CompletionReceipt)
    # The final report's receipt (the envelope-linked one) wins.
    assert claim.envelope is not None


def test_extract_completion_claim_returns_none_without_a_receipt() -> None:
    # Plain prose carries no claim.
    assert extract_completion_claim("All done, nothing to claim.") is None
    # The nested envelope (the marker without the receipt's
    # satisfied_requirement_ids key) is not the claim.
    envelope_only = json.dumps({
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "requirements": [{"requirement_id": "check:pytest", "template": "pytest"}],
    })
    assert extract_completion_claim(envelope_only) is None
    # A marker mention without a JSON object carries no claim.
    assert (
        extract_completion_claim(
            "the marker completion_envelope_v1 appears in prose only"
        )
        is None
    )


def test_extract_completion_claim_rides_a_grammar_violating_payload_raw() -> None:
    # An unknown field is a grammar violation: the raw payload rides so the
    # truth table classifies it as a disclosure fact, never an exception.
    claim = extract_completion_claim(
        json.dumps({
            "marker": COMPLETION_ENVELOPE_MARKER,
            "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
            "satisfied_requirement_ids": ["check:pytest"],
            "bogus": 1,
        })
    )
    assert isinstance(claim, dict)
    assert claim["bogus"] == 1


def test_operative_completion_claim_is_gated_on_a_healthy_turn() -> None:
    accumulator = SubagentRunAccumulator()
    accumulator._response.append(_claiming_report("check:pytest"))
    # A healthy completed turn's publication carries the claim.
    assert operative_completion_claim(accumulator, turn_completed=True) is not None
    # A cancelled or failed run's primary disposition stays authoritative:
    # no claim rides.
    assert operative_completion_claim(accumulator, turn_completed=False) is None


# ----------------------------------------------------------------------
# Unit level: the rejection decision
# ----------------------------------------------------------------------


def _classification(
    rejection: CompletionRejectionReason | None,
) -> CompletionAuditClassification:
    return CompletionAuditClassification(
        claim_fidelity=CompletionClaimFidelity.CLAIM_CONTRADICTED,
        disclosure_completeness=CompletionDisclosure.DISCLOSURE_COMPLETE,
        mandate_satisfaction=CompletionMandateSatisfaction.MANDATE_SATISFIED,
        declared_outcome=CompletionDeclaredOutcome.OUTCOME_COMPLETED,
        primary_runtime_failure=CompletionRuntimeFailure.RUNTIME_HEALTHY,
        rejection=rejection,
    )


def _blocked_result(rejection: CompletionRejectionReason | None) -> TaskResult:
    return TaskResult(
        response="[Completion audit rejected: check:pytest: contradicted.]",
        turns_used=1,
        completed=False,
        completion_audit=_classification(rejection),
    )


def _context(
    *, status: RunStatus = RunStatus.COMPLETED, audit_mode: AuditMode = "enforce"
) -> BackgroundExecutionContext:
    contract = _freeze(
        audit_mode=audit_mode, completion_check=[CompletionCheckTemplate.PYTEST]
    )
    run_record = SimpleNamespace(status=status, completion_contract=contract)
    record = SimpleNamespace(
        work_package_id=None, predecessor_agent_id=None, recovery_chain_id=None
    )
    return cast(
        BackgroundExecutionContext,
        SimpleNamespace(
            agent_id="agent-1",
            run_id="run-1",
            turn_id="turn-1",
            run_record=run_record,
            record=record,
        ),
    )


def test_rejection_envelope_agrees_with_the_audit_stamp() -> None:
    envelope = rejection_envelope(
        _context(), _blocked_result(CompletionRejectionReason.CLAIM_CONTRADICTED)
    )
    assert envelope is not None
    assert envelope.agent_id == "agent-1"
    assert envelope.run_id == "run-1"
    failure = envelope.failure
    assert failure.failure_class is FailureClass.COMPLETION_REJECTION
    assert failure.phase is FailurePhase.FINALIZE
    assert failure.confidence is FailureConfidence.MEASURED
    assert "completion_rejection_claim_contradicted" in failure.evidence_codes
    assert "completion_audit_rejection" in failure.evidence_codes
    # The envelope path inherits the wire mirror.
    public = _public_failure_envelope(envelope)
    assert public is not None
    assert public.failure.failure_class is PublicFailureClass.COMPLETION_REJECTION
    # PROHIBITED with its own distinct reason code: a replacement is
    # authorized, never a warm same-agent auto-resume.
    assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
    assert envelope.recovery.reason_codes == ["completion_rejection_no_auto_recovery"]


def test_rejection_envelope_never_fires_in_stamp_mode() -> None:
    # Escalation requires an enforce-mode contradiction disposition; stamp
    # mode never escalates.
    assert (
        rejection_envelope(
            _context(audit_mode="stamp"),
            _blocked_result(CompletionRejectionReason.CLAIM_CONTRADICTED),
        )
        is None
    )


def test_rejection_envelope_never_fires_for_a_disclosure_rejection() -> None:
    # A missing receipt (an honest abort/blocker outcome) is a disclosure
    # fact, never a worker-side disposition.
    assert (
        rejection_envelope(
            _context(), _blocked_result(CompletionRejectionReason.RECEIPT_UNVERIFIABLE)
        )
        is None
    )


def test_rejection_envelope_never_replaces_a_cancellation() -> None:
    assert (
        rejection_envelope(
            _context(status=RunStatus.CANCELLED),
            _blocked_result(CompletionRejectionReason.CLAIM_CONTRADICTED),
        )
        is None
    )


def test_rejection_envelope_never_replaces_a_primary_runtime_failure() -> None:
    assert (
        rejection_envelope(
            _context(status=RunStatus.FAILED),
            _blocked_result(CompletionRejectionReason.CLAIM_CONTRADICTED),
        )
        is None
    )


def test_rejection_envelope_never_fires_on_a_completed_result() -> None:
    result = _blocked_result(CompletionRejectionReason.CLAIM_CONTRADICTED)
    completed = result.model_copy(update={"completed": True})
    assert rejection_envelope(_context(), completed) is None


# ----------------------------------------------------------------------
# Integration: the background monitor's rejection lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enforce_mode_rejection_finalizes_the_run_with_the_measured_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The four surfaces agree: completed=False, a failed run status, the
    measured rejection envelope, and the audit stamp.
    """
    child_backend = FakeBackend([
        mock_llm_chunk(content=_claiming_report("check:pytest"))
    ])
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="enforce"
    )
    # The finalization's bound check FAILS: the claim is contradicted (C5).
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        # completed=False + the bounded, actionable rejection message.
        assert result.completed is False
        assert result.response.startswith("[Completion audit rejected:")
        assert "check:pytest: contradicted" in result.response
        # The audit stamp agrees.
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
        assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_CONTRADICTED
        # The measured rejection envelope rides the result.
        envelope = result.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.COMPLETION_REJECTION
        assert envelope.failure.confidence is FailureConfidence.MEASURED
        assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
        assert result.run_status is RunStatus.FAILED
        assert result.recovery is not None
        assert result.stop_reason is RunStopReason.ERROR
        # The run record agrees: FAILED with the same envelope stored on the
        # run (the canonical store — the predecessor-retention surface).
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.FAILED
        assert run.stop_reason is RunStopReason.ERROR
        assert run.failure is envelope
        # The audit survives storage and retrieval unchanged.
        stored = await registry.get_agent_result(ack.agent_id, ack.run_id)
        assert stored is not None
        assert stored.completion_audit == audit
        assert stored.failure is envelope
        # The rejected run never suspends: the record returns to idle.
        await _wait_until(lambda: record.state is _AgentState.IDLE)
        assert record.state is not _AgentState.SUSPENDED
        # No same-agent resume: the explicit resume refuses.
        resume = await registry.resume_agent(
            ack.agent_id,
            ack.run_id,
            f"{ack.agent_id}:{ack.run_id}",
            requester_session_id=parent.session_id,
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
        assert resume.reason_codes == ["not_suspended"]
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_rejected_run_never_becomes_a_warm_auto_resume_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The WP4 automatic controller never schedules a resume for a typed
    audit rejection: the envelope is PROHIBITED, so the episode stops with
    no timer armed.
    """
    child_backend = FakeBackend([
        mock_llm_chunk(content=_claiming_report("check:pytest"))
    ])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    config = _config_with_mode("enforce")
    config = config.model_copy(
        update={
            "recovery": RecoverySectionConfig(subagents=RecoveryConfig()),
            "subagents": config.subagents.model_copy(
                update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
            ),
        }
    )
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    wakeup = _GatedWakeup()
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        notify_agents=AsyncMock(),
        clock=_ManualClock(),
        wakeup=wakeup,
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert result.completed is False
        assert result.failure is not None
        assert result.failure.failure.failure_class is FailureClass.COMPLETION_REJECTION
        record = registry._agent_records[ack.agent_id]
        controller = registry._recovery_controller
        await _wait_until(lambda: record.state is _AgentState.IDLE)
        # The rejection's episode stopped at the eligibility gate: no
        # automatic resume was ever scheduled.
        assert controller._episodes == {}
        assert wakeup.calls == []
        assert len(record.run_history) == 1
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_stamp_mode_rejection_only_stamps_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stamp mode (the default) leaves behavior unchanged: the contradiction
    rides the stamp, the publication is never blocked, and no rejection
    envelope is built.
    """
    child_backend = FakeBackend([
        mock_llm_chunk(content=_claiming_report("check:pytest"))
    ])
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="stamp"
    )
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        # The publication is unchanged: the worker's own report, completed.
        assert result.completed is True
        assert "Work complete." in result.response
        # The contradiction rides the stamp, honestly.
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
        # No rejection envelope, no failed run status.
        assert result.failure is None
        assert result.run_status is not RunStatus.FAILED
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.COMPLETED
        assert run.failure is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_honest_report_without_a_receipt_is_never_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An honest report carrying no receipt is a disclosure fact (the
    mandate unaddressed), never a worker-side rejection: the publication is
    never blocked and no envelope is built.
    """
    child_backend = FakeBackend([mock_llm_chunk(content="All done, nothing to claim.")])
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="enforce"
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert result.completed is True
        assert result.response == "All done, nothing to claim."
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is CompletionRejectionReason.RECEIPT_UNVERIFIABLE
        assert (
            audit.disclosure_completeness is CompletionDisclosure.DISCLOSURE_INCOMPLETE
        )
        assert result.failure is None
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.COMPLETED
        assert run.failure is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_grammar_violating_receipt_is_a_disclosure_fact_never_an_escalation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed receipt rides raw through the truth table: the grammar
    violation classifies as disclosure_incomplete, never an escalation.
    """
    report = "Work complete.\n" + json.dumps({
        "marker": COMPLETION_ENVELOPE_MARKER,
        "schema_version": COMPLETION_CONTRACT_SCHEMA_VERSION,
        "satisfied_requirement_ids": ["check:pytest"],
        "bogus": 1,
    })
    child_backend = FakeBackend([mock_llm_chunk(content=report)])
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="enforce"
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        # Never blocked: the publication completes with the worker's report.
        assert result.completed is True
        assert result.failure is None
        audit = result.completion_audit
        assert audit is not None
        assert (
            audit.disclosure_completeness is CompletionDisclosure.DISCLOSURE_INCOMPLETE
        )
        assert audit.rejection is CompletionRejectionReason.ENVELOPE_UNKNOWN_FIELD
        assert audit.claim_fidelity is not CompletionClaimFidelity.CLAIM_CONTRADICTED
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_cancelled_run_preserves_its_reason_and_carries_no_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled run keeps its cancellation disposition: the reason is
    preserved, no claim rides, the stub finalization stamps the honest
    unavailable evidence-status, and no rejection envelope is built.
    """
    child_backend = BlockingBackend()
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="enforce"
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="bg-cancelled", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        record = registry._agent_records[ack.agent_id]
        await asyncio.wait_for(child_backend.started.wait(), 5)
        outcome = await registry.cancel_run(
            record.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert outcome.outcome is CancelOutcome.STOP_REQUESTED
        result = await asyncio.wait_for(_wait_result(registry, record.agent_id), 5)
        assert result.completed is False
        assert result.stop_reason is RunStopReason.USER_CANCELLED
        assert "run cancelled (user_cancelled)" in result.response
        # No rejection envelope: the cancellation is the primary disposition.
        assert result.failure is None
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.CANCELLED
        assert run.failure is None
        # The wired composition stamps the honest unavailable evidence-status
        # (the stub finalization), with no worker-side attribution.
        audit = result.completion_audit
        assert audit is not None
        assert (
            audit.primary_runtime_failure
            is CompletionRuntimeFailure.RUNTIME_UNAVAILABLE
        )
        assert audit.rejection is None
        assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
        # A cancelled run never spawns verification.
        assert spawn.commands == []
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_primary_runtime_failure_is_never_replaced_by_a_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A primary runtime failure stays authoritative: the WP1 failure
    envelope is preserved and never relabeled as a completion rejection.
    """
    child_backend = ImmediateFailureBackend(RuntimeError("provider exploded"))
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="enforce"
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert result.completed is False
        # The primary failure envelope is preserved, not relabeled.
        envelope = result.failure
        assert envelope is not None
        assert envelope.failure.failure_class is not FailureClass.COMPLETION_REJECTION
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.status is RunStatus.FAILED
        assert run.failure is envelope
        # The stamp carries no worker-side attribution (no claim rides on a
        # failed run).
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is not CompletionRejectionReason.CLAIM_CONTRADICTED
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_replacement_claim_against_the_rejected_predecessor_is_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Predecessor retention: the rejected run's envelope authorizes a
    cross-agent replacement through the existing lineage (the P11 policy),
    while the failed agent is never resumed.
    """
    child_backend = FakeBackend([
        mock_llm_chunk(content=_claiming_report("check:pytest"))
    ])
    replacement_backend = FakeBackend([mock_llm_chunk(content="replacement done")])
    _LoopBackends(monkeypatch, [child_backend, replacement_backend])
    config = _config_with_mode("enforce")
    config = config.model_copy(
        update={
            "recovery": RecoverySectionConfig(subagents=RecoveryConfig()),
            "subagents": config.subagents.model_copy(
                update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
            ),
        }
    ).attach_catalog_snapshot(CatalogSnapshot(_replacement_catalog(), "p12-test"))
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        notify_agents=AsyncMock(),
        clock=_ManualClock(),
        wakeup=_GatedWakeup(),
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="launch", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        rejected = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert rejected.completed is False
        assert rejected.failure is not None
        assert (
            rejected.failure.failure.failure_class is FailureClass.COMPLETION_REJECTION
        )
        # The rejected predecessor stays live (retained) with its envelope.
        record = registry._agent_records[ack.agent_id]
        await _wait_until(lambda: record.state is _AgentState.IDLE)
        # The replacement claim is authorized through the existing lineage.
        claim_ack = await _launch(
            registry,
            parent,
            task="Mandated work",
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model",
                work_package_id="wp-1",
                predecessor_agent_id=ack.agent_id,
            ),
        )
        assert claim_ack.agent_id is not None and claim_ack.agent_id != ack.agent_id
        assert claim_ack.work_package_id == "wp-1"
        assert claim_ack.predecessor_agent_id == ack.agent_id
        assert claim_ack.replacement is not None
        assert claim_ack.replacement.disposition == "selected"
        assert (
            "completion_rejection_replacement_authorized"
            in claim_ack.replacement.reason_codes
        )
        assert (
            "new_agent_lineage_no_same_agent_retry"
            in claim_ack.replacement.reason_codes
        )
        # The per-package budget is consumed exactly once; the predecessor's
        # recovery is superseded (never resumed).
        assert registry._package_replacements["wp-1"] == 1
        predecessor = registry._agent_records[ack.agent_id]
        assert predecessor.recovery_superseded is True
    finally:
        await registry.drain_children()
        await parent.aclose()


# ----------------------------------------------------------------------
# Integration: the audit stamp's survival and the persisted-result shape
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_stamp_survives_metadata_replacement_storage_wait_and_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit stamp survives the monitor's metadata replacement, the
    result store, the wait, and retrieval — and round-trips through the
    persisted dump shape.
    """
    child_backend = FakeBackend([
        mock_llm_chunk(content=_claiming_report("check:pytest"))
    ])
    registry, parent = await _registry_with_parent(
        monkeypatch, child_backend, mode="stamp"
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(registry, _mandated_args(), ctx)
        assert ack.agent_id is not None and ack.run_id is not None
        # The wait surface carries the stamp.
        waited = await _wait_result(registry, ack.agent_id, ack.run_id)
        audit = waited.completion_audit
        assert audit is not None
        # Amendment 14: the final-state crediting branch is gone — an
        # exit-0 check with no observed mutation and no collection-time
        # witness never credits the worker, so the claim stays unverified
        # without receipt-path evidence. The stamp (and its survival) is
        # the point here, not the fidelity value.
        assert audit.claim_fidelity is CompletionClaimFidelity.CLAIM_UNVERIFIED
        assert audit.rejection is None
        # The retrieval surface (storage) carries the same stamp.
        stored = await registry.get_agent_result(ack.agent_id, ack.run_id)
        assert stored is not None
        assert stored.completion_audit == audit
        # The run record's result (after the metadata replacement and the
        # stop-reason model_copy) carries the same stamp.
        record = registry._agent_records[ack.agent_id]
        run = _find_run(record, ack.run_id)
        assert run.result is not None
        assert run.result.completion_audit == audit
        # The persisted dump shape carries the stamp, and a round-trip
        # preserves it.
        dump = waited.model_dump()
        assert "completion_audit" in dump
        assert dump["completion_audit"] == audit.model_dump()
        round_tripped = TaskResult.model_validate(waited.model_dump())
        assert round_tripped.completion_audit == audit
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_audit_field_is_omitted_from_dumps_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Persisted-result compatibility: a run composed without an audit
    surface keeps the old dump shape — the field is omitted entirely.
    """
    child_backend = FakeBackend([mock_llm_chunk(content="plain done")])
    registry, parent = await _registry_with_parent(monkeypatch, child_backend)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(task="Plain work", identity="coder", background=True, retain=True),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert result.completed is True
        assert result.completion_audit is None
        assert "completion_audit" not in result.model_dump()
        assert "completion_audit" not in result.model_dump_json()
    finally:
        await registry.drain_children()
        await parent.aclose()


# ----------------------------------------------------------------------
# Integration: the foreground path
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_foreground_enforce_mode_rejection_blocks_the_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The foreground call site wires the same composition surface: an
    enforce-mode rejection blocks the foreground publication with the
    bounded message and the audit stamp.
    """
    child_backend = FakeBackend([
        mock_llm_chunk(content=_claiming_report("check:pytest"))
    ])
    registry, parent = await _foreground_registry_with_shell(
        monkeypatch, child_backend, mode="enforce"
    )
    spawn = _SpawnRecorder(returncode=1)
    _install_spawn(monkeypatch, spawn)
    try:
        ctx = InvokeContext(tool_call_id="fg-1", session_id=parent.session_id)
        items = [
            item
            async for item in registry.run(
                TaskArgs(
                    task="Mandated work",
                    identity="coder",
                    background=False,
                    completion_check=[CompletionCheckTemplate.PYTEST],
                ),
                ctx,
            )
        ]
        result = cast(TaskResult, items[-1])
        assert result.completed is False
        assert result.response.startswith("[Completion audit rejected:")
        assert "check:pytest: contradicted" in result.response
        audit = result.completion_audit
        assert audit is not None
        assert audit.rejection is CompletionRejectionReason.CLAIM_CONTRADICTED
        # The foreground path carries no run record: no failure envelope.
        assert result.failure is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_cancelled_run_preserves_the_terminal_turns_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """P12 reason preservation: a cancelled foreground run settles with the
    terminal turn's own reason (here a LIMIT stop -> budget exceeded), never
    the USER_CANCELLED default — and the wired composition stamps the honest
    unavailable evidence-status with no worker-side attribution.
    """
    child_backend = BlockingBackend()
    registry, parent = await _foreground_registry_with_shell(
        monkeypatch, child_backend, mode="enforce"
    )
    spawn = _SpawnRecorder()
    _install_spawn(monkeypatch, spawn)
    # The terminal turn carries a LIMIT stop: the settled turn's own reason
    # reaches the result.
    real_wait = TurnController.wait_for_operation

    async def _limit_terminal_turn(self: TurnController, turn_id: str) -> Any:
        turn = await real_wait(self, turn_id)
        if turn is not None and turn.status is PublicTurnStatus.INTERRUPTED:
            turn = turn.model_copy(update={"stop_reason": PublicTurnStopReason.LIMIT})
        return turn

    monkeypatch.setattr(TurnController, "wait_for_operation", _limit_terminal_turn)
    try:
        ctx = InvokeContext(tool_call_id="fg-cancelled", session_id=parent.session_id)
        items: list[Any] = []

        async def consume() -> None:
            async for item in registry.run(
                TaskArgs(
                    task="Mandated work",
                    identity="coder",
                    background=False,
                    completion_check=[CompletionCheckTemplate.PYTEST],
                ),
                ctx,
            ):
                items.append(item)

        task = asyncio.ensure_future(consume())
        await asyncio.wait_for(child_backend.started.wait(), 5)
        child_runtime = next(iter(registry._children.values()))
        active = child_runtime.turns.active_turn
        assert active is not None
        child_runtime.turns.interrupt(
            TurnInterruptParams(
                session_id=active.session_id, expected_turn_id=active.id
            )
        )
        await asyncio.wait_for(task, 10)
        result = cast(TaskResult, items[-1])
        # The terminal turn's reason is preserved, not the default.
        assert result.completed is False
        assert result.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert "run cancelled (budget_exceeded)" in result.response
        # The wired composition stamps the honest unavailable evidence-status
        # (the stub finalization), with no worker-side attribution.
        audit = result.completion_audit
        assert audit is not None
        assert (
            audit.primary_runtime_failure
            is CompletionRuntimeFailure.RUNTIME_UNAVAILABLE
        )
        assert audit.rejection is None
        # A cancelled run never spawns verification.
        assert spawn.commands == []
    finally:
        await registry.drain_children()
        await parent.aclose()
