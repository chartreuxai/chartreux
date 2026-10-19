"""WP2 — effect reconciliation for the suspended reconcile barrier.

Covers the reconcile half of the WP2 acceptance: an unknown-outcome failure
(``RECONCILE_REQUIRED``) blocks replay until an externally verified
``reconcile_result`` submission clears the barrier — pinned to the failed run
AND its transcript boundary, so stale or duplicate evidence cannot clear a
newer barrier — or flips it to prohibited when verification shows
unrepeatable damage (handoff §1.4/§1.8: SUSPENDED[reconcile_required] →
eligible | prohibited; the prohibit exit is acceptance-tested here, not just
the clear exit; replacement remains permitted at any point). The published
failure envelope stays immutable; reconcile flips the EFFECTIVE view tracked
on the record and shown in summaries.

The harness and its scripted-backends factory are shared with
``test_explicit_resume.py``: the full app-server surface drives the parent's
``task`` tool, children run behind per-loop scripted backends, and the fault
is the WP1 typed EMPTY_RESPONSE transient with no tool calls — UNKNOWN tool
effects, hence RECONCILE_REQUIRED. The clearing submission is the empty
sequence: the complete zero-entry accounting ("one verified_effects entry
per unknown tool call" — there are none).

Review R5: the submission is compared with the failed run's OUTSTANDING tool
calls. A clearing submission must account for exactly those calls — an
unrelated id, an incomplete subset, a repeated id, or (with calls
outstanding) an empty submission is STALE and never clears the barrier. A
verified damage report always prohibits (the prohibit direction is
fail-safe). The child path is non-streaming, so the UNRESOLVED-barrier
transcript (a published assistant message whose tool calls never settled) is
seeded directly in the dedicated R5 test.
"""

from __future__ import annotations

import pytest

from chartreux.app_server._sessions import _AgentState
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.subagents import (
    AgentAvailability,
    EffectVerification,
    LaunchOutcome,
    ReconcileOutcome,
    ResumeOutcome,
    TaskArgs,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext
from chartreux.core.tools.builtins.reconcile_result import (
    ReconcileResult,
    ReconcileResultArgs,
    ReconcileResultConfig,
)
from tests.app_server._subagent_test_support import _background_result
from tests.mock.utils import collect_result
from tests.recovery.test_explicit_resume import (
    _empty_fault,
    _final_outcome,
    _launch_child,
    _settle_child,
    _start_harness,
    _until,
)


def _settled_effect(call_id: str) -> EffectVerification:
    """One externally verified settled tool effect."""
    return EffectVerification(tool_call_id=call_id, disposition="settled")


def _damage_effect(call_id: str) -> EffectVerification:
    """One externally verified unrepeatable-damage tool effect."""
    return EffectVerification(tool_call_id=call_id, disposition="unrepeatable_damage")


def _outstanding_call_message(*call_ids: str) -> LLMMessage:
    """One assistant message requesting the given tool calls (never settled).

    Seeds the UNRESOLVED-barrier transcript state directly: the child path
    is non-streaming, so the partial-message publication that would leave a
    tool call outstanding end-to-end is a streaming-path behavior (review R5
    pin fixture).
    """
    return LLMMessage(
        role=Role.assistant,
        tool_calls=[
            ToolCall(
                id=call_id,
                index=index,
                function=FunctionCall(name="todo", arguments="{}"),
            )
            for index, call_id in enumerate(call_ids)
        ],
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_clear_clears_the_barrier_and_unblocks_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clear exit: a run/boundary-pinned settled submission clears the
    barrier, flips the effective eligibility to ELIGIBLE in summaries, and
    unblocks the explicit resume.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _final_outcome("reconciled done")]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # Unknown tool outcomes block replay until reconciliation.
        blocked = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED
        assert blocked.eligibility is RecoveryEligibility.RECONCILE_REQUIRED

        # A submission pinned to the wrong transcript boundary is stale.
        wrong_boundary = await registry.reconcile_result(
            agent_id,
            run_id,
            "turn-not-the-boundary",
            [_settled_effect("call-1")],
            requester_session_id=harness.requester_session_id,
        )
        assert wrong_boundary.outcome is ReconcileOutcome.STALE
        assert wrong_boundary.reason_codes == ["stale_transcript_boundary"]
        assert record.reconcile_outcome is None

        # The pinned settled submission clears the barrier. The barrier's
        # failed run left no tool call outstanding, so the complete
        # accounting is the empty sequence.
        cleared = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        assert cleared.eligibility is RecoveryEligibility.ELIGIBLE
        assert cleared.reason_codes == ["effects_verified_settled"]
        assert cleared.transcript_boundary == boundary
        assert record.reconcile_outcome == ReconcileOutcome.CLEARED.value
        assert record.reconcile_reason_codes == ["effects_verified_settled"]
        assert record.last_reconciled == (run_id, boundary)
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.availability is AgentAvailability.SUSPENDED
        assert summary.recovery_eligibility is RecoveryEligibility.ELIGIBLE

        # A duplicate submission is a no-op.
        duplicate = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert duplicate.outcome is ReconcileOutcome.DUPLICATE
        assert record.reconcile_outcome == ReconcileOutcome.CLEARED.value
        assert record.last_reconciled == (run_id, boundary)

        # The cleared barrier unblocks the explicit resume.
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "reconciled done"
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_result_tool_clears_a_zero_call_barrier_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 V4: the PUBLIC tool port accepts the zero-entry submission
    end-to-end — a failed run with NO tool call outstanding is cleared by
    the empty accounting through reconcile_result itself (not only through
    the internal port), so the no-calls reconcile barrier is publicly
    clearable and the explicit resume is unblocked.

    FAILS pre-fix: ReconcileResultArgs enforced min_length=1 — the
    zero-entry submission was rejected at the public port with a
    ValidationError, leaving the barrier unclearable except by retask
    (replace_run=True) or a damage report.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _final_outcome("reconciled done")]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # The public tool port accepts the empty zero-entry accounting and
        # forwards it to the barrier: the no-calls barrier CLEARS.
        tool = ReconcileResult(
            config_getter=lambda: ReconcileResultConfig(), state=BaseToolState()
        )
        ctx = InvokeContext(
            tool_call_id="reconcile",
            session_id=harness.requester_session_id,
            subagent_manager=registry,
        )
        result = await collect_result(
            tool.run(
                ReconcileResultArgs(
                    agent_id=agent_id,
                    expected_run_id=run_id,
                    transcript_boundary=boundary,
                    verified_effects=[],
                ),
                ctx,
            )
        )
        assert result.outcome is ReconcileOutcome.CLEARED
        assert result.eligibility is RecoveryEligibility.ELIGIBLE
        assert result.reason_codes == ["effects_verified_settled"]
        assert record.reconcile_outcome == ReconcileOutcome.CLEARED.value
        assert record.last_reconciled == (run_id, boundary)

        # The cleared barrier unblocks the explicit resume.
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "reconciled done"
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_prohibit_flips_eligibility_and_replacement_still_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prohibit exit: verified unrepeatable damage flips the EFFECTIVE
    eligibility to PROHIBITED (the published envelope stays immutable),
    resume is rejected, and replacement remains permitted. The damage
    report prohibits even though its id matches no outstanding call — the
    prohibit direction is fail-safe (review R5).
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _final_outcome("replacement done")]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # Verification showing unrepeatable damage prohibits recovery.
        prohibited = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_damage_effect("call-1")],
            requester_session_id=harness.requester_session_id,
        )
        assert prohibited.outcome is ReconcileOutcome.PROHIBITED
        assert prohibited.eligibility is RecoveryEligibility.PROHIBITED
        assert prohibited.reason_codes == ["verified_unrepeatable_damage", "call-1"]
        assert record.reconcile_outcome == ReconcileOutcome.PROHIBITED.value
        assert record.reconcile_reason_codes == [
            "verified_unrepeatable_damage",
            "call-1",
        ]
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.recovery_eligibility is RecoveryEligibility.PROHIBITED

        # Resume is rejected — but the PUBLISHED envelope is immutable.
        rejected = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert rejected.outcome is ResumeOutcome.NOT_RESUMABLE
        assert rejected.eligibility is RecoveryEligibility.PROHIBITED
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        failed_run = next(
            item
            for item in [record.current_run, *record.run_history]
            if item is not None and item.run_id == run_id
        )
        assert (
            failed_run.failure.recovery.eligibility
            is RecoveryEligibility.RECONCILE_REQUIRED
        )

        # Replacement remains permitted at any point: it supersedes the
        # suspended recovery and completes on the SAME agent identity.
        launch = await _background_result(
            registry,
            TaskArgs(task="replacement work", agent_id=agent_id, replace_run=True),
            InvokeContext(
                tool_call_id="replace-1", session_id=harness.requester_session_id
            ),
        )
        assert launch.launch_outcome is LaunchOutcome.LAUNCHED
        assert launch.status == "launched"
        assert launch.metadata is not None
        assert launch.metadata["replaced_run_id"] == run_id
        assert launch.metadata["replacement_run_id"] == launch.run_id
        assert launch.run_id is not None and launch.run_id != run_id
        assert registry._agent_records[agent_id] is record
        assert record.recovery_superseded

        replacement = await _settle_child(harness, agent_id, launch.run_id)
        assert replacement.completed
        assert replacement.response == "replacement done"
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_stale_submission_cannot_clear_a_newer_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stale evidence cannot clear a newer barrier: after the first barrier
    is cleared and resumed, a second reconcile_required failure starts a NEW
    barrier (reconcile state reset); a submission pinned to the OLD run is
    STALE (not DUPLICATE) and leaves the new barrier untouched.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _empty_fault()]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        first_envelope = failed.failure
        assert first_envelope is not None
        first_boundary = first_envelope.progress.transcript_boundary
        assert first_boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # Clear the first barrier and resume into a second failed run.
        cleared = await registry.reconcile_result(
            agent_id,
            run_id,
            first_boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        second_run_id = resume.successor_run_id
        assert second_run_id is not None and second_run_id != run_id
        second_failed = await _settle_child(harness, agent_id, second_run_id)
        assert not second_failed.completed
        second_envelope = second_failed.failure
        assert second_envelope is not None
        assert (
            second_envelope.recovery.eligibility
            is RecoveryEligibility.RECONCILE_REQUIRED
        )
        second_boundary = second_envelope.progress.transcript_boundary
        assert second_boundary is not None
        assert second_boundary != first_boundary
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # The new barrier starts from its own envelope: the previous clear
        # never carries into it.
        assert record.reconcile_outcome is None
        assert record.reconcile_reason_codes == []
        assert record.latest_run_id == second_run_id

        # A submission pinned to the OLD run is stale — the run pin wins
        # over the duplicate pin — and cannot clear the newer barrier.
        stale = await registry.reconcile_result(
            agent_id,
            run_id,
            first_boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert stale.outcome is ReconcileOutcome.STALE
        assert stale.reason_codes == ["stale_expected_run_id"]
        assert record.reconcile_outcome is None
        assert record.last_reconciled == (run_id, first_boundary)
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.recovery_eligibility is RecoveryEligibility.RECONCILE_REQUIRED

        # The new barrier still blocks replay.
        blocked = await registry.resume_agent(
            agent_id,
            second_run_id,
            "k",
            requester_session_id=harness.requester_session_id,
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_duplicate_and_wrong_boundary_change_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A duplicate submission and a wrong-boundary submission after a clear
    are both no-ops: the record's reconciliation state is unchanged.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault()]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        cleared = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        assert record.reconcile_outcome == ReconcileOutcome.CLEARED.value
        assert record.reconcile_reason_codes == ["effects_verified_settled"]
        assert record.last_reconciled == (run_id, boundary)

        # Duplicate: same run + same boundary, already reconciled. The
        # duplicate pin is evaluated BEFORE the outstanding-calls comparison
        # (review R5), so the unrelated "call-2" id never reaches it.
        duplicate = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("call-2")],
            requester_session_id=harness.requester_session_id,
        )
        assert duplicate.outcome is ReconcileOutcome.DUPLICATE
        assert duplicate.eligibility is RecoveryEligibility.ELIGIBLE

        # Wrong boundary on the same run: stale, and the boundary check
        # precedes the duplicate check (and the outstanding-calls check).
        wrong_boundary = await registry.reconcile_result(
            agent_id,
            run_id,
            "turn-not-the-boundary",
            [_settled_effect("call-3")],
            requester_session_id=harness.requester_session_id,
        )
        assert wrong_boundary.outcome is ReconcileOutcome.STALE
        assert wrong_boundary.reason_codes == ["stale_transcript_boundary"]

        # Neither submission changed the record.
        assert record.reconcile_outcome == ReconcileOutcome.CLEARED.value
        assert record.reconcile_reason_codes == ["effects_verified_settled"]
        assert record.last_reconciled == (run_id, boundary)
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.recovery_eligibility is RecoveryEligibility.ELIGIBLE
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_not_pending_and_forbidden_dispositions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NOT_PENDING (unknown agent, idle agent — no barrier) and FORBIDDEN
    (wrong requester) dispositions of reconcile_result.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_final_outcome("idle done")], [_empty_fault()]],
        launches=["launch the idle child", "launch the barrier child"],
    )
    registry = harness.registry
    try:
        # Unknown agent → NOT_PENDING.
        unknown = await registry.reconcile_result(
            "agent-999",
            "run-x",
            "turn-1",
            [_settled_effect("call-1")],
            requester_session_id=harness.requester_session_id,
        )
        assert unknown.outcome is ReconcileOutcome.NOT_PENDING

        # An idle (completed) retained agent has no barrier → NOT_PENDING.
        idle_id, idle_run = await _launch_child(harness, "launch the idle child", 1)
        idle_done = await _settle_child(harness, idle_id, idle_run)
        assert idle_done.completed
        idle_record = registry._agent_records[idle_id]
        await _until(lambda: idle_record.state is _AgentState.IDLE)
        not_pending = await registry.reconcile_result(
            idle_id,
            idle_run,
            "turn-1",
            [_settled_effect("call-1")],
            requester_session_id=harness.requester_session_id,
        )
        assert not_pending.outcome is ReconcileOutcome.NOT_PENDING

        # A suspended barrier: only the owning parent may reconcile.
        bar_id, bar_run = await _launch_child(harness, "launch the barrier child", 2)
        bar_failed = await _settle_child(harness, bar_id, bar_run)
        assert not bar_failed.completed
        assert bar_failed.failure is not None
        boundary = bar_failed.failure.progress.transcript_boundary
        assert boundary is not None
        bar_record = registry._agent_records[bar_id]
        await _until(lambda: bar_record.state is _AgentState.SUSPENDED)

        forbidden = await registry.reconcile_result(
            bar_id,
            bar_run,
            boundary,
            [_settled_effect("call-1")],
            requester_session_id="other-session",
        )
        assert forbidden.outcome is ReconcileOutcome.FORBIDDEN
        assert bar_record.reconcile_outcome is None

        # The owning parent's submission is accepted (the barrier itself is
        # covered by the dedicated clear/prohibit tests).
        cleared = await registry.reconcile_result(
            bar_id,
            bar_run,
            boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_unrelated_submission_never_clears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R5 (the reviewer's counterexample): against a barrier whose
    failed run left NO tool call outstanding, a submission naming an
    unrelated id is STALE — it verifies nothing and never clears the
    barrier. Only the complete (here: empty) accounting clears it.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _final_outcome("cleared done")]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # An unrelated id verifies nothing about the (empty) outstanding
        # set — STALE, never CLEARED.
        unrelated = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("call-1")],
            requester_session_id=harness.requester_session_id,
        )
        assert unrelated.outcome is ReconcileOutcome.STALE
        assert unrelated.reason_codes == ["unknown_tool_call_ids", "call-1"]

        # Neither does a list of unrelated ids.
        unrelated_many = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("call-1"), _settled_effect("call-2")],
            requester_session_id=harness.requester_session_id,
        )
        assert unrelated_many.outcome is ReconcileOutcome.STALE
        assert unrelated_many.reason_codes == [
            "unknown_tool_call_ids",
            "call-1",
            "call-2",
        ]

        # The rejected submissions left the barrier untouched and it still
        # blocks replay.
        assert record.reconcile_outcome is None
        assert record.last_reconciled is None
        blocked = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED

        # The complete zero-entry accounting clears the barrier (control).
        cleared = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        assert cleared.eligibility is RecoveryEligibility.ELIGIBLE
        assert cleared.reason_codes == ["effects_verified_settled"]
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_reconcile_submission_must_account_for_every_outstanding_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review R5 (full matrix): with tool calls outstanding, the clearing
    submission must account for exactly those calls — an empty submission,
    an incomplete subset, an unrelated id, or a repeated id is STALE and
    never clears the barrier; only the exact full set clears it. The
    outstanding-call transcript is seeded directly (the child path is
    non-streaming, so this state is a streaming-path behavior).
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _final_outcome("cleared done")]],
        launches=["launch the barrier child"],
    )
    registry = harness.registry
    try:
        agent_id, run_id = await _launch_child(harness, "launch the barrier child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # Seed the UNRESOLVED transcript: two requested tool calls, neither
        # settled — the outstanding set is {"todo-2", "todo-3"}.
        record.runtime.agent_loop.messages.append(
            _outstanding_call_message("todo-2", "todo-3")
        )

        # An empty submission leaves both calls unverified.
        empty = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [],
            requester_session_id=harness.requester_session_id,
        )
        assert empty.outcome is ReconcileOutcome.STALE
        assert empty.reason_codes == [
            "unverified_outstanding_tool_calls",
            "todo-2",
            "todo-3",
        ]

        # An incomplete subset leaves one outstanding call unverified.
        subset = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("todo-2")],
            requester_session_id=harness.requester_session_id,
        )
        assert subset.outcome is ReconcileOutcome.STALE
        assert subset.reason_codes == ["unverified_outstanding_tool_calls", "todo-3"]

        # An unrelated id is unknown AND leaves the calls unverified.
        unrelated = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("call-9")],
            requester_session_id=harness.requester_session_id,
        )
        assert unrelated.outcome is ReconcileOutcome.STALE
        assert unrelated.reason_codes == [
            "unverified_outstanding_tool_calls",
            "todo-2",
            "todo-3",
            "unknown_tool_call_ids",
            "call-9",
        ]

        # A repeated id is not two verifications.
        repeated = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("todo-2"), _settled_effect("todo-2")],
            requester_session_id=harness.requester_session_id,
        )
        assert repeated.outcome is ReconcileOutcome.STALE
        assert repeated.reason_codes == ["duplicate_tool_call_id", "todo-2"]

        # None of the rejected submissions touched the barrier.
        assert record.reconcile_outcome is None
        assert record.last_reconciled is None
        blocked = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED

        # The exact full set clears the barrier (control).
        cleared = await registry.reconcile_result(
            agent_id,
            run_id,
            boundary,
            [_settled_effect("todo-2"), _settled_effect("todo-3")],
            requester_session_id=harness.requester_session_id,
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        assert cleared.eligibility is RecoveryEligibility.ELIGIBLE
        assert cleared.reason_codes == ["effects_verified_settled", "todo-2", "todo-3"]
        assert record.reconcile_reason_codes == [
            "effects_verified_settled",
            "todo-2",
            "todo-3",
        ]
    finally:
        await harness.session.close()
