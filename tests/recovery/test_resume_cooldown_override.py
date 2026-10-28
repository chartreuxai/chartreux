"""C6b — the cooldown override through the explicit resume tool path.

Wires the C1 forced-attempt permit through the explicit ``resume_agent``
admission: an explicit ``override_cooldown=True`` request forces the
un-stick attempt past a temporary (never permanent) unavailability and
mints the endpoint-bound permit from the ROOT ledger at the successor's
preparation, binding it to the successor's first model call (C4's minting
thread). The default path never sets the flag and never mints (F6); a
ledger-denied mint (spacing, exhaustion, an attempt in flight) rolls the
admission back to the suspended state with the resume budget and the
transport allowance intact.

Coverage: the forced/resolved contrast with rejected availability (which
consumes neither resume nor transport allowance); budget, spacing, and
concurrency across fresh agents on one shared endpoint episode;
reconciliation (an override never bypasses RECONCILE_REQUIRED); the pinned
run and single-flight idempotency (one mint per admission); the strict
tool argument; and the _sessions.py half of C6b — the bounded un-stick
diagnostics riding the FOREGROUND TaskResult's metadata, closing C4's
known limitation (they previously rode only the background
acknowledgment).

The unavailability scenarios stub the recovery controller's read-only
``_availability`` snapshot at the same boundary C6a's tests use; the
ledger, its episode, and every mint/denial are the real root-shared
machinery. All backends are offline stubs; no network, no real sleeps.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chartreux.app_server._recovery_controller import RecoveryController, _Availability
from chartreux.app_server._sessions import SessionRuntimeRegistry, _AgentState
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.recovery_override import LedgerOutcome, OverrideDenial
from chartreux.core.subagents import ResumeOutcome, TaskArgs, TaskResult
from chartreux.core.tools.base import ToolError
from chartreux.core.tools.builtins.resume_agent import ResumeAgentArgs
from tests.app_server._subagent_test_support import _ManualClock
from tests.app_server.test_task_cooldown_override import (
    _arm_deployment_cooldown,
    _ctx,
    _open_episode,
    _override_registry,
)
from tests.recovery.test_explicit_resume import (
    _empty_fault,
    _final_outcome,
    _Gated,
    _Harness,
    _launch_child,
    _settle_child,
    _start_harness,
    _until,
    _write_outcome,
)

#: The endpoint of the shipped test catalog's only provider (the child's
#: committed deployment lives on ``mistral`` at this api_base).
CHILD_ENDPOINT = "https://api.mistral.ai/v1"


def _ledger(harness: _Harness) -> Any:
    """The root-owned ledger every runtime copy shares (C1)."""
    return harness.parent.config_orchestrator.cooldown_override_ledger


def _spy_reserve(
    monkeypatch: pytest.MonkeyPatch, ledger: Any
) -> list[tuple[str | None, LedgerOutcome]]:
    """Record every ledger reserve (the mint surface) with its outcome."""
    calls: list[tuple[str | None, LedgerOutcome]] = []
    original = ledger.reserve

    def reserve(endpoint: str | None) -> LedgerOutcome:
        outcome = original(endpoint)
        calls.append((endpoint, outcome))
        return outcome

    monkeypatch.setattr(ledger, "reserve", reserve)
    return calls


def _stub_temporary_unavailability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The controller's read-only availability snapshot reports a temporary
    (never permanent) unavailability — C6a's stubbed boundary.
    """

    def _unavailable(_self: Any, _record: Any) -> _Availability:
        return _Availability(
            available=False,
            permanent=False,
            next_check_delay=12.0,
            endpoint_blocked=False,
        )

    monkeypatch.setattr(RecoveryController, "_availability", _unavailable)


async def _await_suspended(registry: Any, agent_id: str) -> None:
    """Wait until the agent's record reports the warm suspended state."""
    record = registry._agent_records[agent_id]
    await _until(lambda: record.state is _AgentState.SUSPENDED)


def _assert_suspended_untouched(record: Any, run_id: str) -> None:
    """A rejected admission consumed nothing: still suspended, no resume
    budget, no pending admission, no successor for the failed run.
    """
    assert record.state is _AgentState.SUSPENDED
    assert record.same_agent_resumes_used == 0
    assert record.pending_resume is None
    assert record.latest_run_id == run_id
    successors = [
        item for item in record.run_history if item.resumed_from_run_id == run_id
    ]
    assert successors == []


async def _foreground_run(
    registry: SessionRuntimeRegistry, parent: AgentLoop, args: TaskArgs
) -> TaskResult:
    """Drive one foreground task run and return its terminal TaskResult."""
    events = [
        event
        async for event in registry.run(args, _ctx(parent, f"foreground-{args.task}"))
    ]
    result = events[-1]
    assert isinstance(result, TaskResult)
    return result


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_override_resume_forces_past_temporary_unavailability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The C6b core flow: an explicit ``override_cooldown`` resume forces the
    admission past a temporary unavailability and mints exactly one
    endpoint-bound permit; without the flag the same admission is rejected
    and never requests a permit (F6).
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        episode = registry._recovery_controller._episodes[agent_id]
        assert episode.timer is not None

        _stub_temporary_unavailability(monkeypatch)
        calls = _spy_reserve(monkeypatch, ledger)
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True

        # Without the explicit flag the admission is rejected before any
        # budget is consumed, never requests a permit, and leaves the
        # pending automatic timer armed.
        rejected = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=harness.requester_session_id
        )
        assert rejected.outcome is ResumeOutcome.NOT_RESUMABLE
        assert rejected.reason_codes == ["model_unavailable"]
        _assert_suspended_untouched(record, run_id)
        assert calls == []
        assert episode.timer is not None

        # The explicit override forces past the temporary unavailability and
        # mints exactly one permit bound to the committed deployment's
        # endpoint; the successor completes on the forced deployment.
        resume = await registry.resume_agent(
            agent_id,
            run_id,
            "k",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert "forced_un_stick_attempt" in (resume.reason_codes or [])
        assert resume.eligibility is RecoveryEligibility.ELIGIBLE
        assert resume.same_agent_resumes_used == 1
        assert resume.successor_run_id is not None
        assert resume.successor_run_id != run_id
        assert len(calls) == 1
        endpoint, outcome = calls[0]
        assert endpoint == CHILD_ENDPOINT
        assert outcome.denial is None
        assert outcome.permit is not None

        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        # The successor's success is a MEASURED endpoint recovery: it retires
        # the episode (the pinned boundary — only a new measured failure
        # re-opens it). The stub backend fires no transport-start
        # notification, so the attempt released the permit pretransport and
        # the in-flight slot is free — proven by a fresh measured failure
        # opening a new episode whose reserve mints.
        assert ledger.episode_status(CHILD_ENDPOINT) is None
        assert ledger.reserve(CHILD_ENDPOINT).denial is OverrideDenial.NO_ACTIVE_EPISODE
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True
        free = ledger.reserve(CHILD_ENDPOINT)
        assert free.permit is not None
        ledger.release(free.permit)
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_override_resume_denied_while_an_attempt_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrency across fresh agents: ONE forced attempt in flight per
    endpoint — a second agent's explicit override is denied by the ledger
    (ATTEMPT_IN_FLIGHT), and the denied admission rolls back to the
    suspended state with the resume budget and transport allowance intact;
    once the in-flight attempt settles, the second override succeeds.
    """
    gate = asyncio.Event()
    gated = _Gated(_final_outcome("done one"), gate)
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [_write_outcome("todo-1", "item-1"), _empty_fault(), gated],
            [
                _write_outcome("todo-2", "item-2"),
                _empty_fault(),
                _final_outcome("done two"),
            ],
        ],
        launches=["launch one", "launch two"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    try:
        first_id, first_run = await _launch_child(harness, "launch one", 1)
        second_id, second_run = await _launch_child(harness, "launch two", 2)
        for agent_id, run_id in ((first_id, first_run), (second_id, second_run)):
            failed = await _settle_child(harness, agent_id, run_id)
            assert not failed.completed
            await _await_suspended(registry, agent_id)

        _stub_temporary_unavailability(monkeypatch)
        calls = _spy_reserve(monkeypatch, ledger)
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True

        # The first agent's forced attempt mints and holds the endpoint's
        # single in-flight slot while its successor is mid-call.
        resume = await registry.resume_agent(
            first_id,
            first_run,
            "k1",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        await asyncio.wait_for(gated.started.wait(), timeout=5)
        assert len(calls) == 1

        # The second agent's forced attempt is denied: one in flight.
        with pytest.raises(ToolError, match="in flight"):
            await registry.resume_agent(
                second_id,
                second_run,
                "k2",
                requester_session_id=harness.requester_session_id,
                override_cooldown=True,
            )
        second = registry._agent_records[second_id]
        _assert_suspended_untouched(second, second_run)
        status = ledger.episode_status(CHILD_ENDPOINT)
        assert status is not None
        assert status.starts_used == 0
        # The denied reserve is the second call and mutated nothing.
        assert len(calls) == 2
        assert calls[-1][1].denial is OverrideDenial.ATTEMPT_IN_FLIGHT

        # The in-flight attempt settles (releasing the permit pretransport).
        # Its success is a measured endpoint recovery, which RETIRES the
        # episode (the pinned boundary) — a genuinely new measured failure
        # opens a fresh one, and the second override then mints its own
        # (the freed slot would deny ATTEMPT_IN_FLIGHT otherwise).
        gate.set()
        done = await _settle_child(harness, first_id, resume.successor_run_id)
        assert done.completed
        assert done.response == "done one"
        assert ledger.episode_status(CHILD_ENDPOINT) is None
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True
        second_resume = await registry.resume_agent(
            second_id,
            second_run,
            "k2",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert second_resume.outcome is ResumeOutcome.SCHEDULED
        assert second_resume.successor_run_id is not None
        assert len(calls) == 3
        done_two = await _settle_child(
            harness, second_id, second_resume.successor_run_id
        )
        assert done_two.completed
        assert done_two.response == "done two"
    finally:
        gate.set()
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_override_resume_spacing_denial_consumes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Spacing: with one forced start already consumed in the endpoint's
    episode, a second explicit override inside the spacing window is denied
    by the ledger (SPACING_EXCEEDED); the denied mint consumes no transport
    allowance and the admission rolls back with the resume budget intact.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    clock = _ManualClock()
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        _stub_temporary_unavailability(monkeypatch)
        calls = _spy_reserve(monkeypatch, ledger)
        monkeypatch.setattr(ledger, "_clock", clock)
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True
        # One forced start already consumed in this episode.
        first = ledger.reserve(CHILD_ENDPOINT)
        assert first.permit is not None
        assert ledger.start(first.permit).phase is not None
        minted = len(calls)

        with pytest.raises(ToolError, match="spacing floor"):
            await registry.resume_agent(
                agent_id,
                run_id,
                "k",
                requester_session_id=harness.requester_session_id,
                override_cooldown=True,
            )
        _assert_suspended_untouched(record, run_id)
        assert len(calls) == minted + 1
        assert calls[-1][1].denial is OverrideDenial.SPACING_EXCEEDED
        status = ledger.episode_status(CHILD_ENDPOINT)
        assert status is not None
        assert status.starts_used == 1
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_override_resume_exhausted_episode_denial_consumes_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Budget: with the episode's forced-start allowance (2) exhausted, an
    explicit override is denied by the ledger (EPISODE_EXHAUSTED); the
    denied mint consumes nothing and the admission rolls back with the
    resume budget intact.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    clock = _ManualClock()
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        _stub_temporary_unavailability(monkeypatch)
        calls = _spy_reserve(monkeypatch, ledger)
        monkeypatch.setattr(ledger, "_clock", clock)
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True
        # The episode's whole forced-start allowance is already consumed.
        for _ in range(2):
            reserved = ledger.reserve(CHILD_ENDPOINT)
            assert reserved.permit is not None
            assert ledger.start(reserved.permit).phase is not None
            clock.advance(30)
        minted = len(calls)

        with pytest.raises(ToolError, match="budget is exhausted"):
            await registry.resume_agent(
                agent_id,
                run_id,
                "k",
                requester_session_id=harness.requester_session_id,
                override_cooldown=True,
            )
        _assert_suspended_untouched(record, run_id)
        assert len(calls) == minted + 1
        assert calls[-1][1].denial is OverrideDenial.EPISODE_EXHAUSTED
        status = ledger.episode_status(CHILD_ENDPOINT)
        assert status is not None
        assert status.starts_used == 2
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_override_resume_blocked_by_reconciliation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconciliation: an explicit override never bypasses the
    RECONCILE_REQUIRED barrier — the admission is rejected before the
    availability/mint path and never requests a permit.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[[_empty_fault(), _final_outcome("reconciled done")]],
        launches=["launch the child"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        _stub_temporary_unavailability(monkeypatch)
        calls = _spy_reserve(monkeypatch, ledger)
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True

        blocked = await registry.resume_agent(
            agent_id,
            run_id,
            "k",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED
        assert blocked.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        _assert_suspended_untouched(record, run_id)
        assert calls == []
    finally:
        await harness.session.close()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_override_resume_pinned_run_and_single_flight_mint_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned run/idempotency: a stale ``expected_run_id`` is rejected
    without a mint; concurrent same-key calls deduplicate onto the ONE
    admission (which mints exactly once) and a different key is refused
    while that admission is in flight.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("resumed done"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    prepare_gate = asyncio.Event()
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        _stub_temporary_unavailability(monkeypatch)
        calls = _spy_reserve(monkeypatch, ledger)
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True

        # A stale run pin is rejected before any mint.
        stale = await registry.resume_agent(
            agent_id,
            "run-bogus",
            "k",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert stale.outcome is ResumeOutcome.NOT_RESUMABLE
        assert stale.reason_codes == ["stale_expected_run_id"]
        _assert_suspended_untouched(record, run_id)
        assert calls == []

        # Hold the admission's prepare phase: the mint has not run yet.
        original_prepare = registry._execution.prepare_background

        async def gated_prepare(*args: Any, **kwargs: Any) -> Any:
            await prepare_gate.wait()
            return await original_prepare(*args, **kwargs)

        monkeypatch.setattr(registry._execution, "prepare_background", gated_prepare)
        resume_task = asyncio.ensure_future(
            registry.resume_agent(
                agent_id,
                run_id,
                "k",
                requester_session_id=harness.requester_session_id,
                override_cooldown=True,
            )
        )
        await _until(lambda: record.pending_resume is not None)

        # Same key + same run: deduplicated onto the in-flight admission.
        same_key = await registry.resume_agent(
            agent_id,
            run_id,
            "k",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert same_key.outcome is ResumeOutcome.ALREADY_SCHEDULED
        assert record.pending_resume is not None
        assert same_key.successor_run_id == record.pending_resume.successor_run_id

        # A different key while the admission is in flight: refused.
        other_key = await registry.resume_agent(
            agent_id,
            run_id,
            "k2",
            requester_session_id=harness.requester_session_id,
            override_cooldown=True,
        )
        assert other_key.outcome is ResumeOutcome.ALREADY_RESUMED
        # The deduplicated calls never reached the mint.
        assert calls == []

        prepare_gate.set()
        resume = await resume_task
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        assert "forced_un_stick_attempt" in (resume.reason_codes or [])
        assert len(calls) == 1
        assert calls[0][1].permit is not None

        successor = await _settle_child(harness, agent_id, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"
    finally:
        prepare_gate.set()
        await harness.session.close()


@pytest.mark.asyncio
async def test_foreground_override_result_carries_cooldown_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The _sessions.py half of C6b: a foreground override launch mints and
    enforces normally, and the bounded un-stick diagnostics ride the
    foreground TaskResult's metadata exactly like the background
    acknowledgment (C4's known limitation, closed here).
    """
    registry, parent, _backends = await _override_registry(monkeypatch)
    try:
        # An ordinary foreground launch carries no override diagnostics.
        ordinary = await _foreground_run(
            registry,
            parent,
            TaskArgs(
                task="ordinary", identity="coder", background=False, model="small"
            ),
        )
        assert ordinary.completed
        assert "cooldown_override" not in (ordinary.metadata or {})

        _arm_deployment_cooldown(parent)
        _open_episode(parent)
        forced = await _foreground_run(
            registry,
            parent,
            TaskArgs(
                task="unstick",
                identity="coder",
                background=False,
                model="small",
                override_cooldown=True,
            ),
        )
        assert forced.completed
        metadata = forced.metadata
        assert metadata is not None
        diagnostics = metadata["cooldown_override"]
        assert diagnostics["endpoint"] == "one.test"
        assert diagnostics["provider"] == "one"
        assert diagnostics["wire_name"] == "small"
        assert diagnostics["exclusions_bypassed"] == ["deployment-cooldown"]
        assert diagnostics["remaining_starts"] == 1
    finally:
        await registry.close()
        await parent.aclose()


def test_resume_override_cooldown_is_strict_and_defaults_false() -> None:
    """The resume tool's override flag is a strict boolean defaulting to
    False (C6b: strict through the manager protocol).
    """
    defaulted = ResumeAgentArgs(agent_id="agent-1", expected_run_id="run-1")
    assert defaulted.override_cooldown is False
    with pytest.raises(ValueError):
        ResumeAgentArgs.model_validate({
            "agent_id": "agent-1",
            "expected_run_id": "run-1",
            "override_cooldown": "true",
        })


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_failed_post_commit_admission_rearms_the_episode_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C6a rollback liveness: a committed explicit resume stands the
    automatic schedule down; when the successor admission then fails (here:
    the ledger denies the mint because the endpoint's single in-flight slot
    is held), the rollback restores the SUSPENDED record AND re-arms the
    episode's timer. A suspended agent never produces the new-failure
    envelope that would re-arm it, so without the re-arm the automatic
    recovery route is dead for that agent.
    """
    harness = await _start_harness(
        monkeypatch,
        child_scripts=[
            [
                _write_outcome("todo-1", "item"),
                _empty_fault(),
                _final_outcome("never reached"),
            ]
        ],
        launches=["launch the child"],
    )
    registry = harness.registry
    ledger = _ledger(harness)
    held = None
    try:
        agent_id, run_id = await _launch_child(harness, "launch the child", 1)
        failed = await _settle_child(harness, agent_id, run_id)
        assert not failed.completed
        record = registry._agent_records[agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        episode = registry._recovery_controller._episodes[agent_id]
        # Precondition: the automatic controller armed its backoff timer.
        assert episode.timer is not None

        # Open an endpoint episode and hold its single in-flight slot, so
        # the resume's mint is denied with ATTEMPT_IN_FLIGHT.
        assert ledger.note_endpoint_failure(CHILD_ENDPOINT) is True
        held = ledger.reserve(CHILD_ENDPOINT)
        assert held.permit is not None

        # The explicit override resume commits (budget consumed, timer stood
        # down, slot reserved) and then fails inside the successor's
        # preparation when the mint is denied.
        with pytest.raises(ToolError, match="in flight"):
            await registry.resume_agent(
                agent_id,
                run_id,
                "k",
                requester_session_id=harness.requester_session_id,
                override_cooldown=True,
            )

        # The rollback restored the record and the budget...
        assert record.state is _AgentState.SUSPENDED
        assert record.same_agent_resumes_used == 0
        assert record.pending_resume is None
        # ...AND re-armed the automatic schedule: the timer survives the
        # failed post-commit admission, scheduled for the failed run.
        assert episode.timer is not None
        assert episode.scheduled_failed_run_id == run_id
        assert episode.active
    finally:
        if held is not None:
            ledger.release(held.permit)
        await harness.session.close()
