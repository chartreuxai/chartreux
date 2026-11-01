"""O1: the agent-summary payload carries the latest run's audit stamp."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._model import validate_wire
from chartreux.app_server._sessions import RunRecord, SessionRuntimeRegistry
from chartreux.app_server.models import PublicCompletionAudit
from chartreux.app_server.protocol import AgentsUpdateParams
from chartreux.core.completion_contract import (
    CompletionAuditClassification,
    CompletionClaimFidelity,
    CompletionDeclaredOutcome,
    CompletionDisclosure,
    CompletionMandateSatisfaction,
    CompletionRejectionReason,
    CompletionRuntimeFailure,
)
from chartreux.core.subagents import RunStatus, TaskResult
from tests.app_server._subagent_test_support import (
    _idle_record,
    _ManualClock,
    _retention_registry,
)


def _audit(**overrides: object) -> CompletionAuditClassification:
    values: dict[str, object] = {
        "claim_fidelity": CompletionClaimFidelity.CLAIM_MATCHED,
        "disclosure_completeness": CompletionDisclosure.DISCLOSURE_COMPLETE,
        "mandate_satisfaction": CompletionMandateSatisfaction.MANDATE_SATISFIED,
        "declared_outcome": CompletionDeclaredOutcome.OUTCOME_COMPLETED,
        "primary_runtime_failure": CompletionRuntimeFailure.RUNTIME_HEALTHY,
    }
    values.update(overrides)
    return CompletionAuditClassification.model_validate(values)


def _stamped_run(
    agent_id: str, run_id: str, audit: CompletionAuditClassification | None
) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        agent_id=agent_id,
        identity="coder",
        status=RunStatus.COMPLETED,
        completion_task=asyncio.get_running_loop().create_future(),
        result=TaskResult(
            response="done",
            turns_used=1,
            completed=True,
            agent_id=agent_id,
            run_id=run_id,
            completion_audit=audit,
        ),
    )


async def _emitted_payload(
    registry: SessionRuntimeRegistry, notify: AsyncMock
) -> AgentsUpdateParams:
    await registry._emit_agents_update()
    call = notify.await_args
    assert call is not None
    agents = call.args[0]
    return AgentsUpdateParams(
        event_id=0, session_id="session-1", emitted_at=0, agents=agents
    )


@pytest.mark.asyncio
async def test_agents_update_carries_the_latest_runs_audit_stamp() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    record = _idle_record("agent-1", clock=clock)
    run = _stamped_run(
        "agent-1",
        "run-1",
        _audit(rejection=CompletionRejectionReason.RECEIPT_UNVERIFIABLE),
    )
    record.run_history.append(run)
    record.latest_run_id = run.run_id
    registry._agent_records[record.agent_id] = record

    payload = await _emitted_payload(registry, notify)

    assert len(payload.agents) == 1
    stamp = payload.agents[0].completion_audit
    assert isinstance(stamp, PublicCompletionAudit)
    assert stamp.schema_version == 1
    # The wire enums mirror the core vocabulary value-for-value; they are
    # distinct types (the public layer cannot import core).
    assert stamp.claim_fidelity.value == CompletionClaimFidelity.CLAIM_MATCHED.value
    assert (
        stamp.primary_runtime_failure.value
        == CompletionRuntimeFailure.RUNTIME_HEALTHY.value
    )
    assert stamp.rejection is not None
    assert stamp.rejection.value == CompletionRejectionReason.RECEIPT_UNVERIFIABLE.value
    # The payload round-trips: the widget reads this exact wire field.
    wire = validate_wire(AgentsUpdateParams, payload.model_dump(mode="json"))
    assert wire.agents[0].completion_audit is not None
    assert wire.agents[0].completion_audit.claim_fidelity == "claim_matched"
    assert payload.model_dump(mode="json")["agents"][0]["completionAudit"][
        "claimFidelity"
    ] == ("claim_matched")
    # No chronology or stamp text reaches the parent-facing summary: the
    # stamp rides its own typed field and the summary text is untouched.
    assert payload.agents[0].current_task_summary == "initial"
    assert "Claim" not in (payload.agents[0].current_task_summary or "")


@pytest.mark.asyncio
async def test_agents_update_omits_the_stamp_for_a_previous_run_after_a_retask() -> (
    None
):
    clock = _ManualClock()
    registry = _retention_registry(clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    record = _idle_record("agent-1", clock=clock)
    finished = _stamped_run("agent-1", "run-1", _audit())
    record.run_history.append(finished)
    record.latest_run_id = finished.run_id
    registry._agent_records[record.agent_id] = record
    assert (await _emitted_payload(registry, notify)).agents[0].completion_audit

    # The same handle re-tasks: a NEW current run whose result does not exist
    # yet. The finished run keeps its stamped result in history, and the
    # previous run's stamp must NOT be republished for the new run.
    retask = RunRecord(
        run_id="run-2",
        agent_id="agent-1",
        identity="coder",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
    )
    record.current_run = retask
    record.latest_run_id = retask.run_id
    record.state = record.state.__class__.RUNNING
    notify.reset_mock()

    payload = await _emitted_payload(registry, notify)

    assert payload.agents[0].latest_run_id == "run-2"
    assert payload.agents[0].completion_audit is None
    assert finished.result is not None
    assert finished.result.completion_audit is not None


@pytest.mark.asyncio
async def test_agents_update_clears_the_stamp_after_result_expiry() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    record = _idle_record("agent-1", clock=clock)
    run = _stamped_run("agent-1", "run-1", _audit())
    record.run_history.append(run)
    record.latest_run_id = run.run_id
    registry._agent_records[record.agent_id] = record

    # The results collaborator's count-based expiry drops the stored result
    # and marks the run expired; the run record's copy is cleared with it.
    registry._expired_results.add((record.agent_id, run.run_id))
    run.result = None

    payload = await _emitted_payload(registry, notify)

    assert payload.agents[0].result_expired
    assert payload.agents[0].completion_audit is None


@pytest.mark.asyncio
async def test_agents_update_omits_the_stamp_for_an_expired_run_with_a_result() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    record = _idle_record("agent-1", clock=clock)
    run = _stamped_run("agent-1", "run-1", _audit())
    record.run_history.append(run)
    record.latest_run_id = run.run_id
    registry._agent_records[record.agent_id] = record

    # The expiry marker alone withholds the stamp: the stamp tracks the
    # RETAINED result, never a result the server has retired.
    registry._expired_results.add((record.agent_id, run.run_id))

    payload = await _emitted_payload(registry, notify)

    assert run.result is not None
    assert payload.agents[0].completion_audit is None


@pytest.mark.asyncio
async def test_agents_update_omits_the_stamp_after_release() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    record = _idle_record("agent-1", clock=clock)
    run = _stamped_run("agent-1", "run-1", _audit())
    record.run_history.append(run)
    record.latest_run_id = run.run_id
    registry._agent_records[record.agent_id] = record
    assert (await _emitted_payload(registry, notify)).agents[0].completion_audit
    notify.reset_mock()

    await registry.release_agent(record.agent_id)

    payload = await _emitted_payload(registry, notify)

    assert [agent.agent_id for agent in payload.agents] == []
    assert all(agent.completion_audit is None for agent in payload.agents)


@pytest.mark.asyncio
async def test_agents_update_omits_the_stamp_without_a_retained_result() -> None:
    clock = _ManualClock()
    registry = _retention_registry(clock)
    notify = AsyncMock()
    registry._notify_agents = notify
    record = _idle_record("agent-1", clock=clock)
    record.run_history.append(
        RunRecord(
            run_id="run-1",
            agent_id="agent-1",
            identity="coder",
            status=RunStatus.COMPLETED,
            completion_task=asyncio.get_running_loop().create_future(),
        )
    )
    record.latest_run_id = "run-1"
    registry._agent_records[record.agent_id] = record

    payload = await _emitted_payload(registry, notify)

    assert payload.agents[0].completion_audit is None
