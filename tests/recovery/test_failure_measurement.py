"""WP0 F5 — measurement repairs, verified against the F2 fixtures.

Two repairs land in WP0 F5 and are verified here:

1. The synthetic zero-turn fallback is gone: a monitor failure after real
   work measures turns from observed work (assistant messages), never from a
   synthetic zero (``chartreux/app_server/_subagent_execution.py``).
2. The private typed failure observation is preserved before the message-only
   ``PublicError`` projection discards it: the turn-phase capture
   (``chartreux/app_server/_turns.py``) is linked to the failed run by the
   background monitor, and a monitor failure itself is captured with phase
   ``"monitor"`` (``chartreux/core/recovery_observation.py`` — the public
   envelope is WP1's scope; this carrier stays private).

The public projection is unchanged: the failed run's ``TaskResult`` still
carries the message-only ``[Subagent error: ...]`` response.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chartreux.app_server._subagent_execution import SubagentExecution
from chartreux.app_server._turns import TurnController
from chartreux.core.llm.exceptions import BackendError, PayloadSummary
from chartreux.core.llm.failures import FailureCategory
from chartreux.core.recovery_models import FailureClass, FailurePhase
from chartreux.core.recovery_observation import run_observation, turn_observation
from chartreux.core.subagents import RunStopReason, TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    GatedFailureBackend,
    _background_result,
    _wait_result,
    _zero_retention_registry,
)


def _backend_error() -> BackendError:
    """A typed provider fault carrying request-ID/endpoint/status evidence."""
    return BackendError(
        provider="mistral",
        endpoint="https://api.mistral.ai/v1/chat/completions",
        status=503,
        reason="service unavailable",
        headers={"x-request-id": "req-wp0-f5"},
        body_text='{"error": {"code": "overloaded_error"}}',
        parsed_error='{"error": {"code": "overloaded_error"}}',
        model="zai-glm-5-3",
        payload_summary=PayloadSummary(
            model="zai-glm-5-3",
            message_count=1,
            approx_chars=10,
            temperature=0,
            has_tools=False,
            tool_choice=None,
        ),
    )


@pytest.mark.asyncio
async def test_f5_monitor_failure_measures_observed_turns(
    monkeypatch: pytest.MonkeyPatch, clean_observation_store
) -> None:
    """A monitor failure after one completed turn measures turns_used == 1.

    The child completes one turn (one assistant message), then the monitor's
    terminal-turn wait faults. The repaired fallback counts observed work:
    ``turns_used == 1`` (the pre-repair synthetic value was 0). The monitor
    failure itself is preserved as a typed observation (phase ``"monitor"``),
    retrievable by ``(agent_id, run_id)``, and the public projection is
    unchanged.
    """
    child = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, child)
    original_wait = TurnController.wait_for_operation

    async def faulting_wait(self: TurnController, turn_id: str) -> Any:
        await original_wait(self, turn_id)
        raise RuntimeError("monitor observation fault")

    # Patch before the launch: the background monitor is the only
    # wait_for_operation caller on this path.
    monkeypatch.setattr(TurnController, "wait_for_operation", faulting_wait)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child.started.wait(), timeout=2)
        child.release.set()
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )

        # Zero-turn fix: turns are measured from observed work, not a
        # synthetic zero.
        assert result.completed is False
        assert result.turns_used == 1
        assert result.stop_reason is RunStopReason.ERROR
        # The public projection is unchanged (message-only).
        assert "[Subagent error: monitor observation fault]" in result.response
        assert "Child completed" in result.response
        # The monitor failure is preserved as typed evidence, run-keyed.
        observation = run_observation(launch.agent_id, launch.run_id)
        assert observation is not None
        assert observation.phase == "monitor"
        assert observation.exception_type == "RuntimeError"
        assert observation.message == "monitor observation fault"
        assert observation.turn_id is not None
        assert observation.agent_id == launch.agent_id
        assert observation.run_id == launch.run_id
    finally:
        child.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_f5_turn_failure_preserves_typed_evidence(
    monkeypatch: pytest.MonkeyPatch, clean_observation_store
) -> None:
    """A child turn failure preserves typed evidence; the projection is unchanged.

    The child's transport faults with a typed 503. The turn-phase capture
    keeps the exception identity, the bounded cause chain, the low-level
    classification, and the request-ID/endpoint/status evidence; the monitor
    links it to the failed run. The public ``TaskResult`` projection is
    unchanged (message-only ``[Subagent error: ...]``), and ``turns_used`` is
    the measured zero (no assistant message was produced — this zero is
    observed, not synthetic).
    """
    child = GatedFailureBackend(_backend_error())
    registry, parent, _ = await _zero_retention_registry(monkeypatch, child)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child.started.wait(), timeout=2)
        child.release.set()
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )

        # Public projection unchanged.
        assert result.completed is False
        assert result.turns_used == 0
        assert result.stop_reason is RunStopReason.ERROR
        assert "[Subagent error: " in result.response
        assert "503" in result.response

        # Typed evidence preserved and linked to the failed run.
        observation = run_observation(launch.agent_id, launch.run_id)
        assert observation is not None
        assert observation.phase == "turn"
        assert observation.exception_type == "RuntimeError"
        assert observation.request_id == "req-wp0-f5"
        assert observation.endpoint == "https://api.mistral.ai/v1/chat/completions"
        assert observation.status == 503
        # The structured code lives in the low-level classification (the
        # BackendError carries it inside failure_info, not as a raw
        # exception attribute); the raw-attribute extraction finds
        # status/endpoint/request-id but not the code.
        assert observation.error_code is None
        assert observation.classification is not None
        assert observation.classification.category is FailureCategory.OVERLOAD
        assert observation.classification.failover_eligible is True
        assert observation.classification.error_code == "overloaded_error"
        assert observation.classification.status == 503
        # The bounded cause chain reaches the typed provider fault.
        assert any(
            cause.exception_type == "BackendError" for cause in observation.cause_chain
        )
        # The turn-keyed capture is retrievable by the child's turn id.
        record = registry._agent_records[launch.agent_id]
        completed = record.runtime.turns._completed_turns[-1]
        turn_level = turn_observation(completed.id)
        assert turn_level is not None
        assert turn_level.phase == "turn"
        assert turn_level.request_id == "req-wp0-f5"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_f5_zero_request_launch_failure_shape(
    monkeypatch: pytest.MonkeyPatch, clean_observation_store
) -> None:
    """The L1/L2 shape: a cooldown-blocked child dies with zero requests.

    The shared availability cooldown excludes the child's only deployment, so
    the admitted run's turn fails at admission: zero transport requests, zero
    turns, and a typed ``AllDeploymentsUnavailableError`` observation preserved
    run-keyed — the launch_failure evidence WP1's envelope must carry without
    log parsing.
    """
    child = BlockingBackend()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, child)
    try:
        # The root runtime's agent loop is the parent loop: arm the shared
        # availability registry the child's turns admit against.
        parent.config_orchestrator.availability_registry.record_failure(
            "glm-5-3", "mistral"
        )
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )

        # Zero requests, zero turns — measured, not synthetic.
        assert len(child.requests_messages) == 0
        assert result.completed is False
        assert result.turns_used == 0
        assert result.stop_reason is RunStopReason.ERROR
        assert "All deployments for model 'glm-5-3' are unavailable" in result.response
        assert "cooldown" in result.response

        # The launch_failure evidence is preserved run-keyed.
        observation = run_observation(launch.agent_id, launch.run_id)
        assert observation is not None
        assert observation.phase == "turn"
        assert observation.exception_type == "AllDeploymentsUnavailableError"
        assert observation.agent_id == launch.agent_id
        assert observation.run_id == launch.run_id
    finally:
        child.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_f5_secondary_failure_preserves_the_primary_evidence(
    monkeypatch: pytest.MonkeyPatch, clean_observation_store
) -> None:
    """Review R12: a SECONDARY result/finalization failure never displaces
    the primary failure's evidence.

    The child's turn faults with a typed 503 overload; the monitor then
    faults while publishing that result (a secondary failure for the same
    run). The run's published envelope, the result's recovery reference, and
    the run-keyed observation all keep the PRIMARY evidence — the typed
    provider overload at the LLM-call phase — never an INTERNAL-class
    monitor envelope.
    """
    child = GatedFailureBackend(_backend_error())
    registry, parent, _ = await _zero_retention_registry(monkeypatch, child)
    original_update = SubagentExecution._result_update
    updates = 0

    def faulting_update(*args: Any, **kwargs: Any) -> Any:
        # _result_update(context, failure_envelope, run_status) — a
        # staticmethod, so the arguments arrive unshifted.
        nonlocal updates
        envelope = args[1] if len(args) > 1 else kwargs.get("failure_envelope")
        if envelope is not None:
            updates += 1
            if updates == 1:
                raise RuntimeError("secondary result failure")
        return original_update(*args, **kwargs)

    # Armed before the launch: the try-block publication carries the primary
    # envelope, so the FIRST update faults; the except-block publication
    # (the secondary failure path) passes through.
    monkeypatch.setattr(
        SubagentExecution, "_result_update", staticmethod(faulting_update)
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="t1", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(child.started.wait(), timeout=2)
        child.release.set()
        result = await asyncio.wait_for(
            _wait_result(registry, launch.agent_id, launch.run_id), timeout=5
        )

        # The PRIMARY failure stays authoritative: the typed provider
        # overload observed at the LLM-call phase — never the INTERNAL-class
        # monitor envelope of the secondary failure.
        assert result.completed is False
        envelope = result.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.PROVIDER_OVERLOADED
        assert envelope.failure.phase is FailurePhase.LLM_CALL
        assert result.recovery is not None
        assert result.recovery.event_id == envelope.event_id

        # The run record's published envelope is the same primary evidence.
        record = registry._agent_records[launch.agent_id]
        run = next(
            (
                item
                for item in [record.current_run, *record.run_history]
                if item is not None and item.run_id == launch.run_id
            ),
            None,
        )
        assert run is not None
        assert run.failure is not None
        assert run.failure.event_id == envelope.event_id

        # The run-keyed observation is the FIRST (primary) one — the
        # secondary monitor observation never displaces it.
        observation = run_observation(launch.agent_id, launch.run_id)
        assert observation is not None
        assert observation.phase == "turn"
        assert observation.request_id == "req-wp0-f5"
        assert updates == 2
    finally:
        child.release.set()
        await registry.drain_children()
        await parent.aclose()
