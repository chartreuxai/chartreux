"""DEFECT-D / DEFECT-F repairs: the production receipt-sink wiring and the
retained-launch mint rollback.

DEFECT-D (major): the P2 receipt sink was never installed in production.
The child loop initializes ``_completion_evidence_sink`` to ``None`` and the
emission site drops receipts when the sink is ``None``; both production run
paths (the background admission in ``_subagent_execution.prepare_background``
and the foreground admission in ``_sessions._run_admitted``) created the run
accumulator but never wired the sink, so every post-tool receipt was minted
and discarded and the C2 receipt audit found no matching command. The repair
installs ``accumulator.completion_evidence_sink()`` on the child loop at run
admission on BOTH paths and clears it at run teardown (the loop persists
across a retained agent's runs; a stale sink must never receive a later
run's receipts — P3 requires a fresh accumulator per run). These tests
dispatch REAL child runs through the registry and assert the receipts land
in the run accumulator's chronology.

DEFECT-F (major): a denied override mint stranded a retained agent as
RUNNING. On retained reuse the record flips to RUNNING inside the registry
lock, but the C4 mint ran outside every rollback-protected region, so a
ledger denial (or a model-resolution failure) left the record permanently
busy with no run — reuse raised AgentBusyError, release returned CONFLICT,
and the reaper only expires SUSPENDED records. The repair moves the mint
inside the rollback-protected region: on any raise the minted permit (if one
was produced) is released, ``_rollback_reused_claim`` restores IDLE with the
prior idle_since, and the error re-raises.

All backends are offline stubs; no network, no real sleeps.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry, _AgentState
from chartreux.core.agent_loop_hooks import CompletionEvidenceReceipt
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.subagents import SubagentRunAccumulator, TaskArgs, TaskResult
from chartreux.core.tools.base import InvokeContext, ToolError
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _ManualClock,
    _retention_registry,
    _wait_result,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

ONE_ENDPOINT = "https://one.test"
TWO_ENDPOINT = "https://two.test"

pytestmark = pytest.mark.timeout(60)


def _todo_write_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(
            name="todo",
            arguments=json.dumps({
                "action": "write",
                "todos": [{"id": "1", "content": "item", "status": "pending"}],
            }),
        ),
    )


def _capture_sinks(monkeypatch: pytest.MonkeyPatch) -> list[SubagentRunAccumulator]:
    """Record every accumulator whose receipt sink the production wiring
    requests (exactly one call per run admission, on the fixed code).
    """
    captured: list[SubagentRunAccumulator] = []
    real = SubagentRunAccumulator.completion_evidence_sink

    def _recording(
        self: SubagentRunAccumulator,
    ) -> Callable[[CompletionEvidenceReceipt], None]:
        captured.append(self)
        return real(self)

    monkeypatch.setattr(SubagentRunAccumulator, "completion_evidence_sink", _recording)
    return captured


async def _sink_registry(
    monkeypatch: pytest.MonkeyPatch, child_backend: FakeBackend
) -> tuple[SessionRuntimeRegistry, Any]:
    """A real registry whose child executes one todo tool call, then finishes."""
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "todo"]), backend=FakeBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


def _assert_receipt_pairing(accumulator: SubagentRunAccumulator) -> None:
    """The run accumulator's chronology holds exactly one entry: the child's
    post-tool receipt paired with the observed result event of the SAME
    invocation (the executor-minted, run-scoped occurrence ID).
    """
    entries = list(accumulator._chronology._entries.values())
    assert len(entries) == 1
    entry = entries[0]
    assert entry.receipt is not None
    assert entry.receipt.tool_call_id == "todo-1"
    assert entry.receipt.tool_name == "todo"
    assert entry.receipt.occurrence_id
    assert entry.result_event is not None
    assert entry.result_event.occurrence_id == entry.receipt.occurrence_id


@pytest.mark.asyncio
async def test_background_run_emits_receipts_into_its_run_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFECT-D regression (background path): run admission installs the P2
    receipt sink on the child loop, so a real dispatched child run's post-tool
    receipts land in ITS run accumulator — and teardown clears the sink so a
    retained loop carries none afterwards.
    """
    captured = _capture_sinks(monkeypatch)
    child_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_todo_write_call("todo-1")])],
        [mock_llm_chunk(content="child done")],
    ])
    registry, parent = await _sink_registry(monkeypatch, child_backend)
    try:
        ctx = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
        ack = await _background_result(
            registry,
            TaskArgs(task="tool work", identity="coder", background=True, retain=True),
            ctx,
        )
        assert ack.agent_id is not None and ack.run_id is not None
        result = await _wait_result(registry, ack.agent_id, ack.run_id)
        assert result.completed is True
        # The production wiring requested the run accumulator's sink exactly
        # once — at run admission.
        assert len(captured) == 1
        _assert_receipt_pairing(captured[0])
        # Run teardown cleared the sink: the retained loop carries none, so
        # no later run's receipts can reach this spent accumulator.
        record = registry._agent_records[ack.agent_id]
        assert record.runtime.agent_loop._completion_evidence_sink is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_run_emits_receipts_into_its_run_accumulator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFECT-D regression (foreground path): the foreground admission wires
    the same sink, so a real dispatched foreground child run's post-tool
    receipts land in its run accumulator.
    """
    captured = _capture_sinks(monkeypatch)
    child_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_todo_write_call("todo-1")])],
        [mock_llm_chunk(content="child done")],
    ])
    registry, parent = await _sink_registry(monkeypatch, child_backend)
    try:
        ctx = InvokeContext(tool_call_id="fg-1", session_id=parent.session_id)
        result = cast(
            TaskResult,
            [
                item
                async for item in registry.run(
                    TaskArgs(task="tool work", identity="coder", background=False), ctx
                )
            ][-1],
        )
        assert result.completed is True
        assert len(captured) == 1
        _assert_receipt_pairing(captured[0])
    finally:
        await registry.drain_children()
        await parent.aclose()


# ----------------------------------------------------------------------
# DEFECT-F: a denied override mint on a retained retask restores IDLE
# ----------------------------------------------------------------------


def _override_config() -> Any:
    return build_test_vibe_config(
        active_model="small",
        include_project_context=False,
        include_prompt_detail=False,
        enabled_tools=["task"],
        tools={"task": {"permission": "always"}},
        providers=[
            {"name": "one", "api_base": ONE_ENDPOINT, "backend": "generic"},
            {"name": "two", "api_base": TWO_ENDPOINT, "backend": "generic"},
        ],
        models=[
            {
                "name": "small",
                "alias": "small",
                "provider": "one",
                "thinking": "off",
                "supported_thinking_levels": ["off", "high"],
            },
            {
                "name": "other",
                "alias": "other",
                "provider": "two",
                "thinking": "off",
                "supported_thinking_levels": ["off", "high"],
            },
        ],
    )


async def _override_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SessionRuntimeRegistry, Any]:
    """A real registry whose child backends are inspectable offline stubs."""

    def create_backend(*, provider: Any, **_kwargs: Any) -> FakeBackend:
        return FakeBackend([mock_llm_chunk(content=provider.name)])

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=_override_config(), backend=FakeBackend())
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _session_id: 0, clock=_ManualClock()
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


def _ledger(parent: Any) -> Any:
    return parent.config_orchestrator.cooldown_override_ledger


def _ctx(parent: Any, tool_call_id: str = "override") -> InvokeContext:
    return InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id)


@pytest.mark.asyncio
async def test_denied_override_mint_on_retained_retask_restores_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFECT-F regression: a ledger denial at the retained retask's mint
    must NOT strand the record RUNNING — the rollback-protected region
    restores IDLE with the prior idle_since, releases no permit (none was
    minted), and leaves the agent reusable.
    """
    registry, parent = await _override_registry(monkeypatch)
    clock = _ManualClock()
    ledger = _ledger(parent)
    monkeypatch.setattr(ledger, "_clock", clock)
    try:
        first = await _background_result(
            registry,
            TaskArgs(
                task="first",
                identity="coder",
                background=True,
                retain=True,
                model="small",
            ),
            _ctx(parent, "first"),
        )
        assert first.agent_id is not None and first.run_id is not None
        await _wait_result(registry, first.agent_id, first.run_id)
        record = registry._agent_records[first.agent_id]
        assert record.state is _AgentState.IDLE
        idle_since = record.idle_since
        assert idle_since is not None
        # Arm the spacing floor: one consumed start denies the next reserve.
        assert ledger.note_endpoint_failure(ONE_ENDPOINT) is True
        reserved = ledger.reserve(ONE_ENDPOINT)
        assert reserved.permit is not None
        assert ledger.start(reserved.permit).phase is not None
        # The retained retask's explicit override mint is denied (spacing
        # floor) AFTER the record flipped to RUNNING.
        with pytest.raises(ToolError, match="spacing floor"):
            await _background_result(
                registry,
                TaskArgs(
                    task="second",
                    agent_id=first.agent_id,
                    background=True,
                    model="small",
                    override_cooldown=True,
                ),
                _ctx(parent, "retask"),
            )
        # The record is back to IDLE with its idle_since restored — not
        # stranded RUNNING with no run.
        assert record.state is _AgentState.IDLE
        assert record.idle_since == idle_since
        # No permit stranded: past the spacing window the endpoint's single
        # in-flight slot is free — a reserve mints instead of denying
        # ATTEMPT_IN_FLIGHT.
        clock.advance(31)
        outcome = ledger.reserve(ONE_ENDPOINT)
        assert outcome.denial is None
        assert outcome.permit is not None
        ledger.release(outcome.permit)
        # The agent is reusable: an ordinary retask launches and completes.
        second = await _background_result(
            registry,
            TaskArgs(
                task="second", agent_id=first.agent_id, background=True, model="small"
            ),
            _ctx(parent, "retask-2"),
        )
        assert second.agent_id is not None
        assert second.agent_id == first.agent_id
        assert second.run_id != first.run_id
        result = await _wait_result(registry, second.agent_id, second.run_id)
        assert result.completed is True
        assert record.state is _AgentState.IDLE
    finally:
        await registry.close()
        await parent.aclose()
