"""B2a: per-run baseline anchor written at subagent run admission."""

from __future__ import annotations

import asyncio

import pytest

from chartreux.app_server._subagent_execution import (
    _CUMULATIVE_STAT_FIELDS,
    _transfer_cumulative_stats,
)
from chartreux.core.config.models import RecoveryConfig
from chartreux.core.session_types import AgentStats
from chartreux.core.subagents import RetentionPolicy, TaskArgs
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    GatedSequenceBackend,
    _background_result,
    _zero_retention_registry,
)
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_replacement_policy import (
    _claim_config,
    _EndpointTimeout,
    _launch,
    _launch_failed_package,
    _replacement_registry,
    _WriteThenFaultBackend,
)

_ANCHOR_FIELDS = (
    "run_anchor_steps",
    "run_anchor_known_cost_total",
    "run_anchor_session_tokens",
    "run_anchor_has_unknown_cost",
)


def _capture_at_admission(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record the stats snapshot of every anchor_run_baseline call, in order."""
    captured: list[tuple] = []
    original = AgentStats.anchor_run_baseline

    def capture(self: AgentStats) -> None:
        captured.append((
            self.steps,
            self.known_cost_total,
            self.session_total_llm_tokens,
            self.has_unknown_cost,
        ))
        original(self)

    monkeypatch.setattr(AgentStats, "anchor_run_baseline", capture)
    return captured


@pytest.mark.asyncio
async def test_run_anchor_written_at_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = GatedSequenceBackend([mock_llm_chunk(content="work")])
    started, release = backend.add_gate()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    captured = _capture_at_admission(monkeypatch)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="work", background=True),
            InvokeContext(tool_call_id="fresh", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        await asyncio.wait_for(started.wait(), 2)
        stats = registry._agent_records[launch.agent_id].runtime.agent_loop.stats
        # A fresh loop's stats are all zero at admission.
        assert captured[-1] == (0, 0.0, 0, False)
        assert stats.run_anchor_steps == 0
        assert stats.run_anchor_known_cost_total == 0.0
        assert stats.run_anchor_session_tokens == 0
        assert stats.run_anchor_has_unknown_cost is False
    finally:
        release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_run_anchor_reflects_transferred_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blocking = BlockingBackend()
    registry, parent = await _replacement_registry(
        monkeypatch,
        RecoveryConfig(),
        [_WriteThenFaultBackend(_EndpointTimeout("https://beta.test/v1")), blocking],
    )
    captured = _capture_at_admission(monkeypatch)
    try:
        agent_id, _run_id, failed = await _launch_failed_package(registry, parent)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility.value == "eligible"
        predecessor = registry._agent_records[agent_id]
        predecessor.runtime.agent_loop.stats.steps = 5
        predecessor.runtime.agent_loop.stats.known_cost_total = 2.5
        predecessor.runtime.agent_loop.stats.session_prompt_tokens = 100
        predecessor.runtime.agent_loop.stats.session_completion_tokens = 50
        ack = await _launch(
            registry,
            parent,
            tool_call_id="claim",
            config=_claim_config(
                model="alt-model", work_package_id="wp-1", predecessor_agent_id=agent_id
            ),
        )
        assert ack.agent_id is not None and ack.agent_id != agent_id
        await asyncio.wait_for(blocking.started.wait(), 2)
        stats = registry._agent_records[ack.agent_id].runtime.agent_loop.stats
        # The transfer seeded the cumulative counters with the predecessor's
        # consumption, and the anchor — written downstream of the transfer —
        # captured exactly that inherited baseline.
        assert captured[-1] == (5, 2.5, 150, False)
        assert stats.run_anchor_steps == 5
        assert stats.run_anchor_known_cost_total == 2.5
        assert stats.run_anchor_session_tokens == 150
        assert stats.run_anchor_has_unknown_cost is False
        # The session maxes still see the transferred consumption: the
        # counters carry it (a turn step of the blocked run has been added).
        assert stats.steps >= 5
        assert stats.session_total_llm_tokens >= 150
    finally:
        blocking.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_run_anchor_reanchored_on_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = GatedSequenceBackend([
        mock_llm_chunk(content="one"),
        mock_llm_chunk(content="two"),
    ])
    started_first, release_first = backend.add_gate()
    started_second, release_second = backend.add_gate()
    registry, parent, _ = await _zero_retention_registry(monkeypatch, backend)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=100, max_idle_agents=10, retain_agents=True
    )
    captured = _capture_at_admission(monkeypatch)
    ctx = InvokeContext(tool_call_id="reuse", session_id=parent.session_id)
    try:
        first = await _background_result(
            registry, TaskArgs(task="one", background=True, retain=True), ctx
        )
        assert first.agent_id is not None
        record = registry._agent_records[first.agent_id]
        await asyncio.wait_for(started_first.wait(), 2)
        release_first.set()
        assert record.current_run is not None
        await asyncio.wait_for(record.current_run.completion_task, 5)
        loop = record.runtime.agent_loop
        stats = loop.stats
        stats.steps += 3
        stats.known_cost_total += 1.5
        stats.session_completion_tokens += 40
        # The first run's anchor predates the bump.
        assert stats.run_anchor_steps == 0
        second = await _background_result(
            registry,
            TaskArgs(task="two", background=True, retain=True, agent_id=first.agent_id),
            ctx,
        )
        assert second.agent_id == first.agent_id
        await asyncio.wait_for(started_second.wait(), 2)
        assert record.runtime.agent_loop is loop
        # The reused instance re-anchored from the accumulated stats...
        baseline = captured[-1]
        assert (
            stats.run_anchor_steps,
            stats.run_anchor_known_cost_total,
            stats.run_anchor_session_tokens,
            stats.run_anchor_has_unknown_cost,
        ) == baseline
        assert baseline != (0, 0.0, 0, False)
        assert baseline[0] >= 3
        assert baseline[1] >= 1.5
        assert baseline[2] >= 40
        # ...while the cumulative counters kept growing (no reset).
        assert stats.steps >= baseline[0]
        assert stats.session_total_llm_tokens >= baseline[2]
    finally:
        release_second.set()
        await registry.drain_children()
        await parent.aclose()


def test_run_anchor_fields_not_cumulative_transferred() -> None:
    assert not set(_ANCHOR_FIELDS) & set(_CUMULATIVE_STAT_FIELDS)
    source = AgentStats(
        steps=5,
        known_cost_total=2.5,
        session_prompt_tokens=100,
        session_completion_tokens=50,
    )
    source.anchor_run_baseline()
    target = AgentStats()
    target.run_anchor_steps = 7
    target.run_anchor_known_cost_total = 3.5
    target.run_anchor_session_tokens = 700
    target.run_anchor_has_unknown_cost = True
    _transfer_cumulative_stats(source, target)
    assert target.steps == 5
    assert target.known_cost_total == 2.5
    assert target.session_total_llm_tokens == 150
    assert target.run_anchor_steps == 7
    assert target.run_anchor_known_cost_total == 3.5
    assert target.run_anchor_session_tokens == 700
    assert target.run_anchor_has_unknown_cost is True


def test_legacy_stats_without_anchor_fields_parse() -> None:
    legacy = {
        "steps": 3,
        "session_prompt_tokens": 100,
        "session_completion_tokens": 50,
        "session_cached_tokens": 10,
        "tool_calls_agreed": 1,
        "tool_calls_rejected": 0,
        "tool_calls_hook_denied": 0,
        "tool_calls_failed": 0,
        "tool_calls_succeeded": 2,
        "known_cost_total": 1.25,
        "has_unknown_cost": False,
        "context_tokens": 200,
        "last_turn_prompt_tokens": 30,
        "last_turn_completion_tokens": 20,
        "last_turn_cached_tokens": 5,
        "last_turn_duration": 1.5,
        "tokens_per_second": 12.5,
        "input_price_per_million": 1.0,
        "output_price_per_million": 2.0,
    }
    stats = AgentStats.model_validate(legacy)
    for field_name in _ANCHOR_FIELDS:
        assert getattr(stats, field_name) is None
    assert stats.steps == 3
    assert stats.known_cost_total == 1.25
    assert stats.has_unknown_cost is False
    assert stats.session_total_llm_tokens == 150
    assert stats.context_tokens == 200
    # The token-only legacy shape (the existing before-validator's target).
    token_only = AgentStats.model_validate({
        "steps": 2,
        "session_prompt_tokens": 10,
        "session_completion_tokens": 5,
    })
    assert token_only.has_unknown_cost is True
    assert token_only.run_anchor_known_cost_total is None


def test_reset_context_state_preserves_anchors() -> None:
    stats = AgentStats(
        steps=3,
        known_cost_total=2.5,
        session_prompt_tokens=100,
        session_completion_tokens=50,
        context_tokens=999,
    )
    stats.anchor_run_baseline()
    stats.reset_context_state()
    assert stats.context_tokens == 0
    assert stats.run_anchor_steps == 3
    assert stats.run_anchor_known_cost_total == 2.5
    assert stats.run_anchor_session_tokens == 150
    assert stats.run_anchor_has_unknown_cost is False
