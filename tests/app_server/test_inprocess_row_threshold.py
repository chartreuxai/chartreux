"""H4a pins: an IN-PROCESS subagent row renders usage against its threshold.

H3 gave every subagent row a compaction-threshold denominator, but only the
process-child route populated it: an in-process row still rendered
``45k/—`` because its summary carried no trigger. The fix is the core side --
``AgentSummary.context_threshold`` -- plus the in-process producer, which
reads the trigger from the child loop's own active-model projection (the same
computation the main status bar uses) and projects it onto the wire row.

The companion half is fail-closed: an unresolvable or untrustworthy trigger
stays unknown (``45k/—``), and a nonpositive trigger (compaction off) is
normalized to unknown by the wire model, so no row ever measures against the
hard window sitting beside it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.subagents import AgentAvailability, AgentSummary, TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.ui.context_display import format_context
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _config,
    _idle_record,
    _ManualClock,
    _retention_registry,
)
from tests.conftest import build_test_agent_loop
from tests.stubs.fake_backend import FakeBackend


def _stub_record(agent_id: str, threshold: object):
    """One idle record whose child loop answers the given trigger shape.

    The record also carries a real 1M window, so every fail-closed pin proves
    the row does NOT fall back to it when the trigger states no denominator.
    """
    record = _idle_record(agent_id, clock=_ManualClock())
    record.context_window = 1_000_000
    # The record's runtime is a stand-in, never a real child loop, so the
    # stub config carries only the projection the producer reads.
    cast(Any, record.runtime.agent_loop).config = SimpleNamespace(
        get_active_model=lambda: SimpleNamespace(auto_compact_threshold=threshold)
    )
    return record


def _summary_for(registry: SessionRuntimeRegistry, agent_id: str) -> AgentSummary:
    return next(
        summary
        for summary in registry._agent_summaries()
        if summary.agent_id == agent_id
    )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_inprocess_child_row_renders_its_own_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A real in-process launch's row measures usage at the child's trigger.

    The child's active model states a 500k trigger inside a 1M window, so the
    rendered denominator is the trigger -- not the window beside it. The
    pre-H4a base carries no core field at all and renders ``45k/—``.
    """
    backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    registry.bind_root(registry._build_child_runtime(parent))
    args = TaskArgs(task="work", identity="coder", background=True, retain=True)
    ctx = InvokeContext(tool_call_id="h4a-row", session_id=parent.session_id)
    try:
        acknowledgement = await _background_result(registry, args, ctx)
        assert acknowledgement.agent_id is not None
        record = registry._agent_records[acknowledgement.agent_id]
        # The usage numerator is child telemetry; the denominator is what this
        # pin is about, and it comes from the child's own active-model config.
        record.context_tokens = 45_000
        child_active = record.runtime.agent_loop.config.get_active_model()

        summary = _summary_for(registry, acknowledgement.agent_id)
        assert summary.context_threshold == child_active.auto_compact_threshold
        assert summary.context_threshold == 500_000
        assert summary.context_window == 1_000_000
        assert summary.context_threshold != summary.context_window

        row = next(
            item
            for item in registry._agents_update_payload()
            if item.agent_id == acknowledgement.agent_id
        )
        assert row.context_threshold == 500_000
        assert row.context_window == 1_000_000
        assert format_context(row.context_tokens, row.context_threshold) == (
            "45k/500k (9%)"
        )
    finally:
        backend.release.set()
        await registry.close()
        await parent.aclose()


def test_inprocess_rows_read_their_own_child_config() -> None:
    """The producer reads each row's child loop, never a shared constant."""
    registry = _retention_registry()
    first = _stub_record("agent-1", 321_000)
    second = _stub_record("agent-2", 654_000)
    registry._agent_records[first.agent_id] = first
    registry._agent_records[second.agent_id] = second

    rows = {row.agent_id: row for row in registry._agents_update_payload()}

    assert rows["agent-1"].context_threshold == 321_000
    assert rows["agent-2"].context_threshold == 654_000
    # Imported here so a pre-H4a tree fails THIS pin instead of the module
    # import, keeping the companion pins behaviorally discriminating.
    from chartreux.app_server._sessions import _record_compact_threshold

    assert _record_compact_threshold(first) == 321_000


def test_inprocess_row_threshold_fails_closed() -> None:
    """Unknown/disabled triggers render ``45k/—``, never the window.

    An unresolvable active model, an absent trigger, a nonpositive trigger
    (compaction off) and every malformed shape leave the denominator unknown.
    The wire model normalizes a nonpositive trigger to unknown on its own, so
    the row is pinned through the projected payload.
    """
    registry = _retention_registry()
    records = {
        "absent": _stub_record("agent-absent", None),
        "disabled": _stub_record("agent-disabled", 0),
        "negative": _stub_record("agent-negative", -5),
        "string": _stub_record("agent-string", "500000"),
        "boolean": _stub_record("agent-boolean", True),
        "fractional": _stub_record("agent-fractional", 1.5),
    }
    unresolvable = _idle_record("agent-unresolvable", clock=_ManualClock())
    unresolvable.context_window = 1_000_000

    def _unresolvable() -> object:
        raise ValueError("active model is not resolvable")

    cast(Any, unresolvable.runtime.agent_loop).config = SimpleNamespace(
        get_active_model=_unresolvable
    )
    records["unresolvable"] = unresolvable
    for record in records.values():
        registry._agent_records[record.agent_id] = record

    rows = {row.agent_id: row for row in registry._agents_update_payload()}

    for agent_id, row in rows.items():
        # The summary may retain a nonpositive trigger as "compaction off"; the
        # wire model turns it into unknown, and NO row renders a percentage
        # against the hard window.
        assert row.context_threshold is None, agent_id
        assert row.context_window == 1_000_000, agent_id
        assert format_context(45_000, row.context_threshold) == "45k/—", agent_id


def test_core_summary_threshold_defaults_to_unknown() -> None:
    """The core field defaults to None, so a pre-H4a producer fails closed."""
    summary = AgentSummary(
        agent_id="agent-1",
        availability=AgentAvailability.IDLE,
        current_run_id=None,
        current_run_status=None,
    )
    assert summary.context_threshold is None
