"""J2/J3 pins: the threshold denominator on every subagent row.

H3/H4a gave the row a compaction-threshold denominator: a subagent row renders
``usage/threshold``, never ``usage/window``. Two producers still dropped it.

J2 is the process route's tool surface: ``check_agents`` builds core
``AgentSummary`` objects while the wire inventory read carries the field, so a
process child read through the tool surface had no denominator at all even
though the same handle's wire row did.

J3 is the in-process closed row: the terminal tombstone froze usage and the
window but not the trigger, so an agent that lived on as a closed row lost its
denominator and rendered unknown.

Both pins hold the two surfaces to the live row's value and both fail-closed
halves to ``None`` -- never the window -- so a row without a trustworthy
trigger renders ``45k/—``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from chartreux.app_server.root_agent import AgentChildHost
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.models import SubagentsConfig
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext
from chartreux.ui.context_display import format_context
from tests.app_server._subagent_test_support import (
    _idle_record,
    _ManualClock,
    _retention_registry,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

_THRESHOLD = 500_000
_WINDOW = 1_000_000
_TOKENS = 45_000


class _SpawnedChild:
    """One spawned child double: no wire client, so no telemetry is observed."""

    def __init__(self, agent_id: str, session_id: str) -> None:
        self.agent_id = agent_id
        self.session_id = session_id
        self.child_session_dir = None
        self.wire_client = None
        self.closes = 0
        self.alive = True

    async def close(self) -> None:
        self.closes += 1
        self.alive = False


class _Spawn:
    """The spawn adapter these pins publish through."""

    def __init__(self) -> None:
        self.spawned: list[_SpawnedChild] = []

    async def spawn(self, spec: Any) -> _SpawnedChild:
        child = _SpawnedChild(spec.agent_id, f"child-session-{len(self.spawned) + 1}")
        self.spawned.append(child)
        return child


async def _published_handle(tmp_path: Path) -> tuple[Any, AgentChildHost, Any]:
    """Publish one retained process child and answer its host and handle."""
    config = build_test_vibe_config(
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
        )
    )
    config = config.model_copy(update={"subagents": SubagentsConfig()})
    loop = build_test_agent_loop(
        config=config, backend=FakeBackend([[mock_llm_chunk(content="ack")]])
    )
    host = AgentChildHost(loop, spawn=_Spawn())
    args = TaskArgs(
        task="Inspect the project", identity="coder", background=True, retain=True
    )
    ctx = InvokeContext(tool_call_id="j2-row", session_id=loop.session_id)
    handle = await host.publish(args, ctx, attempt_id="attempt-j2", retained=True)
    return loop, host, handle


# ---------------------------------------------------------------------------
# J2: the process route's tool surface carries the same row the wire carries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_process_route_check_agents_carries_the_row_threshold(
    tmp_path: Path,
) -> None:
    """``check_agents`` answers the thinking level and the threshold, like the wire."""
    loop, host, handle = await _published_handle(tmp_path)
    try:
        handle.effective_thinking = "high"
        handle.context_tokens = _TOKENS
        handle.context_window = _WINDOW
        handle.context_threshold = _THRESHOLD

        summaries = {item.agent_id: item for item in await host.check_agents()}
        summary = summaries[handle.agent_id]
        assert summary.effective_thinking == "high"
        assert summary.context_threshold == _THRESHOLD
        assert summary.context_tokens == _TOKENS
        assert summary.context_window == _WINDOW

        rows = {row.agent_id: row for row in host.inventory_snapshot().agents}
        row = rows[handle.agent_id]
        # The tool surface and the wire row describe one handle identically.
        assert row.effective_thinking == summary.effective_thinking
        assert row.context_threshold == summary.context_threshold
        assert row.context_tokens == summary.context_tokens
        assert row.context_window == summary.context_window
        # And the denominator is the trigger, never the window beside it.
        assert row.context_threshold != row.context_window
        assert format_context(row.context_tokens, row.context_threshold) == (
            "45k/500k (9%)"
        )
    finally:
        await host.close()
        await loop.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_process_route_row_states_unknown_when_nothing_was_observed(
    tmp_path: Path,
) -> None:
    """An unobserved trigger and thinking level stay None on both surfaces.

    The child reported nothing (the double has no wire client), so the honest
    answer is unknown on the tool surface too -- never a value borrowed from
    the parent and never the window.
    """
    loop, host, handle = await _published_handle(tmp_path)
    try:
        handle.context_tokens = _TOKENS
        handle.context_window = _WINDOW

        summaries = {item.agent_id: item for item in await host.check_agents()}
        summary = summaries[handle.agent_id]
        assert summary.context_threshold is None
        assert summary.effective_thinking is None

        rows = {row.agent_id: row for row in host.inventory_snapshot().agents}
        row = rows[handle.agent_id]
        assert row.context_threshold is None
        assert row.effective_thinking is None
        assert format_context(row.context_tokens, row.context_threshold) == "45k/—"
    finally:
        await host.close()
        await loop.aclose()


# ---------------------------------------------------------------------------
# J3: a settled in-process row keeps the denominator its live row read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_settled_in_process_row_keeps_its_threshold() -> None:
    """An evicted record's closed row still renders against its own trigger."""
    clock = _ManualClock()
    registry = _retention_registry(clock)
    record = _idle_record("agent-1", clock=clock)
    record.context_tokens = _TOKENS
    record.context_window = _WINDOW
    cast(Any, record.runtime.agent_loop).config = SimpleNamespace(
        get_active_model=lambda: SimpleNamespace(auto_compact_threshold=_THRESHOLD)
    )
    registry._agent_records[record.agent_id] = record
    try:
        assert await registry._evict_agent(record.agent_id, "ttl")
        tombstone = registry._evicted_agents[record.agent_id].summary
        assert tombstone.context_threshold == _THRESHOLD
        assert tombstone.context_window == _WINDOW
        assert tombstone.context_threshold != tombstone.context_window

        rows = {row.agent_id: row for row in registry._agents_update_payload()}
        row = rows[record.agent_id]
        assert row.availability == "evicted"
        assert row.context_threshold == _THRESHOLD
        assert format_context(row.context_tokens, row.context_threshold) == (
            "45k/500k (9%)"
        )
    finally:
        await registry.drain_children()


@pytest.mark.asyncio
async def test_settled_row_threshold_fails_closed_to_unknown() -> None:
    """A settled row whose trigger cannot be resolved renders ``45k/—``.

    An unresolvable active model, an absent trigger and a trigger the strict
    shape check cannot vouch for all leave the denominator unknown. The
    window the record still carries is never substituted.
    """
    clock = _ManualClock()
    registry = _retention_registry(clock)

    def _unresolvable() -> object:
        raise ValueError("active model is not resolvable")

    def _broken() -> object:
        raise RuntimeError("the child loop's config cannot be read")

    records = []
    for agent_id, config in (
        ("agent-unresolvable", SimpleNamespace(get_active_model=_unresolvable)),
        ("agent-broken", SimpleNamespace(get_active_model=_broken)),
        ("agent-absent", SimpleNamespace(get_active_model=lambda: SimpleNamespace())),
        (
            "agent-string",
            SimpleNamespace(
                get_active_model=lambda: SimpleNamespace(
                    auto_compact_threshold="500000"
                )
            ),
        ),
    ):
        record = _idle_record(agent_id, clock=clock)
        record.context_tokens = _TOKENS
        record.context_window = _WINDOW
        cast(Any, record.runtime.agent_loop).config = config
        registry._agent_records[record.agent_id] = record
        records.append(record)
    try:
        for record in records:
            assert await registry._evict_agent(record.agent_id, "ttl")
        rows = {row.agent_id: row for row in registry._agents_update_payload()}
        for record in records:
            row = rows[record.agent_id]
            assert row.context_threshold is None, record.agent_id
            assert row.context_window == _WINDOW, record.agent_id
            assert format_context(row.context_tokens, row.context_threshold) == (
                "45k/—"
            ), record.agent_id
    finally:
        await registry.drain_children()
