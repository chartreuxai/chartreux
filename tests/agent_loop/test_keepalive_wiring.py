"""v0.8.0 C2: loop-level wiring tests for the keep-alive pilot.

A fake-backend ``AgentLoop`` with ``[keepalive] enabled`` asserts: (a) the
timer task exists only on the root, (b) a real turn triggers
``notify_real_call_*`` and snapshot invalidation, (c) a ping calls the gateway
with ``purpose=KEEPALIVE, account_conversation=False`` and leaves ``messages``,
the session log, and the title cadence untouched, (d) a provider error on a
ping leaves the loop healthy and records an ``error`` stop. Also covered here
per the plan's named test scope: the ``/stats`` keepalive-block assertion and
the ``skip_if_no_key`` pre-timer path.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

import pytest

from chartreux.cli.textual_ui.app import format_keepalive_stats_block
from chartreux.core.agent_loop._keepalive import (
    KeepaliveInvalidation,
    KeepaliveStopReason,
)
from chartreux.core.agent_loop._loop_types import RootAccountingOwner
from chartreux.core.agent_loop.llm_gateway import CallResources, CompletionInputs
from chartreux.core.usage import AsyncUsageWriter, UsagePurpose, UsageRecord
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class _AsyncListSink:
    """Async usage writer facade collecting records in memory."""

    def __init__(self) -> None:
        self.records: list[UsageRecord] = []

    async def append(self, record: UsageRecord) -> Any:
        self.records.append(record)
        return None


def _owner(sink: _AsyncListSink) -> RootAccountingOwner:
    return RootAccountingOwner(
        root_session_id="root-ledger",
        project_key="project",
        writer=cast(AsyncUsageWriter, sink),
    )


def _config(**keepalive: Any):
    return build_test_vibe_config(
        keepalive={"enabled": True, "min_idle_gap_seconds": 0.0, **keepalive}
    )


def _loop(config: Any = None, backend: FakeBackend | None = None, **kwargs: Any):
    return build_test_agent_loop(
        config=config or _config(),
        backend=backend or FakeBackend([mock_llm_chunk(content="ok")]),
        **kwargs,
    )


def _open_gates(loop: Any) -> None:
    """Let the scheduler's owned-dependency gate pass in tests."""
    scheduler = loop.keepalive_scheduler
    assert scheduler is not None
    scheduler._gates = replace(
        scheduler._gates, owned_dependency_runs_active=lambda: True
    )


@pytest.mark.asyncio
async def test_timer_task_exists_only_on_the_root() -> None:
    sink = _AsyncListSink()
    root = _loop(accounting_owner=_owner(sink))
    await root.wait_until_ready()
    try:
        assert root.keepalive_scheduler is not None
        assert root.keepalive_scheduler.enabled
        assert root._keepalive_timer_task is not None
        assert not root._keepalive_timer_task.done()

        child = build_test_agent_loop(
            config=_config(),
            backend=FakeBackend([mock_llm_chunk(content="ok")]),
            is_subagent=True,
            inherited_workspace=root.tool_manager.workspace,
            parent_session_id=root.session_id,
        )
        await child.wait_until_ready()
        try:
            assert child.keepalive_scheduler is None
            assert child._keepalive_timer_task is None
        finally:
            await child.aclose()
    finally:
        await root.aclose()
    # Closing cancelled the timer task.
    assert root._keepalive_timer_task is not None
    assert root._keepalive_timer_task.done()


@pytest.mark.asyncio
async def test_flag_off_builds_no_scheduler_and_no_timer() -> None:
    loop = build_test_agent_loop(
        config=build_test_vibe_config(),  # [keepalive] ships disabled
        backend=FakeBackend([mock_llm_chunk(content="ok")]),
    )
    await loop.wait_until_ready()
    try:
        assert loop.keepalive_scheduler is None
        assert loop._keepalive_timer_task is None
        assert loop.keepalive_stats_view() is None
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_skip_if_no_key_prevents_the_timer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The skip_if_no_key pre-timer path: a missing provider key is detected
    # BEFORE the timer starts and skips fast.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.resolve_api_key", lambda env: None
    )
    sink = _AsyncListSink()
    loop = _loop(accounting_owner=_owner(sink))
    try:
        loop._ensure_keepalive_timer()
        assert loop.keepalive_scheduler is None
        assert loop._keepalive_timer_task is None
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_real_turn_wraps_notify_and_invalidates_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = _AsyncListSink()
    loop = _loop(
        backend=FakeBackend([mock_llm_chunk(content="ok")]),
        accounting_owner=_owner(sink),
    )
    await loop.wait_until_ready()
    try:
        scheduler = loop.keepalive_scheduler
        assert scheduler is not None
        events: list[str] = []
        original_started = scheduler.notify_real_call_started
        original_finished = scheduler.notify_real_call_finished

        def spy_started() -> None:
            events.append("started")
            original_started()

        def spy_finished() -> None:
            events.append("finished")
            original_finished()

        monkeypatch.setattr(scheduler, "notify_real_call_started", spy_started)
        monkeypatch.setattr(scheduler, "notify_real_call_finished", spy_finished)
        async for _event in loop.act("hello"):
            pass

        assert events == ["started", "finished"]
        store = scheduler._snapshot_store
        # New user input invalidated; the post-turn save re-snapshotted.
        assert any(
            inv.trigger is KeepaliveInvalidation.CONVERSATION_ACTIVITY
            for inv in store.invalidations
        )
        current = store.current()
        assert current is not None
        assert current.deployment == loop.config.get_active_model().alias
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_ping_uses_keepalive_purpose_and_leaves_conversation_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([
        [mock_llm_chunk(content="ok")],
        [mock_llm_chunk(content="pong", cached_tokens=10)],
    ])
    sink = _AsyncListSink()
    loop = _loop(backend=backend, accounting_owner=_owner(sink))
    await loop.wait_until_ready()
    try:
        async for _event in loop.act("hello"):
            pass
        scheduler = loop.keepalive_scheduler
        assert scheduler is not None
        _open_gates(loop)

        seen: list[tuple[CompletionInputs, CallResources]] = []
        gateway = loop._llm_gateway
        original_complete = gateway.complete

        async def spy_complete(
            inputs: CompletionInputs, resources: CallResources
        ) -> Any:
            seen.append((inputs, resources))
            return await original_complete(inputs, resources)

        monkeypatch.setattr(gateway, "complete", spy_complete)

        messages_before = list(loop.messages)
        session_tokens_before = loop.stats.session_total_llm_tokens
        context_before = loop.stats.context_tokens

        task = await scheduler.maybe_ping()
        assert task is not None
        record = await task
        assert not record.superseded
        assert record.stop is None

        assert len(seen) == 1
        inputs, _resources = seen[0]
        assert inputs.purpose is UsagePurpose.KEEPALIVE
        assert inputs.account_conversation is False
        assert inputs.tools is None
        # The request is the frozen prefix plus one minimal ping message.
        assert inputs.messages[-1].content == "ping"
        assert len(inputs.messages) == len(messages_before) + 1

        # The ping left the conversation, the session log, and the title
        # cadence untouched: no transcript append, no session-total change.
        assert list(loop.messages) == messages_before
        assert loop.stats.session_total_llm_tokens == session_tokens_before
        assert loop.stats.context_tokens == context_before
        # ...but the ping's own usage accumulated only into keepalive_* stats.
        assert loop.stats.keepalive_prompt_tokens == 10
        # And the accountant wrote its purpose=keepalive ledger row.
        await asyncio.sleep(0)
        assert any(
            row.purpose is UsagePurpose.KEEPALIVE and row.keepalive is not None
            for row in sink.records
        )

        # The /stats keepalive block renders the pinned fields.
        block = format_keepalive_stats_block(loop.keepalive_stats_view())
        assert "## Keepalive" in block
        assert "**Attempts**: 1 / 100" in block
        assert "**Pings sent**: 1" in block
        assert "settled 1" in block
        assert "**Cache coverage**: 100.0%" in block
        assert "**Last stop**: none" in block
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_ping_provider_error_leaves_loop_healthy_and_records_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([[mock_llm_chunk(content="ok")]])
    sink = _AsyncListSink()
    loop = _loop(backend=backend, accounting_owner=_owner(sink))
    await loop.wait_until_ready()
    try:
        async for _event in loop.act("hello"):
            pass
        scheduler = loop.keepalive_scheduler
        assert scheduler is not None
        _open_gates(loop)
        # The next backend call (the ping) fails.
        monkeypatch.setattr(
            backend, "_exception_to_raise", RuntimeError("provider boom")
        )

        task = await scheduler.maybe_ping()
        assert task is not None
        record = await task  # never raises: the error is contained
        assert record.stop is KeepaliveStopReason.ERROR
        assert scheduler.stopped is KeepaliveStopReason.ERROR

        # The loop stays healthy: the timer task is still alive and a fresh
        # cycle (new activity) clears the stop without raising.
        assert loop._keepalive_timer_task is not None
        assert not loop._keepalive_timer_task.done()
        scheduler.note_conversation_activity()
        assert scheduler.stopped is None

        # The /stats block surfaces the stop reason.
        block = format_keepalive_stats_block(loop.keepalive_stats_view())
        assert "**Last stop**:" in block
    finally:
        await loop.aclose()
