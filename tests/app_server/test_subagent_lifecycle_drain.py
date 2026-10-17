"""Drain-ownership behavior tests for the subagent lifecycle."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry
from chartreux.core.subagents import RetentionPolicy
from tests.app_server._subagent_test_support import (
    _GatedWakeup,
    _ManualClock,
    _reaper_registry,
    _rearm,
    _record,
    _tombstone,
)

# ---------------------------------------------------------------------------
# 25. Drain owns displaced reapers
# ---------------------------------------------------------------------------


async def _handoff_with_gated_sink() -> tuple[
    SessionRuntimeRegistry,
    asyncio.Task[None],
    asyncio.Event,
    list[tuple[list[Any], list[Any]]],
]:
    """Drive a reaper handoff whose eviction payload blocks in a gated sink.

    Returns the registry, the displaced reaper task, the sink gate, and the
    delivered payloads. The displaced reaper is left blocked: its eviction
    payload is authorized and permitted, but the sink holds it until the
    gate is released.
    """
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=32,
    )
    record = _record("agent-1", clock=clock)
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    delivered: list[tuple[list[Any], list[Any]]] = []
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def gated_sink(summaries: list[Any], evictions: list[Any]) -> None:
        if evictions:
            # The displaced reaper's eviction payload blocks here until
            # the test releases the gate.
            entered.set()
            await gate.wait()
        delivered.append((list(summaries), list(evictions)))

    registry._notify_agents = gated_sink
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    displaced = registry._reaper_task
    assert displaced is not None
    clock.advance(10)
    wakeup.gates[0].set()
    # The reaper selected the idle victim, handed the schedule to a fresh
    # timer, and its eviction notification now blocks in the gated sink.
    await asyncio.wait_for(entered.wait(), timeout=1)
    return registry, displaced, gate, delivered


@pytest.mark.asyncio
async def test_drain_owns_the_displaced_reaper_until_it_settles() -> None:
    # The round-4 reproduction: the reaper hands its schedule to a fresh
    # timer before awaiting eviction cleanup and unregisters itself — the
    # displaced task was in no drain-owned set, so drain_children returned
    # while it was still live.
    registry, displaced, gate, delivered = await _handoff_with_gated_sink()
    # The handoff happened: a fresh timer owns the schedule and the
    # displaced reaper is registered as drain-owned.
    assert registry._reaper_task is not None
    assert registry._reaper_task is not displaced
    assert displaced in registry._detached_reaper_tasks
    assert not displaced.done()
    # Drain cancels and awaits the displaced reaper (aborting its gated
    # sink call) instead of returning while it is still live.
    await asyncio.wait_for(registry.drain_children(), timeout=1)
    assert displaced.done()
    assert displaced.cancelled()
    assert not registry._detached_reaper_tasks
    assert registry._reaper_task is None
    # Drain's final empty payload is the only delivered inventory.
    assert delivered == [([], [])]
    gate.set()
    await asyncio.sleep(0.01)
    assert delivered == [([], [])]


@pytest.mark.asyncio
async def test_no_inventory_payload_after_drain_final_empty_payload() -> None:
    # The round-4 reproduction (ordering): with the sink gated across the
    # handoff, drain_children returned while the displaced reaper was
    # still live; releasing the gate then delivered the old non-empty
    # payload AFTER drain's final empty payload.
    registry, displaced, gate, delivered = await _handoff_with_gated_sink()
    assert not displaced.done()
    await asyncio.wait_for(registry.drain_children(), timeout=1)
    # Drain's final empty payload is the last delivered inventory...
    assert delivered == [([], [])]
    final_count = len(delivered)
    # ...and releasing the gate delivers nothing more: the displaced
    # reaper was cancelled and awaited by drain, so no stale producer can
    # publish after drain's final empty payload.
    gate.set()
    await asyncio.sleep(0.01)
    assert len(delivered) == final_count
    assert delivered[-1] == ([], [])
    assert displaced.done()


# ---------------------------------------------------------------------------
# 26. Drain-boundary disappearances are still published
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_publishes_disappearance_suppressed_at_the_boundary() -> None:
    # The round-4 drain-guard regression (FIFO lock ordering): the reaper
    # queues on the registry lock BEFORE drain starts; drain sets
    # `_draining_children`; the reaper then prunes the FINAL tombstone and
    # its publication hits the drain guard (suppressed) — so drain's own
    # snapshot sees nothing change and would publish nothing. The
    # disappearance must still be delivered: the final payload is the empty
    # inventory.
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=10,
        max_tombstones=32,
    )
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1", reason="closed", closed_at=0, expires_at=10
    )
    delivered: list[tuple[list[Any], list[Any]]] = []

    async def recording_sink(summaries: list[Any], evictions: list[Any]) -> None:
        delivered.append((list(summaries), list(evictions)))

    registry._notify_agents = recording_sink
    # The client's delivered inventory shows the tombstone...
    await registry._emit_agents_update()
    assert any(summary.agent_id == "agent-1" for summary in delivered[0][0])
    # ...and the tombstone's metadata deadline arms the reaper.
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    clock.advance(10)
    async with registry._registry_lock:
        # The reaper queues on the registry lock BEFORE drain starts...
        wakeup.gates[0].set()
        drain = asyncio.create_task(registry.drain_children())
        # ...and drain sets `_draining_children` while queued behind it.
        await asyncio.sleep(0)
    # FIFO: the reaper proceeds first, prunes the final tombstone, and its
    # publication is suppressed by the drain guard; drain then completes
    # with an empty registry.
    await asyncio.wait_for(drain, timeout=1)
    # The disappearance is published: the final payload is the empty
    # inventory, and nothing arrives after it.
    assert delivered[-1] == ([], [])
    assert len(delivered) == 2
    await asyncio.sleep(0.01)
    assert delivered[-1] == ([], [])


@pytest.mark.asyncio
async def test_drain_publishes_disappearance_cancelled_in_flight() -> None:
    # The second regression reproduction: the reaper prunes the final
    # tombstone and blocks INSIDE its publication (gated sink) when drain
    # starts; drain cancels the reaper, aborting the in-flight publication.
    # The disappearance must still be delivered by drain's final empty
    # inventory.
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=0,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=10,
        max_tombstones=32,
    )
    registry._evicted_agents["agent-1"] = _tombstone(
        "agent-1", reason="closed", closed_at=0, expires_at=10
    )
    delivered: list[tuple[list[Any], list[Any]]] = []

    async def recording_sink(summaries: list[Any], evictions: list[Any]) -> None:
        delivered.append((list(summaries), list(evictions)))

    registry._notify_agents = recording_sink
    # The client's delivered inventory shows the tombstone...
    await registry._emit_agents_update()
    assert any(summary.agent_id == "agent-1" for summary in delivered[0][0])
    # ...and the tombstone's metadata deadline arms the reaper.
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    clock.advance(10)
    # From here on, the first sink call blocks until it is cancelled (drain
    # cancels the reaper before releasing it).
    entered = asyncio.Event()
    never = asyncio.Event()
    calls = 0

    async def gated_sink(summaries: list[Any], evictions: list[Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            await never.wait()
        delivered.append((list(summaries), list(evictions)))

    registry._notify_agents = gated_sink
    wakeup.gates[0].set()
    # The reaper pruned the final tombstone and is blocked inside its
    # publication when drain starts.
    await asyncio.wait_for(entered.wait(), timeout=1)
    reaper = registry._reaper_task
    assert reaper is not None
    drain = asyncio.create_task(registry.drain_children())
    # Drain cancels the reaper (aborting the in-flight publication) and
    # completes with an empty registry.
    await asyncio.wait_for(drain, timeout=1)
    assert reaper.cancelled()
    # The disappearance is published: the final payload is the empty
    # inventory (the reaper's blocked call never delivered), and nothing
    # arrives after it.
    assert delivered[-1] == ([], [])
    assert len(delivered) == 2
    await asyncio.sleep(0.01)
    assert delivered[-1] == ([], [])


# ---------------------------------------------------------------------------
# 27. Ordinary rearming keeps drain ownership of the displaced reaper
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drain_owns_the_reaper_displaced_by_an_ordinary_rearm() -> None:
    # The N6 reproduction: a metadata-only reaper is blocked mid-publication
    # in a cancellation-resistant sink; a real release_agent() call's rearm
    # replaces the live reaper. The displaced reaper belonged to neither
    # `_reaper_task` nor `_detached_reaper_tasks`, so drain_children returned
    # while it was still live, and opening the sink gate then delivered the
    # stale non-empty snapshot AFTER drain's final empty payload.
    clock = _ManualClock(0)
    wakeup = _GatedWakeup()
    registry, _root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    registry._retention_policy = RetentionPolicy(
        idle_ttl_seconds=10,
        max_idle_agents=0,
        retain_agents=False,
        tombstone_ttl_seconds=1000,
        max_tombstones=32,
    )
    # A live record the release_agent() call removes (its idle TTL is
    # disabled so the reaper never selects it as an eviction victim).
    record = _record("agent-keep", clock=clock)
    record.idle_ttl_seconds = 0
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime
    # An expired tombstone (the reaper's metadata-only prune) and a live
    # tombstone (kept until drain so drain emits its final empty payload).
    registry._evicted_agents["agent-old"] = _tombstone(
        "agent-old", reason="closed", closed_at=0, expires_at=10
    )
    registry._evicted_agents["agent-stay"] = _tombstone(
        "agent-stay", reason="closed", closed_at=0, expires_at=1000
    )
    delivered: list[tuple[list[Any], list[Any]]] = []
    entered = asyncio.Event()
    gate = asyncio.Event()
    calls = 0

    async def resistant_sink(summaries: list[Any], evictions: list[Any]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            # Cancellation-resistant: a displaced reaper stays blocked in
            # its in-flight publication until the gate opens.
            entered.set()
            while not gate.is_set():
                try:
                    await gate.wait()
                except asyncio.CancelledError:
                    pass
        delivered.append((list(summaries), list(evictions)))

    registry._notify_agents = resistant_sink
    await _rearm(registry)
    assert await asyncio.wait_for(wakeup.arrivals.get(), timeout=1) == 10
    clock.advance(10)
    wakeup.gates[0].set()
    # The reaper pruned the expired tombstone and is blocked mid-publication.
    await asyncio.wait_for(entered.wait(), timeout=1)
    reaper = registry._reaper_task
    assert reaper is not None
    try:
        # The release_agent() call's rearm replaces the live reaper.
        assert await registry.release_agent("agent-keep")
        # Drain owns the displaced reaper: it cannot return while the
        # reaper is still blocked in its in-flight publication.
        drain = asyncio.create_task(registry.drain_children())
        await asyncio.sleep(0.05)
        gate.set()
        await asyncio.wait_for(drain, timeout=1)
    finally:
        # Never leave the cancellation-resistant sink blocked: the
        # displaced reaper must settle even when an assertion fails.
        gate.set()
    await asyncio.sleep(0.01)
    # The stale snapshot is delivered at most mid-drain — never after
    # drain's final empty payload.
    assert delivered[-1] == ([], [])
    assert len(delivered) == 3
    stale = delivered[1][0]
    assert any(summary.agent_id == "agent-keep" for summary in stale)
    # The displaced reaper was drain-owned (registered at the rearm) and
    # has settled; the drain-owned set is empty again.
    assert reaper.done()
    assert not registry._detached_reaper_tasks
