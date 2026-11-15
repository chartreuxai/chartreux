"""Registry inventory revision + joint-acquisition inventory snapshot (C6a).

v0.7.9 point 5, owner half (`development` design of record §§3.2, 7-8): the
registry carries ONE monotone inventory revision advanced by a synchronous
increment in every mutation's held region, and ``inventory_snapshot()`` reads
the revision and the wire projection together under the SAME joint
acquisition as the quiescence precedent (``_registry_lock`` AND
``acquire_run_state``).

The census below drives every lifecycle class through the real registry
(collaborator paths included, never a mocked delegate) and asserts the
revision advanced — a missed class fails. The exclusion proof races a
run-state transition against concurrent snapshots and asserts the returned
revision is never torn or pre-transition; the floor test pins the
per-incarnation pairing contract: a generation change establishes a fresh
floor and an old incarnation's floor never rejects a new incarnation's
snapshot.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import (
    InventorySnapshot,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicEntryGenerationStatus,
)
from chartreux.app_server.protocol import HistoryEntryAddedParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config.models import RecoveryConfig, RecoverySectionConfig
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.subagents import (
    ReconcileOutcome,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    WaitStatus,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _config,
    _ManualClock,
    _reaper_registry,
    _record,
    _retention_registry,
    _root_registry,
    _run,
    _stored_result,
    _tombstone,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


async def _until(predicate: Any, timeout: float = 10.0) -> None:
    """Poll until ``predicate`` holds; a gate, not a latency bound."""
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def _revision(registry: SessionRuntimeRegistry) -> int:
    return getattr(registry, "_inventory_revision", 0)


def _row(snapshot: InventorySnapshot, agent_id: str) -> Any:
    return next(item for item in snapshot.agents if item.agent_id == agent_id)


class _ServingParentBackend(FakeBackend):
    """A parent backend that answers every notification turn quietly."""

    async def complete(self, **_kwargs: Any) -> LLMChunk:
        return mock_llm_chunk(content="notification handled")

    async def complete_streaming(
        self, **_kwargs: Any
    ) -> AsyncGenerator[LLMChunk, None]:
        yield mock_llm_chunk(content="notification handled")


class _TruncatingChildBackend(FakeBackend):
    """Truncate the child's first stream; complete the resumed successor.

    The truncating response is content bearing with NO finish reason, so the
    sharing gateway raises the typed ``IncompleteStreamError`` at the same
    detection site the truncation suite drives: the run resolves FAILED with
    UNKNOWN tool effects, so recovery requires an explicit reconciliation
    before the successor is admitted.
    """

    def __init__(self) -> None:
        super().__init__([])
        self.calls = 0

    def _next_stream(self) -> list[LLMChunk]:
        self.calls += 1
        if self.calls == 1:
            return [mock_llm_chunk(content="partial answer", stop_reason=None)]
        return [mock_llm_chunk(content="recovered")]

    async def complete(self, **_kwargs: Any) -> LLMChunk:
        aggregate = LLMChunk(message=LLMMessage(role=Role.assistant))
        for chunk in self._next_stream():
            aggregate += chunk
        return aggregate

    async def complete_streaming(
        self, **_kwargs: Any
    ) -> AsyncGenerator[LLMChunk, None]:
        for chunk in self._next_stream():
            yield chunk


def _recovery_registry(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend
) -> tuple[SessionRuntimeRegistry, AgentLoop]:
    """A registry whose in-process child streams behind ``backend``.

    Subagent loops disable streaming at construction, but the truncation
    detection site lives on the sharing gateway's streaming path, so the pin
    drives that path on the child (as in the truncation suite). The recovery
    policy zeroes the cooldown and jitter, so the explicit resume admission
    is deterministic.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    create_child = AgentRuntimeFactory.create_child

    async def streaming_child(
        factory: AgentRuntimeFactory, loop: AgentLoop, agent_name: str, **kwargs: Any
    ) -> AgentLoop:
        child = await create_child(factory, loop, agent_name, **kwargs)
        child.enable_streaming = True
        return child

    monkeypatch.setattr(AgentRuntimeFactory, "create_child", streaming_child)
    policy = RecoveryConfig(
        auto_retries_max=1, cooldowns_seconds=[0.0], jitter_fraction=0.0
    )
    config = _config().model_copy(
        update={"recovery": RecoverySectionConfig(subagents=policy)}
    )
    parent = build_test_agent_loop(
        config=config, backend=_ServingParentBackend(), enable_streaming=True
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


def _launch_task(
    registry: SessionRuntimeRegistry, parent: AgentLoop, tool_call_id: str
) -> Any:
    return _background_result(
        registry,
        TaskArgs(task="inspect", identity="coder", background=True, retain=True),
        InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id),
    )


# ---------------------------------------------------------------------------
# Read parity: the snapshot IS the in-process agents/update projection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inventory_snapshot_matches_visible_summaries_projection() -> None:
    """The snapshot's agents are exactly ``_agents_update_payload()`` over
    ``_visible_summaries`` — the bar's own projection — with the root
    incarnation and revision riding alongside (point 5 §3.1/5).
    """
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock)
    idle = _record("agent-idle", clock=clock)
    running = _record("agent-running", clock=clock)
    running.state = _AgentState.RUNNING
    run = _run(running, "run-running", started_at=clock(), status=RunStatus.RUNNING)
    running.current_run = run
    registry._agent_records[idle.agent_id] = idle
    registry._agent_records[running.agent_id] = running
    registry._evicted_agents["agent-gone"] = _tombstone(
        "agent-gone", reason="closed", closed_at=0, expires_at=None
    )
    stored = _stored_result(idle.agent_id, "run-old")
    registry._result_store[(stored.agent_id, stored.run_id)] = stored

    snapshot = await registry.inventory_snapshot()

    assert snapshot.incarnation == ("root", 0)
    assert snapshot.revision == _revision(registry)
    # Parity vs the payload projection (the source the TUI bar consumes).
    assert snapshot.agents == tuple(registry._agents_update_payload())
    # Parity vs the visible summaries: same rows, same order.
    assert [item.agent_id for item in snapshot.agents] == [
        summary.agent_id for summary in registry._visible_summaries()
    ]
    assert {item.agent_id for item in snapshot.agents} == {
        "agent-running",
        "agent-idle",
        "agent-gone",
    }
    running_row = _row(snapshot, "agent-running")
    assert running_row.availability == "running"
    assert running_row.current_run_id == "run-running"
    idle_row = _row(snapshot, "agent-idle")
    assert idle_row.availability == "idle"
    assert idle_row.uncollected_run_ids == ["run-old"]


# ---------------------------------------------------------------------------
# Joint exclusion: the run-state transition race (never torn, never stale).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_joint_snapshot_excludes_run_state_transitions() -> None:
    """A snapshot racing a run-state transition blocks on the joint
    acquisition: it can never observe a torn or pre-transition revision.
    """
    clock = _ManualClock()
    registry, _root = _root_registry(clock)
    record = _record("agent-1", clock=clock)
    run = _run(record, "run-1", started_at=clock(), status=RunStatus.RUNNING)
    record.current_run = run
    registry._agent_records[record.agent_id] = record

    pre = await registry.inventory_snapshot()
    assert pre_row_status(pre) == RunStatus.RUNNING.value
    bump_applied = asyncio.Event()
    committed = asyncio.Event()

    async def transition() -> None:
        async with registry.acquire_run_state():
            registry._advance_inventory_revision_locked()
            bump_applied.set()
            # The held region leaves the state change deliberately pending:
            # a snapshot that slipped between bump and write would be torn.
            await committed.wait()
            run.status = RunStatus.COMPLETED

    writer = asyncio.create_task(transition())
    await asyncio.wait_for(bump_applied.wait(), 5)

    snapshots = [asyncio.create_task(registry.inventory_snapshot()) for _ in range(5)]
    for _ in range(25):
        await asyncio.sleep(0)
    assert all(not task.done() for task in snapshots), (
        "the joint snapshot must not complete inside a held run-state region"
    )

    committed.set()
    for task in snapshots:
        landed = await asyncio.wait_for(task, 5)
        # Post-transition: the revision advanced exactly once and describes
        # the completed run — the torn (old revision, new state) pair never
        # appears.
        assert landed.revision == pre.revision + 1
        assert pre_row_status(landed) == RunStatus.COMPLETED.value
    await asyncio.wait_for(writer, 5)

    # After the writer released, a fresh read still pairs the bumped revision
    # with the post-transition state, and a hypothetical pre-transition
    # re-read of the OLD revision row would have shown RUNNING — never the
    # pre-transition revision paired with the post-transition state.
    settled = await registry.inventory_snapshot()
    assert settled.revision == pre.revision + 1
    assert pre_row_status(settled) == RunStatus.COMPLETED.value


def pre_row_status(snapshot: InventorySnapshot) -> Any:
    return _row(snapshot, "agent-1").current_run_status


# ---------------------------------------------------------------------------
# The census: the revision advances on every lifecycle class.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_revision_advances_on_launch_terminal_and_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """spawn/admit/terminal/collection: every stage bumps the revision at a
    stable gate, and an already-collected fetch does NOT bump (no spurious
    advancement).
    """
    backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=_ServingParentBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        pre_launch = _revision(registry)
        launch = await _launch_task(registry, parent, "launch")
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(backend.started.wait(), 10)
        # spawn + admission RUNNING + acceptance writes all bumped.
        after_admit = _revision(registry)
        assert after_admit > pre_launch
        running = await registry.inventory_snapshot()
        assert _row(running, launch.agent_id).availability == "running"

        backend.release.set()
        record = registry._agent_records[launch.agent_id]
        await _until(
            lambda: (
                (launch.agent_id, launch.run_id) in registry._result_store
                and record.state is _AgentState.IDLE
            )
        )
        # The terminal resolution, the FINALIZING/IDLE resident transitions,
        # and the store publication each bumped along the monitor's flight
        # (the flight completes before the stable end state is observable).
        after_terminal = _revision(registry)
        assert after_terminal > after_admit

        # Collection: the uncollected list change bumps exactly once; the
        # repeated fetch changes nothing visible, so nothing bumps.
        fetched = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert fetched is not None
        assert fetched.completed
        after_fetch = _revision(registry)
        assert after_fetch > after_terminal
        again = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert again is not None
        assert _revision(registry) == after_fetch
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_revision_advances_on_cancel_of_a_running_agent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cancel: the authoritative stop bumps at the arbitration write, and
    the cancellation resolution + terminal transition bump again.
    """
    backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=_ServingParentBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        launch = await _launch_task(registry, parent, "launch")
        assert launch.agent_id is not None
        await asyncio.wait_for(backend.started.wait(), 10)

        stopped = await registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert stopped.outcome.value == "stop_requested"
        # The stopping latch advanced the revision before any terminal state.
        after_cancel = _revision(registry)
        assert after_cancel > 0

        backend.release.set()
        outcome = await registry.wait_for_agent(launch.agent_id, launch.run_id)
        assert outcome.status is WaitStatus.COMPLETED
        assert outcome.result is not None
        assert not outcome.result.completed
        # The cancellation resolution and terminal transition bumped beyond
        # the latch write.
        assert _revision(registry) > after_cancel
        await backend.stopped.wait()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(90)
async def test_revision_advances_on_suspension_reconcile_and_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """suspension/reconcile/resume: a truncated run resolves FAILED and
    SUSPENDS (bumps), the barrier blocks a resume without changing the
    revision, the explicit reconcile clears it (bump), and the admitted
    resume runs a successor (bumps at admission and successor settlement).
    """
    backend = _TruncatingChildBackend()
    registry, parent = _recovery_registry(monkeypatch, backend)
    try:
        before = _revision(registry)
        launch = await _launch_task(registry, parent, "truncate")
        assert launch.agent_id is not None and launch.run_id is not None
        # The waiter is pinned behind the run's completion task, so the whole
        # flight (FAILED terminal resolution, store publication, warm
        # suspension) has bumped by the time the failed result returns.
        failed = await _wait_result(registry, launch.agent_id, launch.run_id)
        assert not failed.completed
        after_failure = _revision(registry)
        assert after_failure > before

        record = registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        after_suspend = _revision(registry)
        assert after_suspend >= after_failure

        failed_run = next(
            item
            for item in [record.current_run, *record.run_history]
            if item is not None and item.run_id == launch.run_id
        )
        envelope = failed_run.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED

        # The blocked explicit admission mutates nothing: no bump.
        blocked = await registry.resume_agent(
            launch.agent_id,
            launch.run_id,
            "explicit-before-reconcile",
            requester_session_id=parent.session_id,
        )
        assert blocked.outcome is ResumeOutcome.RECONCILE_REQUIRED
        assert _revision(registry) == after_suspend

        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        reconciled = await registry.reconcile_result(
            launch.agent_id,
            launch.run_id,
            boundary,
            [],
            requester_session_id=parent.session_id,
        )
        assert reconciled.outcome is ReconcileOutcome.CLEARED
        after_reconcile = _revision(registry)
        assert after_reconcile > after_suspend

        admitted = await registry.resume_agent(
            launch.agent_id,
            launch.run_id,
            "explicit-after-reconcile",
            requester_session_id=parent.session_id,
        )
        assert admitted.outcome is ResumeOutcome.SCHEDULED
        assert admitted.successor_run_id is not None
        assert admitted.successor_run_id != launch.run_id
        # The committed admission (SUSPENDED -> RESTORING) bumped in its
        # own held region.
        after_resume = _revision(registry)
        assert after_resume > after_reconcile

        done = await _wait_result(registry, launch.agent_id, admitted.successor_run_id)
        assert done.completed
        assert done.response == "recovered"
        # The successor's admission and terminal resolution bumped again.
        assert _revision(registry) > after_resume
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_revision_advances_on_eviction_release_result_fetch_and_expiry() -> None:
    """evict/release/fetch: each collaborator mutation path bumps the
    revision in its own held region (retention, release, results,
    suspension expiry).
    """
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock)
    evicted = _record("agent-evict", clock=clock)
    registry._agent_records[evicted.agent_id] = evicted
    registry._children[evicted.session_id] = evicted.runtime

    before_evict = _revision(registry)
    assert await asyncio.wait_for(registry._evict_agent(evicted.agent_id, "ttl"), 5)
    assert _revision(registry) > before_evict

    released = _record("agent-release", clock=clock)
    registry._agent_records[released.agent_id] = released
    before_release = _revision(registry)
    outcome = await asyncio.wait_for(registry.release_agent(released.agent_id), 5)
    assert outcome.value == "released"
    assert _revision(registry) > before_release

    collected = _record("agent-collect", clock=clock)
    registry._agent_records[collected.agent_id] = collected
    stored = _stored_result(collected.agent_id, "run-collect")
    registry._result_store[(stored.agent_id, stored.run_id)] = stored
    before_fetch = _revision(registry)
    fetched = await registry.get_agent_result(collected.agent_id, stored.run_id)
    assert fetched is not None
    assert _revision(registry) > before_fetch

    expired = _record("agent-expired", clock=clock, state=_AgentState.SUSPENDED)
    registry._agent_records[expired.agent_id] = expired
    registry._children[expired.session_id] = expired.runtime
    before_expiry = _revision(registry)
    registry._suspension_expire_record_locked(expired)
    assert expired.agent_id not in registry._agent_records
    assert _revision(registry) > before_expiry


@pytest.mark.asyncio
async def test_revision_advances_on_drain_and_orphan_reap() -> None:
    """drain/reap: the wholesale inventory clears and the orphan sweep each
    advance the revision in their synchronous mutation windows.
    """
    clock = _ManualClock()
    registry, _root = _reaper_registry(clock)
    drained = _record("agent-drain", clock=clock)
    registry._agent_records[drained.agent_id] = drained
    registry._children[drained.session_id] = drained.runtime

    before_drain = _revision(registry)
    await asyncio.wait_for(registry.drain_children(), 5)
    after_drain = _revision(registry)
    assert after_drain > before_drain
    empty = await registry.inventory_snapshot()
    assert empty.agents == ()
    assert empty.revision == _revision(registry)

    orphan_runtime = MagicMock()
    orphan_runtime.close = AsyncMock()
    orphan_runtime.agent_loop.parent_session_id = "gone-session"
    orphan_runtime.agent_loop.session_logger.session_dir = None
    orphan_record = _record("agent-orphan", clock=clock)
    orphan_record.runtime = orphan_runtime
    registry._agent_records[orphan_record.agent_id] = orphan_record
    registry._children[orphan_record.session_id] = orphan_runtime

    before_reap = _revision(registry)
    report = await asyncio.wait_for(registry.reap_orphaned_children(), 5)
    assert report.reaped == 1
    assert _revision(registry) > before_reap
    assert orphan_record.agent_id not in registry._agent_records


@pytest.mark.asyncio
async def test_revision_advances_on_compaction_flag_capture() -> None:
    """compaction: the compaction flag writes bump inside the capture's
    locked region, and a notification that changes nothing does NOT bump.
    """
    clock = _ManualClock()
    registry, _root = _root_registry(clock)
    record = _record("agent-compact", clock=clock)
    record.state = _AgentState.RUNNING
    run = _run(record, "run-compact", started_at=clock(), status=RunStatus.RUNNING)
    record.current_run = run
    record.runtime._closed = False
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    entry = PublicCheckpointEntry(
        id="cp-1",
        session_id=record.session_id,
        kind="compaction",
        created_at=0,
        updated_at=0,
        generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
    )
    params = HistoryEntryAddedParams(
        event_id=0, session_id=record.session_id, emitted_at=0, entry=entry
    )
    before = _revision(registry)
    await registry._capture_child_notification(record.runtime, record, run, params)
    assert record.compacting is True
    after = _revision(registry)
    assert after > before

    # A non-compaction (or unchanged) capture bumps nothing.
    unrelated = HistoryEntryAddedParams(
        event_id=1,
        session_id=record.session_id,
        emitted_at=0,
        entry=PublicCheckpointEntry(
            id="cp-2",
            session_id=record.session_id,
            kind="checkpoint",
            created_at=1,
            updated_at=1,
            generation_status=PublicEntryGenerationStatus.IN_PROGRESS,
        ),
    )
    await registry._capture_child_notification(record.runtime, record, run, unrelated)
    assert _revision(registry) == after


# ---------------------------------------------------------------------------
# Generation change resets the revision-floor pairing (per incarnation).
# ---------------------------------------------------------------------------


class _RevisionFloor:
    """The per-incarnation acceptance floor (design §3.3): identical
    (session_id, generation) compares revisions numerically; a new
    incarnation establishes a fresh floor and is never compared numerically
    against the old one.
    """

    def __init__(self) -> None:
        self.incarnation: tuple[str, int] | None = None
        self.revision: int | None = None

    def accept(self, snapshot: InventorySnapshot) -> bool:
        if self.incarnation is None or self.incarnation != snapshot.incarnation:
            self.incarnation = snapshot.incarnation
            self.revision = snapshot.revision
            return True
        if snapshot.revision >= (self.revision or 0):
            self.revision = max(self.revision or 0, snapshot.revision)
            return True
        return False


@pytest.mark.asyncio
async def test_generation_change_resets_the_revision_floor_pairing() -> None:
    """rotation: the snapshot carries the NEW incarnation with a strictly
    higher revision; the floor resets with the identity, so an old
    incarnation's floor never rejects a new incarnation's snapshot — and
    within one incarnation a stale revision is still rejected.
    """
    clock = _ManualClock()
    registry, root = _reaper_registry(clock)
    record = _record("agent-rotate", clock=clock)
    registry._agent_records[record.agent_id] = record

    first = await registry.inventory_snapshot()
    assert first.incarnation == ("root", 0)
    assert _row(first, record.agent_id).availability == "idle"

    root.agent_loop._session_generation = 1
    registry.begin_root_generation()

    second = await registry.inventory_snapshot()
    # The counter is monotone across the rotation, and the retired
    # generation's records left the projection.
    assert second.incarnation == ("root", 1)
    assert second.revision > first.revision
    assert second.agents == ()

    floor = _RevisionFloor()
    assert floor.accept(first)
    # Within the first incarnation, a stale revision is rejected.
    stale = InventorySnapshot(
        session_id="root", generation=0, revision=first.revision - 1, agents=()
    )
    assert not floor.accept(stale)
    # The rotation's snapshot carries a NEW incarnation: the floor resets
    # with the identity — accepted regardless of the numeric revision.
    assert floor.accept(second)
    # Even a numerically lower revision from a fresh incarnation is accepted:
    # revisions are never compared across incarnations.
    synthetic_fresh = InventorySnapshot(
        session_id="other-session", generation=0, revision=1, agents=()
    )
    assert floor.accept(synthetic_fresh)
    # The new incarnation's own floor then governs it numerically.
    stale_new = InventorySnapshot(
        session_id="other-session", generation=0, revision=0, agents=()
    )
    assert not floor.accept(stale_new)
