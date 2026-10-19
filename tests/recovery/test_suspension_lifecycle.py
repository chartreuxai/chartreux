"""WP3 — bounded warm suspension lifecycle (v1 is WARM-ONLY, resolved Q6).

Covers the WP3 acceptance: the suspension TTL/count/byte caps are SEPARATE
from the idle-retention and tombstone caps, with deterministic cap selection
and expiry; suspension TTL expiry runs on the shared reaper's timer WITHOUT
any parent model turn and publishes the reduced inventory through
``agents/update`` only; expiry removes recovery eligibility WITHOUT deleting
the transcript, artifacts, or stored results (it converts the record to a
``closed`` terminal tombstone through the normal ownership paths); the
provider connection is released at suspension while the conversation stays
warm; cancellation, release, generation change, and shutdown invalidate
pending recovery; normal success still closes ephemeral agents; a
parent-process restart forfeits suspended agents (non-resumable, no
dangling scheduled recovery); the storage seam is warm-only (no checkpoint
writer, durable scheduler, restoration lease, or cold path); and the
warm-conversation byte accounting is deterministic and bounded. WP1 E4 #2 is
pinned here too: a closed agent's tombstone summary carries the latest
failed run's public failure envelope.

Harness: registry-level (no app server) — a real parent loop drives
``registry.run`` launches, child backends come from the patched
``chartreux.core.agent_loop._loop.create_backend`` facade, and the shared
reaper runs on a manual clock behind a gated wakeup so TTL expiry is fully
deterministic. The connection-release test reuses the per-loop scripted
backend factory from ``test_explicit_resume.py``. The recoverable fault is
the WP1 typed EMPTY_RESPONSE transient with no tool calls (RECONCILE_REQUIRED
— it suspends without arming an availability cooldown).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
import pytest

from chartreux.app_server._sessions import (
    AgentRecord,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server._suspension_lifecycle import (
    SuspensionStorage,
    WarmOnlySuspensionStorage,
    _warm_conversation_bytes,
)
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.config import ChartreuxConfigSchema, SessionLoggingConfig
from chartreux.core.config.models import RecoveryConfig, RecoverySectionConfig
from chartreux.core.llm_models import FunctionCall, LLMMessage, Role, ToolCall
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.subagents import (
    AgentAvailability,
    CancelOutcome,
    ReconcileOutcome,
    ReleaseAgentOutcome,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    ImmediateFailureBackend,
    _background_result,
    _config,
    _GatedWakeup,
    _ManualClock,
    _reaper_registry,
    _record,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_explicit_resume import (
    _ChildBackendFactory,
    _empty_fault,
    _final_outcome,
    _ScriptedBackend,
)
from tests.stubs.fake_backend import FakeBackend


async def _until(predicate: Any, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def _suspension_config(
    recovery: RecoveryConfig, session_logging: SessionLoggingConfig | None = None
) -> ChartreuxConfigSchema:
    """Test config: the given recovery policy, zeroed idle-retention caps.

    Zeroing the idle TTL/cap keeps the shared reaper's ONLY deadline the
    suspension TTL, so the gated wakeup observes exactly the suspension arm
    (and, after expiry, exactly the tombstone arm). WP4: the child policy
    lives in the ``[recovery.subagents]`` subsection of the ``[recovery]``
    container.
    """
    config = _config(session_logging=session_logging)
    return config.model_copy(
        update={
            "recovery": RecoverySectionConfig(subagents=recovery),
            "subagents": config.subagents.model_copy(
                update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
            ),
        }
    )


async def _suspension_registry(
    monkeypatch: pytest.MonkeyPatch,
    recovery: RecoveryConfig,
    create_backend: Callable[..., Any],
    *,
    session_logging: SessionLoggingConfig | None = None,
) -> tuple[SessionRuntimeRegistry, AgentLoop, _ManualClock, _GatedWakeup, list[str]]:
    """Registry-level harness: real parent loop, gated reaper, manual clock.

    ``create_backend`` is the patched ``create_backend`` facade (a plain
    backend-returning lambda or a per-loop scripted factory). The returned
    ``notifications`` list records every parent ``turns.start`` text, so a
    test can pin that no parent model turn was started.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    parent = build_test_agent_loop(
        config=_suspension_config(recovery, session_logging), backend=FakeBackend()
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        notify_agents=AsyncMock(),
        clock=clock,
        wakeup=wakeup,
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    notifications: list[str] = []
    start = root.turns.start

    def record_notification(params: TurnStartParams):
        notifications.extend(
            block.text
            for block in params.message
            if isinstance(block, TextContentBlock)
        )
        return start(params)

    monkeypatch.setattr(root.turns, "start", record_notification)
    registry.bind_root(root)
    return registry, parent, clock, wakeup, notifications


async def _launch(
    registry: SessionRuntimeRegistry,
    parent: AgentLoop,
    task: str,
    *,
    retain: bool | None = None,
    tool_call_id: str = "launch",
) -> TaskResult:
    arguments: dict[str, Any] = {
        "task": task,
        "agent_type": "worker",
        "background": True,
    }
    if retain is not None:
        arguments["retain"] = retain
    return await _background_result(
        registry,
        TaskArgs(**arguments),
        InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id),
    )


async def _suspend_child(
    registry: SessionRuntimeRegistry,
    parent: AgentLoop,
    *,
    task: str = "fail",
    tool_call_id: str = "launch",
) -> tuple[str, str, TaskResult, AgentRecord]:
    """Launch a recoverably failing retained child and await its suspension."""
    launch = await _launch(
        registry, parent, task, retain=True, tool_call_id=tool_call_id
    )
    assert launch.agent_id is not None and launch.run_id is not None
    failed = await _wait_result(registry, launch.agent_id, launch.run_id)
    assert not failed.completed
    record = registry._agent_records[launch.agent_id]
    await _until(lambda: record.state is _AgentState.SUSPENDED)
    return launch.agent_id, launch.run_id, failed, record


class _RecordingStore:
    """A ``SuspensionStorage`` seam double recording every seam call."""

    def __init__(self) -> None:
        self.captured: list[str] = []
        self.restored: list[str] = []
        self.discarded: list[str] = []

    def capture(self, record: AgentRecord) -> None:
        self.captured.append(record.agent_id)

    def restore(self, agent_id: str) -> AgentRecord | None:
        self.restored.append(agent_id)
        return None

    def discard(self, agent_id: str) -> None:
        self.discarded.append(agent_id)


def _user_message(text: str) -> LLMMessage:
    return LLMMessage(role=Role.user, content=text)


def _suspended_record(
    agent_id: str, *, clock: _ManualClock, suspended_at: float, content: str
) -> AgentRecord:
    """A fabricated SUSPENDED record holding one sized user message."""
    record = _record(agent_id, clock=clock, state=_AgentState.SUSPENDED)
    record.suspended_at = suspended_at
    record.runtime.agent_loop.messages.append(_user_message(content))
    record.suspended_warm_bytes = _warm_conversation_bytes(
        record.runtime.agent_loop.messages
    )
    return record


def _incoming_record(
    agent_id: str, *, clock: _ManualClock, content: str
) -> AgentRecord:
    """A fabricated resident record about to suspend (still FINALIZING)."""
    record = _record(agent_id, clock=clock, state=_AgentState.FINALIZING)
    record.runtime.agent_loop.messages.append(_user_message(content))
    return record


def _empty_fault_backend() -> ImmediateFailureBackend:
    return ImmediateFailureBackend(EmptyLLMResponseError("test-provider", "test-model"))


# ---------------------------------------------------------------------------
# Declared caps and the byte accounting (config + accounting pins)
# ---------------------------------------------------------------------------


def test_suspension_byte_cap_defaults_and_validation() -> None:
    """The suspension byte cap is a separate declared limit (WP1b/WP3).

    Pins the declared default alongside the existing suspension-limit pins
    (``test_recovery_config.py`` stays untouched — outside this package's
    manifest): a strict nonnegative int, default 64 MiB, zero accepted (zero
    disables byte accounting).
    """
    assert RecoveryConfig().max_suspended_bytes == 67108864
    with pytest.raises(ValidationError, match="max_suspended_bytes"):
        RecoveryConfig(max_suspended_bytes=-1)
    with pytest.raises(ValidationError, match="max_suspended_bytes"):
        RecoveryConfig.model_validate({"max_suspended_bytes": 1.5})
    assert RecoveryConfig(max_suspended_bytes=0).max_suspended_bytes == 0


def test_warm_conversation_bytes_is_deterministic_and_itemized() -> None:
    """The byte accounting is a defined, deterministic estimate (WP3).

    Per message: a fixed 64-byte overhead plus the role length plus the
    itemized payload fields (content, reasoning, tool-call ids/names/
    arguments, tool result ids, names). Same conversation -> same number.
    """
    empty: list[LLMMessage] = []
    assert _warm_conversation_bytes(empty) == 0

    user = _user_message("hello")
    # 64 overhead + len("user") + len("hello")
    assert _warm_conversation_bytes([user]) == 64 + 4 + 5
    # Deterministic: the same conversation accounts identically.
    assert _warm_conversation_bytes([user]) == _warm_conversation_bytes([user])

    assistant = LLMMessage(
        role=Role.assistant,
        content="",
        tool_calls=[
            ToolCall(
                id="call-1",
                function=FunctionCall(name="todo", arguments='{"action":"read"}'),
            )
        ],
    )
    # 64 + len("assistant") + id + name + arguments
    assert _warm_conversation_bytes([assistant]) == (
        64 + 9 + len("call-1") + len("todo") + len('{"action":"read"}')
    )

    tool = LLMMessage(role=Role.tool, content="ok", tool_call_id="call-1", name="todo")
    # 64 + len("tool") + content + tool_call_id + name
    assert _warm_conversation_bytes([tool]) == 64 + 4 + 2 + 6 + 4

    reasoning = LLMMessage(role=Role.assistant, content="x", reasoning_content="think")
    # 64 + len("assistant") + content + reasoning
    assert _warm_conversation_bytes([reasoning]) == 64 + 9 + 1 + 5

    # Monotonic growth with content size; a bare message never raises.
    small = _warm_conversation_bytes([_user_message("a")])
    large = _warm_conversation_bytes([_user_message("a" * 100)])
    assert large > small
    assert _warm_conversation_bytes([LLMMessage(role=Role.system)]) == 64 + 6


def test_warm_only_storage_seam_persists_nothing() -> None:
    """The v1 storage seam is warm-only: no checkpoint, no cold restore.

    ``WarmOnlySuspensionStorage`` persists nothing — ``restore`` always
    reports absent, which is what forfeits a suspended agent across a
    parent-process restart (Q6). There is no checkpoint writer, durable
    scheduler, or restoration lease behind the seam.
    """
    store: SuspensionStorage = WarmOnlySuspensionStorage()
    record = _record("agent-1", clock=_ManualClock())
    store.capture(record)
    assert store.restore("agent-1") is None
    store.discard("agent-1")


# ---------------------------------------------------------------------------
# TTL expiry end-to-end (expiry without a parent turn; eligibility removed
# without deleting transcript/artifacts/results)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspension_ttl_expiry_closes_without_a_parent_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warm window expires on the reaper's timer, with no parent turn.

    The shared idle/metadata reaper arms for the earliest suspension
    deadline; the elapsed record converts to a ``closed`` terminal tombstone
    through the normal ownership paths, and the reduced inventory is
    published through ``agents/update`` only. The failed run's stored
    result and the on-disk transcript survive; recovery eligibility is gone
    (resume NOT_RESUMABLE).
    """
    registry, parent, clock, wakeup, notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(suspension_ttl_seconds=10.0),
        lambda **_: _empty_fault_backend(),
        session_logging=SessionLoggingConfig(
            enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
        ),
    )
    store = _RecordingStore()
    registry._suspension._store = store
    try:
        agent_id, run_id, failed, record = await _suspend_child(registry, parent)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        # The suspension captured the deterministic warm-byte accounting
        # and handed the record to the storage seam.
        assert record.suspended_at == clock()
        assert record.suspended_warm_bytes is not None
        assert record.suspended_warm_bytes > 0
        assert store.captured == [agent_id]
        transcript = record.runtime.agent_loop.session_logger.messages_filepath
        await _until(lambda: transcript.is_file())
        # The reaper armed for the suspension deadline — the ONLY deadline
        # (idle retention is zeroed, no tombstones yet).
        await _until(lambda: bool(wakeup.calls) and wakeup.calls[-1] == 10)
        # Snapshot the parent-turn surface: the settlement's failure
        # notification is the only parent activity, and it lands before the
        # suspension settles.
        await _until(
            lambda: bool(notifications) or bool(parent._pending_injected_messages)
        )
        turns_before = list(notifications)
        injected_before = len(parent._pending_injected_messages)
        notify = registry._notify_agents
        assert isinstance(notify, AsyncMock)
        notify_before = notify.await_count

        clock.advance(10)
        wakeup.gates[-1].set()
        reaper = registry._reaper_task
        assert reaper is not None
        await asyncio.wait_for(reaper, timeout=5)

        # Expiry removed the record and stored a CLOSED tombstone — the
        # failed run's public envelope is carried into the terminal
        # metadata (WP1 E4 #2).
        assert agent_id not in registry._agent_records
        tombstone = registry._evicted_agents[agent_id]
        assert tombstone.reason == "closed"
        assert tombstone.summary.availability is AgentAvailability.CLOSED
        assert tombstone.summary.last_run_status is RunStatus.FAILED
        # Expiry invents no stop reason: the failure's own reason (ERROR)
        # is preserved — unlike a cancellation, which records its own.
        assert tombstone.summary.stop_reason is RunStopReason.ERROR
        assert tombstone.summary.last_failure is not None
        assert tombstone.summary.last_failure.event_id == envelope.event_id
        # The latest-run index survives, so the result resolves by handle.
        assert registry._latest_run_ids[agent_id] == run_id
        # The storage-seam state is discarded at expiry.
        assert store.discarded == [agent_id]

        # Recovery eligibility is gone: resume reports the terminal
        # metadata.
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE

        # The stored result AND the on-disk transcript survive expiry.
        result = await registry.get_agent_result(agent_id, run_id)
        assert result is not None
        assert not result.completed
        assert result.failure is not None
        assert result.failure.event_id == envelope.event_id
        assert (await registry.get_agent_result(agent_id)) is not None
        assert transcript.is_file()

        # The expiry notification is the agents/update publication — the
        # last notification shows the agent closed, carrying the envelope.
        summary = next(
            item for item in await registry.check_agents() if item.agent_id == agent_id
        )
        assert summary.availability is AgentAvailability.CLOSED
        assert summary.last_failure is not None
        assert summary.last_failure.event_id == envelope.event_id
        assert notify.await_count > notify_before
        await_args = notify.await_args
        assert await_args is not None
        wire_summaries, _evictions = await_args.args
        wire_summary = next(
            item for item in wire_summaries if item.agent_id == agent_id
        )
        assert wire_summary.availability == "closed"
        assert wire_summary.last_failure is not None
        assert wire_summary.last_failure.event_id == envelope.event_id

        # NO parent model turn was started by the expiry: the turn surface
        # is unchanged since the settlement snapshot.
        assert notifications == turns_before
        assert len(parent._pending_injected_messages) == injected_before
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_zero_suspension_ttl_disables_warm_suspension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero suspension TTL disables warm suspension entirely (WP3).

    A recoverable failure then follows the normal path: an unretained agent
    closes (``closed`` tombstone), a retained agent returns to idle — no
    SUSPENDED record, no suspension deadline, no captured warm bytes.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(suspension_ttl_seconds=0.0),
        lambda **_: _empty_fault_backend(),
    )
    try:
        ephemeral = await _launch(
            registry, parent, "ephemeral fail", tool_call_id="ephemeral"
        )
        retained = await _launch(
            registry, parent, "retained fail", retain=True, tool_call_id="retained"
        )
        assert ephemeral.agent_id is not None and retained.agent_id is not None
        # The unretained failure closes; the retained failure idles.
        await _until(lambda: ephemeral.agent_id not in registry._agent_records)
        await _until(lambda: ephemeral.agent_id in registry._evicted_agents)
        closed = registry._evicted_agents[ephemeral.agent_id]
        assert closed.summary.availability is AgentAvailability.CLOSED
        record = registry._agent_records[retained.agent_id]
        await _until(lambda: record.state is _AgentState.IDLE)
        assert record.suspended_at is None
        assert record.suspended_warm_bytes is None
        assert registry._suspension_next_expiry_locked() is None
        # The failed results are still published.
        assert (await registry.get_agent_result(ephemeral.agent_id)) is not None
        assert (await registry.get_agent_result(retained.agent_id)) is not None
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# Count and byte caps (separate from retention; deterministic selection)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspended_count_cap_admits_one_warm_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The suspended-count cap is separate from retention (WP3).

    With ``max_suspended_agents=1`` exactly one of two recoverably failed
    retained children suspends; the other sees the cap and returns to idle.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(max_suspended_agents=1),
        lambda **_: _empty_fault_backend(),
    )
    try:
        first = await _launch(
            registry, parent, "first fail", retain=True, tool_call_id="first"
        )
        second = await _launch(
            registry, parent, "second fail", retain=True, tool_call_id="second"
        )
        assert first.agent_id is not None and second.agent_id is not None
        first_record = registry._agent_records[first.agent_id]
        second_record = registry._agent_records[second.agent_id]
        await _until(
            lambda: (
                first_record.state is _AgentState.SUSPENDED
                or first_record.state is _AgentState.IDLE
            )
        )
        await _until(
            lambda: (
                second_record.state is _AgentState.SUSPENDED
                or second_record.state is _AgentState.IDLE
            )
        )
        records = (first_record, second_record)
        assert {record.state for record in records} == {
            _AgentState.SUSPENDED,
            _AgentState.IDLE,
        }
        suspended = [
            record for record in records if record.state is _AgentState.SUSPENDED
        ]
        idle = [record for record in records if record.state is _AgentState.IDLE]
        assert len(suspended) == 1 and len(idle) == 1
        assert suspended[0].suspended_warm_bytes is not None
        assert idle[0].suspended_warm_bytes is None
        assert idle[0].suspended_at is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_oversized_conversation_never_suspends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A record whose conversation alone exceeds the byte cap is refused.

    The deterministic accounting is computed before the transition: with a
    1-byte cap any real conversation is oversized, so the recoverable
    failure idles (retained) instead of suspending.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(max_suspended_bytes=1),
        lambda **_: _empty_fault_backend(),
    )
    try:
        launch = await _launch(registry, parent, "oversized fail", retain=True)
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.IDLE)
        assert record.suspended_at is None
        assert record.suspended_warm_bytes is None
        failed = await registry.get_agent_result(launch.agent_id, launch.run_id)
        assert failed is not None and not failed.completed
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_byte_cap_expires_the_oldest_suspended_record() -> None:
    """Deterministic byte-cap selection: the OLDEST suspended record expires.

    ``(suspended_at, numeric agent-id)`` ordering — the same convention as
    the retention caps — makes room for the new suspension; the total
    accounting stays within the cap.
    """
    clock = _ManualClock(100)
    registry, _root = _reaper_registry(clock)
    # One user message of 100 chars: 64 + len("user") + 100 = 168 bytes.
    oldest = _suspended_record(
        "agent-1", clock=clock, suspended_at=10, content="a" * 100
    )
    middle = _suspended_record(
        "agent-2", clock=clock, suspended_at=20, content="b" * 100
    )
    newest = _suspended_record(
        "agent-3", clock=clock, suspended_at=30, content="c" * 100
    )
    registry._agent_records.update({
        oldest.agent_id: oldest,
        middle.agent_id: middle,
        newest.agent_id: newest,
    })
    incoming = _incoming_record("agent-4", clock=clock, content="d" * 100)
    registry._agent_records[incoming.agent_id] = incoming
    # The cap fits exactly three of the four records.
    config = RecoveryConfig(max_suspended_bytes=504)
    async with registry._registry_lock:
        admitted = registry._suspension.admit_locked(incoming, config)
        assert admitted is True
        # The hook's transition (mirrored): the record suspends with its
        # captured bytes.
        incoming.state = _AgentState.SUSPENDED
        incoming.suspended_at = clock()
        assert incoming.suspended_warm_bytes == 168
        assert set(registry._agent_records) == {"agent-2", "agent-3", "agent-4"}
        assert registry._suspension.suspended_total_bytes_locked() == 504
    # The oldest record expired into a CLOSED tombstone; the rest stay warm.
    assert "agent-1" not in registry._agent_records
    tombstone = registry._evicted_agents["agent-1"]
    assert tombstone.reason == "closed"
    assert tombstone.summary.availability is AgentAvailability.CLOSED
    await registry.drain_children()


@pytest.mark.asyncio
async def test_byte_cap_tie_break_uses_the_numeric_agent_id() -> None:
    """Equal ``suspended_at`` ties break on the numeric agent id (WP3)."""
    clock = _ManualClock(100)
    registry, _root = _reaper_registry(clock)
    # Insertion order is deliberately NOT the tie-break order.
    later = _suspended_record(
        "agent-2", clock=clock, suspended_at=10, content="a" * 100
    )
    earlier = _suspended_record(
        "agent-1", clock=clock, suspended_at=10, content="b" * 100
    )
    registry._agent_records.update({later.agent_id: later, earlier.agent_id: earlier})
    incoming = _incoming_record("agent-3", clock=clock, content="c" * 100)
    registry._agent_records[incoming.agent_id] = incoming
    # The cap fits exactly one incumbent plus the incoming record.
    config = RecoveryConfig(max_suspended_bytes=336)
    async with registry._registry_lock:
        admitted = registry._suspension.admit_locked(incoming, config)
        assert admitted is True
        assert set(registry._agent_records) == {"agent-2", "agent-3"}
    assert "agent-1" not in registry._agent_records
    assert "agent-1" in registry._evicted_agents
    await registry.drain_children()


@pytest.mark.asyncio
async def test_zero_byte_cap_disables_byte_accounting() -> None:
    """A zero byte cap disables the accounting; the capture still happens."""
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _incoming_record("agent-1", clock=clock, content="x" * 1000)
    registry._agent_records[record.agent_id] = record
    config = RecoveryConfig(max_suspended_bytes=0)
    async with registry._registry_lock:
        admitted = registry._suspension.admit_locked(record, config)
        assert admitted is True
        assert record.suspended_warm_bytes == 64 + 4 + 1000
    await registry.drain_children()


@pytest.mark.asyncio
async def test_byte_cap_refuses_an_oversized_record() -> None:
    """A record alone over the cap is never suspended (WP3)."""
    clock = _ManualClock(0)
    registry, _root = _reaper_registry(clock)
    record = _incoming_record("agent-1", clock=clock, content="x" * 100)
    registry._agent_records[record.agent_id] = record
    config = RecoveryConfig(max_suspended_bytes=64)  # below one message
    async with registry._registry_lock:
        admitted = registry._suspension.admit_locked(record, config)
    assert admitted is False
    # Refused without side effects: no tombstone, no captured bytes.
    assert record.suspended_warm_bytes is None
    assert "agent-1" in registry._agent_records
    assert "agent-1" not in registry._evicted_agents
    await registry.drain_children()


@pytest.mark.asyncio
async def test_expiry_selects_only_elapsed_suspended_records() -> None:
    """Expiry never touches a RESTORING (mid-admission) or unelapsed record."""
    clock = _ManualClock(100)
    registry, _root = _reaper_registry(clock)
    elapsed = _suspended_record("agent-1", clock=clock, suspended_at=0, content="a")
    restoring = _suspended_record("agent-2", clock=clock, suspended_at=0, content="b")
    restoring.state = _AgentState.RESTORING
    future = _suspended_record("agent-3", clock=clock, suspended_at=90, content="c")
    registry._agent_records.update({
        elapsed.agent_id: elapsed,
        restoring.agent_id: restoring,
        future.agent_id: future,
    })
    config = RecoveryConfig(suspension_ttl_seconds=10.0)
    async with registry._registry_lock:
        expired = registry._suspension.expire_elapsed_locked(10.0, config)
        assert [record.agent_id for record in expired] == ["agent-1"]
        assert set(registry._agent_records) == {"agent-2", "agent-3"}
        # The earliest deadline arms for the unelapsed record only.
        assert registry._suspension.next_expiry_locked(config) == 100.0
    assert "agent-1" in registry._evicted_agents
    await registry.drain_children()


# ---------------------------------------------------------------------------
# Warm connection release at suspension (conversation stays warm)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suspension_releases_the_provider_connection_but_stays_warm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warm means: connection released, conversation resident (handoff §1.5).

    At suspension the failed backend is retired through its ownership-aware
    lifetime (closed exactly once) and a fresh backend is installed, while
    the runtime stays open with its conversation. The sibling owns a
    separate backend lifetime and is never touched. The warm conversation
    still resumes: after the reconcile barrier clears, the successor
    completes on the fresh connection.
    """
    factory = _ChildBackendFactory(
        monkeypatch,
        [
            [_empty_fault(), _final_outcome("resumed done")],
            [_final_outcome("sibling done")],
        ],
    )
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(suspension_ttl_seconds=3600.0),
        factory._create_backend,
    )
    try:
        agent_a, run_a, failed_a, record_a = await _suspend_child(
            registry, parent, task="fail warm"
        )
        envelope = failed_a.failure
        assert envelope is not None
        loop_a = record_a.runtime.agent_loop
        backends_a = factory.created[id(loop_a)]
        # Construction throwaway, the failed attempt backend, the
        # suspension-time refresh backend.
        assert len(backends_a) == 3
        failed_backend = backends_a[1]
        assert failed_backend.calls >= 1
        # WP3: the provider connection is released AT suspension — the
        # failed backend is retired through its ownership-aware lifetime.
        await _until(lambda: failed_backend.aexit_calls >= 1)
        assert failed_backend.aexit_calls == 1
        # The conversation stays warm: the runtime is open, the
        # conversation is resident, and a fresh (never-called) backend is
        # installed.
        fresh_backend = cast(_ScriptedBackend, loop_a._backend_lifetime.active)
        assert fresh_backend is backends_a[2]
        assert fresh_backend.calls == 0
        assert record_a.runtime._closed is not True
        assert len(loop_a.messages) >= 1
        assert record_a.initial_task_summary == "fail warm"
        assert record_a.suspended_warm_bytes is not None
        assert record_a.suspended_warm_bytes > 0

        # The sibling owns a separate backend lifetime and is untouched.
        launch_b = await _launch(
            registry, parent, "sibling work", retain=True, tool_call_id="launch-b"
        )
        assert launch_b.agent_id is not None and launch_b.run_id is not None
        sibling_result = await _wait_result(
            registry, launch_b.agent_id, launch_b.run_id
        )
        assert sibling_result.completed
        assert sibling_result.response == "sibling done"
        record_b = registry._agent_records[launch_b.agent_id]
        loop_b = record_b.runtime.agent_loop
        sibling_backend = cast(_ScriptedBackend, loop_b._backend_lifetime.active)
        assert sibling_backend.calls >= 1
        assert sibling_backend.aexit_calls == 0

        # The warm conversation still resumes: clear the reconcile barrier
        # and admit the successor on the fresh connection.
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        # The failed run left no tool call outstanding, so the complete
        # accounting is the empty sequence (review R5).
        cleared = await registry.reconcile_result(
            agent_a, run_a, boundary, [], requester_session_id=parent.session_id
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        resume = await registry.resume_agent(
            agent_a, run_a, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.SCHEDULED
        assert resume.successor_run_id is not None
        successor = await _wait_result(registry, agent_a, resume.successor_run_id)
        assert successor.completed
        assert successor.response == "resumed done"

        # The failed connection was closed exactly once (at suspension); the
        # resume retired the suspension-time fresh backend instead, and the
        # sibling's backend was never closed.
        assert failed_backend.aexit_calls == 1
        assert sibling_backend.aexit_calls == 0
        assert loop_b._backend_lifetime.active is sibling_backend
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# Invalidation paths: cancellation, release, generation change, shutdown
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_invalidates_a_suspended_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authoritative cancellation invalidates pending recovery (WP3).

    A foreign non-user cancel falls through to the run-results authority
    check (FORBIDDEN — the barrier stays suspended); a user cancel
    converts the barrier to a ``closed`` tombstone carrying the stop reason
    and the failure envelope, reports STOP_REQUESTED, and keeps the stored
    result.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch, RecoveryConfig(), lambda **_: _empty_fault_backend()
    )
    store = _RecordingStore()
    registry._suspension._store = store
    try:
        agent_id, run_id, failed, record = await _suspend_child(registry, parent)
        envelope = failed.failure
        assert envelope is not None
        assert store.captured == [agent_id]
        # A foreign requester with a non-user reason is refused by the
        # run-results authority check; the barrier stays suspended.
        foreign = await registry.cancel_run(
            agent_id,
            reason=RunStopReason.ORCHESTRATOR_CANCELLED,
            requester_session_id="foreign-session",
        )
        assert foreign.outcome is CancelOutcome.FORBIDDEN
        assert record.state is _AgentState.SUSPENDED
        # The authoritative user cancel invalidates the pending recovery.
        notify = registry._notify_agents
        assert isinstance(notify, AsyncMock)
        notify_before = notify.await_count
        cancelled = await registry.cancel_run(
            agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id="foreign-session",
        )
        assert cancelled.outcome is CancelOutcome.STOP_REQUESTED
        assert cancelled.run_id == run_id
        assert cancelled.stop_reason is RunStopReason.USER_CANCELLED
        assert agent_id not in registry._agent_records
        tombstone = registry._evicted_agents[agent_id]
        assert tombstone.reason == "closed"
        assert tombstone.summary.availability is AgentAvailability.CLOSED
        assert tombstone.summary.stop_reason is RunStopReason.USER_CANCELLED
        assert tombstone.summary.last_failure is not None
        assert tombstone.summary.last_failure.event_id == envelope.event_id
        assert store.discarded == [agent_id]
        # The failed run's stored result survives; the agent is not
        # resumable.
        result = await registry.get_agent_result(agent_id, run_id)
        assert result is not None and not result.completed
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
        # The cancellation published the reduced inventory.
        assert notify.await_count > notify_before
        await_args = notify.await_args
        assert await_args is not None
        wire_summaries, _evictions = await_args.args
        wire_summary = next(
            item for item in wire_summaries if item.agent_id == agent_id
        )
        assert wire_summary.availability == "closed"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_release_invalidates_a_suspended_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Release invalidates pending recovery and forfeits the seam state.

    The unguarded release pops the record, discards the suspension
    storage-seam state, purges the stored result, and closes the warm
    runtime; the released handle is not resumable.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch, RecoveryConfig(), lambda **_: _empty_fault_backend()
    )
    store = _RecordingStore()
    registry._suspension._store = store
    try:
        agent_id, run_id, _failed, record = await _suspend_child(registry, parent)
        assert store.captured == [agent_id]
        outcome = await registry.release_agent(agent_id)
        assert outcome is ReleaseAgentOutcome.RELEASED
        assert agent_id not in registry._agent_records
        assert agent_id not in registry._evicted_agents
        assert record.runtime._closed is True
        assert store.discarded == [agent_id]
        # Release purges the stored result (unlike expiry): the read is an
        # unknown-run error, not a missing result.
        with pytest.raises(UnknownAgentError):
            await registry.get_agent_result(agent_id, run_id)
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_release_during_in_flight_admission_rolls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A release during an in-flight resume admission rolls it back (WP3).

    The admission holds the record in RESTORING with a pending resume
    operation; releasing the agent invalidates the operation with the
    record — the admission's residency revalidation fails, no successor run
    is admitted, and the handle is not resumable afterwards.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch, RecoveryConfig(), lambda **_: _empty_fault_backend()
    )
    store = _RecordingStore()
    registry._suspension._store = store
    gate = asyncio.Event()
    try:
        agent_id, run_id, failed, record = await _suspend_child(registry, parent)
        envelope = failed.failure
        assert envelope is not None
        boundary = envelope.progress.transcript_boundary
        assert boundary is not None
        # The failed run left no tool call outstanding, so the complete
        # accounting is the empty sequence (review R5).
        cleared = await registry.reconcile_result(
            agent_id, run_id, boundary, [], requester_session_id=parent.session_id
        )
        assert cleared.outcome is ReconcileOutcome.CLEARED
        # Hold the admission's prepare phase so the release lands while the
        # resume operation is in flight.
        original_prepare = registry._execution.prepare_background

        async def gated_prepare(*args: Any, **kwargs: Any) -> Any:
            await gate.wait()
            return await original_prepare(*args, **kwargs)

        monkeypatch.setattr(registry._execution, "prepare_background", gated_prepare)
        task = asyncio.create_task(
            registry.resume_agent(
                agent_id, run_id, "k", requester_session_id=parent.session_id
            )
        )
        await _until(lambda: record.pending_resume is not None)
        assert record.state is _AgentState.RESTORING
        outcome = await registry.release_agent(agent_id)
        assert outcome is ReleaseAgentOutcome.RELEASED
        assert agent_id not in registry._agent_records
        gate.set()
        with pytest.raises(UnknownAgentError):
            await task
        # No successor run was admitted on the released agent.
        assert record.latest_run_id == run_id
        assert not any(
            item.resumed_from_run_id == run_id for item in record.run_history
        )
        assert store.discarded == [agent_id]
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
        with pytest.raises(UnknownAgentError):
            await registry.get_agent_result(agent_id, run_id)
    finally:
        gate.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_generation_change_invalidates_suspended_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A root-generation change retires suspended records (WP3).

    The retired generation forfeits its records (and their storage-seam
    state) through the normal ownership paths; the armed suspension
    deadline does not survive — no reaper is registered for the retired
    record — and the agent is not resumable under the new generation.
    """
    registry, parent, clock, wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(suspension_ttl_seconds=10.0),
        lambda **_: _empty_fault_backend(),
    )
    store = _RecordingStore()
    registry._suspension._store = store
    try:
        agent_id, run_id, _failed, record = await _suspend_child(registry, parent)
        assert store.captured == [agent_id]
        await _until(lambda: bool(wakeup.calls) and wakeup.calls[-1] == 10)
        notify = registry._notify_agents
        assert isinstance(notify, AsyncMock)
        parent._session_generation += 1
        registry.begin_root_generation()
        # Join the retired runtime's close and the retirement publication.
        await asyncio.gather(*tuple(registry._teardown_tasks))
        assert agent_id not in registry._agent_records
        assert store.discarded == [agent_id]
        assert record.runtime._closed is True
        # The armed suspension deadline did not survive: no reaper is
        # registered, and advancing the clock publishes nothing.
        assert registry._reaper_task is None
        notify.reset_mock()
        clock.advance(10_000)
        if wakeup.gates:
            wakeup.gates[-1].set()
        await asyncio.sleep(0.05)
        notify.assert_not_awaited()
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_shutdown_drain_cancels_pending_suspension_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shutdown (drain) invalidates pending recovery: no scheduled expiry.

    ``drain_children`` cancels the reaper — the armed suspension deadline
    never fires — closes the warm resident through the normal ownership
    paths, and clears the inventory; afterwards the clock can advance
    arbitrarily without any publication, and the agent is not resumable.
    """
    registry, parent, clock, wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(suspension_ttl_seconds=10.0),
        lambda **_: _empty_fault_backend(),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(registry, parent)
        await _until(lambda: bool(wakeup.calls) and wakeup.calls[-1] == 10)
        notify = registry._notify_agents
        assert isinstance(notify, AsyncMock)
        await registry.drain_children()
        # The reaper is gone; the warm runtime was closed by the drain.
        assert registry._reaper_task is None
        assert record.runtime._closed is True
        assert agent_id not in registry._agent_records
        # Nothing publishes after the drain, however far the clock advances.
        notify.reset_mock()
        clock.advance(10_000)
        if wakeup.gates:
            wakeup.gates[-1].set()
        await asyncio.sleep(0.05)
        notify.assert_not_awaited()
        resume = await registry.resume_agent(
            agent_id, run_id, "k", requester_session_id=parent.session_id
        )
        assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_parent_restart_forfeits_suspended_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A parent-process restart forfeits suspended agents (Q6, warm-only).

    The fresh registry (the restarted process) knows no record and arms NO
    scheduled recovery — nothing is restored from persisted state; the
    warm-only seam reports absent; resume reports NOT_RESUMABLE.
    """
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch,
        RecoveryConfig(suspension_ttl_seconds=10.0),
        lambda **_: _empty_fault_backend(),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(registry, parent)
        assert record.state is _AgentState.SUSPENDED
        # The "restarted process": a fresh registry, clock, wakeup, and root.
        restarted = SessionRuntimeRegistry(
            AsyncMock(),
            AsyncMock(),
            lambda _session_id: 0,
            notify_agents=AsyncMock(),
            clock=_ManualClock(),
            wakeup=_GatedWakeup(),
        )
        fresh_parent = build_test_agent_loop(
            config=_suspension_config(RecoveryConfig(suspension_ttl_seconds=10.0)),
            backend=FakeBackend(),
        )
        fresh_root = restarted._build_child_runtime(fresh_parent)
        fresh_root.turns._projector = MagicMock()
        fresh_root.turns.link_subagent = AsyncMock()
        restarted.bind_root(fresh_root)
        try:
            # No dangling scheduled recovery: the fresh registry arms no
            # reaper (nothing was restored).
            assert restarted._reaper_task is None
            # The warm-only seam reports nothing to restore.
            assert restarted._suspension._store.restore(agent_id) is None
            assert await restarted.check_agents() == []
            resume = await restarted.resume_agent(
                agent_id, run_id, "k", requester_session_id=fresh_parent.session_id
            )
            assert resume.outcome is ResumeOutcome.NOT_RESUMABLE
        finally:
            await restarted.drain_children()
            await fresh_parent.aclose()
    finally:
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# Normal success and the tombstone envelope (E4 #2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_normal_success_still_closes_ephemeral_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal success follows the original retention policy unchanged (WP3).

    An unretained successful child closes (``closed`` tombstone, no failure
    envelope, no suspension state); a retained one returns to idle. The
    suspension lifecycle never intercepts a successful completion.
    """
    backend = FakeBackend([
        [mock_llm_chunk(content="child done")],
        [mock_llm_chunk(content="child done")],
    ])
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch, RecoveryConfig(), lambda **_: backend
    )
    try:
        ephemeral = await _launch(
            registry, parent, "ephemeral work", tool_call_id="eph"
        )
        retained = await _launch(
            registry, parent, "retained work", retain=True, tool_call_id="ret"
        )
        assert ephemeral.agent_id is not None and retained.agent_id is not None
        # The unretained success closes with a CLOSED tombstone.
        await _until(lambda: ephemeral.agent_id not in registry._agent_records)
        await _until(lambda: ephemeral.agent_id in registry._evicted_agents)
        tombstone = registry._evicted_agents[ephemeral.agent_id]
        assert tombstone.reason == "closed"
        assert tombstone.summary.availability is AgentAvailability.CLOSED
        assert tombstone.summary.last_run_status is RunStatus.COMPLETED
        assert tombstone.summary.last_failure is None
        # The retained success returns to idle, resident and reusable.
        record = registry._agent_records[retained.agent_id]
        await _until(lambda: record.state is _AgentState.IDLE)
        assert record.availability is AgentAvailability.IDLE
        assert record.suspended_at is None
        # Both results are published.
        done = await registry.get_agent_result(ephemeral.agent_id)
        assert done is not None and done.completed
        assert done.response == "child done"
        idle_result = await registry.get_agent_result(retained.agent_id)
        assert idle_result is not None and idle_result.completed
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_closed_tombstone_carries_the_failure_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1 E4 #2: a closed (unretained) agent's tombstone summary carries
    the latest failed run's public typed failure envelope.

    A non-recoverable failure (PROHIBITED eligibility) never suspends; the
    close-on-completion tombstone still shows WHY the run failed — the same
    latest-first search the live summary uses.
    """
    backend = ImmediateFailureBackend(RuntimeError("child backend failed"))
    registry, parent, _clock, _wakeup, _notifications = await _suspension_registry(
        monkeypatch, RecoveryConfig(), lambda **_: backend
    )
    try:
        launch = await _launch(registry, parent, "prohibited fail")
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(registry, launch.agent_id, launch.run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
        await _until(lambda: launch.agent_id not in registry._agent_records)
        await _until(lambda: launch.agent_id in registry._evicted_agents)
        tombstone = registry._evicted_agents[launch.agent_id]
        assert tombstone.reason == "closed"
        assert tombstone.summary.availability is AgentAvailability.CLOSED
        assert tombstone.summary.last_run_status is RunStatus.FAILED
        assert tombstone.summary.last_failure is not None
        assert tombstone.summary.last_failure.event_id == envelope.event_id
        # The agents/update wire projection carries the same envelope.
        notify = registry._notify_agents
        assert isinstance(notify, AsyncMock)
        await registry._emit_agents_update()
        await_args = notify.await_args
        assert await_args is not None
        wire_summaries, _evictions = await_args.args
        wire_summary = next(
            item for item in wire_summaries if item.agent_id == launch.agent_id
        )
        assert wire_summary.availability == "closed"
        assert wire_summary.last_failure is not None
        assert wire_summary.last_failure.event_id == envelope.event_id
    finally:
        await registry.drain_children()
        await parent.aclose()
