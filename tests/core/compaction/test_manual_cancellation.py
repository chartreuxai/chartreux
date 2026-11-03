"""Manual-compaction publication and settlement safety (v0.7.2 package A).

Every provider and persistence operation here is a local gated fake: the
summarizer and the save callable are injected, and the loop-level tests stub
the manager's summarizer, so no test can reach a real provider. The repository
conftest additionally blocks non-loopback sockets for the whole session.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import contextlib
from typing import Any

import pytest

from chartreux.core.compaction.manager import (
    CompactionManager,
    CompactionOutcome,
    CompactionSettlementError,
    CompletionFn,
)
from chartreux.core.file_revision import full_revision
from chartreux.core.file_state import MAX_TRACKED_FILES, FileStateStore
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role
from chartreux.core.message_list import MessageList
from chartreux.core.middleware import ResetReason
from chartreux.core.session_types import AgentStats
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)
from tests.mock.utils import mock_llm_chunk

SUMMARY = "all done"
MANUAL_SUMMARY = "manual summary"


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------


async def _noop() -> None:
    return None


async def _wait_on_gate(gate: asyncio.Event, *, suppress: bool) -> None:
    """Wait on a test gate, optionally swallowing a delivered cancellation.

    ``suppress`` models a provider that swallows the cancellation and still
    returns a usable result; without it the cancellation propagates like a
    cooperative provider's.
    """
    if suppress:
        with contextlib.suppress(asyncio.CancelledError):
            await gate.wait()
    else:
        await gate.wait()


class _ImmediateProvider:
    """Summarizer stand-in that returns a scripted summary at once."""

    def __init__(self, summary: str = SUMMARY) -> None:
        self._chunk = mock_llm_chunk(content=f"<summary>{summary}</summary>")
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> LLMChunk:
        self.calls += 1
        return self._chunk


class _GatedProvider:
    """Summarizer stand-in the test cancels while it is still waiting."""

    def __init__(
        self, *, gate: asyncio.Event, suppress: bool, summary: str = SUMMARY
    ) -> None:
        self._gate = gate
        self._suppress = suppress
        self._chunk = mock_llm_chunk(content=f"<summary>{summary}</summary>")
        self.started = asyncio.Event()
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> LLMChunk:
        self.calls += 1
        self.started.set()
        await _wait_on_gate(self._gate, suppress=self._suppress)
        return self._chunk


class _CancelOnReturnProvider:
    """Summarizer stand-in that leaves a cancellation request outstanding."""

    def __init__(self, summary: str = SUMMARY) -> None:
        self._chunk = mock_llm_chunk(content=f"<summary>{summary}</summary>")
        self.calls = 0

    async def __call__(self, **_kwargs: Any) -> LLMChunk:
        self.calls += 1
        task = asyncio.current_task()
        assert task is not None
        task.cancel()
        # No await follows, so the request stays undelivered even though a
        # usable summary is about to come back.
        return self._chunk


class _GatedSave:
    """Persistence stand-in the test can cancel and then release."""

    def __init__(self, *, fail: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.completed = 0
        self._fail = fail
        self.error = OSError("disk unavailable")

    async def __call__(self) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        self.completed += 1
        if self._fail:
            raise self.error


class _FailingSave:
    def __init__(self) -> None:
        self.calls = 0
        self.error = OSError("disk unavailable")

    async def __call__(self) -> None:
        self.calls += 1
        raise self.error


class _RecordingSave:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1


async def _async_noop() -> None:
    return None


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _conversation() -> MessageList:
    return MessageList([
        LLMMessage(role=Role.system, content="sys"),
        LLMMessage(role=Role.user, content="oldest ask"),
        LLMMessage(role=Role.assistant, content="work"),
        LLMMessage(role=Role.user, content="newest ask"),
    ])


def _boundaries(messages: MessageList) -> list[LLMMessage]:
    return [message for message in messages if message.context_boundary == "compaction"]


def _over_cap_store() -> FileStateStore:
    store = FileStateStore()
    for index in range(MAX_TRACKED_FILES + 5):
        store.record_revision(
            f"/repo/file_{index}.py", full_revision(f"{index}".encode())
        )
    assert store.is_over_cap()
    return store


def _build_manager(
    complete: CompletionFn,
    *,
    save: Callable[[], Awaitable[None]] | None = None,
    file_state: FileStateStore | None = None,
    messages: MessageList | None = None,
) -> CompactionManager:
    stats = AgentStats()
    config = build_test_vibe_config(models=make_test_models(auto_compact_threshold=999))
    return CompactionManager(
        messages=messages if messages is not None else _conversation(),
        stats_getter=lambda: stats,
        config_getter=lambda: config,
        complete=complete,
        available_tools=lambda: [],
        tool_choice=lambda: "auto",
        save=save if save is not None else _noop,
        file_state_getter=(lambda: file_state) if file_state is not None else None,
    )


def _track_trims(
    monkeypatch: pytest.MonkeyPatch, manager: CompactionManager
) -> list[int]:
    calls: list[int] = []

    async def _trim() -> None:
        calls.append(1)

    monkeypatch.setattr(manager, "_trim_archived_prefix", _trim)
    return calls


def _lingering_tasks() -> set[asyncio.Task[Any]]:
    current = asyncio.current_task()
    return {
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    }


async def _drain_until(predicate: Callable[[], bool]) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not observed")


def _immediate_summarize(summary: str) -> Callable[..., Awaitable[str]]:
    async def _summarize(*_args: Any, **_kwargs: Any) -> str:
        return summary

    return _summarize


def _gated_summarize(
    *, started: asyncio.Event, gate: asyncio.Event, suppress: bool, summary: str
) -> Callable[..., Awaitable[str]]:
    async def _summarize(*_args: Any, **_kwargs: Any) -> str:
        started.set()
        await _wait_on_gate(gate, suppress=suppress)
        return summary

    return _summarize


def _capture_settlement(
    monkeypatch: pytest.MonkeyPatch, manager: CompactionManager
) -> list[asyncio.Task[Any]]:
    """Record the settlement task so a test can act on its completion instant.

    The recording wrapper delegates to the real settlement, so the save and the
    trim behave exactly as in production; only the task handle is new.
    """
    settlements: list[asyncio.Task[Any]] = []
    settle = manager._settle_published

    async def _recording_settle() -> None:
        current = asyncio.current_task()
        assert current is not None
        settlements.append(current)
        await settle()

    monkeypatch.setattr(manager, "_settle_published", _recording_settle)
    return settlements


def _build_loop() -> Any:
    agent = build_test_agent_loop(
        config=build_test_vibe_config(
            models=make_test_models(auto_compact_threshold=999)
        )
    )
    agent.messages.extend([
        LLMMessage(role=Role.user, content="old ask"),
        LLMMessage(role=Role.assistant, content="old work"),
        LLMMessage(role=Role.user, content="latest ask"),
    ])
    return agent


# --------------------------------------------------------------------------
# pre-publication cancellation
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suppressing_cancellation_before_publication_mutates_nothing() -> None:
    gate = asyncio.Event()
    provider = _GatedProvider(gate=gate, suppress=True)
    store = _over_cap_store()
    messages = _conversation()
    save = _GatedSave()
    manager = _build_manager(provider, save=save, file_state=store, messages=messages)
    outcome = CompactionOutcome()

    task = asyncio.create_task(manager.compact(manual=True, outcome=outcome))
    await provider.started.wait()
    task.cancel()
    # Release the (cancellation-suppressing) provider: it still returns a usable
    # summary, so only the outstanding request can stop publication.
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.calls == 1
    assert _boundaries(messages) == []
    assert store.is_over_cap()
    assert save.calls == 0
    assert outcome.published is False
    assert outcome.cancellation_observed is True
    assert _lingering_tasks() == set()


@pytest.mark.asyncio
async def test_undelivered_cancellation_after_the_summary_is_not_published() -> None:
    provider = _CancelOnReturnProvider()
    messages = _conversation()
    save = _RecordingSave()
    manager = _build_manager(provider, save=save, messages=messages)
    outcome = CompactionOutcome()

    task = asyncio.create_task(manager.compact(manual=True, outcome=outcome))
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _boundaries(messages) == []
    assert save.calls == 0
    assert outcome.published is False
    assert outcome.cancellation_observed is True


@pytest.mark.asyncio
async def test_cooperative_cancellation_propagates_before_publication() -> None:
    gate = asyncio.Event()
    provider = _GatedProvider(gate=gate, suppress=False)
    store = _over_cap_store()
    messages = _conversation()
    save = _RecordingSave()
    manager = _build_manager(provider, save=save, file_state=store, messages=messages)
    outcome = CompactionOutcome()

    task = asyncio.create_task(manager.compact(manual=True, outcome=outcome))
    await provider.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.calls == 1
    assert _boundaries(messages) == []
    assert store.is_over_cap()
    assert save.calls == 0
    assert outcome.published is False
    assert outcome.cancellation_observed is True


# --------------------------------------------------------------------------
# post-publication settlement
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_cancellation_during_settlement_still_publishes_once_and_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _ImmediateProvider()
    messages = _conversation()
    save = _GatedSave()
    manager = _build_manager(provider, save=save, messages=messages)
    trims = _track_trims(monkeypatch, manager)
    outcome = CompactionOutcome()

    task = asyncio.create_task(manager.compact(manual=True, outcome=outcome))
    await save.started.wait()
    assert outcome.published is True
    assert outcome.cancellation_observed is False

    task.cancel()
    await _drain_until(lambda: outcome.cancellation_observed)
    task.cancel()
    await asyncio.sleep(0)
    save.release.set()

    summary = await task

    assert summary == SUMMARY
    assert task.cancelled() is False
    assert outcome.published is True
    assert outcome.cancellation_observed is True
    # Once each: the boundary was published once and settled once despite the
    # repeated cancellations.
    assert len(_boundaries(messages)) == 1
    assert save.calls == 1
    assert save.completed == 1
    assert len(trims) == 1
    assert _lingering_tasks() == set()


@pytest.mark.asyncio
async def test_settlement_failure_is_typed_and_keeps_the_published_boundary() -> None:
    provider = _ImmediateProvider()
    messages = _conversation()
    save = _FailingSave()
    manager = _build_manager(provider, save=save, messages=messages)
    outcome = CompactionOutcome()

    with pytest.raises(CompactionSettlementError) as info:
        await manager.compact(manual=True, outcome=outcome)

    assert info.value.published is True
    assert info.value.summary_length == len(SUMMARY)
    assert info.value.__cause__ is save.error
    assert outcome.published is True
    assert outcome.cancellation_observed is False
    assert len(_boundaries(messages)) == 1
    assert save.calls == 1


@pytest.mark.asyncio
async def test_settlement_failure_after_absorbed_cancellation_is_typed() -> None:
    provider = _ImmediateProvider()
    messages = _conversation()
    save = _GatedSave(fail=True)
    manager = _build_manager(provider, save=save, messages=messages)
    outcome = CompactionOutcome()

    task = asyncio.create_task(manager.compact(manual=True, outcome=outcome))
    await save.started.wait()
    task.cancel()
    await _drain_until(lambda: outcome.cancellation_observed)
    save.release.set()

    with pytest.raises(CompactionSettlementError) as info:
        await task

    assert info.value.published is True
    assert info.value.summary_length == len(SUMMARY)
    assert info.value.__cause__ is save.error
    assert outcome.published is True
    assert outcome.cancellation_observed is True
    assert len(_boundaries(messages)) == 1


@pytest.mark.asyncio
async def test_settlement_failure_at_the_completion_instant_is_not_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation landing after the settlement finished must not win.

    The cancel fires from the settlement task's own done callback, so it reaches
    the shielded await in the instant after the settlement completed, here with
    the save failure. The already-decided outcome is the one reported: the
    typed settlement error, not a bare cancellation.
    """
    provider = _ImmediateProvider()
    messages = _conversation()
    save = _GatedSave(fail=True)
    manager = _build_manager(provider, save=save, messages=messages)
    settlements = _capture_settlement(monkeypatch, manager)
    outcome = CompactionOutcome()

    task = asyncio.create_task(manager.compact(manual=True, outcome=outcome))
    await save.started.wait()
    assert len(settlements) == 1
    settlements[0].add_done_callback(lambda _done: task.cancel())
    save.release.set()

    with pytest.raises(CompactionSettlementError) as info:
        await task

    assert info.value.published is True
    assert info.value.summary_length == len(SUMMARY)
    assert info.value.__cause__ is save.error
    assert outcome.published is True
    assert outcome.cancellation_observed is True
    assert len(_boundaries(messages)) == 1
    assert save.calls == 1
    assert save.completed == 1
    assert _lingering_tasks() == set()


# --------------------------------------------------------------------------
# the automatic path stays unchanged
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_compaction_keeps_the_raw_save_failure() -> None:
    provider = _ImmediateProvider()
    messages = _conversation()
    save = _FailingSave()
    manager = _build_manager(provider, save=save, messages=messages)

    with pytest.raises(OSError) as info:
        await manager.compact()

    assert info.value is save.error
    assert save.calls == 1
    assert len(_boundaries(messages)) == 1


@pytest.mark.asyncio
async def test_auto_compaction_does_not_consult_a_pending_cancellation() -> None:
    gate = asyncio.Event()
    provider = _GatedProvider(gate=gate, suppress=True)
    messages = _conversation()
    save = _RecordingSave()
    manager = _build_manager(provider, save=save, messages=messages)

    task = asyncio.create_task(manager.compact())
    await provider.started.wait()
    task.cancel()
    gate.set()

    assert await task == SUMMARY
    assert save.calls == 1
    assert len(_boundaries(messages)) == 1


# --------------------------------------------------------------------------
# loop integration: carrier, bookkeeping, retry save
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_loop_bookkeeping_completes_after_an_absorbed_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _build_loop()
    manager = agent.compaction_manager
    save = _GatedSave()
    trims = _track_trims(monkeypatch, manager)
    monkeypatch.setattr(manager, "_summarize", _immediate_summarize(MANUAL_SUMMARY))
    monkeypatch.setattr(manager, "_save", save)
    resets: list[ResetReason] = []
    marks: list[int] = []
    monkeypatch.setattr(
        agent.middleware_pipeline,
        "reset",
        lambda reset_reason=ResetReason.STOP: resets.append(reset_reason),
    )
    monkeypatch.setattr(
        agent._title_controller, "mark_compaction", lambda: marks.append(1)
    )

    task = asyncio.create_task(agent.compact(manual=True))
    await save.started.wait()
    assert agent.compaction_outcome.published is True
    task.cancel()
    await _drain_until(lambda: agent.compaction_outcome.cancellation_observed)
    save.release.set()

    summary = await task

    assert summary == MANUAL_SUMMARY
    assert task.cancelled() is False
    assert agent.compaction_outcome.published is True
    assert agent.compaction_outcome.cancellation_observed is True
    assert save.calls == 1
    assert save.completed == 1
    assert len(trims) == 1
    assert resets == [ResetReason.COMPACT]
    assert marks == [1]
    assert len(_boundaries(agent.messages)) == 1
    assert _lingering_tasks() == set()


@pytest.mark.asyncio
async def test_loop_bookkeeping_completes_when_cancellation_follows_a_settled_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancellation that lands after the settlement finished still settles.

    The cancel fires from the settlement task's done callback, so it reaches the
    shielded await one instant after the settlement completed successfully. The
    settled summary is returned and the caller's bookkeeping runs; re-raising
    here would discard the summary and skip every step below it.
    """
    agent = _build_loop()
    manager = agent.compaction_manager
    save = _GatedSave()
    trims = _track_trims(monkeypatch, manager)
    monkeypatch.setattr(manager, "_summarize", _immediate_summarize(MANUAL_SUMMARY))
    monkeypatch.setattr(manager, "_save", save)
    settlements = _capture_settlement(monkeypatch, manager)
    resets: list[ResetReason] = []
    marks: list[int] = []
    monkeypatch.setattr(
        agent.middleware_pipeline,
        "reset",
        lambda reset_reason=ResetReason.STOP: resets.append(reset_reason),
    )
    monkeypatch.setattr(
        agent._title_controller, "mark_compaction", lambda: marks.append(1)
    )

    task = asyncio.create_task(agent.compact(manual=True))
    await save.started.wait()
    assert agent.compaction_outcome.published is True
    assert len(settlements) == 1
    settlements[0].add_done_callback(lambda _done: task.cancel())
    save.release.set()

    summary = await task

    assert summary == MANUAL_SUMMARY
    assert task.cancelled() is False
    assert agent.compaction_outcome.published is True
    assert agent.compaction_outcome.cancellation_observed is True
    assert save.calls == 1
    assert save.completed == 1
    assert len(trims) == 1
    assert resets == [ResetReason.COMPACT]
    assert marks == [1]
    assert len(_boundaries(agent.messages)) == 1
    assert _lingering_tasks() == set()


@pytest.mark.asyncio
async def test_loop_pre_publication_cancellation_skips_the_retry_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _build_loop()
    manager = agent.compaction_manager
    gate = asyncio.Event()
    started = asyncio.Event()
    monkeypatch.setattr(
        manager,
        "_summarize",
        _gated_summarize(
            started=started, gate=gate, suppress=True, summary=MANUAL_SUMMARY
        ),
    )
    real_save = agent._save_messages
    save_calls: list[int] = []

    async def _counting_save(*, allow_empty: bool = False) -> None:
        save_calls.append(1)
        await real_save(allow_empty=allow_empty)

    monkeypatch.setattr(agent, "_save_messages", _counting_save)

    task = asyncio.create_task(agent.compact(manual=True))
    await started.wait()
    task.cancel()
    gate.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _boundaries(agent.messages) == []
    assert agent.compaction_outcome.published is False
    assert agent.compaction_outcome.cancellation_observed is True
    # Only the pre-compaction save: a cancellation never triggers the retry.
    assert len(save_calls) == 1


@pytest.mark.asyncio
async def test_loop_retry_save_failure_preserves_the_settlement_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _build_loop()
    manager = agent.compaction_manager
    monkeypatch.setattr(manager, "_summarize", _immediate_summarize(MANUAL_SUMMARY))
    save = _FailingSave()
    monkeypatch.setattr(manager, "_save", save)
    real_save = agent._save_messages
    calls: list[int] = []

    async def _first_ok_then_fail(*, allow_empty: bool = False) -> None:
        calls.append(1)
        if len(calls) == 1:
            await real_save(allow_empty=allow_empty)
            return
        raise save.error

    monkeypatch.setattr(agent, "_save_messages", _first_ok_then_fail)

    with pytest.raises(CompactionSettlementError) as info:
        await agent.compact(manual=True)

    assert info.value.published is True
    assert info.value.summary_length == len(MANUAL_SUMMARY)
    assert info.value.__cause__ is save.error
    # The boundary stays published and the pre-compaction save plus one retry ran.
    assert len(_boundaries(agent.messages)) == 1
    assert save.calls == 1
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_loop_auto_compaction_never_marks_the_manual_carrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _build_loop()
    manager = agent.compaction_manager
    monkeypatch.setattr(manager, "_summarize", _immediate_summarize(MANUAL_SUMMARY))
    monkeypatch.setattr(manager, "_save", _RecordingSave())
    monkeypatch.setattr(manager, "_trim_archived_prefix", _async_noop)
    agent._compaction_outcome.published = True
    agent._compaction_outcome.cancellation_observed = True

    # The automatic caller's exact call: no manual keyword, no carrier.
    summary = await agent.compact()

    assert summary == MANUAL_SUMMARY
    assert agent.compaction_outcome.published is False
    assert agent.compaction_outcome.cancellation_observed is False
    assert len(_boundaries(agent.messages)) == 1
