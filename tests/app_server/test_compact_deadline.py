"""Manual-compaction deadline classification and backend state adoption.

These tests drive ``CoreRequestHandler._compact`` against the real reserve, turn
controller and root-session coordinator, replacing only the loop's ``compact``
coroutine with a gated local fake: the summarization is the one step that would
otherwise reach a provider. The deadline is injected through the module constant
so the fast path and the expiry path differ by no more than that budget.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server import _handler as handler_module
from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._execution import SessionExecution, SessionExecutionKind
from chartreux.app_server._handler import (
    CoreRequestHandler,
    ResumeOrchestration,
    RootLifecycle,
)
from chartreux.app_server._root_session import RootSessionCoordinator
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._turns import TurnController
from chartreux.app_server.models import PublicCheckpointEntry, PublicSessionState
from chartreux.app_server.protocol import (
    ProtocolErrorCode,
    SessionCompactParams,
    SessionCompactResponse,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.compaction import CompactionFailedError
from chartreux.core.compaction.manager import CompactionSettlementError
from tests.conftest import build_test_agent_loop

SUMMARY = "SUMMARY"
# A deadline that cannot fire while a gated fake is held open.
LIVE_DEADLINE_SECONDS = 30.0


@dataclass
class CompactRig:
    loop: AgentLoop
    execution: SessionExecution
    turns: TurnController
    coordinator: RootSessionCoordinator
    history: SessionHistory
    handler: CoreRequestHandler

    @property
    def params(self) -> SessionCompactParams:
        return SessionCompactParams(session_id=self.loop.session_id)

    def checkpoints(self) -> list[PublicCheckpointEntry]:
        return [
            entry
            for entry in self.history.base
            if isinstance(entry, PublicCheckpointEntry)
        ]

    async def aclose(self) -> None:
        await self.loop.aclose()


def build_rig() -> CompactRig:
    loop = build_test_agent_loop()
    execution = SessionExecution()
    history = SessionHistory(base=[])
    coordinator = RootSessionCoordinator(
        loop, Mock(), Mock(), Mock(return_value=0), history
    )

    def snapshot_state() -> PublicSessionState:
        return coordinator.public_state(
            current_history=[], callbacks=[], turns=[], retrying=None, history_limit=200
        )

    turns = TurnController(
        loop, AsyncMock(), AsyncMock(), execution, Mock(), snapshot_state=snapshot_state
    )
    handler = CoreRequestHandler(
        loop,
        turns,
        execution,
        AsyncMock(),
        Mock(),
        Mock(),
        coordinator,
        RootLifecycle(replace=AsyncMock(), adopt=AsyncMock()),
        ResumeOrchestration(
            runtime_factory=Mock(),
            current_event_id=Mock(return_value=0),
            spawn_resume_task=Mock(),
        ),
    )
    return CompactRig(loop, execution, turns, coordinator, history, handler)


def install_compact(monkeypatch: pytest.MonkeyPatch, rig: CompactRig, fake) -> None:
    monkeypatch.setattr(rig.loop, "compact", fake)


def set_deadline(monkeypatch: pytest.MonkeyPatch, seconds: float) -> None:
    monkeypatch.setattr(handler_module, "_MANUAL_COMPACTION_TIMEOUT_SECONDS", seconds)


def state_of(rig: CompactRig) -> PublicSessionState:
    return rig.coordinator.public_state(
        current_history=[], callbacks=[], turns=[], retrying=None, history_limit=200
    )


def published_settlement_state(failure: RequestFailure) -> PublicSessionState:
    """Read the post-replacement state a published settlement failure carries."""
    data = failure.data
    assert isinstance(data, dict)
    assert data["reason"] == "persistence"
    return PublicSessionState.model_validate(data["state"])


@pytest.mark.asyncio
async def test_deadline_expiry_without_publication_is_a_typed_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    seen_manual: list[bool] = []

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        seen_manual.append(manual)
        rig.loop.compaction_outcome.published = False
        await asyncio.Event().wait()
        return "unreachable"

    install_compact(monkeypatch, rig, compact)

    async def quick_compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.data == {"reason": "timeout"}
        # The reserve is released after the cancellation, so the next request
        # is admitted instead of inheriting the stale owner.
        assert rig.execution.active is None
        install_compact(monkeypatch, rig, quick_compact)
        response, _ = await rig.handler._compact(rig.params)
        assert response.summary == SUMMARY
    finally:
        await rig.aclose()
    assert seen_manual == [True]


@pytest.mark.asyncio
async def test_reserve_holds_the_lifecycle_slot_until_the_call_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        entered.set()
        await gate.wait()
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    install_compact(monkeypatch, rig, compact)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)
        active = rig.execution.active
        assert active is not None
        assert active.kind is SessionExecutionKind.LIFECYCLE
        assert active.id.startswith("compact:")
        with pytest.raises(RequestFailure) as conflict:
            await rig.handler.dispatch(
                "session/compact", {"sessionId": rig.loop.session_id}
            )
        assert conflict.value.code is ProtocolErrorCode.CONFLICT
        gate.set()
        response, _ = await asyncio.wait_for(call, 5)
        assert response.summary == SUMMARY
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_published_boundary_that_suppresses_expiry_still_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    cancellations: list[bool] = []

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A provider that finishes after swallowing the cooperative
            # cancellation still returns a usable summary.
            cancellations.append(True)
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    install_compact(monkeypatch, rig, compact)
    try:
        response, _ = await rig.handler._compact(rig.params)
        assert cancellations == [True]
        assert response.summary == SUMMARY
        assert [entry.kind for entry in rig.checkpoints()] == ["compaction"]
        assert rig.checkpoints()[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_provider_failure_after_expiry_is_classified_as_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A provider that converts the deadline's cancellation into its own
            # failure is still an expired run.
            raise CompactionFailedError("tool_call") from None
        raise AssertionError("the gated wait must only end by cancellation")

    install_compact(monkeypatch, rig, compact)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.data == {"reason": "timeout"}
        assert rig.checkpoints() == []
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_expired_deadline_with_external_cancellation_is_still_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    expired = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The deadline's own cancellation lands first and is swallowed, so
            # the external cancellation arrives on an already-expired deadline
            # and the second cancel keeps the exit from raising TimeoutError.
            expired.set()
        await asyncio.Event().wait()
        return "unreachable"

    install_compact(monkeypatch, rig, compact)

    async def quick_compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(expired.wait(), 5)
        call.cancel()
        with pytest.raises(RequestFailure) as failure:
            await call
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.data == {"reason": "timeout"}
        # The racing cancellation is the cause, and expiry governed the
        # classification rather than the interrupted-settlement path.
        assert isinstance(failure.value.__cause__, asyncio.CancelledError)
        assert rig.checkpoints() == []
        assert rig.execution.active is None
        install_compact(monkeypatch, rig, quick_compact)
        response, _ = await rig.handler._compact(rig.params)
        assert response.summary == SUMMARY
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_expired_published_compaction_surfacing_as_timeout_still_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # The published boundary's own settlement is what timed out, so
            # expiry must not be mistaken for an unpublished run.
            raise TimeoutError("settlement timed out") from None
        raise AssertionError("the gated wait must only end by cancellation")

    install_compact(monkeypatch, rig, compact)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is None
        adopted = published_settlement_state(failure.value)
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        # An interrupted settlement claims no summary length.
        assert checkpoints[0].details is None
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_expired_published_compaction_interrupted_by_cancel_still_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    expired = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Both fired with a published boundary: the settlement path owns
            # the classification, not deadline expiry.
            expired.set()
        await asyncio.Event().wait()
        return "unreachable"

    install_compact(monkeypatch, rig, compact)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(expired.wait(), 5)
        call.cancel()
        with pytest.raises(RequestFailure) as failure:
            await call
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is None
        adopted = published_settlement_state(failure.value)
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details is None
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_provider_timeout_without_expiry_is_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        raise TimeoutError("provider read timeout")

    install_compact(monkeypatch, rig, compact)
    try:
        with pytest.raises(TimeoutError, match="provider read timeout"):
            await rig.handler._compact(rig.params)
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_unpublished_compaction_failure_keeps_its_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        raise CompactionFailedError("empty_summary")

    install_compact(monkeypatch, rig, compact)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.data == {"reason": "empty_summary"}
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_external_cancellation_before_publication_settles_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    entered = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        entered.set()
        await asyncio.Event().wait()
        return "unreachable"

    install_compact(monkeypatch, rig, compact)
    order: list[str] = []
    real_reset = rig.turns.reset

    async def reset():
        order.append("reset")
        await real_reset()
        return lambda: order.append("reset-callback")

    replace_idle = Mock()
    monkeypatch.setattr(rig.turns, "reset", reset)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        assert order == ["reset", "reset-callback"]
        # No checkpoint: nothing was compacted, so nothing may claim it.
        replace_idle.assert_not_called()
        assert rig.history.base == []
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_published_settlement_failure_reports_persistence_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    settlement = CompactionSettlementError(published=True, summary_length=len(SUMMARY))

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        raise settlement

    install_compact(monkeypatch, rig, compact)
    order: list[str] = []
    real_reset = rig.turns.reset
    real_replace = rig.coordinator.replace_idle

    async def reset():
        order.append("reset")
        await real_reset()
        return lambda: order.append("reset-callback")

    def replace_idle(*args, **kwargs):
        order.append("replace")
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(rig.turns, "reset", reset)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is settlement
        data = failure.value.data
        assert isinstance(data, dict)
        assert data["reason"] == "persistence"
        adopted = PublicSessionState.model_validate(data["state"])
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        # The reset callback runs before the replacement that can still fail.
        assert order == ["reset", "reset-callback", "replace"]
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_published_settlement_failure_without_replacement_omits_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    settlement = CompactionSettlementError(published=True, summary_length=3)

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        raise settlement

    def replace_idle(*args, **kwargs):
        raise RuntimeError("session replacement failed")

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is settlement
        data = failure.value.data
        assert isinstance(data, dict)
        assert data == {"reason": "persistence"}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_success_path_reset_failure_is_a_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handoff failure after publication is typed, never a raw exception."""
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    reset_error = RuntimeError("turn reset failed")
    faults = [True]

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    real_reset = rig.turns.reset

    async def reset():
        if faults[0]:
            raise reset_error
        return await real_reset()

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.turns, "reset", reset)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        # The handoff failure is the cause, not a replacement for it.
        assert failure.value.__cause__ is reset_error
        # The replacement still ran, so the post-replacement state is settled.
        adopted = published_settlement_state(failure.value)
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
        # The reserve is released, so the next request is admitted.
        faults[0] = False
        response, _ = await rig.handler._compact(rig.params)
        assert response.summary == SUMMARY
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_success_path_replacement_failure_omits_the_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    replacement_error = RuntimeError("session replacement failed")

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    install_compact(monkeypatch, rig, compact)
    order: list[str] = []
    real_reset = rig.turns.reset

    async def reset():
        order.append("reset")
        await real_reset()
        return lambda: order.append("reset-callback")

    def replace_idle(*args, **kwargs):
        order.append("replace")
        raise replacement_error

    monkeypatch.setattr(rig.turns, "reset", reset)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is replacement_error
        data = failure.value.data
        assert isinstance(data, dict)
        # Without a settled replacement there is no state to hand over, so the
        # client has to reconcile later.
        assert data == {"reason": "persistence"}
        # The deferred queue reset still publishes once the handoff fails, the
        # settled reset is not redone, and the replacement is retried only to
        # settle the state the error wants to hand over.
        assert order == ["reset", "replace", "reset-callback", "replace"]
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_expired_success_path_reset_failure_is_a_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row 3 governs identically once a boundary is published."""
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    reset_error = RuntimeError("turn reset failed")

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # A provider that finishes after swallowing the cancellation still
            # returns a usable summary on an expired deadline.
            pass
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    async def reset():
        raise reset_error

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.turns, "reset", reset)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is reset_error
        adopted = published_settlement_state(failure.value)
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_expired_success_path_replacement_failure_omits_the_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    replacement_error = RuntimeError("session replacement failed")

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    def replace_idle(*args, **kwargs):
        raise replacement_error

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is replacement_error
        assert failure.value.data == {"reason": "persistence"}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_published_failure_survives_a_reset_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The typed persistence error wins over a failed settlement reset."""
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    settlement = CompactionSettlementError(published=True, summary_length=len(SUMMARY))
    reset_error = RuntimeError("turn reset failed")

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        raise settlement

    async def reset():
        raise reset_error

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.turns, "reset", reset)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        # The reset failure is not what the client is told: the settlement
        # error stays the cause and the replacement still settles the state.
        assert failure.value.__cause__ is settlement
        adopted = published_settlement_state(failure.value)
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_published_failure_survives_a_reset_callback_raise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising deferred queue reset never masks the typed error."""
    rig = build_rig()
    set_deadline(monkeypatch, LIVE_DEADLINE_SECONDS)
    settlement = CompactionSettlementError(published=True, summary_length=len(SUMMARY))
    callback_error = RuntimeError("queue reset callback failed")
    order: list[str] = []
    failing = [True]

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        if failing[0]:
            raise settlement
        return SUMMARY

    real_reset = rig.turns.reset
    real_replace = rig.coordinator.replace_idle

    async def reset():
        order.append("reset")
        await real_reset()

        def after_reset() -> None:
            order.append("reset-callback")
            raise callback_error

        return after_reset

    def replace_idle(*args, **kwargs):
        order.append("replace")
        return real_replace(*args, **kwargs)

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.turns, "reset", reset)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        # The settlement error stays the cause: the callback failure is logged
        # and swallowed instead of taking its place.
        assert failure.value.__cause__ is settlement
        adopted = published_settlement_state(failure.value)
        checkpoints = [
            entry
            for entry in adopted.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        # The callback ran, and the replacement still ran after it.
        assert order == ["reset", "reset-callback", "replace"]
        assert rig.execution.active is None
        # The reserve is released, so the next admission is admitted.
        failing[0] = False
        response, _ = await rig.handler._compact(rig.params)
        assert response.summary == SUMMARY
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_expired_published_failure_without_replacement_omits_the_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    set_deadline(monkeypatch, 0.02)
    settlement = CompactionSettlementError(published=True, summary_length=3)
    replacement_error = RuntimeError("session replacement failed")

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = True
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise settlement from None
        raise AssertionError("the gated wait must only end by cancellation")

    def replace_idle(*args, **kwargs):
        raise replacement_error

    install_compact(monkeypatch, rig, compact)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is settlement
        # Neither handoff settled, so there is no state to reconcile with yet.
        assert failure.value.data == {"reason": "persistence"}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


class _FailingCompactHandler:
    def __init__(self, failure: RequestFailure) -> None:
        self.failure = failure

    async def dispatch(self, method: str, raw_params: dict[str, object]) -> object:
        raise self.failure


def build_backend(handler: _FailingCompactHandler) -> SessionBackendImpl:
    return SessionBackendImpl(
        Mock(), Mock(), Mock(), cast(CoreRequestHandler, handler), Mock(), Mock()
    )


@pytest.mark.asyncio
async def test_backend_adopts_the_state_of_a_failed_compaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    try:
        expected = state_of(rig)
    finally:
        await rig.aclose()
    failure = RequestFailure(
        ProtocolErrorCode.COMPACTION_FAILED,
        "Compaction was published but could not be persisted",
        {
            "reason": "persistence",
            "state": expected.model_dump(mode="json", by_alias=True),
        },
    )
    backend = build_backend(_FailingCompactHandler(failure))

    with pytest.raises(SessionBackendError) as forwarded:
        await backend._request(
            "session/compact",
            SessionCompactParams(session_id="session-1"),
            SessionCompactResponse,
        )
    assert forwarded.value.code is ProtocolErrorCode.COMPACTION_FAILED
    assert forwarded.value.data == failure.data
    assert backend._projection is not None
    assert backend._projection.state == expected


@pytest.mark.asyncio
async def test_backend_keeps_the_typed_error_for_a_malformed_state() -> None:
    failure = RequestFailure(
        ProtocolErrorCode.COMPACTION_FAILED,
        "Compaction was published but could not be persisted",
        {"reason": "persistence", "state": {"history": "not-a-list"}},
    )
    backend = build_backend(_FailingCompactHandler(failure))

    with pytest.raises(SessionBackendError) as forwarded:
        await backend._request(
            "session/compact",
            SessionCompactParams(session_id="session-1"),
            SessionCompactResponse,
        )
    assert forwarded.value.code is ProtocolErrorCode.COMPACTION_FAILED
    assert forwarded.value.data == failure.data
    assert backend._projection is None


@pytest.mark.asyncio
async def test_backend_ignores_state_without_a_compaction_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    try:
        state = state_of(rig)
    finally:
        await rig.aclose()
    failure = RequestFailure(
        ProtocolErrorCode.CONFLICT,
        "Session lifecycle transition is active",
        {"state": state.model_dump(mode="json", by_alias=True)},
    )
    backend = build_backend(_FailingCompactHandler(failure))

    with pytest.raises(SessionBackendError) as forwarded:
        await backend._request(
            "session/compact",
            SessionCompactParams(session_id="session-1"),
            SessionCompactResponse,
        )
    assert forwarded.value.code is ProtocolErrorCode.CONFLICT
    assert backend._projection is None
