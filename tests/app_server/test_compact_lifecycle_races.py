"""Cross-boundary races around a manual compaction (v0.7.2 package D).

These tests drive the *integrated* lifecycle: the handler's LIFECYCLE reserve,
the turn router that shares it, the real compaction manager, and the client-side
projection. Only the summarizer, the persistence call and the archive trim are
replaced with local gated fakes, so no test can reach a provider.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from chartreux.app_server import _handler as handler_module
from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._execution import SessionExecution, SessionExecutionKind
from chartreux.app_server._handler import (
    _MANUAL_COMPACTION_TIMEOUT_SECONDS,
    CoreRequestHandler,
    ResumeOrchestration,
    RootLifecycle,
)
from chartreux.app_server._root_session import RootSessionCoordinator
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._session_backend_port import SessionBackendError
from chartreux.app_server._session_history import SessionHistory
from chartreux.app_server._session_resources import _COMPACT_REQUEST_TIMEOUT_SECONDS
from chartreux.app_server._turns import SteerReceipt, TurnController
from chartreux.app_server.models import (
    PublicCheckpointEntry,
    PublicSessionState,
    PublicTurn,
    PublicTurnStatus,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    AgentsCancelResponse,
    CancelOutcome,
    ProtocolErrorCode,
    SessionCompactParams,
    SessionCompactResponse,
    SessionResumeParams,
    SessionResumeResponse,
    TurnInterruptParams,
    TurnInterruptResponse,
    TurnStartParams,
    TurnSteerParams,
    TurnSteerResponse,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import LLMMessage, Role
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    make_test_models,
)

SUMMARY = "the compacted summary"
# The client must outlive the server's cooperative-cancellation budget: the
# server requests cancellation at its deadline, so a client that gives up first
# would abandon the settlement it is waiting for.
CLIENT_SETTLEMENT_MARGIN_SECONDS = 30.0


def test_the_client_margin_outlives_the_server_deadline() -> None:
    """Drift guard: a shorter client timeout would abandon settlement.

    Both constants are imported at module import, so dropping or renaming either
    one fails collection instead of silently disabling the invariant.
    """
    assert (
        _MANUAL_COMPACTION_TIMEOUT_SECONDS + CLIENT_SETTLEMENT_MARGIN_SECONDS
        <= _COMPACT_REQUEST_TIMEOUT_SECONDS
    )


# --------------------------------------------------------------------------
# rig
# --------------------------------------------------------------------------


class CompactRig:
    def __init__(self) -> None:
        self.loop: AgentLoop = build_test_agent_loop(
            config=build_test_vibe_config(
                models=make_test_models(auto_compact_threshold=999)
            )
        )
        self.loop.messages.extend([
            LLMMessage(role=Role.user, content="oldest ask"),
            LLMMessage(role=Role.assistant, content="old work"),
            LLMMessage(role=Role.user, content="newest ask"),
        ])
        self.execution = SessionExecution()
        self.history = SessionHistory(base=[])
        self.coordinator = RootSessionCoordinator(
            self.loop, Mock(), Mock(), Mock(return_value=0), self.history
        )
        # Attached like a live client, so the guarded methods a compaction must
        # not disturb can actually be routed.
        self.coordinator.attach(self.loop.session_id)

        def snapshot_state() -> PublicSessionState:
            return self.state()

        self.turns = TurnController(
            self.loop,
            AsyncMock(),
            AsyncMock(),
            self.execution,
            Mock(),
            snapshot_state=snapshot_state,
        )
        self.handler = CoreRequestHandler(
            self.loop,
            self.turns,
            self.execution,
            AsyncMock(),
            Mock(),
            Mock(),
            self.coordinator,
            RootLifecycle(replace=AsyncMock(), adopt=AsyncMock()),
            ResumeOrchestration(
                runtime_factory=Mock(),
                current_event_id=Mock(return_value=0),
                spawn_resume_task=Mock(),
            ),
        )

    @property
    def params(self) -> SessionCompactParams:
        return SessionCompactParams(session_id=self.loop.session_id)

    def state(self) -> PublicSessionState:
        return self.coordinator.public_state(
            current_history=self.turns.history,
            callbacks=[],
            turns=[],
            retrying=None,
            history_limit=200,
        )

    def checkpoints(self) -> list[PublicCheckpointEntry]:
        return [
            entry
            for entry in self.history.base
            if isinstance(entry, PublicCheckpointEntry)
        ]

    async def aclose(self) -> None:
        await self.loop.aclose()


def build_rig() -> CompactRig:
    return CompactRig()


class GatedSave:
    """Persistence stand-in the test holds open across the settlement window."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    async def __call__(self) -> None:
        self.calls += 1
        self.started.set()
        await self.release.wait()


class FailingSave:
    def __init__(self) -> None:
        self.error = OSError("disk unavailable")

    async def __call__(self) -> None:
        raise self.error


def install_immediate_summarizer(
    monkeypatch: pytest.MonkeyPatch, rig: CompactRig, summary: str = SUMMARY
) -> None:
    """Answer the summarization locally and keep the persistence a no-op."""

    async def summarize(*_args: Any, **_kwargs: Any) -> str:
        return summary

    async def save() -> None:
        return None

    manager = rig.loop.compaction_manager
    monkeypatch.setattr(manager, "_summarize", summarize)
    monkeypatch.setattr(manager, "_save", save)


def install_trim_noop(monkeypatch: pytest.MonkeyPatch, rig: CompactRig) -> None:
    async def trim() -> None:
        return None

    monkeypatch.setattr(rig.loop.compaction_manager, "_trim_archived_prefix", trim)


def install_gated_compact(
    monkeypatch: pytest.MonkeyPatch, rig: CompactRig
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the manual compaction inside the reserve, before it publishes."""
    entered = asyncio.Event()
    gate = asyncio.Event()

    async def compact(extra: str = "", *, manual: bool = False) -> str:
        rig.loop.compaction_outcome.published = False
        entered.set()
        await gate.wait()
        rig.loop.compaction_outcome.published = True
        return SUMMARY

    monkeypatch.setattr(rig.loop, "compact", compact)
    return entered, gate


def lifecycle_conflict(failure: pytest.ExceptionInfo[RequestFailure]) -> bool:
    return failure.value.code is ProtocolErrorCode.CONFLICT and str(
        failure.value
    ).startswith("Session lifecycle transition is active")


# --------------------------------------------------------------------------
# the input-rejection window
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ordinary_input_and_a_second_compact_conflict_while_held(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    entered, gate = install_gated_compact(monkeypatch, rig)
    before = list(rig.loop.messages)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)
        owner = rig.execution.active
        assert owner is not None
        assert owner.kind is SessionExecutionKind.LIFECYCLE

        with pytest.raises(RequestFailure) as rejected_input:
            await rig.handler.dispatch(
                "turn/start",
                TurnStartParams(
                    session_id=rig.loop.session_id,
                    message=[TextContentBlock(text="interrupting input")],
                ).model_dump(mode="json", by_alias=True),
            )
        assert lifecycle_conflict(rejected_input)
        # A rejected input never reaches the turn layer.
        assert rig.turns.active_turn is None
        assert list(rig.loop.messages) == before

        with pytest.raises(RequestFailure) as rejected_compact:
            await rig.handler.dispatch(
                "session/compact", {"sessionId": rig.loop.session_id}
            )
        assert lifecycle_conflict(rejected_compact)
        # Neither rejection stole or cleared the reserve.
        assert rig.execution.active is owner

        gate.set()
        response, _ = await asyncio.wait_for(call, 5)
        assert response.summary == SUMMARY
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_the_conflict_holds_across_publication_and_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The published boundary settles under the same reserve that admitted it."""
    rig = build_rig()
    install_immediate_summarizer(monkeypatch, rig)
    install_trim_noop(monkeypatch, rig)
    save = GatedSave()
    monkeypatch.setattr(rig.loop.compaction_manager, "_save", save)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(save.started.wait(), 5)
        owner = rig.execution.active
        assert owner is not None and owner.kind is SessionExecutionKind.LIFECYCLE

        # Cross-boundary evidence: the manager published into the live message
        # list, and only the settlement is outstanding.
        boundaries = [
            message
            for message in rig.loop.messages
            if message.context_boundary == "compaction"
        ]
        assert len(boundaries) == 1
        assert rig.checkpoints() == []

        with pytest.raises(RequestFailure) as rejected_input:
            await rig.handler.dispatch(
                "turn/start",
                TurnStartParams(
                    session_id=rig.loop.session_id,
                    message=[TextContentBlock(text="during settlement")],
                ).model_dump(mode="json", by_alias=True),
            )
        assert lifecycle_conflict(rejected_input)
        with pytest.raises(RequestFailure) as rejected_compact:
            await rig.handler.dispatch(
                "session/compact", {"sessionId": rig.loop.session_id}
            )
        assert lifecycle_conflict(rejected_compact)
        assert rig.execution.active is owner

        save.release.set()
        response, _ = await asyncio.wait_for(call, 5)
        assert response.summary == SUMMARY
        assert [entry.kind for entry in rig.checkpoints()] == ["compaction"]
        assert rig.checkpoints()[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_an_active_turn_conflicts_before_the_reserve_is_taken() -> None:
    rig = build_rig()
    rig.turns._active_turn = PublicTurn(
        id="turn-1",
        session_id=rig.loop.session_id,
        status=PublicTurnStatus.IN_PROGRESS,
        started_at=0,
    )
    before = list(rig.loop.messages)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.CONFLICT
        assert str(failure.value) == "Cannot compact while a turn is active"
        # Refused before admission: nothing published, no reserve taken.
        assert rig.execution.active is None
        assert list(rig.loop.messages) == before
        assert rig.checkpoints() == []
    finally:
        await rig.aclose()


# --------------------------------------------------------------------------
# exemptions: the methods that must still be routable under the reserve
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_interrupt_is_exempt_from_the_lifecycle_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    entered, gate = install_gated_compact(monkeypatch, rig)
    interrupt = Mock(return_value=TurnInterruptResponse())
    monkeypatch.setattr(rig.turns, "interrupt", interrupt)
    params = TurnInterruptParams(
        session_id=rig.loop.session_id, expected_turn_id="turn-1"
    )
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)

        result = await rig.handler.dispatch(
            "turn/interrupt", params.model_dump(mode="json", by_alias=True)
        )
        assert isinstance(result.response, TurnInterruptResponse)
        interrupt.assert_called_once_with(params)
        assert rig.execution.active is not None

        gate.set()
        await asyncio.wait_for(call, 5)
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_agent_cancel_is_exempt_from_the_lifecycle_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    entered, gate = install_gated_compact(monkeypatch, rig)

    async def cancel_run(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            outcome=CancelOutcome.STOP_REQUESTED, run_id="run-1", stop_reason=None
        )

    monkeypatch.setattr(rig.handler._sessions, "cancel_run", cancel_run)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)

        result = await rig.handler.dispatch(
            "agents/cancel", {"agentId": "agent-1", "runId": "run-1"}
        )
        assert isinstance(result.response, AgentsCancelResponse)
        assert result.response.outcome is CancelOutcome.STOP_REQUESTED
        assert rig.execution.active is not None

        gate.set()
        await asyncio.wait_for(call, 5)
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_known_steer_receipt_is_exempt_from_the_lifecycle_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    entered, gate = install_gated_compact(monkeypatch, rig)
    params = TurnSteerParams(
        session_id=rig.loop.session_id,
        expected_turn_id="turn-1",
        message=[TextContentBlock(text="already delivered")],
        idempotency_key="steer-1",
    )

    async def settled() -> TurnSteerResponse:
        return TurnSteerResponse()

    receipt_task = asyncio.create_task(settled())
    await receipt_task
    rig.turns._steer_receipts["steer-1"] = SteerReceipt(
        params=params, task=receipt_task
    )
    assert rig.turns.known_steer_receipt(params)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)

        result = await rig.handler.dispatch(
            "turn/steer", params.model_dump(mode="json", by_alias=True)
        )
        assert isinstance(result.response, TurnSteerResponse)
        assert result.response.accepted is True
        assert rig.execution.active is not None

        gate.set()
        await asyncio.wait_for(call, 5)
    finally:
        await rig.aclose()


# --------------------------------------------------------------------------
# reconnect / resume
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_is_refused_while_held_and_reports_settlement_afterwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A reconnect neither clears the reserve nor hides the settled state."""
    rig = build_rig()
    entered, gate = install_gated_compact(monkeypatch, rig)
    resume = SessionResumeParams(session_id=rig.loop.session_id)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(entered.wait(), 5)
        owner = rig.execution.active

        with pytest.raises(RequestFailure) as refused:
            await rig.handler.dispatch(
                "session/resume", resume.model_dump(mode="json", by_alias=True)
            )
        assert lifecycle_conflict(refused)
        assert rig.execution.active is owner

        gate.set()
        _, _ = await asyncio.wait_for(call, 5)

        # The resumed session is the authoritative view after settlement.
        result = await rig.handler.dispatch(
            "session/resume", resume.model_dump(mode="json", by_alias=True)
        )
        assert isinstance(result.response, SessionResumeResponse)
        state = result.response.state
        assert isinstance(state, PublicSessionState)
        checkpoints = [
            entry
            for entry in state.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


# --------------------------------------------------------------------------
# persistence: handler state, backend forwarding and client agreement
# --------------------------------------------------------------------------


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
async def test_persistence_failure_state_agrees_across_handler_and_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    install_immediate_summarizer(monkeypatch, rig)
    install_trim_noop(monkeypatch, rig)
    save = FailingSave()
    monkeypatch.setattr(rig.loop.compaction_manager, "_save", save)
    order: list[str] = []
    real_reset = rig.turns.reset
    real_replace = rig.coordinator.replace_idle

    async def reset() -> Any:
        order.append("reset")
        await real_reset()
        return lambda: order.append("reset-callback")

    def replace_idle(*args: Any, **kwargs: Any) -> Any:
        order.append("replace")
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(rig.turns, "reset", reset)
    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert isinstance(failure.value.__cause__, Exception)
        data = failure.value.data
        assert isinstance(data, dict)
        assert data["reason"] == "persistence"
        settled = PublicSessionState.model_validate(data["state"])
        checkpoints = [
            entry
            for entry in settled.history or []
            if isinstance(entry, PublicCheckpointEntry)
        ]
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        # The reset callback runs before the replacement that can still fail.
        assert order == ["reset", "reset-callback", "replace"]
        assert rig.execution.active is None

        # The client-side session view adopts exactly the state the server
        # reported, so a persistence error leaves client and server agreed.
        backend = build_backend(_FailingCompactHandler(failure.value))
        with pytest.raises(SessionBackendError) as forwarded:
            await backend._request(
                "session/compact", rig.params, SessionCompactResponse
            )
        assert forwarded.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert forwarded.value.data == data
        assert backend._projection is not None
        assert backend._projection.state == settled
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_success_records_the_summary_length_and_releases_the_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    install_immediate_summarizer(monkeypatch, rig)
    install_trim_noop(monkeypatch, rig)
    order: list[str] = []
    real_reset = rig.turns.reset

    async def reset() -> Any:
        order.append("reset")
        await real_reset()
        return lambda: order.append("reset-callback")

    monkeypatch.setattr(rig.turns, "reset", reset)
    try:
        response, after_response = await rig.handler._compact(rig.params)
        assert response.summary == SUMMARY
        checkpoints = rig.checkpoints()
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        # The reset callback is deferred to the response, not dropped.
        assert order == ["reset"]
        assert after_response is not None
        after_response()
        assert order == ["reset", "reset-callback"]
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_failing_handoff_after_a_real_publication_is_a_persistence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The settlement of a published boundary is typed, never a raw failure."""
    rig = build_rig()
    install_immediate_summarizer(monkeypatch, rig)
    install_trim_noop(monkeypatch, rig)
    replacement_error = RuntimeError("session replacement failed")

    def replace_idle(*_args: Any, **_kwargs: Any) -> Any:
        raise replacement_error

    monkeypatch.setattr(rig.coordinator, "replace_idle", replace_idle)
    try:
        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.__cause__ is replacement_error
        # The boundary exists in the live conversation, but nothing handed the
        # session over, so the client reconciles instead of retrying.
        assert rig.loop.compaction_outcome.published is True
        assert failure.value.data == {"reason": "persistence"}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


@pytest.mark.asyncio
async def test_a_stale_publication_flag_never_classifies_the_next_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpublished retry is a timeout, not the previous attempt's boundary.

    The carrier is read at the entry of the handler's reserve and classified by
    on every expiry exit, so a publication retained from an earlier attempt
    would settle a boundary this attempt never wrote.
    """
    rig = build_rig()
    install_immediate_summarizer(monkeypatch, rig)
    install_trim_noop(monkeypatch, rig)
    never = asyncio.Event()
    try:
        first, _ = await rig.handler._compact(rig.params)
        assert first.summary == SUMMARY
        assert rig.loop.compaction_outcome.published is True
        assert len(rig.checkpoints()) == 1

        async def stalled_summarize(*_args: Any, **_kwargs: Any) -> str:
            await never.wait()
            return "unreachable"

        monkeypatch.setattr(
            rig.loop.compaction_manager, "_summarize", stalled_summarize
        )
        monkeypatch.setattr(handler_module, "_MANUAL_COMPACTION_TIMEOUT_SECONDS", 0.02)

        with pytest.raises(RequestFailure) as failure:
            await rig.handler._compact(rig.params)
        # Nothing was published here, so the entry reset is what makes the
        # expiry classify as a timeout instead of settling a phantom boundary.
        assert rig.loop.compaction_outcome.published is False
        assert failure.value.code is ProtocolErrorCode.COMPACTION_FAILED
        assert failure.value.data == {"reason": "timeout"}
        assert len(rig.checkpoints()) == 1
        assert rig.execution.active is None
    finally:
        await rig.aclose()


# --------------------------------------------------------------------------
# external cancellation inside the reserve
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_external_cancellation_during_settlement_still_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = build_rig()
    install_immediate_summarizer(monkeypatch, rig)
    install_trim_noop(monkeypatch, rig)
    save = GatedSave()
    monkeypatch.setattr(rig.loop.compaction_manager, "_save", save)
    try:
        call = asyncio.create_task(rig.handler._compact(rig.params))
        await asyncio.wait_for(save.started.wait(), 5)
        assert rig.loop.compaction_outcome.published is True

        call.cancel()
        await _drain_until(lambda: rig.loop.compaction_outcome.cancellation_observed)
        # The reserve is still held while the interrupted settlement runs.
        assert rig.execution.active is not None
        with pytest.raises(RequestFailure) as rejected_input:
            await rig.handler.dispatch(
                "turn/start",
                TurnStartParams(
                    session_id=rig.loop.session_id,
                    message=[TextContentBlock(text="still refused")],
                ).model_dump(mode="json", by_alias=True),
            )
        assert lifecycle_conflict(rejected_input)

        save.release.set()
        response, _ = await asyncio.wait_for(call, 5)
        # The published boundary is reported as success, never as an
        # interrupted compaction: it exists in the live conversation.
        assert response.summary == SUMMARY
        assert rig.loop.compaction_outcome.published is True
        checkpoints = rig.checkpoints()
        assert [entry.kind for entry in checkpoints] == ["compaction"]
        assert checkpoints[0].details == {"summaryLength": len(SUMMARY)}
        assert rig.execution.active is None
    finally:
        await rig.aclose()


async def _drain_until(predicate: Any) -> None:
    for _ in range(1000):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition was not observed")
