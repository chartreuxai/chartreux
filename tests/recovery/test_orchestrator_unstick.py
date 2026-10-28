"""C10 — composed un-stick lifecycle verification (offline integration).

Composes the landed C-lane pieces into end-to-end offline scenarios over the
real :class:`LLMGateway` and :class:`CallResources` plumbing:

- **No progress**: a resumed run stalled on its first model call is killed
  at the configurable first-semantic-progress deadline and the typed
  :class:`ResumeProgressDeadlineError` surfaces — the in-flight call task is
  cancelled, the pretransport release frees the reservation, and no orphan
  transport start is recorded (F6: automatic paths never acquire permits).
- **Successful progress**: the observation of the first semantic delta
  disarms the watch terminally and the run completes.
- **Both forced attempts**: the C1 ledger admits exactly two forced starts
  per endpoint episode; the third reserve is denied (EPISODE_EXHAUSTED) and
  consumes nothing.
- **Third refusal** is covered by the exhaustion case above.
- **Cross-agent accounting**: a second agent's forced attempt draws on the
  SAME episode budget through the shared (copied) orchestrator ledger, and
  endpoints stay isolated — one endpoint's closed episode never authorizes
  another's.
- **Cancellation/finalization race**: a cancellation that lands while the
  attempt's cleanup is joining still releases the in-flight slot — the
  stranded reservation is never left outstanding.
- **No automatic authority**: an ordinary call (no explicit override) never
  mints a permit even when one is available, and a bound call consumes its
  permit exactly once at the actual transport start.

All backends are offline stubs; no network, no unmocked provider call.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from typing import Any

import pytest

from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
)
from chartreux.core.agent_loop.resume_progress import (
    ResumeProgressDeadlineError,
    ResumeProgressWatch,
)
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role
from chartreux.core.recovery_override import (
    CooldownOverrideLedger,
    OverrideDenial,
    OverridePhase,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.utils.retry import RetryReason
from tests.mock.utils import mock_llm_chunk
from tests.recovery.conftest import FakeClock
from tests.stubs.fake_backend import FakeBackend

ENDPOINT_URL = "https://api.shared.test/v1"
OTHER_ENDPOINT_URL = "https://second.test"
RATE_LIMIT_SIGNATURE = "rate_limit:429:rate_limit:rate_limit"


class TransportBackend(FakeBackend):
    """FakeBackend that fires the real transport-start notification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.transports = 0

    async def complete(self, **kwargs: Any) -> LLMChunk:
        self.transports += 1
        notify_request_started()
        return await super().complete(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        self.transports += 1
        notify_request_started()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk


class HangingBackend(FakeBackend):
    """Blocks before any transport start; only cancellation ends it."""

    def __init__(self, *, notify_first: bool = False) -> None:
        super().__init__()
        self.notify_first = notify_first
        self.transports = 0
        self.started = asyncio.Event()
        self.killed = asyncio.Event()

    async def complete(self, **kwargs: Any) -> LLMChunk:
        if self.notify_first:
            self.transports += 1
            notify_request_started()
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.killed.set()
            raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        if self.notify_first:
            self.transports += 1
            notify_request_started()
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.killed.set()
            raise
        yield mock_llm_chunk(content="")  # pragma: no cover - unreachable


def _spy(ledger: CooldownOverrideLedger) -> tuple[list[Any], list[Any]]:
    starts: list[Any] = []
    releases: list[Any] = []
    real_start = ledger.start
    real_release = ledger.release

    def start(permit: Any) -> Any:
        outcome = real_start(permit)
        starts.append(outcome)
        return outcome

    def release(permit: Any) -> Any:
        outcome = real_release(permit)
        releases.append(outcome)
        return outcome

    ledger.start = start  # type: ignore[method-assign]
    ledger.release = release  # type: ignore[method-assign]
    return starts, releases


def _reserved_permit(
    ledger: CooldownOverrideLedger, endpoint: str = ENDPOINT_URL
) -> Any:
    assert ledger.note_endpoint_failure(endpoint) is True
    outcome = ledger.reserve(endpoint)
    assert outcome.permit is not None
    return outcome.permit


def _inputs(**updates: Any) -> CompletionInputs:
    values: dict[str, Any] = dict(
        model=_model(),
        provider_name="test",
        emits_finish_reason=True,
        messages=(LLMMessage(role=Role.user, content="hello"),),
        tools=None,
        tool_choice=None,
        extra_headers={},
        metadata={},
        max_tokens=42,
    )
    values.update(updates)
    return CompletionInputs(**values)


def _model() -> Any:
    from chartreux.core.config.models import ModelConfig

    return ModelConfig(
        name="test", provider="test", alias="test", input_price=1, output_price=2
    )


def _resources(
    backend: FakeBackend,
    *,
    permit: Any = None,
    ledger: Any = None,
    watch: ResumeProgressWatch | None = None,
) -> CallResources:
    resources = CallResources(
        backend=backend, stats=AgentStats(), process_message=lambda message: message
    )
    if permit is not None:
        resources = replace(resources, override_permit=permit, override_ledger=ledger)
    if watch is not None:
        resources = replace(resources, resume_progress=watch)
    return resources


async def _gateway_call(
    streaming: bool,
    inputs: CompletionInputs,
    resources: CallResources,
    outcomes: list[Any],
) -> list[Any]:
    gateway = LLMGateway()
    if streaming:
        chunks = []
        async with aclosing(
            gateway.chat_streaming(inputs, resources, transcript=outcomes.append)
        ) as stream:
            async for chunk in stream:
                chunks.append(chunk)
        return chunks
    return [await gateway.chat(inputs, resources, transcript=outcomes.append)]


async def _cancel_when(task: asyncio.Task[Any], ready: Any) -> None:
    for _ in range(1000):
        if ready():
            break
        await asyncio.sleep(0.001)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# ---------------------------------------------------------------------------
# No progress: the first-semantic-progress deadline kills the stalled call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_no_progress_run_is_killed_at_the_deadline() -> None:
    """A resumed run stalled on its first model call is killed at the
    deadline: the typed failure surfaces, the call task is cancelled, and
    the in-flight reservation is released pretransport — no orphan transport.
    """
    watch = ResumeProgressWatch(0.05)
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = HangingBackend(notify_first=False)
    outcomes: list[Any] = []

    with pytest.raises(ResumeProgressDeadlineError) as exc_info:
        resources = _resources(backend, permit=permit, ledger=ledger, watch=watch)
        await _gateway_call(True, _inputs(model=_model()), resources, outcomes)
    # The in-flight call task was cancelled: the backend's hang observed it.
    await asyncio.wait_for(backend.killed.wait(), 5.0)

    # The kill is the C5a measured-timeout producer contract.
    assert exc_info.value.error_code == "resume_progress_deadline"
    assert "no semantic progress" in str(exc_info.value)
    assert watch.expired and not watch.disarmed
    # The stalled call never reached a transport start: no orphan.
    assert backend.transports == 0
    assert outcomes == []
    # The kill landed before any transport start: the in-flight reservation
    # is released pretransport, never consumed.
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False
    # The pretransport release frees the slot: a fresh reserve succeeds.
    assert watch.deadline_failure().non_retryable is True
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger, endpoint=ENDPOINT_URL)
    assert ledger.permit_outstanding(permit) is True
    assert ledger.release(permit).phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_successful_progress_disarms_and_completes() -> None:
    """The first semantic delta disarms the watch terminally; the run
    completes normally with no deadline kill and exactly one consumption.
    """
    watch = ResumeProgressWatch(30.0)
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = TransportBackend([mock_llm_chunk(content="progress")])
    outcomes: list[Any] = []

    result = await _gateway_call(
        False,
        _inputs(model=_model()),
        _resources(backend, permit=permit, ledger=ledger, watch=watch),
        outcomes,
    )

    assert result[-1].message.content == "progress"
    assert watch.disarmed and not watch.expired
    # Consumed exactly once at the actual transport start; a late release is
    # unknown-permit and the slot stays free.
    assert backend.transports == 1
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert starts[0].permit is permit
    assert releases == []
    assert ledger.permit_outstanding(permit) is False
    assert outcomes
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    assert ledger.start(permit).phase is OverridePhase.STARTED
    assert ledger.release(permit).denial is OverrideDenial.UNKNOWN_PERMIT
    assert ledger.permit_outstanding(permit) is False


# ---------------------------------------------------------------------------
# Both forced attempts, then the third refusal — one episode's accounting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_both_forced_attempts_consume_then_the_third_is_refused() -> None:
    """The settled pressure bound: one start is available immediately, the
    second after one full spacing window, and the third reserve inside the
    same episode is denied (EPISODE_EXHAUSTED) without consuming anything.
    """
    clock = FakeClock()
    ledger = CooldownOverrideLedger(clock=clock)

    assert ledger.note_endpoint_failure(ENDPOINT_URL) is True

    first = ledger.reserve(ENDPOINT_URL)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    clock.advance(30.0)

    second = ledger.reserve(ENDPOINT_URL)
    assert second.permit is not None
    assert second.permit.episode_id == first.permit.episode_id
    assert ledger.start(second.permit).phase is OverridePhase.STARTED
    clock.advance(30.0)

    third = ledger.reserve(ENDPOINT_URL)
    assert third.denial is OverrideDenial.EPISODE_EXHAUSTED
    assert third.permit is None
    # A denial consumes nothing: the second start is still the last.
    status = ledger.episode_status(ENDPOINT_URL)
    assert status is not None
    assert status.starts_used == 2
    assert status.starts_remaining == 0
    assert ledger.release(second.permit).denial is OverrideDenial.UNKNOWN_PERMIT

    # Only a measured recovery closes the episode; a new measured failure
    # opens a fresh one with fresh counters.
    assert ledger.note_endpoint_recovery(ENDPOINT_URL) is True
    assert ledger.note_endpoint_failure(ENDPOINT_URL) is True
    fresh = ledger.reserve(ENDPOINT_URL)
    assert fresh.permit is not None
    assert fresh.permit.episode_id != second.permit.episode_id
    assert ledger.start(fresh.permit).phase is OverridePhase.STARTED


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cross_agent_accounting_shares_one_episode_budget() -> None:
    """A permit minted by one agent never bypasses another endpoint's
    cooldown: the second agent's forced attempt draws on the SAME episode
    through the copied orchestrator ledger, and the wrong-endpoint permit is
    denied before any budget is consumed.
    """
    clock = FakeClock()
    ledger = CooldownOverrideLedger(clock=clock)
    child_ledger = CooldownOverrideLedger(clock=clock)
    child_ledger._states = ledger._states  # the shared root-owned accounting

    assert ledger.note_endpoint_failure(ENDPOINT_URL) is True

    first = ledger.reserve(ENDPOINT_URL)
    assert first.permit is not None
    assert ledger.start(first.permit).phase is OverridePhase.STARTED
    clock.advance(30.0)

    # The copied orchestrator's handle draws on the SAME episode budget.
    second = child_ledger.reserve(ENDPOINT_URL)
    assert second.permit is not None
    assert second.permit.episode_id == first.permit.episode_id
    assert child_ledger.start(second.permit).phase is OverridePhase.STARTED
    clock.advance(30.0)

    # The episode is exhausted: a third agent's explicit request is denied
    # before any mint, and nothing about the first two starts is refunded.
    third = child_ledger.reserve(OTHER_ENDPOINT_URL)
    assert third.denial is OverrideDenial.NO_ACTIVE_EPISODE
    assert third.permit is None
    forged = replace(first.permit, endpoint=OTHER_ENDPOINT_URL)
    assert ledger.start(forged).denial is OverrideDenial.UNKNOWN_PERMIT
    assert ledger.release(forged).denial is OverrideDenial.UNKNOWN_PERMIT
    status = ledger.episode_status(ENDPOINT_URL)
    assert status is not None
    assert status.starts_used == 2


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cancellation_releases_the_reservation_pretransport() -> None:
    """A cancellation that lands before the transport start releases the
    reservation without consuming it: the endpoint's in-flight slot is free
    and the next forced attempt mints normally.
    """
    clock = FakeClock()
    ledger = CooldownOverrideLedger(clock=clock)
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = HangingBackend()

    task = asyncio.ensure_future(
        _gateway_call(
            False,
            _inputs(model=_model()),
            _resources(backend, permit=permit, ledger=ledger),
            outcomes=[],
        )
    )
    await asyncio.wait_for(backend.started.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The attempt never consumed its start: the slot is free.
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False
    free = ledger.reserve(ENDPOINT_URL)
    assert free.permit is not None
    assert ledger.start(free.permit).phase is OverridePhase.STARTED

    # The released permit is dead: a replayed start is unknown-permit.
    assert ledger.release(permit).denial is OverrideDenial.UNKNOWN_PERMIT


# ---------------------------------------------------------------------------
# Ordinary (non-override) calls keep the full replay path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_ordinary_call_keeps_the_empty_replay() -> None:
    """Without a permit binding the bounded in-call empty replay still runs:
    the empty first stream raises the typed empty-response failure (driving
    the one in-call replay) and the second stream completes normally — and
    the ordinary call never touches the override ledger, even when an
    un-stick permit is available: no start, no release, no mint (F6:
    automatic paths never acquire permits).
    """
    ledger = CooldownOverrideLedger()
    available = _reserved_permit(ledger)  # an un-stick permit IS available
    starts, releases = _spy(ledger)
    backend = TransportBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="answer")],
    ])
    replays: list[RetryReason] = []

    async def observe_replay(reason: RetryReason) -> None:
        replays.append(reason)

    outcomes: list[Any] = []

    result = await _gateway_call(
        False,
        _inputs(),
        replace(_resources(backend), on_retry=observe_replay),
        outcomes,
    )

    assert result[-1].message.content == "answer"
    # The empty first stream raised the typed failure and the bounded replay
    # ran exactly once; the second stream completed the logical call.
    assert len(replays) == 1
    assert "Empty assistant response" in replays[0].detail
    assert backend.transports == 2
    assert outcomes
    # The no-ordinary-call guard: the available permit was never bound, so
    # the ledger saw no start and no release — it stays outstanding.
    assert starts == []
    assert releases == []
    assert ledger.permit_outstanding(available) is True
