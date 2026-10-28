"""C3 — consume the override permit at transport start; prohibit forced replay.

The C1 permit minted for an explicit un-stick request is CONSUMED at the
ACTUAL transport start — the backend's real transport-start notification
(``notify_request_started``), never at coordinator entry — and released when
the attempt ends before any transport started (preparation failure,
pretransport backend error, pretransport cancellation). A bound permit also
denies the logical call's bounded in-call empty replay, and the forced
candidate (the one serving the permit's canonical endpoint) is terminal: its
failure is raised, never replayed on an alternative provider. Ordinary calls
(no binding) keep the empty replay and the full failover loop.

All backends are offline stubs; no network, no real sleeps.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import aclosing
from dataclasses import replace
from types import MethodType
from typing import Any, cast

import httpx
import pytest

from chartreux.core.agent_loop.errors import (
    EmptyLLMResponseError,
    EmptyReplayBudgetStop,
    ImagesNotSupportedError,
    OverrideStartDeniedError,
)
from chartreux.core.agent_loop.llm_gateway import (
    CallResources,
    CompletionInputs,
    LLMGateway,
)
from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
from chartreux.core.config.models import ModelConfig
from chartreux.core.llm.backend.generic import notify_request_started
from chartreux.core.llm.failures import FailureCategory, FailureInfo, RequestRetryBudget
from chartreux.core.llm_models import (
    ImageAttachment,
    InlineImageSource,
    LLMChunk,
    LLMMessage,
    Role,
)
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.model_catalog.schema import ModelCatalog
from chartreux.core.recovery_override import (
    CooldownOverrideLedger,
    OverrideDenial,
    OverridePhase,
)
from chartreux.core.session_types import AgentStats
from chartreux.core.utils.retry import async_generator_retry, async_retry
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

ENDPOINT_URL = "https://api.shared.test/v1"
OTHER_ENDPOINT_URL = "https://second.test"


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

    async def complete(self, **kwargs: Any) -> LLMChunk:
        if self.notify_first:
            self.transports += 1
            notify_request_started()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")  # pragma: no cover

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        if self.notify_first:
            self.transports += 1
            notify_request_started()
        await asyncio.Event().wait()
        yield mock_llm_chunk(content="")  # pragma: no cover - unreachable


class RealRetryBackend(FakeBackend):
    """A backend whose request paths carry the REAL retry decorators.

    Mirrors the GenericBackend construction (chartreux/core/llm/backend/
    generic.py:378-393): each send is wrapped in the real
    ``async_retry``/``async_generator_retry`` with the budget the
    coordinator selected at backend construction, so a retryable transport
    error retries while that budget allows — the layer the forced
    attempt's zeroed budget must disable.
    """

    def __init__(self, budget: RequestRetryBudget, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.transports = 0
        self._send = async_retry(tries=None, budget=budget)(self._send_once)
        self._send_stream = async_generator_retry(tries=None, budget=budget)(
            self._send_stream_once
        )

    async def _send_once(self, **kwargs: Any) -> LLMChunk:
        self.transports += 1
        notify_request_started()
        return await super().complete(**kwargs)

    async def _send_stream_once(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        self.transports += 1
        notify_request_started()
        async for chunk in super().complete_streaming(**kwargs):
            yield chunk

    async def complete(self, **kwargs: Any) -> LLMChunk:
        return await self._send(**kwargs)

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk]:
        async for chunk in self._send_stream(**kwargs):
            yield chunk


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


def _model() -> ModelConfig:
    return ModelConfig(
        name="test", provider="test", alias="test", input_price=1, output_price=2
    )


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


def _resources(
    backend: FakeBackend, *, permit: Any = None, ledger: Any = None
) -> CallResources:
    resources = CallResources(
        backend=backend, stats=AgentStats(), process_message=lambda m: m
    )
    if permit is not None:
        resources = replace(resources, override_permit=permit, override_ledger=ledger)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_permit_is_consumed_at_actual_transport_start(streaming: bool) -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = TransportBackend([mock_llm_chunk(content="answer")])
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    result = await _gateway_call(streaming, _inputs(), resources, outcomes)

    assert result[-1].message.content == "answer"
    assert len(backend.requests_messages) == 1
    # Consumed at the actual transport start, exactly once, never released.
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert starts[0].permit is permit
    assert releases == []
    # Consumed: the permit no longer bypasses and a late release is denied.
    assert ledger.permit_outstanding(permit) is False
    assert ledger.release(permit).denial is OverrideDenial.UNKNOWN_PERMIT


@pytest.mark.asyncio
async def test_repeated_transport_notification_consumes_once() -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)

    class DoubleNotifyBackend(TransportBackend):
        async def complete(self, **kwargs: Any) -> LLMChunk:
            notify_request_started()
            notify_request_started()
            return await super().complete(**kwargs)

    backend = DoubleNotifyBackend([mock_llm_chunk(content="answer")])
    outcomes: list[Any] = []
    await _gateway_call(
        False, _inputs(), _resources(backend, permit=permit, ledger=ledger), outcomes
    )
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_preparation_failure_releases_the_reservation(streaming: bool) -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = TransportBackend([mock_llm_chunk(content="answer")])
    image = ImageAttachment(
        source=InlineImageSource(data="image"), alias="image.png", mime_type="image/png"
    )
    inputs = _inputs(
        model=_model().model_copy(update={"supports_images": False}),
        messages=(LLMMessage(role=Role.user, content="look", images=[image]),),
    )
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    with pytest.raises(ImagesNotSupportedError):
        await _gateway_call(streaming, inputs, resources, outcomes)

    # The failure happened before any transport start: no consumption, and
    # the reservation frees without consuming the start allowance.
    assert backend.transports == 0
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_pretransport_backend_failure_releases_the_reservation(
    streaming: bool,
) -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    # FakeBackend raises BEFORE the transport-start notification.
    backend = FakeBackend(exception_to_raise=RuntimeError("prep boom"))
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    with pytest.raises(RuntimeError, match="prep boom"):
        await _gateway_call(streaming, _inputs(), resources, outcomes)

    assert backend.requests_messages == []
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_retryable_error_consumes_and_does_not_retry(streaming: bool) -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = TransportBackend(exception_to_raise=httpx.ConnectError("down"))
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    # A non-empty retryable transport failure maps to the gateway's
    # RuntimeError wrapper (cause-chained to the ConnectError).
    with pytest.raises(RuntimeError, match="down"):
        await _gateway_call(streaming, _inputs(), resources, outcomes)

    # One transport start; the start stays consumed; no replay followed.
    assert backend.transports == 1
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_empty_response_consumes_and_never_replays(streaming: bool) -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    # The second stream is the empty replay that MUST NOT happen.
    backend = TransportBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="answer")],
    ])
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    with pytest.raises(EmptyLLMResponseError) as excinfo:
        await _gateway_call(streaming, _inputs(), resources, outcomes)

    assert type(excinfo.value) is EmptyReplayBudgetStop
    assert backend.transports == 1
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_ordinary_call_keeps_the_empty_replay(streaming: bool) -> None:
    """Without a binding the bounded in-call empty replay still runs."""
    backend = TransportBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="answer")],
    ])
    resources = _resources(backend)
    outcomes: list[Any] = []

    result = await _gateway_call(streaming, _inputs(), resources, outcomes)

    assert result[-1].message.content == "answer"
    assert backend.transports == 2


@pytest.mark.asyncio
async def test_pretransport_cancellation_releases_the_reservation() -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = HangingBackend()
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    task = asyncio.ensure_future(_gateway_call(False, _inputs(), resources, outcomes))
    await _cancel_when(task, lambda: False)

    assert backend.transports == 0
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT


@pytest.mark.asyncio
async def test_post_start_cancellation_consumes_and_never_releases() -> None:
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    backend = HangingBackend(notify_first=True)
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    task = asyncio.ensure_future(_gateway_call(True, _inputs(), resources, outcomes))
    await _cancel_when(task, lambda: backend.transports > 0)

    assert backend.transports == 1
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


def _catalog() -> CatalogSnapshot:
    return CatalogSnapshot(
        ModelCatalog.model_validate({
            "providers": {
                "test-first": {"api_base": ENDPOINT_URL},
                "test-second": {"api_base": OTHER_ENDPOINT_URL},
            },
            "models": {
                "base": {
                    "deployments": [
                        {"provider": "test-first", "name": "first"},
                        {"provider": "test-second", "name": "second"},
                    ]
                }
            },
            "roles": {},
        }),
        "test",
    )


def _agent(*, streaming: bool = False, retry_window: float | None = None) -> Any:
    snapshot = _catalog()
    raw: dict[str, Any] = {"active_model": "base"}
    if retry_window is not None:
        # A small shared retry window keeps a REGRESSED forced attempt (a
        # non-zeroed budget) failing fast: the first backoff (~0.5s) fits
        # inside it, so the unfixed layer retries once before giving up.
        raw["api_retry_max_elapsed_time"] = retry_window
    config = ChartreuxConfigSchema.model_validate(
        raw, context={"catalog_snapshot": snapshot}
    ).attach_catalog_snapshot(snapshot)
    return build_test_agent_loop(
        config=config, backend=FakeBackend(), enable_streaming=streaming
    )


def _failover_backends(agent: Any, first: FakeBackend, second: FakeBackend) -> None:
    def backend_for_attempt(
        _self: Any, model: ModelConfig, _budget: object
    ) -> FakeBackend:
        return {"test-first": first, "test-second": second}[model.provider]

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]


def _messages() -> list[LLMMessage]:
    return [LLMMessage(role=Role.user, content="hello")]


async def _coordinator_call(agent: Any, streaming: bool, **kwargs: Any) -> Any:
    coordinator = agent._completion_coordinator
    common = dict(
        messages=_messages(),
        tools=None,
        tool_choice=None,
        call_type=None,
        fallback_model=agent.config.get_active_model(),
    )
    if streaming:
        stream = coordinator.attempt(
            transcript=True, streaming=True, **common, **kwargs
        )
        chunks = []
        async with aclosing(stream) as it:
            async for chunk in it:
                chunks.append(chunk)
        return chunks
    return [await coordinator.complete(transcript=False, **common, **kwargs)]


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_forced_candidate_runs_and_consumes_at_transport_start(
    streaming: bool,
) -> None:
    agent = _agent(streaming=streaming)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    first = TransportBackend([mock_llm_chunk(content="answer-first")])
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)

    result = await _coordinator_call(
        agent, streaming, override_permit=permit, override_ledger=ledger
    )

    assert result[-1].message.content == "answer-first"
    # The forced candidate (the permit's endpoint) served the call; the
    # alternative provider was never touched.
    assert first.transports == 1
    assert second.transports == 0
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_forced_attempt_never_fails_over_to_the_alternative_provider(
    streaming: bool,
) -> None:
    agent = _agent(streaming=streaming)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    first = TransportBackend(exception_to_raise=httpx.ConnectError("down"))
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)

    # The mapped wrapper carries the original message; the forced attempt is
    # terminal either way.
    with pytest.raises(RuntimeError, match="down"):
        await _coordinator_call(
            agent, streaming, override_permit=permit, override_ledger=ledger
        )

    # The forced attempt is terminal: the alternative provider never ran.
    assert first.transports == 1
    assert second.transports == 0
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_forced_pretransport_failure_releases_and_stops(streaming: bool) -> None:
    agent = _agent(streaming=streaming)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    first = FakeBackend(exception_to_raise=RuntimeError("prep boom"))
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)

    with pytest.raises(Exception, match="prep boom"):
        await _coordinator_call(
            agent, streaming, override_permit=permit, override_ledger=ledger
        )

    assert second.transports == 0
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_forced_empty_response_consumes_and_never_replays_or_fails_over(
    streaming: bool,
) -> None:
    agent = _agent(streaming=streaming)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    first = TransportBackend([
        [mock_llm_chunk(content="")],
        [mock_llm_chunk(content="answer-first")],
    ])
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)

    with pytest.raises(EmptyLLMResponseError) as excinfo:
        await _coordinator_call(
            agent, streaming, override_permit=permit, override_ledger=ledger
        )

    assert type(excinfo.value) is EmptyReplayBudgetStop
    assert first.transports == 1
    assert second.transports == 0
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_ordinary_failover_is_unaffected(streaming: bool) -> None:
    """No binding: the full candidate loop (failover) still runs."""
    agent = _agent(streaming=streaming)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    first = FakeBackend(exception_to_raise=httpx.ConnectError("down"))
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)

    result = await _coordinator_call(agent, streaming)

    assert result[-1].message.content == "answer-second"
    assert second.transports == 1


def test_enumeration_sees_the_forced_deployment_through_the_binding() -> None:
    agent = _agent()
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    coordinator = agent._completion_coordinator
    messages = _messages()
    active_model = agent.config.get_active_model()

    # Without the binding the cooled-down deployment stays excluded.
    assert [
        candidate.resolved.deployment.provider
        for candidate in coordinator.failover_candidates(messages, active_model)
    ] == ["test-second"]

    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    coordinator._active_override = (permit, ledger)
    try:
        assert [
            candidate.resolved.deployment.provider
            for candidate in coordinator.failover_candidates(messages, active_model)
        ] == ["test-first", "test-second"]
    finally:
        coordinator._active_override = None


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_mismatched_permit_is_never_bound_and_gets_released(
    streaming: bool,
) -> None:
    """A permit whose endpoint no deployment serves never binds to a
    candidate's transport; the safety net releases it so the in-flight slot
    cannot strand.
    """
    agent = _agent(streaming=streaming)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger, endpoint="https://nowhere.test")
    starts, releases = _spy(ledger)
    first = TransportBackend([mock_llm_chunk(content="answer-first")])
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)

    result = await _coordinator_call(
        agent, streaming, override_permit=permit, override_ledger=ledger
    )

    # The permit's endpoint is not among the surviving candidates... the
    # cooled-down test-first stays excluded, the ordinary candidate serves
    # the call WITHOUT the binding, and the reservation is released.
    assert result[-1].message.content == "answer-second"
    assert starts == []
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False


# ---------------------------------------------------------------------------
# Production wiring: measured outcomes drive the ledger's episode lifecycle
# (DEFECT-1 regression), and the gateway honors start denials at transport
# start (DEFECT-2 regression: both stranding windows plus the safety net)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_measured_outcomes_drive_the_ledger_episode_lifecycle(
    streaming: bool,
) -> None:
    """DEFECT-1 regression (production wiring): every measured endpoint
    failure/recovery feeds the root-shared override ledger through the
    coordinator — a real measured failure opens the endpoint's episode (a
    subsequent reserve mints), and a real measured recovery retires it.
    """
    agent = _agent(streaming=streaming)
    ledger = agent.config_orchestrator.cooldown_override_ledger
    registry = agent.config_orchestrator.availability_registry
    # Lift the cooldown duration so the failed deployment is selectable
    # again once its measured recovery lands.
    registry.initial_cooldown = 0

    # No measured failure yet: no episode, so a reserve is denied.
    assert ledger.reserve(ENDPOINT_URL).denial is OverrideDenial.NO_ACTIVE_EPISODE

    # A real measured failure on the first deployment (a failover-eligible
    # CONNECTION error) opens the episode on the root-shared ledger through
    # the production coordinator path; the failover to the second deployment
    # succeeds on the OTHER endpoint.
    first = TransportBackend(exception_to_raise=httpx.ConnectError("down"))
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    _failover_backends(agent, first, second)
    result = await _coordinator_call(agent, streaming)
    assert result[-1].message.content == "answer-second"

    # The production path opened the episode: the mint now succeeds.
    minted = ledger.reserve(ENDPOINT_URL)
    assert minted.permit is not None
    ledger.release(minted.permit)

    # A real measured recovery on the SAME endpoint (the first deployment
    # serving the next call) retires the episode.
    recovered = TransportBackend([mock_llm_chunk(content="answer-first")])
    _failover_backends(agent, recovered, second)
    result = await _coordinator_call(agent, streaming)
    assert result[-1].message.content == "answer-first"
    assert ledger.reserve(ENDPOINT_URL).denial is OverrideDenial.NO_ACTIVE_EPISODE


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_episode_closed_at_transport_start_aborts_before_transport(
    streaming: bool,
) -> None:
    """DEFECT-2 regression (stranding window A): a measured endpoint
    recovery lands between reserve and transport start. The gateway honors
    the ledger's EPISODE_CLOSED denial — the call aborts BEFORE any
    transport (no request is recorded) and the pretransport release frees
    the in-flight slot, so a genuinely new failure's reserve mints.
    """
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    # A measured endpoint recovery lands between reserve and transport start.
    assert ledger.note_endpoint_recovery(ENDPOINT_URL) is True
    starts, releases = _spy(ledger)
    backend = TransportBackend([mock_llm_chunk(content="answer")])
    resources = _resources(backend, permit=permit, ledger=ledger)
    outcomes: list[Any] = []

    with pytest.raises(OverrideStartDeniedError) as exc_info:
        await _gateway_call(streaming, _inputs(), resources, outcomes)

    assert exc_info.value.denial is OverrideDenial.EPISODE_CLOSED
    # No transport went out: the backend recorded no request.
    assert backend.requests_messages == []
    # The start was denied (never counted) and the reservation was released
    # through the pretransport path: the in-flight slot is free.
    assert len(starts) == 1
    assert starts[0].denial is OverrideDenial.EPISODE_CLOSED
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT
    assert ledger.permit_outstanding(permit) is False
    # A genuinely new measured failure opens a fresh episode and its first
    # reserve mints: the un-stick override is alive again (no stranded slot).
    assert ledger.note_endpoint_failure(ENDPOINT_URL) is True
    fresh = ledger.reserve(ENDPOINT_URL)
    assert fresh.permit is not None
    ledger.release(fresh.permit)


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_spacing_denial_at_transport_start_aborts_before_transport(
    streaming: bool,
) -> None:
    """DEFECT-2 regression (stranding window B): the spacing floor denies
    the transport start (start #1 at T=29, reserve #2 admitted at T=30 by
    the pressure bound, transport start at T=30.5). The gateway honors the
    SPACING_EXCEEDED denial — no second transport runs inside the spacing
    window, the start is never counted, and the reservation is released
    pretransport so the slot frees.
    """
    now = [0.0]
    ledger = CooldownOverrideLedger(clock=lambda: now[0])
    assert ledger.note_endpoint_failure(ENDPOINT_URL) is True
    # Start #1 at T=29 (admitted at T=0 by the pressure bound).
    first = ledger.reserve(ENDPOINT_URL)
    assert first.permit is not None
    now[0] = 29.0
    started = ledger.start(first.permit)
    assert started.phase is OverridePhase.STARTED
    # Reserve #2 at T=30: the pressure bound admits it (1 + floor(30/30) = 2).
    now[0] = 30.0
    second = ledger.reserve(ENDPOINT_URL)
    assert second.permit is not None
    assert second.denial is None
    starts, releases = _spy(ledger)
    # The transport start at T=30.5: 1.5s after start #1, far inside the
    # 30s floor — the ledger denies it and the gateway aborts.
    now[0] = 30.5
    backend = TransportBackend([mock_llm_chunk(content="answer")])
    resources = _resources(backend, permit=second.permit, ledger=ledger)
    outcomes: list[Any] = []

    with pytest.raises(OverrideStartDeniedError) as exc_info:
        await _gateway_call(streaming, _inputs(), resources, outcomes)

    assert exc_info.value.denial is OverrideDenial.SPACING_EXCEEDED
    # No second transport went out and the start was never counted: the
    # >=30s spacing invariant holds (only start #1 is recorded).
    assert backend.requests_messages == []
    status = ledger.episode_status(ENDPOINT_URL)
    assert status is not None
    assert status.starts_used == 1
    # The denied start's reservation was released pretransport: the slot
    # frees without refunding the consumed start.
    assert len(starts) == 1
    assert starts[0].denial is OverrideDenial.SPACING_EXCEEDED
    assert len(releases) == 1
    assert releases[0].phase is OverridePhase.RELEASED_PRETRANSPORT
    free = ledger.reserve(ENDPOINT_URL)
    assert free.permit is not None
    ledger.release(free.permit)


@pytest.mark.asyncio
async def test_safety_net_releases_a_stranded_episode_closed_permit() -> None:
    """DEFECT-2 regression (safety net): a permit stranded by an episode
    close is no longer "outstanding" by the episode check, but it is still
    the endpoint's live reservation. The coordinator's safety net releases
    on the ledger's TOKEN check, so the in-flight slot can never strand.
    """
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    # The episode closes while the permit is outstanding: stranded.
    assert ledger.note_endpoint_recovery(ENDPOINT_URL) is True
    assert ledger.permit_outstanding(permit) is False
    coordinator = _agent()._completion_coordinator
    coordinator._release_unconsumed_override(permit, ledger)
    # The slot is free: a genuinely new measured failure opens a fresh
    # episode whose first reserve mints (a stranded slot would deny
    # ATTEMPT_IN_FLIGHT).
    assert ledger.note_endpoint_failure(ENDPOINT_URL) is True
    fresh = ledger.reserve(ENDPOINT_URL)
    assert fresh.permit is not None
    ledger.release(fresh.permit)


# ---------------------------------------------------------------------------
# Retry-budget defect regression: the forced attempt never retries — the
# backend's REAL retry decorators cannot fire under a permit.
# ---------------------------------------------------------------------------


def test_forced_branch_zeroes_the_budget_but_lone_deployment_keeps_it() -> None:
    """The forced-specific branch: a forced attempt's budget is zero even
    when it is also the lone candidate, while the lone-deployment rule
    still returns the shared budget unchanged.
    """
    coordinator = _agent()._completion_coordinator
    parent = RequestRetryBudget(10.0)

    forced = coordinator.attempt_retry_budget(
        parent, has_alternative=False, forced=True
    )
    lone = coordinator.attempt_retry_budget(parent, has_alternative=False)

    assert forced.exhausted is True
    assert forced.max_elapsed_time == 0.0
    assert lone is parent


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_forced_attempt_zeroes_the_backend_retry_budget(streaming: bool) -> None:
    """Retry-budget defect regression: the forced candidate's backend is
    constructed with a ZERO retry budget, so its real retry decorators
    cannot fire under a permit — exactly one transport on a retryable
    error, whatever the shared retry window allows.
    """
    agent = _agent(streaming=streaming, retry_window=1.0)
    agent.config_orchestrator.availability_registry.record_failure(
        "base", "test-first", FailureInfo(FailureCategory.TIMEOUT)
    )
    ledger = CooldownOverrideLedger()
    permit = _reserved_permit(ledger)
    starts, releases = _spy(ledger)
    second = TransportBackend([mock_llm_chunk(content="answer-second")])
    budgets: list[RequestRetryBudget] = []
    created: list[RealRetryBackend] = []

    def backend_for_attempt(
        _self: Any, model: ModelConfig, budget: object
    ) -> FakeBackend:
        budgets.append(cast(RequestRetryBudget, budget))
        if model.provider == "test-first":
            backend = RealRetryBackend(
                cast(RequestRetryBudget, budget),
                exception_to_raise=httpx.ConnectError("down"),
            )
            created.append(backend)
            return backend
        return second

    agent._backend_for_attempt = MethodType(backend_for_attempt, agent)  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="down"):
        await _coordinator_call(
            agent, streaming, override_permit=permit, override_ledger=ledger
        )

    # The forced candidate's construction budget is zeroed (distinct from
    # the lone-deployment rule): the real retry decorators cannot fire
    # under a permit — exactly one transport on a retryable error.
    assert len(budgets) == 1
    assert budgets[0].exhausted is True
    assert budgets[0].max_elapsed_time == 0.0
    assert len(created) == 1
    assert created[0].transports == 1
    # The forced attempt stays terminal: the alternative provider never ran.
    assert second.transports == 0
    # The single transport start consumed the permit, exactly once.
    assert len(starts) == 1
    assert starts[0].phase is OverridePhase.STARTED
    assert releases == []
