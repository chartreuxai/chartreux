"""Reusable WP0 recovery-test fixtures (opt-in, function-scoped only).

The v0.5.6 recovery plan builds the reusable fake-clock and
controlled-transport fixtures here so every recovery package shares one
implementation. Two parallel packages (P and G0) land tests in this directory
at the same time, so every fixture in this module is OPT-IN: tests request it
by name. There is deliberately no ``autouse`` fixture and no session-global
patching in this file — a fixture requested by one test must never change the
environment of a concurrently developed test that does not ask for it.

Fixtures:

- ``fake_clock``: a manual monotonic clock with ``advance()`` for deterministic
  cooldown/budget windows (no real sleeps).
- ``controlled_transport``: a factory for :class:`ControlledTransport`, a
  scripted transport that records every transport start (the authoritative
  transport-attempt source), supports per-attempt gates (hold one attempt in
  flight), scripted per-attempt outcomes (chunk streams, mid-stream faults,
  in-stream gates), transport-internal retries with ``on_retry`` notices
  (mirroring the production backend wrappers' ``async_retry``), and an
  ``__aexit__`` counter (borrowed-backend cleanup observation).
- ``clean_observation_store``: clears the private failure-observation store
  before and after the test (test isolation for WP0 F5's carrier).
- ``acp_recovery_agent``: an ACP agent harness whose agent loop is built in
  the test scope, so tests can reach the loop's shared availability registry
  and committed model identity.

All transports here are controlled fakes: they never open a network
connection, so an unmocked provider call cannot occur (fail-closed egress).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Iterable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import pytest

from chartreux.acp.agent import ChartreuxAcpAgent
from chartreux.app_server.local import LocalHarnessOptions
from chartreux.app_server.session import AppServerSession
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.llm_models import LLMChunk, LLMMessage, Role
from chartreux.core.recovery_observation import clear_observations
from chartreux.core.utils.retry import RetryReason
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import start_test_app_server
from tests.stubs.fake_backend import FakeBackend
from tests.stubs.fake_client import FakeClient


class FakeClock:
    """Manual monotonic clock; deterministic without real sleeps."""

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.fixture
def fake_clock() -> FakeClock:
    """Opt-in manual monotonic clock for cooldown/budget windows."""
    return FakeClock()


# One scripted outcome per transport attempt. An outcome is either a
# ``BaseException`` (TERMINAL: raised immediately, no transport-internal
# retry) or a sequence of items, where each item is an ``LLMChunk`` (streamed),
# a ``BaseException`` (raised mid-stream, after the chunks before it), or an
# ``asyncio.Event`` (awaited mid-stream — an in-stream gate). Wrap an
# exception in ``RetryableFault`` to make it retryable transport-internally.


class RetryableFault:
    """Marks a scripted exception as retryable transport-internally.

    A bare ``BaseException`` outcome is terminal. Wrapping it in
    ``RetryableFault`` mirrors the production backend wrapper's
    ``async_retry``: the transport retries while outcomes remain, noticing
    each retry through ``on_retry`` and sleeping ``retry_delay`` between
    attempts (the wrapper's backoff).
    """

    def __init__(self, exc: Exception) -> None:
        self.exc = exc


Outcome = BaseException | RetryableFault | Sequence[Any]


class ControlledTransport(FakeBackend):
    """Scripted controlled transport for WP0 fault injection.

    Every attempt is recorded in ``requests_messages`` BEFORE the scripted
    outcome applies — that list is the authoritative transport-start source
    for the F2 evidence matrix (one entry per real transport attempt).

    Scripted exceptions are retried transport-internally while outcomes
    remain, mirroring the production backend wrappers' ``async_retry``
    (``chartreux/core/llm/backend/mistral.py``, ``generic.py``): each retry is
    noticed through ``on_retry`` (the same channel ``FakeBackend`` uses for
    ``notice_retries``), and ``retry_delay`` models the wrapper's backoff
    sleep between attempts. The gateway itself sees ONE logical call per
    ``complete``/``complete_streaming`` invocation, exactly as with a real
    backend wrapper.

    Attempt indexes are GLOBAL across the transport: every transport start —
    from any logical call, including transport-internal retries — draws the
    next index from one counter. ``gate_attempt(index)`` therefore holds the
    index-th transport start in flight (event barriers instead of real
    cooldown sleeps), scripted outcomes are consumed in transport-wide attempt
    order, and ``attempt_hook`` runs at each attempt start (used to arm the
    shared availability cooldown mid-turn, after the turn's admission has
    already passed).
    """

    def __init__(self, outcomes: Iterable[Outcome] = ()) -> None:
        super().__init__()
        self._outcomes: list[Outcome] = list(outcomes)
        self.started: list[asyncio.Event] = []
        self.releases: list[asyncio.Event] = []
        self._gated: set[int] = set()
        self._attempt_count = 0
        self.aexit_calls = 0
        self.stream_started = asyncio.Event()
        self.stream_closed = asyncio.Event()
        self.retry_delay = 0.0
        self.attempt_hook: Callable[[int], None] | None = None

    def add_outcome(self, outcome: Outcome) -> None:
        """Append one scripted outcome (the next attempt consumes it)."""
        self._outcomes.append(outcome)

    def gate_attempt(self, index: int) -> tuple[asyncio.Event, asyncio.Event]:
        """Hold attempt ``index`` in flight until its release event is set."""
        while len(self.started) <= index:
            self.started.append(asyncio.Event())
            self.releases.append(asyncio.Event())
        self._gated.add(index)
        return self.started[index], self.releases[index]

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.aexit_calls += 1
        return None

    def _outcome(self, index: int) -> Outcome:
        return self._outcomes[index] if index < len(self._outcomes) else []

    async def _attempt(self, messages: Sequence[LLMMessage]) -> Outcome:
        """Record one transport start, honor the gate, return the outcome."""
        index = self._attempt_count
        self._attempt_count += 1
        self._requests_messages.append(list(messages))
        if self.attempt_hook is not None:
            self.attempt_hook(index)
        if index in self._gated:
            self.started[index].set()
            await self.releases[index].wait()
        return self._outcome(index)

    async def _next_outcome(self, messages: Sequence[LLMMessage]) -> Any:
        """Advance the scripted retry plan, noticing each retry."""
        while True:
            outcome = await self._attempt(messages)
            if isinstance(outcome, RetryableFault):
                if self._attempt_count >= len(self._outcomes):
                    raise outcome.exc
                if self.on_retry is not None:
                    await self.on_retry(RetryReason.from_error(outcome.exc))
                if self.retry_delay:
                    await asyncio.sleep(self.retry_delay)
                continue
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    async def complete(
        self, *, messages: Sequence[LLMMessage], **kwargs: Any
    ) -> LLMChunk:
        outcome = await self._next_outcome(messages)
        chunk_agg = LLMChunk(message=LLMMessage(role=Role.assistant))
        for item in outcome:
            if isinstance(item, BaseException):
                raise item
            chunk_agg += item
        return chunk_agg

    async def complete_streaming(
        self, *, messages: Sequence[LLMMessage], **kwargs: Any
    ) -> AsyncGenerator[LLMChunk, None]:
        self.stream_started.set()
        try:
            outcome = await self._next_outcome(messages)
            for item in outcome:
                if isinstance(item, BaseException):
                    raise item
                if isinstance(item, asyncio.Event):
                    await item.wait()
                    continue
                yield item
        finally:
            self.stream_closed.set()


@pytest.fixture
def controlled_transport() -> Callable[..., ControlledTransport]:
    """Opt-in factory for scripted controlled transports."""

    def _make(outcomes: Iterable[Outcome] = ()) -> ControlledTransport:
        return ControlledTransport(outcomes)

    return _make


@pytest.fixture
def clean_observation_store() -> Any:
    """Opt-in isolation for the private failure-observation store."""
    clear_observations()
    yield
    clear_observations()


@dataclass
class AcpRecoveryHarness:
    """ACP agent harness with the agent loop built in the test scope."""

    agent: ChartreuxAcpAgent
    client: FakeClient
    loop: AgentLoop
    backend: ControlledTransport


@pytest.fixture
def acp_recovery_agent() -> AcpRecoveryHarness:
    """Opt-in ACP harness exposing the loop (registry + committed model).

    Mirrors ``tests/acp/conftest.py``'s ``acp_agent_loop`` fixture but builds
    the loop outside ``start_session`` so the test can arm the shared
    availability registry between turns.
    """
    backend = ControlledTransport([[mock_llm_chunk(content="Hi")]])
    loop = build_test_agent_loop(backend=backend, enable_streaming=True)

    async def start_session(options: LocalHarnessOptions) -> AppServerSession:
        return await AppServerSession.start(
            start_test_app_server(loop),
            client_info=options.client.info,
            capabilities=options.client.capabilities,
            session_options=options.session_options,
        )

    agent = ChartreuxAcpAgent(session_starter=start_session)
    client = FakeClient()
    agent.on_connect(client)
    client.on_connect(agent)
    return AcpRecoveryHarness(agent=agent, client=client, loop=loop, backend=backend)
