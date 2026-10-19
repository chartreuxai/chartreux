"""WP4 — the bounded automatic recovery controller for child subagents.

Covers the WP4 acceptance: the controller observes every settled background
run at the monitor's completion transition and — only when
``[recovery.subagents] enabled`` is set EXPLICITLY (never relying on the
development default; WP6 promotes it per OD5) — schedules a cancellable
backoff timer on the registry's clock/wakeup and admits the successor through
the SAME WP2 ``resume_agent`` admission an explicit request uses (one
admission operation, never a second scheduler). The first/second stall use the
configured cooldowns with the configured jitter, a longer Retry-After and any
deployment/endpoint cooldown postpone admission, and ONE episode budget is
shared across the stalls (exhaustion stops recovery; the package-lineage
recurrence counter spans replacements). The controller runs INDEPENDENTLY of
parent turns (proven on a fake overnight idle — no parent model call occurs
between the failure settling and the successor's admission).
Endpoint-correlated throttling blocks recovery onto ALL affected-endpoint
deployments regardless of model/provider naming, from a READ-ONLY eligibility
view that never claims or releases the single-flight probes (cheap in-call
retries stay telemetry only). NO auto-resume for cancellation/budget stops/
authorization denial/preflight/task blockers/permanent config errors/
unknown-internal without a transient cause. The launch_failure policy rows
(handoff §2.3) are BOTH pinned: a transient launch_failure gets ONE bounded
startup retry preserving identity and brief; a REPEATED launch failure STOPS
the relaunch loop. Spend/turn/permission budgets never reset. The
failure/scheduled/successor-start/success/exhaustion milestones emit exactly
once each. The aggregate episode deadline is ADVISORY (Q2-negative: no new
admission after it, never an in-flight kill). Cancellation, release, and
shutdown cancel pending schedules; a config reload never extends an active
episode.

Harness: registry-level (no app server) — a real parent loop drives
``registry.run`` launches, child backends come from a per-loop scripted
factory behind the patched ``chartreux.core.agent_loop._loop.create_backend``
facade, and the controller's timers run on the registry's manual clock behind
a gated wakeup, so every cooldown is fully deterministic. The shared
availability registry is replaced with one on the manual clock (child
orchestrators share it through the config orchestrator copy), so cooldown and
endpoint-correlation gaps are exact. The recoverable fault is the WP1 typed
EMPTY_RESPONSE transient after a settled todo write (tool effects SETTLED →
ELIGIBLE; the same fault with no tool call is RECONCILE_REQUIRED and covered
by the allowlist matrix). The controller's jitter draw is pinned to the exact
midpoint, making every scheduled delay exact.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import logging
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server.models import TextContentBlock
from chartreux.app_server.protocol import TurnStartParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ChartreuxConfigSchema
from chartreux.core.config.models import (
    RecoveryConfig,
    RecoveryEligibleClass,
    RecoverySectionConfig,
)
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.llm_models import LLMChunk, Role
from chartreux.core.model_catalog.availability import AvailabilityRegistry
from chartreux.core.recovery_models import (
    EnvelopeDeployment,
    EnvelopeFailure,
    EnvelopeProgress,
    EnvelopeRecovery,
    EnvelopeRequest,
    FailureClass,
    FailureConfidence,
    FailureEnvelope,
    FailurePhase,
    RecoveryEligibility,
)
from chartreux.core.subagents import (
    CancelOutcome,
    ReleaseAgentOutcome,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
)
from chartreux.core.tools.base import InvokeContext
from tests.app_server._subagent_test_support import (
    _background_result,
    _config,
    _GatedWakeup,
    _ManualClock,
    _record,
    _run,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import mock_llm_chunk
from tests.recovery.test_explicit_resume import (
    _empty_fault,
    _final_outcome,
    _Gated,
    _Outcome,
    _ScriptedBackend,
    _write_outcome,
)
from tests.stubs.fake_backend import FakeBackend

#: The endpoint of the test catalog's only provider (the child's committed
#: deployment lives on ``mistral`` at this api_base).
CHILD_ENDPOINT = "https://api.mistral.ai/v1"

#: The structured signature the completion coordinator derives for a 429
#: ``rate_limit_exceeded`` fault (category:status:code:type).
RATE_LIMIT_SIGNATURE = "rate_limit:429:rate_limit_exceeded:"


@pytest.fixture(autouse=True)
def _controller_events_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """The controller's milestone lines are INFO on the ``vibe`` logger."""
    caplog.set_level(logging.INFO, logger="vibe")


async def _until(predicate: Any, timeout: float = 5.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


def _controller_config(recovery: RecoveryConfig) -> ChartreuxConfigSchema:
    """Test config: the given ``[recovery.subagents]`` policy, todo-enabled
    tools, zeroed idle-retention caps (the reaper's only deadline is the
    suspension TTL).
    """
    config = _config(enabled_tools=["task", "todo"])
    return config.model_copy(
        update={
            "recovery": RecoverySectionConfig(subagents=recovery),
            "subagents": config.subagents.model_copy(
                update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
            ),
        }
    )


class _Factory:
    """Per-loop scripted outcomes behind the patched create_backend facade.

    Scripts bind to loops in launch order — the first call from an unbound
    loop is that loop's construction — and every created backend is recorded.
    ``rearm`` runs on every construction/refresh call (no ``retry_budget``
    kwarg): the repeated-launch-failure test re-arms the shared cooldown at
    resume-time backend refresh.
    """

    def __init__(self, scripts: list[list[_Outcome]], *, rearm: Any = None) -> None:
        self._pending = [list(script) for script in scripts]
        self._bound: dict[int, list[_Outcome]] = {}
        self.created: list[_ScriptedBackend] = []
        self._rearm = rearm

    def __call__(self, **kwargs: Any) -> Any:
        on_retry = kwargs.get("on_retry")
        loop = getattr(on_retry, "__self__", None)
        key = id(loop) if loop is not None else -1
        if key not in self._bound:
            if not self._pending:
                raise AssertionError("create_backend called by an unscripted loop")
            self._bound[key] = self._pending.pop(0)
        if "retry_budget" in kwargs:
            script = self._bound[key]
            if not script:
                raise AssertionError("child backend script exhausted")
            backend = _ScriptedBackend(script.pop(0))
        else:
            if self._rearm is not None:
                self._rearm()
            backend = _ScriptedBackend(None)
        self.created.append(backend)
        return backend


class _ServingParentBackend(FakeBackend):
    """A parent backend that answers every notification turn with terminal
    content.

    The monitor starts a parent turn for every settled child; an empty
    ``FakeBackend`` fails those turns (each adding gateway in-call empty-replay
    telemetry unrelated to the child under test). Serving terminal content
    keeps the parent's model-call surface quiet and deterministic.
    """

    def __init__(self) -> None:
        super().__init__([])
        self._terminal = mock_llm_chunk(content="parent notification handled")

    async def complete(self, **kwargs: Any) -> LLMChunk:
        await self.notice_retries()
        self._requests_messages.append(list(kwargs["messages"]))
        return self._terminal

    async def complete_streaming(self, **kwargs: Any) -> AsyncGenerator[LLMChunk, None]:
        await self.notice_retries()
        self._requests_messages.append(list(kwargs["messages"]))
        yield self._terminal


class _WakeupDriver:
    """Releases the next pending gated wakeup arm for a given delay."""

    def __init__(self, wakeup: _GatedWakeup) -> None:
        self._wakeup = wakeup
        self._released: set[int] = set()

    def release(self, delay: float) -> None:
        for index, call in enumerate(self._wakeup.calls):
            if index not in self._released and call == delay:
                self._released.add(index)
                self._wakeup.gates[index].set()
                return
        raise AssertionError(f"no pending wakeup arm for delay {delay!r}")


class _ControllerHarness:
    """The registry-level controller harness."""

    def __init__(
        self,
        registry: SessionRuntimeRegistry,
        parent: AgentLoop,
        clock: _ManualClock,
        wakeup: _GatedWakeup,
        notifications: list[str],
        parent_backend: FakeBackend,
    ) -> None:
        self.registry = registry
        self.parent = parent
        self.clock = clock
        self.wakeup = wakeup
        self.notifications = notifications
        self.parent_backend = parent_backend
        self.driver = _WakeupDriver(wakeup)

    @property
    def controller(self) -> Any:
        return self.registry._recovery_controller

    @property
    def requester_session_id(self) -> str:
        """The owning parent's session id — the only resume authority."""
        return self.parent.session_id

    @property
    def shared_registry(self) -> AvailabilityRegistry:
        """The availability registry shared by the root and every child."""
        return self.parent.config_orchestrator.availability_registry

    def runs(self, agent_id: str) -> list[RunRecord]:
        record = self.registry._agent_records[agent_id]
        return [
            run for run in [record.current_run, *record.run_history] if run is not None
        ]


async def _controller_registry(
    monkeypatch: pytest.MonkeyPatch, recovery: RecoveryConfig, create_backend: Any
) -> _ControllerHarness:
    """Build the registry-level harness: real parent loop, manual clock, gated
    wakeup, and a manual-clock shared availability registry (child
    orchestrators share it through the config orchestrator copy, so cooldown
    and endpoint-correlation gaps are exact). The controller's jitter draw is
    pinned to the exact midpoint, making every scheduled delay exact.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    parent_backend = _ServingParentBackend()
    parent = build_test_agent_loop(
        config=_controller_config(recovery), backend=parent_backend
    )
    parent.config_orchestrator._availability_registry = AvailabilityRegistry(
        clock=clock
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
    registry._recovery_controller._jitter_draw = lambda: 0.5
    return _ControllerHarness(
        registry, parent, clock, wakeup, notifications, parent_backend
    )


async def _launch(
    harness: _ControllerHarness, task: str, *, tool_call_id: str = "launch"
) -> TaskResult:
    """Launch a retained background worker child."""
    return await _background_result(
        harness.registry,
        TaskArgs(task=task, agent_type="worker", background=True, retain=True),
        InvokeContext(tool_call_id=tool_call_id, session_id=harness.parent.session_id),
    )


async def _suspend_child(
    harness: _ControllerHarness, *, task: str = "fail", tool_call_id: str = "launch"
) -> tuple[str, str, TaskResult, AgentRecord]:
    """Launch a recoverably failing retained child and await its suspension
    (the controller's observation runs in the same locked transition).
    """
    launch = await _launch(harness, task, tool_call_id=tool_call_id)
    assert launch.agent_id is not None and launch.run_id is not None
    failed = await _wait_result(harness.registry, launch.agent_id, launch.run_id)
    assert not failed.completed
    record = harness.registry._agent_records[launch.agent_id]
    await _until(lambda: record.state is _AgentState.SUSPENDED)
    await _until(lambda: bool(harness.wakeup.calls))
    return launch.agent_id, launch.run_id, failed, record


async def _settle(
    harness: _ControllerHarness, agent_id: str, run_id: str
) -> TaskResult:
    """Await a child run's settlement and the record leaving RUNNING."""
    result = await _wait_result(harness.registry, agent_id, run_id)
    record = harness.registry._agent_records[agent_id]
    await _until(lambda: record.state is not _AgentState.RUNNING)
    return result


def _successor_of(
    harness: _ControllerHarness, agent_id: str, failed_run_id: str
) -> RunRecord | None:
    """The admitted successor run for a failed run (stable after settlement:
    the run history keeps it after ``current_run`` is cleared).
    """
    return next(
        (
            run
            for run in harness.runs(agent_id)
            if run.resumed_from_run_id == failed_run_id
        ),
        None,
    )


async def _until_admitted(
    harness: _ControllerHarness, agent_id: str, failed_run_id: str
) -> RunRecord:
    """Await the successor's admission (stable: ``latest_run_id`` changes at
    admission and is never cleared, unlike ``current_run``).
    """
    record = harness.registry._agent_records[agent_id]
    await _until(lambda: record.latest_run_id not in (None, failed_run_id))
    successor = _successor_of(harness, agent_id, failed_run_id)
    assert successor is not None
    return successor


def _parent_idle(harness: _ControllerHarness) -> bool:
    """The monitor's notification condition: the parent has no active turn."""
    root = harness.registry._root
    if root is None:
        return True
    turns = root.turns
    return (
        turns.active_turn is None
        and turns._active_task is None
        and not turns.queue_state.items
    )


def _controller_events(caplog: pytest.LogCaptureFixture) -> list[dict[str, str]]:
    """Parsed recovery-controller milestone lines (the exactly-once events)."""
    events: list[dict[str, str]] = []
    prefix = "recovery-controller event="
    for record in caplog.records:
        message = record.getMessage()
        if prefix not in message:
            continue
        rest = message[message.index(prefix) + len(prefix) :]
        event, _, rest = rest.partition(" ")
        fields: dict[str, str] = {"event": event}
        detail_at = rest.index("detail=")
        for part in rest[:detail_at].split():
            key, _, value = part.partition("=")
            fields[key] = value
        fields["detail"] = rest[detail_at + len("detail=") :]
        events.append(fields)
    return events


def _events_of(caplog: pytest.LogCaptureFixture, event: str) -> list[dict[str, str]]:
    return [item for item in _controller_events(caplog) if item["event"] == event]


def _http_error(
    status: int, *, code: str | None = None, retry_after: str | None = None
) -> httpx.HTTPStatusError:
    """A provider-shaped HTTP fault (status, structured code, Retry-After)."""
    detail = {key: value for key, value in {"code": code}.items() if value}
    request = httpx.Request("POST", "https://provider.invalid/chat")
    response = httpx.Response(
        status,
        request=request,
        headers={"Retry-After": retry_after} if retry_after else None,
        json={"error": detail},
    )
    return httpx.HTTPStatusError("provider error", request=request, response=response)


def _envelope(
    agent_id: str,
    run_id: str,
    *,
    failure_class: FailureClass,
    eligibility: RecoveryEligibility,
    cause_category: FailureCategory | None = None,
) -> FailureEnvelope:
    """A fabricated public failure envelope (the allowlist matrix pins)."""
    return FailureEnvelope(
        event_id=f"failure-event:{agent_id}:{run_id}",
        agent_id=agent_id,
        run_id=run_id,
        recovery_chain_id=f"recovery-chain:{agent_id}:{run_id}",
        observed_at=time.time(),
        failure=EnvelopeFailure(
            failure_class=failure_class,
            cause_category=cause_category,
            phase=FailurePhase.LLM_CALL,
            confidence=FailureConfidence.MEASURED,
        ),
        deployment=EnvelopeDeployment(),
        request=EnvelopeRequest(),
        progress=EnvelopeProgress(),
        recovery=EnvelopeRecovery(eligibility=eligibility, reason_codes=["test-pin"]),
    )


def _fabricated_failure(
    harness: _ControllerHarness,
    *,
    failure_class: FailureClass | None = None,
    eligibility: RecoveryEligibility | None = None,
    cause_category: FailureCategory | None = None,
    state: _AgentState = _AgentState.SUSPENDED,
    with_envelope: bool = True,
) -> tuple[AgentRecord, RunRecord]:
    """A fabricated record + failed run carrying the given envelope."""
    record = _record("agent-fab", clock=harness.clock, state=state, retained=True)
    record.initial_task_summary = "fabricated brief"
    run = _run(
        record, "agent-fab-run-1", started_at=harness.clock(), status=RunStatus.FAILED
    )
    if with_envelope:
        assert failure_class is not None and eligibility is not None
        run.failure = _envelope(
            "agent-fab",
            run.run_id,
            failure_class=failure_class,
            eligibility=eligibility,
            cause_category=cause_category,
        )
        record.recovery_chain_id = run.failure.recovery_chain_id
    record.current_run = run
    record.latest_run_id = run.run_id
    return record, run


async def _observe(
    harness: _ControllerHarness, record: AgentRecord, run: RunRecord
) -> None:
    """Drive the controller's observation hook (the monitor holds the lock)."""
    async with harness.registry._registry_lock:
        harness.controller.observe_run_locked(record, run)


def _enabled_policy(**overrides: Any) -> RecoveryConfig:
    """An EXPLICITLY enabled child recovery policy with exact test cooldowns."""
    kwargs: dict[str, Any] = {
        "enabled": True,
        "auto_retries_max": 2,
        "cooldowns_seconds": [5.0, 10.0],
        "jitter_fraction": 0.0,
    }
    kwargs.update(overrides)
    return RecoveryConfig(**kwargs)


# ---------------------------------------------------------------------------
# The enable flag (round-3 symmetry: set EXPLICITLY, never the default)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_disabled_flag_explicitly_set_never_schedules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """``enabled=False`` (set EXPLICITLY) leaves the controller inert (WP4).

    Warm suspension (WP2/WP3) is deliberately NOT gated on the enable flag:
    the child still suspends — but the controller creates no episode, arms no
    timer, and emits no milestone. WP6's promotion must not break this pin.
    """
    harness = await _controller_registry(
        monkeypatch,
        RecoveryConfig(
            enabled=False, auto_retries_max=2, cooldowns_seconds=[5.0, 10.0]
        ),
        _Factory([[_write_outcome("call-1", "step one"), _empty_fault()]]),
    )
    try:
        assert harness.parent.config.recovery.subagents.enabled is False
        agent_id, run_id, failed, record = await _suspend_child(harness)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        # Suspension still happened — the flag gates only the controller.
        assert record.state is _AgentState.SUSPENDED
        # The controller stayed inert: no episode, no timer, no milestones,
        # and the only wakeup arm is the reaper's suspension TTL.
        assert harness.controller._episodes == {}
        assert _controller_events(caplog) == []
        assert harness.wakeup.calls == [1800.0]
        # Nothing was admitted: the failed run is the only run.
        assert [run.run_id for run in harness.runs(agent_id)] == [run_id]
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# Enabled recovery through the SAME admission (events exactly once, budgets)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_enabled_controller_recovers_through_the_same_admission(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An enabled controller recovers a suspended child via ``resume_agent``.

    The automatic admission goes through the SAME WP2 operation an explicit
    request uses (the spy records the controller's ``auto-recovery:``
    idempotency key); exactly one successor run starts on the SAME agent
    identity with the failed run's linkage; the milestones emit exactly once
    each (the gateway's cheap in-call empty replay stays telemetry only — one
    failure event); and the spend/turn/resume budgets never reset.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "step one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ]
        ]),
    )
    resume_calls: list[str] = []
    original_resume = harness.registry.resume_agent

    async def resume_spy(
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> Any:
        resume_calls.append(idempotency_key)
        return await original_resume(
            agent_id,
            expected_run_id,
            idempotency_key,
            requester_session_id=requester_session_id,
        )

    monkeypatch.setattr(harness.registry, "resume_agent", resume_spy)
    try:
        agent_id, run_id, failed, record = await _suspend_child(harness)
        stats = record.runtime.agent_loop.stats
        steps_after_failure = stats.steps
        # The first stall schedules on the first configured cooldown.
        scheduled = _events_of(caplog, "scheduled")
        assert len(scheduled) == 1
        assert scheduled[0]["detail"] == ("delay=5.000 retry_index=0 deadline=advisory")
        harness.driver.release(5.0)
        successor = await _until_admitted(harness, agent_id, run_id)
        assert successor.resumed_from_run_id == run_id
        assert successor.agent_id == agent_id
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        assert done.completed
        assert done.response == "recovered"
        await _until(lambda: record.state is _AgentState.IDLE)

        # The automatic admission used the SAME operation (one auto-recovery
        # key; no explicit call was made).
        assert len(resume_calls) == 1
        assert resume_calls[0].startswith("auto-recovery:")
        # Exactly one successor run on the same identity.
        runs = harness.runs(agent_id)
        assert [run.run_id for run in runs] == [run_id, successor.run_id]
        # The failed run's stored result stays immutable.
        stored = await harness.registry.get_agent_result(agent_id, run_id)
        assert stored is not None
        assert not stored.completed
        assert stored.failure is not None
        assert failed.failure is not None
        assert stored.failure.event_id == failed.failure.event_id

        # Milestones exactly once each; the in-call empty replay produced a
        # single failure event (telemetry only).
        assert len(_events_of(caplog, "failure")) == 1
        assert len(_events_of(caplog, "scheduled")) == 1
        assert len(_events_of(caplog, "successor_start")) == 1
        assert len(_events_of(caplog, "success")) == 1
        assert _events_of(caplog, "exhaustion") == []
        # The child's empty fault was replayed in-call exactly once (cheap
        # retry = telemetry only); the parent's notification turns complete
        # quietly on the serving parent backend, adding no retry telemetry.
        retries = sum(
            "Retrying LLMGateway._complete_attempt" in item.getMessage()
            for item in caplog.records
        )
        assert retries == 1

        # Budgets never reset: the WP2 resume counter advances, the warm
        # loop's cumulative step counter keeps counting across the resume
        # (never reset to the successor's own single turn), the stats object
        # is the same, and the run-1 tool state survives the resume.
        assert record.same_agent_resumes_used == 1
        assert done.turns_used == 1
        assert stats.steps > steps_after_failure
        assert record.runtime.agent_loop.stats is stats
        messages = record.runtime.agent_loop.messages
        assert any(
            message.role is Role.assistant
            and any(call.function.name == "todo" for call in (message.tool_calls or []))
            for message in messages
        )
        assert harness.controller._episodes == {}
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# Cooldown schedule, ONE shared episode budget, exhaustion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cooldown_schedule_and_shared_budget_stop_at_exhaustion(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The first/second stall use the configured cooldowns; ONE episode budget
    is shared; exhaustion stops recovery (WP4).

    With ``auto_retries_max=2`` and cooldowns ``[5, 10]``: the first stall
    schedules at 5s, the second at 10s (retry_index 1), and the third failure
    exhausts the episode — no third successor, one exhaustion milestone. The
    third failure also exhausts the WP2 ``same_agent_resumes_max`` bound, so
    the retained record returns to IDLE (a further recoverable failure no
    longer suspends — the WP2 minimal-warm-bounds behavior).
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _write_outcome("call-2", "two"),
                _empty_fault(),
                _write_outcome("call-3", "three"),
                _empty_fault(),
            ]
        ]),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        # Stall 1 → cooldown[0]; admit successor 1.
        harness.driver.release(5.0)
        successor_1 = await _until_admitted(harness, agent_id, run_id)
        # Stall 2 → cooldown[1]; admit successor 2.
        await _until(lambda: len(harness.wakeup.calls) >= 3)
        harness.driver.release(10.0)
        successor_2 = await _until_admitted(harness, agent_id, successor_1.run_id)
        # Stall 3 → the shared budget is exhausted: no third successor. The
        # WP2 same-agent resume bound (2) is spent too, so the third failure
        # no longer suspends: the retained record returns to IDLE.
        await _until(lambda: harness.controller._episodes == {})
        await _until(lambda: record.state is _AgentState.IDLE)

        scheduled = _events_of(caplog, "scheduled")
        assert [item["detail"] for item in scheduled] == [
            "delay=5.000 retry_index=0 deadline=advisory",
            "delay=10.000 retry_index=1 deadline=advisory",
        ]
        assert len(_events_of(caplog, "failure")) == 3
        assert len(_events_of(caplog, "successor_start")) == 2
        assert _events_of(caplog, "success") == []
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "auto_retries_exhausted"
        # Exactly two successors; the record parks IDLE (never a third run).
        runs = harness.runs(agent_id)
        assert [run.run_id for run in runs] == [
            run_id,
            successor_1.run_id,
            successor_2.run_id,
        ]
        assert record.state is _AgentState.IDLE
        assert record.last_run_status is RunStatus.FAILED
        assert record.same_agent_resumes_used == 2
        assert harness.controller._episodes == {}
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_capacity_rejection_stops_the_automatic_episode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Review R8: a capacity/concurrency admission rejection (a RuntimeError
    from the shared admission) stops the automatic episode with an
    exhaustion milestone instead of dying mid-timer — and the rejected
    agent admits no successor.
    """
    gate = asyncio.Event()
    gated = _Gated(_final_outcome("a recovered"), gate)
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(recovery_concurrency=1),
        _Factory([
            # A: settles a tool effect, fails (suspends), and its resumed
            # successor stays in flight (gated) — the recovery concurrency
            # cap (1) stays occupied when B's timer fires.
            [_write_outcome("call-1", "step one"), _empty_fault(), gated],
            # B: settles a tool effect and fails (suspends).
            [
                _write_outcome("call-1", "step one"),
                _empty_fault(),
                _final_outcome("b done"),
            ],
        ]),
    )
    try:
        agent_a, run_a, _failed_a, record_a = await _suspend_child(
            harness, task="fail a"
        )
        agent_b, run_b, _failed_b, record_b = await _suspend_child(
            harness, task="fail b", tool_call_id="launch-b"
        )
        # Both episodes scheduled on the first configured cooldown (5.0).
        await _until(lambda: len(harness.wakeup.calls) >= 2)
        # A's timer fires first: the successor is admitted and stays active.
        harness.driver.release(5.0)
        await _until(lambda: gated.started.is_set())
        successor_a = await _until_admitted(harness, agent_a, run_a)
        assert successor_a.resumed_from_run_id == run_a
        # B's timer fires: the admission rejects on the occupied concurrency
        # cap, and the episode stops with an exhaustion milestone.
        harness.driver.release(5.0)
        await _until(lambda: agent_b not in harness.controller._episodes)
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["agent_id"] == agent_b
        assert exhaustion[0]["detail"] == "admission_capacity_rejected"
        # The rejected agent admitted no successor and stays suspended.
        assert _successor_of(harness, agent_b, run_b) is None
        assert record_b.state is _AgentState.SUSPENDED
        assert record_a.state is not _AgentState.SUSPENDED
    finally:
        gate.set()
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# A longer Retry-After postpones admission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_longer_retry_after_postpones_admission(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A longer server-directed Retry-After postpones the schedule (WP4).

    A 429 rate-limit fault carrying ``Retry-After: 40`` arms the shared
    registry's cooldown to 40s (past the 30s floor) and the envelope's
    ``retry_after_seconds``; the controller schedules at 40s — past both the
    configured 5s cooldown and the registry gap — and admits only once the
    cooldown has expired.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "step one"),
                _http_error(429, code="rate_limit_exceeded", retry_after="40"),
                _final_outcome("recovered"),
            ]
        ]),
    )
    try:
        agent_id, run_id, failed, record = await _suspend_child(harness)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.PROVIDER_RATE_LIMIT
        assert envelope.failure.cause_category is FailureCategory.RATE_LIMIT
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        assert envelope.request.retry_after_seconds == 40.0
        # The registry armed the longer window: until = now + max(30, 40).
        assert harness.shared_registry.cooldown_until("glm-5-3", "mistral") == 40.0
        scheduled = _events_of(caplog, "scheduled")
        assert len(scheduled) == 1
        assert scheduled[0]["detail"] == (
            "delay=40.000 retry_index=0 deadline=advisory"
        )
        # Still cooling down: the timer re-checks and waits without admitting.
        harness.driver.release(40.0)
        await _until(lambda: len(harness.wakeup.calls) >= 3)
        assert record.current_run is None or record.current_run.run_id == run_id
        assert record.state is _AgentState.SUSPENDED
        # Past the window: the re-armed timer admits the successor.
        harness.clock.advance(41)
        harness.driver.release(40.0)
        successor = await _until_admitted(harness, agent_id, run_id)
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        assert done.completed
        assert done.response == "recovered"
        assert len(_events_of(caplog, "success")) == 1
        assert _events_of(caplog, "exhaustion") == []
        # The wait consumed no budget: exactly one resume.
        assert record.same_agent_resumes_used == 1
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# Overnight independence (no parent model call drives the recovery)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_controller_recovers_on_a_fake_overnight_idle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The controller runs INDEPENDENTLY of parent turns (WP4).

    After the failure settles (and the parent returns to idle), a fake
    overnight passes on the manual clock; releasing the controller's timer
    admits the successor while the parent's model-call surface is unchanged —
    no parent turn, no notification, no injected message.
    """
    response_gate = asyncio.Event()
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "step one"),
                _empty_fault(),
                _Gated(_final_outcome("recovered"), response_gate),
            ]
        ]),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        # The failure's own notification turn has settled; the parent is idle.
        await _until(lambda: _parent_idle(harness))
        calls_before = len(harness.parent_backend.requests_messages)
        notifications_before = list(harness.notifications)
        injected_before = len(harness.parent._pending_injected_messages)
        # A fake overnight: the manual clock advances; nothing else happens.
        harness.clock.advance(3600)
        harness.driver.release(5.0)
        successor = await _until_admitted(harness, agent_id, run_id)
        assert successor.resumed_from_run_id == run_id
        # The successor's model call is in flight (gated): the controller's
        # decision and admission needed NO parent model call.
        assert len(harness.parent_backend.requests_messages) == calls_before
        assert harness.notifications == notifications_before
        assert len(harness.parent._pending_injected_messages) == injected_before
        # Release the successor's response: it completes on its own.
        response_gate.set()
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        assert done.completed
        assert done.response == "recovered"
        assert len(_events_of(caplog, "success")) == 1
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# Endpoint-correlated throttling blocks recovery (read-only probes)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_endpoint_correlation_blocks_recovery_until_it_expires(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Endpoint-correlated throttling blocks recovery onto ALL affected-endpoint
    deployments regardless of model/provider naming (WP4).

    Cross-model rate-limit evidence on ONE endpoint (recorded under a
    DIFFERENT base model and a DIFFERENT provider name than the child's own
    deployment, plus the child's own 429 completing the correlation) arms an
    endpoint-wide cooldown: the controller's READ-ONLY eligibility view sees
    every affected-endpoint deployment excluded, waits for the endpoint
    cooldown WITHOUT consuming budget, and admits only after it expires. The
    controller never claims or releases a single-flight probe
    (``claim_probes=False`` on every eligibility call; zero ``release_probe``
    calls).
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "step one"),
                _http_error(429, code="rate_limit_exceeded"),
                _final_outcome("recovered"),
            ]
        ]),
    )
    # A longer endpoint cooldown than the per-deployment floor.
    harness.parent.config_orchestrator._availability_registry = AvailabilityRegistry(
        clock=harness.clock, endpoint_cooldown=45.0
    )
    # One correlation event under a different base model AND provider name.
    harness.shared_registry.record_failure(
        "mistral-large-4",
        "mirror",
        FailureInfo(
            FailureCategory.RATE_LIMIT, status=429, error_code="rate_limit_exceeded"
        ),
        endpoint=CHILD_ENDPOINT,
        signature=RATE_LIMIT_SIGNATURE,
    )
    # Spy on the controller's eligibility view and the registry's probe port.
    claim_probe_kwargs: list[Any] = []
    from chartreux.app_server import _recovery_controller as controller_module

    original_eligible = controller_module.eligible_deployments

    def eligible_spy(**kwargs: Any) -> Any:
        claim_probe_kwargs.append(kwargs.get("claim_probes"))
        return original_eligible(**kwargs)

    monkeypatch.setattr(
        "chartreux.app_server._recovery_controller.eligible_deployments", eligible_spy
    )
    release_spy = MagicMock(wraps=harness.shared_registry.release_probe)
    harness.shared_registry.release_probe = release_spy
    try:
        agent_id, run_id, failed, record = await _suspend_child(harness)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.PROVIDER_RATE_LIMIT
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        # The child's own 429 completed the cross-model correlation: the
        # endpoint-wide cooldown is armed for the child's endpoint.
        assert harness.shared_registry.endpoint_correlation_active("api.mistral.ai")
        # The controller waits for the endpoint cooldown (45s), not the
        # configured 5s cooldown, and admits nothing while it is armed.
        scheduled = _events_of(caplog, "scheduled")
        assert len(scheduled) == 1
        assert scheduled[0]["detail"] == (
            "delay=45.000 retry_index=0 deadline=advisory"
        )
        harness.driver.release(45.0)
        await _until(lambda: len(harness.wakeup.calls) >= 3)
        assert record.state is _AgentState.SUSPENDED
        assert record.current_run is None or record.current_run.run_id == run_id
        # The read-only wait consumed NO budget: the episode is still armed.
        episode = harness.controller._episodes[agent_id]
        assert episode.auto_retries_used == 0
        # Past the endpoint cooldown: the re-armed timer admits the successor.
        harness.clock.advance(46)
        harness.driver.release(45.0)
        successor = await _until_admitted(harness, agent_id, run_id)
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        assert done.completed
        assert done.response == "recovered"
        # The controller's eligibility view was read-only on every call and
        # never released a probe.
        assert claim_probe_kwargs
        assert all(kw is False for kw in claim_probe_kwargs)
        assert release_spy.call_count == 0
        # The endpoint wait consumed no budget: exactly one resume.
        assert record.same_agent_resumes_used == 1
        assert len(_events_of(caplog, "success")) == 1
        assert _events_of(caplog, "exhaustion") == []
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# The no-auto-resume allowlist (protected classes never schedule)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_no_auto_resume_allowlist_never_schedules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """NO auto-resume for the protected classes, unknown/internal, a
    reconcile-required barrier, a missing envelope, or a non-suspended record
    (WP4 acceptance allowlist).

    Every row is observed by the controller (one failure milestone each) and
    stopped at the gate — no episode survives, no timer is armed, no scheduled
    or exhaustion milestone is emitted.
    """
    harness = await _controller_registry(monkeypatch, _enabled_policy(), _Factory([]))
    prohibited = RecoveryEligibility.PROHIBITED
    rows: list[dict[str, Any]] = [
        {"failure_class": FailureClass.CANCELLATION, "eligibility": prohibited},
        {"failure_class": FailureClass.BUDGET_STOP, "eligibility": prohibited},
        {"failure_class": FailureClass.AUTHORIZATION, "eligibility": prohibited},
        {"failure_class": FailureClass.PREFLIGHT, "eligibility": prohibited},
        {"failure_class": FailureClass.TASK, "eligibility": prohibited},
        {"failure_class": FailureClass.PERMANENT_CONFIG, "eligibility": prohibited},
        {"failure_class": FailureClass.UNKNOWN, "eligibility": prohibited},
        {"failure_class": FailureClass.INTERNAL, "eligibility": prohibited},
        {
            "failure_class": FailureClass.EMPTY_RESPONSE,
            "eligibility": RecoveryEligibility.RECONCILE_REQUIRED,
            "cause_category": FailureCategory.EMPTY_RESPONSE,
        },
        {"with_envelope": False},
        {
            "failure_class": FailureClass.EMPTY_RESPONSE,
            "eligibility": RecoveryEligibility.ELIGIBLE,
            "cause_category": FailureCategory.EMPTY_RESPONSE,
            "state": _AgentState.IDLE,
        },
    ]
    try:
        for row in rows:
            record, run = _fabricated_failure(harness, **row)
            await _observe(harness, record, run)
            # No episode survives, no schedule is armed.
            assert harness.controller._episodes == {}
        assert len(_events_of(caplog, "failure")) == len(rows)
        assert _events_of(caplog, "scheduled") == []
        assert _events_of(caplog, "exhaustion") == []
        assert harness.wakeup.calls == []
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_eligible_class_outside_the_configured_allowlist_never_schedules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An ELIGIBLE transient whose class is outside ``eligible_classes`` is
    never scheduled (WP4): the configured allowlist is enforced on top of the
    envelope's own eligibility, with one exhaustion milestone.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(eligible_classes=[RecoveryEligibleClass.CONNECTION]),
        _Factory([]),
    )
    try:
        record, run = _fabricated_failure(
            harness,
            failure_class=FailureClass.EMPTY_RESPONSE,
            eligibility=RecoveryEligibility.ELIGIBLE,
            cause_category=FailureCategory.EMPTY_RESPONSE,
        )
        await _observe(harness, record, run)
        assert harness.controller._episodes == {}
        assert _events_of(caplog, "scheduled") == []
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "class_not_eligible"
        assert harness.wakeup.calls == []
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# End-to-end protected classes (authorization denial, budget stop, cancel)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_authorization_denial_never_auto_resumes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A 401 authorization denial is PROHIBITED: no suspension, no episode,
    no schedule (WP4 allowlist, end to end).
    """
    harness = await _controller_registry(
        monkeypatch, _enabled_policy(), _Factory([[_http_error(401)]])
    )
    try:
        launch = await _launch(harness, "needs auth")
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _settle(harness, launch.agent_id, launch.run_id)
        envelope = failed.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.AUTHORIZATION
        assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
        record = harness.registry._agent_records[launch.agent_id]
        # Not recoverable: the retained agent returns to idle (no suspension).
        assert record.state is _AgentState.IDLE
        assert harness.controller._episodes == {}
        assert _events_of(caplog, "scheduled") == []
        assert _events_of(caplog, "exhaustion") == []
        assert len(_events_of(caplog, "failure")) == 1
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_budget_stop_never_auto_resumes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A turn-budget stop is never auto-resumed (WP4 allowlist, end to end).

    The child's loop is capped at one turn; the run stops with
    ``budget_exceeded`` after its first settled write. A budget stop resolves
    as a COMPLETED run carrying no failure envelope and no suspension — the
    retained record returns to IDLE — so the controller never observes a
    failure and never schedules (the envelope-level BUDGET_STOP/PROHIBITED
    gate is pinned by the allowlist matrix above).
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _write_outcome("call-2", "two"),
                _write_outcome("call-3", "three"),
            ]
        ]),
    )
    try:
        launch = await _launch(harness, "capped work")
        assert launch.agent_id is not None and launch.run_id is not None
        record = harness.registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.RUNNING)
        # Cap the child's turn budget while its first turn is in flight.
        record.runtime.agent_loop.set_max_turns(1)
        stopped = await _settle(harness, launch.agent_id, launch.run_id)
        run = next(
            item
            for item in harness.runs(launch.agent_id)
            if item.run_id == launch.run_id
        )
        # The stop carries no failure envelope: no transient cause to recover.
        assert stopped.failure is None
        assert stopped.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert run.status is RunStatus.COMPLETED
        assert run.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert run.failure is None
        # The retained record returns to idle (no suspension, no successor).
        assert record.state is _AgentState.IDLE
        assert record.last_run_status is RunStatus.COMPLETED
        assert record.stop_reason is RunStopReason.BUDGET_EXCEEDED
        assert harness.controller._episodes == {}
        assert _controller_events(caplog) == []
        assert [item.run_id for item in harness.runs(launch.agent_id)] == [
            launch.run_id
        ]
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cancellation_never_auto_resumes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A user cancellation settles the run as CANCELLED: the controller never
    observes a failure and never schedules (WP4 allowlist, end to end).

    The cancelled run carries no failure envelope (a requested stop is not a
    failure), so no episode, timer, or milestone exists; the RETAINED agent
    stays resident and returns to IDLE through the normal paths.
    """
    release = asyncio.Event()
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([[_Gated(_write_outcome("call-1", "held"), release)]]),
    )
    try:
        launch = await _launch(harness, "cancel me")
        assert launch.agent_id is not None and launch.run_id is not None
        record = harness.registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.RUNNING)
        cancel = await harness.registry.cancel_run(
            launch.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=harness.requester_session_id,
        )
        assert cancel.outcome is CancelOutcome.STOP_REQUESTED
        release.set()
        cancelled = await _settle(harness, launch.agent_id, launch.run_id)
        # The cancelled run never reached the failure path: no envelope, and
        # the retained agent stays resident (idle) — never evicted, never
        # resumed.
        assert cancelled.failure is None
        assert cancelled.stop_reason is RunStopReason.USER_CANCELLED
        assert launch.agent_id in harness.registry._agent_records
        assert launch.agent_id not in harness.registry._evicted_agents
        assert record.state is _AgentState.IDLE
        assert _controller_events(caplog) == []
        assert harness.controller._episodes == {}
        assert [run.run_id for run in harness.runs(launch.agent_id)] == [launch.run_id]
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# launch_failure policy rows (handoff §2.3) — BOTH rows pinned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_transient_launch_failure_gets_one_bounded_startup_retry(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """launch_failure row 1 (handoff §2.3): a confirmed transient launch_failure
    gets ONE bounded-backoff startup retry preserving identity and brief.

    The shared registry's per-deployment cooldown blocks the child's first
    call at admission (a measured zero-transport-attempt launch failure, all
    exclusions ``cooldown`` → ELIGIBLE); the controller waits out the cooldown
    and admits exactly one successor, which completes on the warm loop with
    the original brief.
    """
    harness = await _controller_registry(
        monkeypatch, _enabled_policy(), _Factory([[_final_outcome("launched ok")]])
    )
    harness.shared_registry.record_failure(
        "glm-5-3", "mistral", FailureInfo(FailureCategory.CONNECTION)
    )
    try:
        launch = await _launch(harness, "fix the launch")
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(harness.registry, launch.agent_id, launch.run_id)
        record = harness.registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        await _until(lambda: bool(harness.wakeup.calls))
        envelope = failed.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.LAUNCH_FAILURE
        assert envelope.failure.phase is FailurePhase.ADMISSION
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        # The bounded backoff waits out the deployment cooldown (30s), not the
        # configured 5s cooldown.
        scheduled = _events_of(caplog, "scheduled")
        assert len(scheduled) == 1
        assert scheduled[0]["detail"] == (
            "delay=30.000 retry_index=0 deadline=advisory"
        )
        # Past the cooldown: the startup retry is admitted.
        harness.clock.advance(31)
        harness.driver.release(30.0)
        successor = await _until_admitted(harness, launch.agent_id, launch.run_id)
        assert successor.resumed_from_run_id == launch.run_id
        assert successor.agent_id == launch.agent_id
        # Identity and brief preserved: the successor re-ran the original task.
        assert successor.task_summary == "fix the launch"
        done = await _wait_result(harness.registry, launch.agent_id, successor.run_id)
        assert done.completed
        assert done.response == "launched ok"
        await _until(lambda: record.state is _AgentState.IDLE)
        assert len(_events_of(caplog, "successor_start")) == 1
        assert len(_events_of(caplog, "success")) == 1
        assert _events_of(caplog, "exhaustion") == []
        assert record.same_agent_resumes_used == 1
        assert harness.controller._lineage == {"brief:worker:fix the launch": 1}
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_repeated_launch_failure_stops_the_relaunch_loop(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """launch_failure row 2 (handoff §2.3): a REPEATED launch failure STOPS the
    relaunch loop — the instant-relaunch pattern (L1/L2, R2/R3) is prevented.

    The factory re-arms the shared cooldown at the resume-time backend refresh,
    so the admitted successor ALSO launch-fails; the controller stops the
    episode with ``repeated_launch_failure`` and parks the agent suspended —
    no second successor, one exhaustion milestone.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        # One empty per-loop script: both runs are blocked at admission (the
        # re-armed cooldown), so no backend call ever pops an outcome.
        _Factory(
            [[]],
            rearm=lambda: harness.shared_registry.record_failure(
                "glm-5-3", "mistral", FailureInfo(FailureCategory.CONNECTION)
            ),
        ),
    )
    harness.shared_registry.record_failure(
        "glm-5-3", "mistral", FailureInfo(FailureCategory.CONNECTION)
    )
    try:
        launch = await _launch(harness, "fix the launch")
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(harness.registry, launch.agent_id, launch.run_id)
        record = harness.registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        await _until(lambda: bool(harness.wakeup.calls))
        envelope = failed.failure
        assert envelope is not None
        assert envelope.failure.failure_class is FailureClass.LAUNCH_FAILURE
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        harness.clock.advance(31)
        harness.driver.release(30.0)
        # Exactly one successor is admitted — and it launch-fails too.
        successor = await _until_admitted(harness, launch.agent_id, launch.run_id)
        second = await _wait_result(harness.registry, launch.agent_id, successor.run_id)
        assert not second.completed
        envelope_2 = second.failure
        assert envelope_2 is not None
        assert envelope_2.failure.failure_class is FailureClass.LAUNCH_FAILURE
        assert envelope_2.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        await _until(lambda: record.state is _AgentState.SUSPENDED)
        await _until(lambda: harness.controller._episodes == {})
        # The relaunch loop STOPPED: no third run, one exhaustion milestone.
        runs = harness.runs(launch.agent_id)
        assert [run.run_id for run in runs] == [launch.run_id, successor.run_id]
        assert record.state is _AgentState.SUSPENDED
        assert len(_events_of(caplog, "successor_start")) == 1
        assert _events_of(caplog, "success") == []
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "repeated_launch_failure"
        assert record.same_agent_resumes_used == 1
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# The package-lineage recurrence counter spans replacements
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_recurrence_counter_spans_replacements(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The recurrence counter spans replacements (WP4): a replacement agent
    re-dispatched with the same brief cannot reset the package budget.

    With ``auto_retries_max=1``, agent A exhausts its episode budget; a NEW
    agent B launched with the SAME task summary (the replacement's package
    identity) is refused at the lineage gate — no schedule for B.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(auto_retries_max=1, cooldowns_seconds=[5.0]),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _write_outcome("call-2", "two"),
                _empty_fault(),
            ],
            [_write_outcome("call-1", "one"), _empty_fault()],
        ]),
    )
    try:
        agent_a, run_a, _failed_a, record_a = await _suspend_child(
            harness, task="pkg-x", tool_call_id="launch-a"
        )
        harness.driver.release(5.0)
        await _until_admitted(harness, agent_a, run_a)
        # A's successor fails too: A's episode budget (1) is exhausted.
        await _until(lambda: harness.controller._episodes == {})
        assert record_a.state is _AgentState.SUSPENDED
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "auto_retries_exhausted"
        assert harness.controller._lineage == {"brief:worker:pkg-x": 1}

        # The replacement: a NEW agent with the SAME brief (package identity).
        agent_b, run_b, failed_b, record_b = await _suspend_child(
            harness, task="pkg-x", tool_call_id="launch-b"
        )
        assert agent_b != agent_a
        envelope = failed_b.failure
        assert envelope is not None
        assert envelope.recovery.eligibility is RecoveryEligibility.ELIGIBLE
        # Refused at the lineage gate: no schedule, no timer, one exhaustion.
        assert harness.controller._episodes == {}
        assert record_b.state is _AgentState.SUSPENDED
        assert [run.run_id for run in harness.runs(agent_b)] == [run_b]
        scheduled = _events_of(caplog, "scheduled")
        assert len(scheduled) == 1  # only A's schedule
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 2
        assert exhaustion[1]["detail"] == "package_recurrence_exhausted"
        assert harness.controller._lineage == {"brief:worker:pkg-x": 1}
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# A config reload never extends an active episode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_config_reload_does_not_extend_an_active_episode(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Episode policy is SNAPSHOTTED at the first failure (WP4): a config
    reload that raises ``auto_retries_max`` mid-episode never extends the
    active episode or replenishes its count.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(auto_retries_max=1, cooldowns_seconds=[5.0]),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _write_outcome("call-2", "two"),
                _empty_fault(),
            ]
        ]),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        episode = harness.controller._episodes[agent_id]
        assert episode.auto_retries_max == 1
        # A config reload mid-episode: the policy now allows five retries.
        object.__setattr__(
            harness.parent.config,
            "recovery",
            RecoverySectionConfig(
                subagents=RecoveryConfig(
                    enabled=True,
                    auto_retries_max=5,
                    cooldowns_seconds=[5.0] * 5,
                    jitter_fraction=0.0,
                )
            ),
        )
        assert harness.parent.config.recovery.subagents.auto_retries_max == 5
        harness.driver.release(5.0)
        await _until_admitted(harness, agent_id, run_id)
        # The successor fails: the SNAPSHOT budget (1) still refuses.
        await _until(lambda: harness.controller._episodes == {})
        assert record.state is _AgentState.SUSPENDED
        runs = harness.runs(agent_id)
        assert len(runs) == 2
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "auto_retries_exhausted"
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# The aggregate episode deadline is ADVISORY (Q2-negative branch)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_advisory_deadline_stops_new_admissions(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The aggregate deadline is ADVISORY (Q2-negative): no NEW admission after
    it — enforced at the timer's re-validation — and the schedule reports it
    as advisory, never a hard bound.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(episode_deadline_seconds=100.0),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ]
        ]),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        episode = harness.controller._episodes[agent_id]
        scheduled = _events_of(caplog, "scheduled")
        assert len(scheduled) == 1
        assert "deadline=advisory" in scheduled[0]["detail"]
        # The deadline passes before the timer fires.
        harness.controller._epoch = lambda: episode.started_at + 200
        harness.driver.release(5.0)
        await _until(lambda: harness.controller._episodes == {})
        # No admission: the record stays suspended, one exhaustion milestone.
        assert record.state is _AgentState.SUSPENDED
        assert [run.run_id for run in harness.runs(agent_id)] == [run_id]
        assert _events_of(caplog, "successor_start") == []
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "episode_deadline_passed"
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_advisory_deadline_never_kills_an_in_flight_successor(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The advisory deadline never kills an in-flight successor (Q2-negative:
    no hard in-flight timeout exists). The successor admitted before the
    deadline completes even after the deadline passes mid-flight.
    """
    response_gate = asyncio.Event()
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(episode_deadline_seconds=100.0),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _Gated(_final_outcome("recovered"), response_gate),
            ]
        ]),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        episode = harness.controller._episodes[agent_id]
        # The timer fires before the deadline: the successor is admitted.
        harness.driver.release(5.0)
        successor = await _until_admitted(harness, agent_id, run_id)
        # The deadline passes while the successor is in flight (gated call).
        harness.controller._epoch = lambda: episode.started_at + 200
        response_gate.set()
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        # No kill: the successor completes and closes the episode with success.
        assert done.completed
        assert done.response == "recovered"
        await _until(lambda: record.state is _AgentState.IDLE)
        assert len(_events_of(caplog, "success")) == 1
        assert _events_of(caplog, "exhaustion") == []
        assert harness.controller._episodes == {}
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# Invalidation: cancellation, release, and shutdown cancel pending schedules
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_cancellation_cancels_the_pending_schedule(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """An authoritative cancellation of a suspended barrier cancels the
    pending automatic recovery FIRST (rollback: cancel timers, retain
    counters/results) — no successor is ever admitted.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ]
        ]),
    )
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        assert len(_events_of(caplog, "scheduled")) == 1
        cancel = await harness.registry.cancel_run(
            agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=harness.requester_session_id,
        )
        assert cancel.outcome is CancelOutcome.STOP_REQUESTED
        await _until(lambda: agent_id not in harness.registry._agent_records)
        # The episode stopped; a late timer release admits nothing.
        harness.driver.release(5.0)
        await asyncio.sleep(0.05)
        assert agent_id not in harness.registry._agent_records
        tombstone = harness.registry._evicted_agents[agent_id]
        assert tombstone.reason == "closed"
        assert _events_of(caplog, "successor_start") == []
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 1
        assert exhaustion[0]["detail"] == "invalidated"
        assert harness.controller._episodes == {}
        # The failed run's stored result survives the cancellation.
        stored = await harness.registry.get_agent_result(agent_id, run_id)
        assert stored is not None
        assert not stored.completed
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_release_and_drain_cancel_pending_schedules(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Release and shutdown cancel pending schedules (rollback: cancel timers,
    retain counters/results) — no successor is admitted for either agent.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ],
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ],
        ]),
    )
    try:
        # Release invalidates the pending recovery FIRST.
        agent_a, run_a, _failed_a, _record_a = await _suspend_child(
            harness, task="release me", tool_call_id="launch-a"
        )
        outcome = await harness.registry.release_agent(agent_a)
        assert outcome is ReleaseAgentOutcome.RELEASED
        assert harness.controller._episodes == {}
        assert agent_a not in harness.registry._agent_records
        # A late timer release admits nothing (the record is gone).
        harness.driver.release(5.0)
        await asyncio.sleep(0.05)
        assert agent_a not in harness.registry._agent_records

        # Shutdown (drain) cancels every pending schedule.
        agent_b, run_b, _failed_b, record_b = await _suspend_child(
            harness, task="drain me", tool_call_id="launch-b"
        )
        assert agent_b in harness.controller._episodes
        await harness.registry.drain_children()
        assert harness.controller._episodes == {}
        assert agent_b not in harness.registry._agent_records
        harness.driver.release(5.0)
        await asyncio.sleep(0.05)
        assert agent_b not in harness.registry._agent_records
        assert record_b.state is not _AgentState.RUNNING

        assert _events_of(caplog, "successor_start") == []
        exhaustion = _events_of(caplog, "exhaustion")
        assert len(exhaustion) == 2
        assert all(item["detail"] == "invalidated" for item in exhaustion)
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


# ---------------------------------------------------------------------------
# Automatic and explicit requests hit the SAME admission operation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_automatic_and_explicit_requests_share_one_admission(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Automatic and explicit requests hit the SAME ``resume_agent`` admission
    (WP4): the controller's timer admits the successor; an explicit request
    for the same failed run afterwards is refused by the SAME operation's
    run-pinned detection (``already_resumed``) — exactly one successor ever
    exists, and the controller's episode closed with the successor's success.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ]
        ]),
    )
    resume_calls: list[tuple[str, Any]] = []
    original_resume = harness.registry.resume_agent

    async def resume_spy(
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> Any:
        result = await original_resume(
            agent_id,
            expected_run_id,
            idempotency_key,
            requester_session_id=requester_session_id,
        )
        resume_calls.append((idempotency_key, result.outcome))
        return result

    monkeypatch.setattr(harness.registry, "resume_agent", resume_spy)
    try:
        agent_id, run_id, failed, record = await _suspend_child(harness)
        # The controller's timer admits the successor (same operation).
        harness.driver.release(5.0)
        successor = await _until_admitted(harness, agent_id, run_id)
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        assert done.completed
        assert done.response == "recovered"
        await _until(lambda: record.state is _AgentState.IDLE)
        # An explicit request for the SAME failed run now hits the SAME
        # admission's run-pinned detection: already_resumed.
        explicit = await harness.registry.resume_agent(
            agent_id,
            run_id,
            "explicit-key",
            requester_session_id=harness.requester_session_id,
        )
        assert explicit.outcome is ResumeOutcome.ALREADY_RESUMED
        assert explicit.successor_run_id == successor.run_id
        # Both requests hit the SAME operation; exactly one successor exists.
        assert len(resume_calls) == 2
        assert resume_calls[0][0].startswith("auto-recovery:")
        assert resume_calls[0][1] is ResumeOutcome.SCHEDULED
        assert resume_calls[1] == ("explicit-key", ResumeOutcome.ALREADY_RESUMED)
        assert [run.run_id for run in harness.runs(agent_id)] == [
            run_id,
            successor.run_id,
        ]
        # The failed run's stored result stays immutable.
        stored = await harness.registry.get_agent_result(agent_id, run_id)
        assert stored is not None
        assert not stored.completed
        assert stored.failure is not None
        assert failed.failure is not None
        assert stored.failure.event_id == failed.failure.event_id
        # The successor's completion closed the episode: one success milestone,
        # no exhaustion.
        assert _events_of(caplog, "exhaustion") == []
        assert len(_events_of(caplog, "successor_start")) == 1
        assert len(_events_of(caplog, "success")) == 1
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(60)
async def test_explicit_resume_during_a_pending_schedule_wins_single_flight(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The mirror race: an explicit resume while the controller's timer is
    pending wins the single-flight admission; the controller's episode ends
    with the successor's completion (its pending timer is cancelled) — exactly
    one successor, no exhaustion.
    """
    harness = await _controller_registry(
        monkeypatch,
        _enabled_policy(),
        _Factory([
            [
                _write_outcome("call-1", "one"),
                _empty_fault(),
                _final_outcome("recovered"),
            ]
        ]),
    )
    resume_calls: list[str] = []
    original_resume = harness.registry.resume_agent

    async def resume_spy(
        agent_id: str,
        expected_run_id: str,
        idempotency_key: str,
        *,
        requester_session_id: str,
    ) -> Any:
        resume_calls.append(idempotency_key)
        return await original_resume(
            agent_id,
            expected_run_id,
            idempotency_key,
            requester_session_id=requester_session_id,
        )

    monkeypatch.setattr(harness.registry, "resume_agent", resume_spy)
    try:
        agent_id, run_id, _failed, record = await _suspend_child(harness)
        episode = harness.controller._episodes[agent_id]
        timer = episode.timer
        assert timer is not None
        # The explicit request wins the admission while the timer is pending.
        explicit = await harness.registry.resume_agent(
            agent_id,
            run_id,
            "explicit-key",
            requester_session_id=harness.requester_session_id,
        )
        assert explicit.outcome is ResumeOutcome.SCHEDULED
        assert len(resume_calls) == 1  # the controller never called
        successor = await _until_admitted(harness, agent_id, run_id)
        assert successor.resumed_from_run_id == run_id
        done = await _wait_result(harness.registry, agent_id, successor.run_id)
        assert done.completed
        assert done.response == "recovered"
        await _until(lambda: record.state is _AgentState.IDLE)
        # The successor's completion closed the episode and cancelled the
        # pending timer: exactly one successor, no exhaustion.
        assert timer.cancelled() or timer.done()
        assert [run.run_id for run in harness.runs(agent_id)] == [
            run_id,
            successor.run_id,
        ]
        assert len(_events_of(caplog, "success")) == 1
        assert _events_of(caplog, "exhaustion") == []
        assert harness.controller._episodes == {}
    finally:
        await harness.registry.drain_children()
        await harness.parent.aclose()
