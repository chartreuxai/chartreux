"""C4 — activate the explicit task override.

The strict ``override_cooldown`` task flag only REQUESTS the un-stick
override: root launch admission (``create_registered`` for a fresh or
replacement launch, ``prepare_background`` for a retained retask) mints the
endpoint-bound permit from the ROOT ``CooldownOverrideLedger`` only for the
explicit request, threads it through the candidate resolution (the cooled
forced deployment stays assignable) and binds it to the child's first model
call (C3 consumes it at the actual transport start). A ledger denial fails
the launch BEFORE any budget is consumed, the bounded diagnostics ride the
launch acknowledgment, and automatic paths never acquire permits (F6).

All backends are offline stubs; no network, no real sleeps.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from chartreux.app_server._sessions import SessionRuntimeRegistry, _AgentState
from chartreux.core.llm.failures import FailureCategory, FailureInfo
from chartreux.core.recovery_override import OverrideDenial
from chartreux.core.subagents import (
    InvalidLaunchModelError,
    LaunchConfig,
    RunStopReason,
    TaskArgs,
)
from chartreux.core.tools.base import InvokeContext, ToolError
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    _background_result,
    _ManualClock,
    _wait_result,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

ONE_ENDPOINT = "https://one.test"
TWO_ENDPOINT = "https://two.test"


def _override_config() -> Any:
    return build_test_vibe_config(
        active_model="small",
        include_project_context=False,
        include_prompt_detail=False,
        enabled_tools=["task"],
        tools={"task": {"permission": "always"}},
        providers=[
            {"name": "one", "api_base": ONE_ENDPOINT, "backend": "generic"},
            {"name": "two", "api_base": TWO_ENDPOINT, "backend": "generic"},
        ],
        models=[
            {
                "name": "small",
                "alias": "small",
                "provider": "one",
                "thinking": "off",
                "supported_thinking_levels": ["off", "high"],
            },
            {
                "name": "other",
                "alias": "other",
                "provider": "two",
                "thinking": "off",
                "supported_thinking_levels": ["off", "high"],
            },
        ],
    )


async def _override_registry(
    monkeypatch: pytest.MonkeyPatch, backend: Any = None
) -> tuple[SessionRuntimeRegistry, Any, list[Any]]:
    """A real registry whose child backends are inspectable offline stubs."""
    backends: list[Any] = []

    def create_backend(*, provider: Any, **_kwargs: Any) -> FakeBackend:
        child_backend = backend or FakeBackend([mock_llm_chunk(content=provider.name)])
        backends.append(child_backend)
        return child_backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=_override_config(), backend=FakeBackend())
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _session_id: 0, clock=_ManualClock()
    )
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent, backends


def _availability(parent: Any) -> Any:
    return parent.config_orchestrator.availability_registry


def _ledger(parent: Any) -> Any:
    return parent.config_orchestrator.cooldown_override_ledger


def _arm_deployment_cooldown(parent: Any) -> None:
    _availability(parent).record_failure(
        "small", "one", FailureInfo(FailureCategory.TIMEOUT)
    )


def _open_episode(parent: Any) -> None:
    assert _ledger(parent).note_endpoint_failure(ONE_ENDPOINT) is True


def _ctx(parent: Any, tool_call_id: str = "override") -> InvokeContext:
    return InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id)


def _override_diagnostics(result: Any) -> dict[str, Any]:
    """The launch acknowledgment's cooldown_override diagnostics (asserted)."""
    metadata = result.metadata
    assert metadata is not None
    return metadata["cooldown_override"]


def _agent_id(result: Any) -> str:
    agent_id = result.agent_id
    assert agent_id is not None
    return agent_id


@pytest.mark.asyncio
async def test_fresh_override_launch_mints_binds_and_reports_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _backends = await _override_registry(monkeypatch)
    _arm_deployment_cooldown(parent)
    args = TaskArgs(
        task="unstick", identity="coder", background=True, retain=True, model="small"
    )
    # Without the explicit request the cooled deployment is not assignable.
    with pytest.raises(InvalidLaunchModelError):
        await _background_result(registry, args, _ctx(parent))
    # The failed ordinary launch opened no episode and minted nothing.
    assert _ledger(parent).episode_status(ONE_ENDPOINT) is None
    try:
        _open_episode(parent)
        minted = await _background_result(
            registry, args.model_copy(update={"override_cooldown": True}), _ctx(parent)
        )
        assert minted.launch_outcome is not None
        diagnostics = _override_diagnostics(minted)
        assert diagnostics["endpoint"] == "one.test"
        assert diagnostics["provider"] == "one"
        assert diagnostics["wire_name"] == "small"
        assert diagnostics["exclusions_bypassed"] == ["deployment-cooldown"]
        # The allowance remaining after this attempt consumes: 2 - 1.
        assert diagnostics["remaining_starts"] == 1
        delay = diagnostics["next_start_delay_seconds"]
        assert delay is not None and 0 < delay <= 31
        # The minted permit bound to the child's first call, and the child
        # ran on the forced deployment.
        await registry.wait_for_agent(_agent_id(minted), minted.run_id)
        ledger = _ledger(parent)
        # The forced attempt SUCCEEDED: a measured endpoint recovery, which
        # RETIRES the episode (the pinned boundary — only a new measured
        # failure re-opens it). The stub backend fires no transport-start
        # notification, so the attempt released the permit pretransport and
        # the in-flight slot is free — proven by a fresh measured failure
        # opening a new episode whose reserve mints.
        assert ledger.episode_status(ONE_ENDPOINT) is None
        assert ledger.reserve(ONE_ENDPOINT).denial is OverrideDenial.NO_ACTIVE_EPISODE
        assert ledger.note_endpoint_failure(ONE_ENDPOINT) is True
        outcome = ledger.reserve(ONE_ENDPOINT)
        assert outcome.permit is not None
        ledger.release(outcome.permit)
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_override_without_episode_is_denied_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _backends = await _override_registry(monkeypatch)
    try:
        with pytest.raises(ToolError, match="no active endpoint episode"):
            await _background_result(
                registry,
                TaskArgs(
                    task="unstick",
                    identity="coder",
                    background=True,
                    model="small",
                    override_cooldown=True,
                ),
                _ctx(parent),
            )
        # Nothing minted, nothing launched.
        assert registry._agent_records == {}
        assert _ledger(parent).episode_status(ONE_ENDPOINT) is None
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_spacing_denial_fails_launch_before_budget_consumption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _backends = await _override_registry(monkeypatch)
    clock = _ManualClock()
    ledger = _ledger(parent)
    monkeypatch.setattr(ledger, "_clock", clock)
    try:
        _open_episode(parent)
        first = ledger.reserve(ONE_ENDPOINT)
        assert first.permit is not None
        assert ledger.start(first.permit).phase is not None
        # The pressure bound admits the second start only after the spacing
        # window: an explicit request right now is denied, consuming nothing.
        with pytest.raises(ToolError, match="spacing floor"):
            await _background_result(
                registry,
                TaskArgs(
                    task="unstick",
                    identity="coder",
                    background=True,
                    model="small",
                    override_cooldown=True,
                ),
                _ctx(parent),
            )
        status = ledger.episode_status(ONE_ENDPOINT)
        assert status is not None
        assert status.starts_used == 1
        assert registry._agent_records == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_exhausted_episode_denial_reports_remaining_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _backends = await _override_registry(monkeypatch)
    clock = _ManualClock()
    ledger = _ledger(parent)
    monkeypatch.setattr(ledger, "_clock", clock)
    try:
        _open_episode(parent)
        for _ in range(2):
            reserved = ledger.reserve(ONE_ENDPOINT)
            assert reserved.permit is not None
            assert ledger.start(reserved.permit).phase is not None
            clock.advance(30)
        with pytest.raises(ToolError, match="forced-start budget is exhausted"):
            await _background_result(
                registry,
                TaskArgs(
                    task="unstick",
                    identity="coder",
                    background=True,
                    model="small",
                    override_cooldown=True,
                ),
                _ctx(parent),
            )
        status = ledger.episode_status(ONE_ENDPOINT)
        assert status is not None
        assert status.starts_used == 2
        assert registry._agent_records == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_permit_is_scoped_to_the_requested_launch_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _backends = await _override_registry(monkeypatch)
    try:
        # An episode on provider one's endpoint never authorizes a launch of
        # the model served by provider two's endpoint.
        _open_episode(parent)
        with pytest.raises(ToolError, match="no active endpoint episode"):
            await _background_result(
                registry,
                TaskArgs(
                    task="unstick",
                    identity="coder",
                    background=True,
                    model="other",
                    override_cooldown=True,
                ),
                _ctx(parent),
            )
        assert registry._agent_records == {}
        # The matching endpoint mints normally.
        _arm_deployment_cooldown(parent)
        minted = await _background_result(
            registry,
            TaskArgs(
                task="unstick",
                identity="coder",
                background=True,
                model="small",
                override_cooldown=True,
            ),
            _ctx(parent),
        )
        assert _override_diagnostics(minted)["endpoint"] == "one.test"
        await registry.wait_for_agent(_agent_id(minted), minted.run_id)
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_automatic_paths_never_acquire_permits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6: an ordinary launch (no explicit request) never mints or binds."""
    registry, parent, _backends = await _override_registry(monkeypatch)
    try:
        launched = await _background_result(
            registry,
            TaskArgs(
                task="ordinary",
                identity="coder",
                background=True,
                retain=True,
                model="small",
            ),
            _ctx(parent),
        )
        assert launched.launch_outcome is not None
        assert "cooldown_override" not in (launched.metadata or {})
        assert _ledger(parent).episode_status(ONE_ENDPOINT) is None
        launched_id = _agent_id(launched)
        await registry.wait_for_agent(launched_id, launched.run_id)
        record = registry._agent_records[launched_id]
        assert record.runtime.agent_loop._completion_coordinator._active_override is (
            None
        )
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_failed_admission_after_mint_releases_the_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch denied after the mint releases the permit (no stranded slot)."""
    registry, parent, _backends = await _override_registry(monkeypatch)
    from chartreux.app_server._subagent_launch import ChildLaunchAdmission
    from chartreux.core.tools.models import ToolPermissionError

    def denied(_self: Any, _parent: Any, _identity: str) -> None:
        raise ToolPermissionError("Task denied for launch identity: coder")

    monkeypatch.setattr(ChildLaunchAdmission, "require_identity_allowed", denied)
    try:
        _open_episode(parent)
        with pytest.raises(ToolPermissionError):
            await _background_result(
                registry,
                TaskArgs(
                    task="unstick",
                    identity="coder",
                    background=True,
                    model="small",
                    override_cooldown=True,
                ),
                _ctx(parent),
            )
        # The minted reservation was released: the slot is free again.
        outcome = _ledger(parent).reserve(ONE_ENDPOINT)
        assert outcome.permit is not None
        _ledger(parent).release(outcome.permit)
        assert registry._agent_records == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reused_launch_mints_through_the_committed_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, _backends = await _override_registry(monkeypatch)
    try:
        first = await _background_result(
            registry,
            TaskArgs(
                task="first",
                identity="coder",
                background=True,
                retain=True,
                model="small",
            ),
            _ctx(parent),
        )
        await registry.wait_for_agent(_agent_id(first), first.run_id)
        _arm_deployment_cooldown(parent)
        _open_episode(parent)
        second = await _background_result(
            registry,
            TaskArgs(
                task="second",
                agent_id=first.agent_id,
                background=True,
                model="small",
                config=LaunchConfig(),
                override_cooldown=True,
            ),
            _ctx(parent, "retask"),
        )
        diagnostics = _override_diagnostics(second)
        assert diagnostics["provider"] == "one"
        assert diagnostics["exclusions_bypassed"] == ["deployment-cooldown"]
        second_id = _agent_id(second)
        await registry.wait_for_agent(second_id, second.run_id)
        record = registry._agent_records[second_id]
        assert record.state is _AgentState.IDLE
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_cancelled_run_releases_the_unconsumed_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancellation before any transport start frees the in-flight slot."""
    registry, parent, backends = await _override_registry(
        monkeypatch, backend=BlockingBackend()
    )
    _open_episode(parent)
    try:
        minted = await _background_result(
            registry,
            TaskArgs(
                task="unstick",
                identity="coder",
                background=True,
                retain=True,
                model="small",
                override_cooldown=True,
            ),
            _ctx(parent),
        )
        agent_id = _agent_id(minted)
        await asyncio.wait_for(backends[0].started.wait(), timeout=5)
        await registry.cancel_run(
            agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        await registry.wait_for_agent(agent_id, minted.run_id)
        ledger = _ledger(parent)
        status = ledger.episode_status(ONE_ENDPOINT)
        assert status is not None
        assert status.starts_used == 0
        outcome = ledger.reserve(ONE_ENDPOINT)
        assert outcome.permit is not None
        ledger.release(outcome.permit)
    finally:
        for backend in backends:
            if hasattr(backend, "release"):
                backend.release.set()
        await registry.close()
        await parent.aclose()


def test_override_cooldown_is_strict_and_defaults_false() -> None:
    defaulted = TaskArgs(task="work", identity="coder")
    assert defaulted.override_cooldown is False
    with pytest.raises(ValueError):
        TaskArgs.model_validate({
            "task": "work",
            "identity": "coder",
            "override_cooldown": "true",
        })


@pytest.mark.asyncio
async def test_measured_child_failure_opens_the_episode_for_the_override_mint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEFECT-1 regression (production wiring): a real measured endpoint
    failure through the child's completion coordinator opens the root-shared
    ledger's episode, so the explicit override mint succeeds — previously the
    mint always denied "no active endpoint episode" because nothing fed the
    ledger at runtime.
    """
    registry, parent, _backends = await _override_registry(
        monkeypatch, backend=FakeBackend(exception_to_raise=httpx.ConnectError("down"))
    )
    try:
        # Before any measured failure the explicit override is denied.
        with pytest.raises(ToolError, match="no active endpoint episode"):
            await _background_result(
                registry,
                TaskArgs(
                    task="unstick",
                    identity="coder",
                    background=True,
                    model="small",
                    override_cooldown=True,
                ),
                _ctx(parent, "denied"),
            )
        # A real measured failure: the child's model call fails through the
        # coordinator (a failover-eligible CONNECTION error on provider one).
        ordinary = await _background_result(
            registry,
            TaskArgs(task="unstick", identity="coder", background=True, model="small"),
            _ctx(parent, "ordinary"),
        )
        failed = await _wait_result(registry, _agent_id(ordinary), ordinary.run_id)
        assert not failed.completed
        # The production path opened the endpoint's episode on the root ledger.
        assert _ledger(parent).episode_status(ONE_ENDPOINT) is not None
        # The explicit override mint now succeeds and reports diagnostics.
        minted = await _background_result(
            registry,
            TaskArgs(
                task="unstick",
                identity="coder",
                background=True,
                model="small",
                override_cooldown=True,
            ),
            _ctx(parent, "forced"),
        )
        diagnostics = _override_diagnostics(minted)
        assert diagnostics["endpoint"] == "one.test"
        assert diagnostics["provider"] == "one"
        assert diagnostics["wire_name"] == "small"
        assert diagnostics["remaining_starts"] == 1
    finally:
        await registry.close()
        await parent.aclose()
