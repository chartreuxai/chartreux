"""Management-surface tests for the 0.5.3 subagent lifecycle (design WP3).

Covers the three WP3 surfaces:

- Names (design §5): the ``{identity}-{model}-{number}`` derivation helper,
  its projection onto ``AgentSummary``/``AgentSummaryModel``/launch
  acknowledgments/completion notifications, and non-relabeling refusals.
- Guarded release (design §3.1, §8): the ``release_agent_guarded`` registry
  matrix, the ``agents/release`` RPC route + handler dispatch, and the
  ``AppServerSession.release_agent`` client method.
- Typed waits (design §6): the activated ``WaitForAgentResult`` contract —
  ``still_running`` as a normal result with turn-yield guidance, the 1500s
  default, explicit-``None`` indefinite waits, pinned run identity under
  races, lease protection, evicted-with-result success, and cancelled
  partial results — plus client-side legacy nameless payload acceptance.
"""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
import re
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

from pydantic import ValidationError
import pytest

from chartreux.app_server._execution import SessionExecutionKind
from chartreux.app_server._session_backend_impl import SessionBackendImpl
from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntimeRegistry,
    _AgentState,
    _AgentTombstone,
)
from chartreux.app_server.client import AppServerClient, AppServerResponseError
from chartreux.app_server.events import AgentsUpdate, ClientProjection
from chartreux.app_server.models import (
    IdleSessionStatus,
    PublicRecoveryEligibility,
    PublicSession,
    PublicSessionState,
    TextContentBlock,
)
from chartreux.app_server.protocol import (
    SERVER_METHODS,
    AgentsReleaseParams,
    AgentsReleaseResponse,
    Notification,
    ProtocolErrorCode,
    ReleaseOutcome,
)
from chartreux.app_server.session import AppServerSession
from chartreux.app_server.transport import memory_transport_pair
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop.errors import EmptyLLMResponseError
from chartreux.core.config.harness_files import HarnessFilesManager
from chartreux.core.llm_models import LLMMessage, Role
from chartreux.core.message_list import MessageList
from chartreux.core.recovery_models import RecoveryEligibility
from chartreux.core.subagents import (
    AgentAvailability,
    AgentIdentityMismatchError,
    AgentSummary,
    EffectVerification,
    ReconcileOutcome,
    ReconcileResult,
    ReleaseAgentOutcome,
    ResumeAgentResult,
    ResumeOutcome,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    WaitStatus,
    derive_agent_name,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext
from chartreux.core.tools.builtins.wait_for_agent import (
    DEFAULT_WAIT_TIMEOUT_SECONDS,
    WaitForAgent,
    WaitForAgentArgs,
    WaitForAgentConfig,
)
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    ImmediateFailureBackend,
    _background_result,
    _config,
    _idle_record,
    _ManualClock,
    _retention_registry,
    _stored_result,
    _wait_result,
)
from tests.conftest import build_test_agent_loop
from tests.mock.utils import collect_result, mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    build_test_app_server,
    legacy_backend,
)
from tests.stubs.fake_backend import FakeBackend
from tests.tools.test_agent_management import FakeSubagentManager

_HANDLE_PATTERN = re.compile(r"agent-\d+")


async def _until(predicate: Any, timeout: float = 3.0) -> None:
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.005)


# ---------------------------------------------------------------------------
# Name derivation (design §5)
# ---------------------------------------------------------------------------


def test_derive_agent_name_prefers_canonical_base_model() -> None:
    assert (
        derive_agent_name(
            "agent-15", "coder", None, "mistral/mistral-large-4", "provider/wire"
        )
        == "coder-mistral-large-4-15"
    )


def test_derive_agent_name_falls_back_to_namespace_stripped_effective_model() -> None:
    assert (
        derive_agent_name("agent-3", "scout", None, None, "mistral/zai-glm-5-3")
        == "scout-zai-glm-5-3-3"
    )


def test_derive_agent_name_omits_model_segment_when_both_models_none() -> None:
    assert derive_agent_name("agent-7", "reviewer", "Reviewer", None, None) == (
        "reviewer-7"
    )


def test_derive_agent_name_builtin_identity_uses_identity_not_display_name() -> None:
    # The builtin coder identity's shipped display name is "Coder"; the name
    # uses the identity reference (design §5).
    assert derive_agent_name("agent-2", "coder", "Coder", "base", None) == (
        "coder-base-2"
    )
    assert derive_agent_name("agent-4", "advisor", "Advisor", None, None) == (
        "advisor-4"
    )


def test_derive_agent_name_user_identity_collapses_display_label_whitespace() -> None:
    assert (
        derive_agent_name(
            "agent-12", "user:scout", "Deep  Scout\tTeam", None, "gpt-6-luna"
        )
        == "Deep-Scout-Team-gpt-6-luna-12"
    )


def test_derive_agent_name_custom_identity_without_label_uses_identity_reference() -> (
    None
):
    assert derive_agent_name("agent-5", "scout", None, None, None) == "scout-5"


def test_derive_agent_name_uses_handle_sequence_digits() -> None:
    assert derive_agent_name("agent-15", "coder", None, None, None) == "coder-15"


@pytest.mark.parametrize(
    ("agent_id", "identity", "label", "base_model", "effective_model"),
    [
        ("agent-1", "coder", None, None, None),
        ("agent-2", "coder", "Coder", "base", "provider/wire"),
        ("agent-3", "scout", "Deep  Scout  Team", None, "mistral/zai-glm-5-3"),
        ("agent-15", "reviewer", "Reviewer", "glm-5.3", None),
        # Adversarial: an identity (or display label) of exactly "agent" would
        # derive the handle shape when no model is known.
        ("agent-7", "agent", None, None, None),
        ("agent-7", "agent", "agent", "gpt-6-luna", None),
        ("agent-7", "user:agent", "agent", None, None),
        ("agent-9", "agent", None, None, "mistral/zai-glm-5-3"),
        ("agent-9", "agent", "  agent  ", None, None),
    ],
)
def test_derived_names_never_parse_as_handles(
    agent_id: str,
    identity: str,
    label: str | None,
    base_model: str | None,
    effective_model: str | None,
) -> None:
    # Names are presentation only: the ``agent-\d+`` handle namespace stays
    # disjoint from derived names (design §5).
    name = derive_agent_name(agent_id, identity, label, base_model, effective_model)
    assert _HANDLE_PATTERN.fullmatch(name) is None
    assert name != agent_id


def test_derive_agent_name_breaks_the_handle_shape_for_agent_identities() -> None:
    # The derivation sanitizes the one configuration that would otherwise
    # collide with the handle namespace: prefixing the identity segment makes
    # the result non-handle-shaped deterministically.
    assert derive_agent_name("agent-7", "agent", None, None, None) == "agent-agent-7"
    # The display-label path (user:-qualified identities) is sanitized the
    # same way when the label is exactly "agent".
    assert derive_agent_name("agent-7", "user:agent", "agent", None, None) == (
        "agent-agent-7"
    )
    # A known model segment already breaks the shape on its own.
    assert (
        derive_agent_name("agent-7", "agent", None, "gpt-6-luna", None)
        == "agent-gpt-6-luna-7"
    )


def test_handle_shaped_agent_toml_is_retired_never_parsed(config_dir: Path) -> None:
    # The ``agent-\d+`` profile-name reservation retired with AgentProfile:
    # agents/*.toml is no longer parsed at all. A handle-shaped file is
    # reported through the retired agents directory and left untouched.
    agents_dir = config_dir / "agents"
    agents_dir.mkdir()
    profile_path = agents_dir / "agent-3.toml"
    original = b'description = "handle-shaped"\n'
    profile_path.write_bytes(original)
    manager = HarnessFilesManager(sources=("user",))
    assert agents_dir in manager.retired_agent_dirs
    # The AgentProfile module retired with it: no loader surface remains.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("chartreux.core.agents.models")
    assert profile_path.read_bytes() == original


# ---------------------------------------------------------------------------
# Name projection (registry → summaries, wire, acknowledgment, notifications)
# ---------------------------------------------------------------------------


def _resident_record(
    registry: SessionRuntimeRegistry,
    agent_id: str,
    *,
    clock: _ManualClock,
    identity: str = "coder",
    identity_display_label: str | None = None,
    base_model: str | None = None,
    effective_model: str | None = None,
    latest_run_id: str | None = "run-1",
) -> AgentRecord:
    record = _idle_record(agent_id, clock=clock)
    record.identity = identity
    # The attached-root tests bind a root with session id "root" and
    # generation 0; resident records belong to it (design §8 authority).
    record.parent_session_id = "root"
    record.identity_display_label = identity_display_label
    record.base_model = base_model
    record.effective_model = effective_model
    record.latest_run_id = latest_run_id
    registry._agent_records[agent_id] = record
    registry._children[record.session_id] = record.runtime
    return record


async def _scout_registry(
    monkeypatch: pytest.MonkeyPatch,
    backend: FakeBackend,
    *,
    identity_name: str = "scout",
) -> tuple[SessionRuntimeRegistry, AgentLoop, list[str]]:
    """A registry whose parent resolves a custom identity prompt.

    The custom identity's prompt file is installed by the autouse conftest
    fixture; agents/*.toml discovery is retired (16a).
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    config = _config(enabled_tools=["task", "todo"])
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    notifications: list[str] = []
    start = root.turns.start

    def record_notification(params: Any) -> Any:
        notifications.extend(
            block.text
            for block in params.message
            if isinstance(block, TextContentBlock)
        )
        return start(params)

    monkeypatch.setattr(root.turns, "start", record_notification)
    registry.bind_root(root)
    return registry, parent, notifications


@pytest.mark.asyncio
async def test_agent_summaries_project_derived_names() -> None:
    clock = _ManualClock(10)
    registry = _retention_registry(clock)
    _resident_record(
        registry,
        "agent-1",
        clock=clock,
        identity="user:scout",
        identity_display_label="Deep  Scout  Team",
        effective_model="mistral/zai-glm-5-3",
    )
    _resident_record(
        registry,
        "agent-2",
        clock=clock,
        identity="coder",
        identity_display_label="Coder",
        base_model="mistral/mistral-large-4",
        effective_model="mistral/mistral-large-4-wire",
    )
    _resident_record(registry, "agent-3", clock=clock, latest_run_id=None)

    summaries = await registry.check_agents()

    assert [summary.name for summary in summaries] == [
        "Deep-Scout-Team-zai-glm-5-3-1",
        "coder-mistral-large-4-2",
        "coder-3",
    ]
    # The name is a required wire field on every emitted summary.
    notify = AsyncMock()
    registry._notify_agents = notify
    await registry._emit_agents_update()
    await_args = notify.await_args
    assert await_args is not None
    emitted = await_args.args[0]
    assert [agent.name for agent in emitted] == [summary.name for summary in summaries]
    for agent in emitted:
        assert "name" in agent.model_dump(mode="json")


@pytest.mark.asyncio
async def test_tombstone_name_survives_metadata_expiry_window() -> None:
    clock = _ManualClock(100)
    registry = _retention_registry(clock)
    record = _resident_record(
        registry,
        "agent-1",
        clock=clock,
        base_model="glm-5.3",
        effective_model="zai/glm-5.3-wire",
    )

    summary = (await registry.check_agents())[0]
    assert summary.name == "coder-glm-5.3-1"

    assert registry._close_unretained_locked(record, None, name="test-close")
    closed = (await registry.check_agents())[0]
    assert closed.availability is AgentAvailability.CLOSED
    assert closed.name == "coder-glm-5.3-1"

    # Within the metadata window the Recent row keeps its frozen name; past
    # the tombstone TTL the row behaves as absent (never a stale name).
    clock.advance(registry._retention_policy.tombstone_ttl_seconds - 1)
    assert (await registry.check_agents())[0].name == "coder-glm-5.3-1"
    clock.advance(2)
    assert await registry.check_agents() == []
    await registry.drain_children()


@pytest.mark.asyncio
async def test_launch_acknowledgment_carries_name_alongside_handle_and_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    registry, parent, _notifications = await _scout_registry(monkeypatch, backend)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="scout work", identity="scout", background=True, retain=True),
            InvokeContext(tool_call_id="scout", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.identity == "scout"
        # A custom identity prompt carries no explicit display name.
        assert record.identity_display_label is None
        expected_name = derive_agent_name(
            record.agent_id,
            record.identity,
            record.identity_display_label,
            record.base_model,
            record.effective_model,
        )
        assert expected_name.startswith("scout-")
        # The launch acknowledgment carries the name as a top-level field and
        # in the acknowledgment text alongside agent_id/run_id.
        assert launch.name == expected_name
        assert launch.response.startswith(
            f"Background agent {expected_name} "
            f"({launch.agent_id}, run {launch.run_id}) launched and running;"
        )
        summary = (await registry.check_agents())[0]
        assert summary.name == expected_name
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_completion_notification_derives_from_its_own_run_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    notifications: list[str] = []
    start = root.turns.start

    def record_notification(params: Any) -> Any:
        notifications.extend(
            block.text
            for block in params.message
            if isinstance(block, TextContentBlock)
        )
        return start(params)

    monkeypatch.setattr(root.turns, "start", record_notification)
    registry.bind_root(root)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="notify me", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        await asyncio.wait_for(backend.started.wait(), 2)
        backend.release.set()
        await _wait_result(registry, launch.agent_id, launch.run_id)
        await _until(lambda: bool(notifications))
        record = registry._agent_records[launch.agent_id]
        expected_name = derive_agent_name(
            record.agent_id,
            record.identity,
            record.identity_display_label,
            record.base_model,
            record.effective_model,
        )
        assert any(
            f"Background agent {expected_name} ({launch.agent_id}, "
            f"run {launch.run_id}) completed." in text
            for text in notifications
        )
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_failed_background_run_carries_envelope_and_last_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP1: a failed background run's stored result carries the public typed
    failure envelope (run_status + recovery reference), and the agents/update
    summary projection carries it as ``last_failure``.
    """
    backend = ImmediateFailureBackend(RuntimeError("child backend failed immediately"))
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="fail", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(registry, launch.agent_id, launch.run_id)
        assert not failed.completed
        assert failed.run_status is RunStatus.FAILED
        envelope = failed.failure
        assert envelope is not None
        assert envelope.agent_id == launch.agent_id
        assert envelope.run_id == launch.run_id
        # The synthetic fault is unfamiliar: unknown reports-and-parks.
        assert envelope.recovery.eligibility is RecoveryEligibility.PROHIBITED
        assert failed.recovery is not None
        assert failed.recovery.event_id == envelope.event_id
        assert failed.recovery.recovery_chain_id == envelope.recovery_chain_id
        # The summary projection (the agents/update source) carries it.
        summary = next(
            item
            for item in await registry.check_agents()
            if item.agent_id == launch.agent_id
        )
        assert summary.last_failure is not None
        assert summary.last_failure.event_id == envelope.event_id
        assert summary.last_run_status is RunStatus.FAILED
        # The agents/update wire projection carries the projected envelope:
        # the public layer cannot import core models, so the server boundary
        # converts the canonical envelope onto the public wire model.
        notify = AsyncMock()
        registry._notify_agents = notify
        await registry._emit_agents_update()
        await_args = notify.await_args
        assert await_args is not None
        summaries, _evictions = await_args.args
        wire_summary = next(
            item for item in summaries if item.agent_id == launch.agent_id
        )
        assert wire_summary.last_failure is not None
        assert wire_summary.last_failure.event_id == envelope.event_id
        assert (
            wire_summary.last_failure.recovery.eligibility
            is PublicRecoveryEligibility.PROHIBITED
        )
        wire_dump = wire_summary.model_dump(mode="json", by_alias=True)
        assert wire_dump["lastFailure"]["event_id"] == envelope.event_id
        assert (
            wire_dump["lastFailure"]["failure"]["class"]
            == envelope.failure.failure_class.value
        )
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_suspended_failure_projects_recovery_state_on_the_wire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """WP2: a recoverably failed retained agent SUSPENDS warm; the summary
    and the agents/update wire projection carry the suspension truthfully —
    availability ``suspended``, the recovery-chain linkage, and the effective
    recovery eligibility — never defaulting to idle.
    """
    backend = ImmediateFailureBackend(
        EmptyLLMResponseError("test-provider", "test-model")
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="fail", identity="coder", background=True, retain=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert launch.agent_id is not None and launch.run_id is not None
        failed = await _wait_result(registry, launch.agent_id, launch.run_id)
        assert not failed.completed
        envelope = failed.failure
        assert envelope is not None
        # The typed empty-response transient with no tool calls: unknown
        # outcomes suspend behind a reconcile barrier.
        assert envelope.recovery.eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        record = registry._agent_records[launch.agent_id]
        await _until(lambda: record.state is _AgentState.SUSPENDED)

        # The summary projection shows the suspension truthfully (not idle).
        summary = next(
            item
            for item in await registry.check_agents()
            if item.agent_id == launch.agent_id
        )
        assert summary.availability is AgentAvailability.SUSPENDED
        assert summary.last_run_status is RunStatus.FAILED
        assert summary.latest_run_id == launch.run_id
        assert summary.recovery_chain_id == envelope.recovery_chain_id
        assert summary.same_agent_resumes_used == 0
        assert summary.recovery_eligibility is RecoveryEligibility.RECONCILE_REQUIRED
        assert summary.last_failure is not None
        assert summary.last_failure.event_id == envelope.event_id

        # The agents/update wire projection carries the same truth (the
        # public layer mirrors the core summary; it cannot import core).
        notify = AsyncMock()
        registry._notify_agents = notify
        await registry._emit_agents_update()
        await_args = notify.await_args
        assert await_args is not None
        summaries, _evictions = await_args.args
        wire_summary = next(
            item for item in summaries if item.agent_id == launch.agent_id
        )
        assert wire_summary.availability == "suspended"
        assert wire_summary.last_run_status == RunStatus.FAILED.value
        assert wire_summary.recovery_chain_id == envelope.recovery_chain_id
        assert wire_summary.same_agent_resumes_used == 0
        assert wire_summary.resumed_from_run_id is None
        assert (
            wire_summary.recovery_eligibility
            is PublicRecoveryEligibility.RECONCILE_REQUIRED
        )
        wire_dump = wire_summary.model_dump(mode="json", by_alias=True)
        assert wire_dump["recoveryChainId"] == envelope.recovery_chain_id
        assert wire_dump["sameAgentResumesUsed"] == 0
        assert wire_dump["recoveryEligibility"] == "reconcile_required"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_refused_reuse_never_relabels_the_resident_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([[mock_llm_chunk(content="scout done")]])
    registry, parent, _notifications = await _scout_registry(monkeypatch, backend)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="scout work", identity="scout", background=True, retain=True),
            InvokeContext(tool_call_id="scout", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        await _wait_result(registry, launch.agent_id, launch.run_id)
        before = (await registry.check_agents())[0]
        assert before.name.startswith("scout-")

        # A refused reuse (identity mismatch) must not relabel the record.
        with pytest.raises(AgentIdentityMismatchError):
            await _background_result(
                registry,
                TaskArgs(
                    task="wrong identity",
                    identity="reviewer",
                    agent_id=launch.agent_id,
                    background=True,
                ),
                InvokeContext(tool_call_id="refused", session_id=parent.session_id),
            )
        after = (await registry.check_agents())[0]
        assert after.name == before.name
        assert after.agent_id == before.agent_id
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_custom_identity_name_uses_the_identity_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Design §5: a custom identity prompt carries no explicit display name,
    # so the derivation always falls back to the identity reference — a
    # my-scout prompt derives my-scout-… (not My-Scout-…).
    backend = BlockingBackend()
    registry, parent, _notifications = await _scout_registry(
        monkeypatch, backend, identity_name="my-scout"
    )
    try:
        launch = await _background_result(
            registry,
            TaskArgs(
                task="scout work", identity="my-scout", background=True, retain=True
            ),
            InvokeContext(tool_call_id="scout", session_id=parent.session_id),
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.identity == "my-scout"
        # The admission capture stores no label for a custom identity prompt.
        assert record.identity_display_label is None
        expected_name = derive_agent_name(
            record.agent_id,
            record.identity,
            record.identity_display_label,
            record.base_model,
            record.effective_model,
        )
        assert expected_name.startswith("my-scout-")
        assert launch.name == expected_name
        assert (await registry.check_agents())[0].name == expected_name
    finally:
        backend.release.set()
        await registry.drain_children()
        await parent.aclose()


# ---------------------------------------------------------------------------
# Guarded release (design §3.1, §8)
# ---------------------------------------------------------------------------


def _attached_registry(
    clock: _ManualClock | None = None, *, generation: int = 0
) -> SessionRuntimeRegistry:
    """A registry with a bound root so guarded release can verify authority.

    The guarded release verifies attached-root/current-generation authority
    (design §8) exactly like the transcript authorization sites, so its
    tests run against a registry whose root is attached — as in production,
    where the agents/release RPC only arrives while a session root is bound.
    """
    registry = _retention_registry(clock)
    root = MagicMock()
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = generation
    registry._root = root
    registry._generation_identity = ("root", generation)
    return registry


def _running_record(
    agent_id: str = "agent-1",
    run_id: str = "run-1",
    *,
    clock: _ManualClock | None = None,
) -> tuple[SessionRuntimeRegistry, AgentRecord, RunRecord]:
    clock = clock or _ManualClock()
    registry = _attached_registry(clock)
    record = _idle_record(agent_id, clock=clock)
    record.parent_session_id = "root"
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.turns.close = AsyncMock()
    run = RunRecord(
        run_id=run_id,
        agent_id=agent_id,
        identity="coder",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
        started_at=clock(),
    )
    record.current_run = run
    record.latest_run_id = run_id
    record.state = _AgentState.RUNNING
    registry._agent_records[agent_id] = record
    registry._children[record.session_id] = record.runtime
    return registry, record, run


@pytest.mark.asyncio
async def test_guarded_release_conflicts_on_running_record() -> None:
    registry, record, run = _running_record()
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.CONFLICT
    # Nothing was stopped or purged.
    assert registry._agent_records["agent-1"] is record
    assert record.current_run is run
    assert not run.completion_task.cancelled()
    assert ("agent-1", "run-1") in registry._result_store
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_conflicts_on_finalizing_record() -> None:
    registry, record, _run = _running_record()
    record.state = _AgentState.FINALIZING

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.CONFLICT
    assert registry._agent_records["agent-1"] is record
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_conflicts_on_stale_expected_run_id() -> None:
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    record = _resident_record(registry, "agent-1", clock=clock, latest_run_id="run-2")
    registry._result_store[("agent-1", "run-2")] = _stored_result("agent-1", "run-2")

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.CONFLICT
    assert registry._agent_records["agent-1"] is record
    assert ("agent-1", "run-2") in registry._result_store
    # The matching expectation releases.
    assert (
        await registry.release_agent_guarded("agent-1", "run-2")
        is ReleaseAgentOutcome.RELEASED
    )
    assert "agent-1" not in registry._agent_records
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_conflict_never_stops_the_new_run() -> None:
    # Stale idle→running race: the UI saw run-1, the agent was reused with
    # run-2. The guarded release conflicts and the new run is untouched.
    registry, record, old_run = _running_record(run_id="run-1")
    new_run = RunRecord(
        run_id="run-2",
        agent_id="agent-1",
        identity="coder",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
    )
    record.run_history.append(old_run)
    record.current_run = new_run
    record.latest_run_id = "run-2"

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.CONFLICT
    assert registry._agent_records["agent-1"] is record
    assert record.current_run is new_run
    assert not new_run.completion_task.cancelled()
    assert not new_run.completion_task.done()
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_conflicts_on_replacement_reservation() -> None:
    registry, record, run = _running_record()
    record.reserved_for_replacement = True

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.CONFLICT
    assert registry._agent_records["agent-1"] is record
    assert not run.completion_task.cancelled()
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_unknown_for_released_agent() -> None:
    clock = _ManualClock(10)
    registry = _retention_registry(clock)
    _resident_record(registry, "agent-1", clock=clock)
    assert await registry.release_agent("agent-1") is ReleaseAgentOutcome.RELEASED

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.UNKNOWN
    assert "agent-1" not in registry._agent_records
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_unknown_for_metadata_expired_tombstone() -> None:
    clock = _ManualClock(100)
    registry = _retention_registry(clock)
    record = _resident_record(registry, "agent-1", clock=clock)
    assert registry._close_unretained_locked(record, None, name="test-close")
    assert "agent-1" in registry._evicted_agents
    clock.advance(registry._retention_policy.tombstone_ttl_seconds + 1)

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.UNKNOWN
    # The guard is non-destructive: the expired metadata is left to the reaper.
    assert "agent-1" in registry._evicted_agents
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_unknown_for_never_seen_agent() -> None:
    registry = _retention_registry()
    assert (
        await registry.release_agent_guarded("agent-99", "run-1")
        is ReleaseAgentOutcome.UNKNOWN
    )
    assert (
        await registry.release_agent_guarded("agent-99") is ReleaseAgentOutcome.UNKNOWN
    )
    await registry.drain_children()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parent_identity",
    [("other-session", 0), ("root", 1)],
    ids=["wrong-parent", "wrong-generation"],
)
async def test_guarded_release_unknown_for_foreign_tombstone(
    parent_identity: tuple[str, int],
) -> None:
    # Attached-root authority (design §8): a tombstone belonging to another
    # parent session or generation is not this root's to purge — the guarded
    # release reports UNKNOWN and leaves the metadata and results alone.
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    registry._evicted_agents["agent-1"] = _AgentTombstone(
        summary=AgentSummary(
            agent_id="agent-1",
            identity="coder",
            availability=AgentAvailability.EVICTED,
            current_run_id="run-1",
            current_run_status=RunStatus.COMPLETED,
            latest_run_id="run-1",
        ),
        child_session_id="session-agent-1",
        parent_identity=parent_identity,
    )
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.UNKNOWN
    # Nothing was purged: the foreign tombstone and its stored result survive.
    assert "agent-1" in registry._evicted_agents
    assert ("agent-1", "run-1") in registry._result_store
    await registry.drain_children()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parent_session_id", "root_generation"),
    [("other-session", 0), ("root", 1)],
    ids=["wrong-parent", "wrong-generation"],
)
async def test_guarded_release_unknown_for_stale_generation_record(
    parent_session_id: str, root_generation: int
) -> None:
    # Attached-root authority (design §8): a resident record from a foreign
    # parent or a stale generation is not this root's to release — UNKNOWN,
    # and the record survives untouched.
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    record = _resident_record(registry, "agent-1", clock=clock)
    record.parent_session_id = parent_session_id
    record.root_generation = root_generation
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.UNKNOWN
    assert registry._agent_records["agent-1"] is record
    assert ("agent-1", "run-1") in registry._result_store
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_idle_record_succeeds_with_full_cleanup() -> None:
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    record = _resident_record(registry, "agent-1", clock=clock)
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")
    registry._expired_results.add(("agent-1", "expired-run"))
    registry._pending_notifications.add(("agent-1", "run-1"))
    registry._wait_leases[("agent-1", "run-1")] = 1
    registry._result_write_tokens[("agent-1", "run-1")] = object()
    registry._latest_run_ids["agent-1"] = "run-1"
    notify = AsyncMock()
    registry._notify_agents = notify

    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.RELEASED
    assert "agent-1" not in registry._agent_records
    assert "agent-1" not in registry._children
    assert not [key for key in registry._result_store if key[0] == "agent-1"]
    assert not [key for key in registry._expired_results if key[0] == "agent-1"]
    assert not [key for key in registry._pending_notifications if key[0] == "agent-1"]
    assert not [key for key in registry._wait_leases if key[0] == "agent-1"]
    assert not [key for key in registry._result_write_tokens if key[0] == "agent-1"]
    assert "agent-1" not in registry._latest_run_ids
    cast(AsyncMock, record.runtime.close).assert_awaited_once()
    # The reduced inventory was published.
    await_args = notify.await_args
    assert await_args is not None
    assert await_args.args[0] == []
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_recent_tombstone_succeeds_and_purges_results() -> None:
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    record = _resident_record(registry, "agent-1", clock=clock)
    assert registry._close_unretained_locked(record, None, name="test-close")
    registry._result_store[("agent-1", "run-1")] = _stored_result("agent-1", "run-1")
    registry._latest_run_ids["agent-1"] = "run-1"
    notify = AsyncMock()
    registry._notify_agents = notify

    # A stale expectation conflicts; the matching one removes the tombstone.
    assert (
        await registry.release_agent_guarded("agent-1", "run-0")
        is ReleaseAgentOutcome.CONFLICT
    )
    assert "agent-1" in registry._evicted_agents
    outcome = await registry.release_agent_guarded("agent-1", "run-1")

    assert outcome is ReleaseAgentOutcome.EVICTED
    assert "agent-1" not in registry._evicted_agents
    assert ("agent-1", "run-1") not in registry._result_store
    assert "agent-1" not in registry._latest_run_ids
    await_args = notify.await_args
    assert await_args is not None
    assert await_args.args[0] == []
    await registry.drain_children()


@pytest.mark.asyncio
async def test_guarded_release_without_expected_run_id_releases_idle() -> None:
    clock = _ManualClock(10)
    registry = _attached_registry(clock)
    _resident_record(registry, "agent-1", clock=clock)

    outcome = await registry.release_agent_guarded("agent-1")

    assert outcome is ReleaseAgentOutcome.RELEASED
    assert "agent-1" not in registry._agent_records
    await registry.drain_children()


@pytest.mark.asyncio
async def test_unguarded_release_agent_stays_available_for_active_instances() -> None:
    # The model tool's release_agent stays unguarded: active instances are
    # released (design §3.1).
    registry, _record, _run = _running_record()

    outcome = await registry.release_agent("agent-1")

    assert outcome is ReleaseAgentOutcome.RELEASED
    assert "agent-1" not in registry._agent_records
    assert "agent-1" not in registry._children
    await asyncio.gather(*tuple(registry._teardown_tasks), return_exceptions=True)
    await registry.drain_children()


# ---------------------------------------------------------------------------
# agents/release RPC route, handler dispatch, and client method (design §8)
# ---------------------------------------------------------------------------


async def _release_rpc_session() -> tuple[
    AppServerClient, AppServerSession, SessionBackendImpl, AgentLoop
]:
    parent = build_test_agent_loop()
    client_transport, server_transport = memory_transport_pair()
    server = build_test_app_server(parent, server_transport)
    client = AppServerClient(client_transport, run_peer=server.serve)
    session = await attach_test_app_server_session(client)
    root = legacy_backend(server)
    return client, session, root, parent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected_reason"),
    [
        (
            ReleaseAgentOutcome.RELEASED,
            "Agent released; retained results and browser access are discarded.",
        ),
        (
            ReleaseAgentOutcome.EVICTED,
            "Terminal metadata removed; retained results and browser access are "
            "discarded.",
        ),
        (
            ReleaseAgentOutcome.CONFLICT,
            "Agent is active, reserved for replacement, or its latest run "
            "changed; nothing was released.",
        ),
        (ReleaseAgentOutcome.UNKNOWN, "Unknown agent; nothing was released."),
    ],
)
async def test_agents_release_route_dispatches_to_guarded_registry_release(
    outcome: ReleaseAgentOutcome, expected_reason: str
) -> None:
    client, session, root, _parent = await _release_rpc_session()
    guard = AsyncMock(return_value=outcome)
    root.children.release_agent_guarded = guard
    try:
        assert "agents/release" in SERVER_METHODS
        response = await client.request(
            "agents/release", {"agentId": "child", "expectedRunId": "run-1"}
        )
        assert response == {"outcome": outcome.value, "reason": expected_reason}
        guard.assert_awaited_once_with("child", "run-1")

        # A missing expectation still dispatches (no identity check).
        guard.reset_mock()
        response = await client.request("agents/release", {"agentId": "child"})
        assert response["outcome"] == outcome.value
        guard.assert_awaited_once_with("child", None)
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_agents_resume_route_dispatches_to_registry_resume() -> None:
    """WP2: the agents/resume RPC route dispatches to the registry's explicit
    resume admission, defaulting the idempotency key to the run pin.
    """
    client, session, root, parent = await _release_rpc_session()
    guard = AsyncMock(
        return_value=ResumeAgentResult(
            outcome=ResumeOutcome.NOT_RESUMABLE,
            agent_id="child",
            run_id="run-1",
            reason="Agent is idle, not suspended",
            reason_codes=["not_suspended"],
        )
    )
    root.children.resume_agent = guard
    try:
        assert "agents/resume" in SERVER_METHODS
        response = await client.request(
            "agents/resume", {"agentId": "child", "expectedRunId": "run-1"}
        )
        assert response["outcome"] == "not_resumable"
        assert response["agentId"] == "child"
        assert response["runId"] == "run-1"
        assert response["reasonCodes"] == ["not_suspended"]
        guard.assert_awaited_once_with(
            "child", "run-1", "child:run-1", requester_session_id=parent.session_id
        )

        # An explicit idempotency key is forwarded verbatim.
        guard.reset_mock()
        response = await client.request(
            "agents/resume",
            {"agentId": "child", "expectedRunId": "run-1", "idempotencyKey": "key-1"},
        )
        assert response["outcome"] == "not_resumable"
        guard.assert_awaited_once_with(
            "child", "run-1", "key-1", requester_session_id=parent.session_id
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_agents_reconcile_route_dispatches_to_registry_reconcile() -> None:
    """WP2: the agents/reconcile RPC route dispatches to the registry's
    run/boundary-pinned reconcile_result barrier.
    """
    client, session, root, parent = await _release_rpc_session()
    guard = AsyncMock(
        return_value=ReconcileResult(
            outcome=ReconcileOutcome.CLEARED,
            agent_id="child",
            run_id="run-1",
            transcript_boundary="turn-1",
            eligibility=RecoveryEligibility.ELIGIBLE,
            reason_codes=["effects_verified_settled", "call-1"],
            reason="All verified effects settled; the barrier is cleared.",
        )
    )
    root.children.reconcile_result = guard
    try:
        assert "agents/reconcile" in SERVER_METHODS
        response = await client.request(
            "agents/reconcile",
            {
                "agentId": "child",
                "expectedRunId": "run-1",
                "transcriptBoundary": "turn-1",
                "verifiedEffects": [{"toolCallId": "call-1", "disposition": "settled"}],
            },
        )
        assert response["outcome"] == "cleared"
        assert response["agentId"] == "child"
        assert response["runId"] == "run-1"
        assert response["transcriptBoundary"] == "turn-1"
        assert response["eligibility"] == "eligible"
        assert response["reasonCodes"] == ["effects_verified_settled", "call-1"]
        guard.assert_awaited_once_with(
            "child",
            "run-1",
            "turn-1",
            [EffectVerification(tool_call_id="call-1", disposition="settled")],
            requester_session_id=parent.session_id,
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_agents_release_rejects_invalid_params() -> None:
    client, session, _root, _parent = await _release_rpc_session()
    try:
        with pytest.raises(AppServerResponseError) as exc:
            await client.request("agents/release", {"expectedRunId": "run-1"})
        assert exc.value.error.code is ProtocolErrorCode.INVALID_PARAMS
        with pytest.raises(AppServerResponseError) as exc:
            await client.request("agents/release", {"agentId": "child", "extra": 1})
        assert exc.value.error.code is ProtocolErrorCode.INVALID_PARAMS
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_agents_release_requires_attached_session() -> None:
    client, session, root, parent = await _release_rpc_session()
    try:
        root.coordinator.attach("stale-session")
        with pytest.raises(AppServerResponseError) as exc:
            await client.request("agents/release", {"agentId": "child"})
        assert exc.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        root.coordinator.attach(parent.session_id)
        await session.close()


@pytest.mark.asyncio
async def test_agents_release_rejected_while_handler_closed() -> None:
    client, session, root, _parent = await _release_rpc_session()
    try:
        root.handler._closed = True
        with pytest.raises(AppServerResponseError) as exc:
            await client.request("agents/release", {"agentId": "child"})
        assert exc.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        root.handler._closed = False
        await session.close()


@pytest.mark.asyncio
async def test_agents_release_keeps_lifecycle_transition_guards() -> None:
    # Release has no cancel-style lifecycle exemption (design §8).
    client, session, root, _parent = await _release_rpc_session()
    owner = root.session.execution.begin(SessionExecutionKind.LIFECYCLE, "lifecycle")
    try:
        with pytest.raises(AppServerResponseError) as exc:
            await client.request(
                "agents/release", {"agentId": "child", "expectedRunId": "run-1"}
            )
        assert exc.value.error.code is ProtocolErrorCode.CONFLICT
    finally:
        root.session.execution.finish(owner)
        await session.close()


@pytest.mark.asyncio
async def test_session_release_agent_forwards_target_and_validates_response() -> None:
    session = object.__new__(AppServerSession)
    client = AsyncMock()
    client.request.return_value = {
        "outcome": "conflict",
        "reason": "Agent is active; nothing was released.",
    }
    session._ensure_attached = AsyncMock(return_value=client)
    response = await session.release_agent("agent", "run-1")
    assert isinstance(response, AgentsReleaseResponse)
    assert response.outcome is ReleaseOutcome.CONFLICT
    assert response.reason is not None
    client.request.assert_awaited_once_with(
        "agents/release", AgentsReleaseParams(agent_id="agent", expected_run_id="run-1")
    )
    # Without an expectation the client still sends the route.
    client.request.reset_mock()
    await session.release_agent("agent")
    client.request.assert_awaited_once_with(
        "agents/release", AgentsReleaseParams(agent_id="agent", expected_run_id=None)
    )
    client.request.return_value = {"outcome": "released", "expected_run_id": "private"}
    with pytest.raises(ValidationError):
        await session.release_agent("agent", "run-1")


# ---------------------------------------------------------------------------
# Typed waits (design §6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_wait_returns_still_running_without_cancelling_the_run() -> None:
    registry, _record, run = _running_record()

    disposition = await registry.wait_for_agent("agent-1", "run-1", timeout=0.01)

    # A timed-out wait is a normal still_running result, not a failure.
    assert disposition.status is WaitStatus.STILL_RUNNING
    assert disposition.result is None
    assert disposition.agent_id == "agent-1"
    assert disposition.run_id == "run-1"
    assert disposition.run_status is RunStatus.RUNNING
    assert disposition.elapsed_seconds is not None
    assert disposition.elapsed_seconds >= 0
    assert disposition.turns_used == 0
    guidance = disposition.guidance or ""
    # Turn-yield guidance: end the turn now, re-invoke in a later turn.
    assert "End your turn" in guidance
    assert "in a later turn" in guidance
    assert "check_agents" in guidance
    assert "same turn" not in guidance.lower()
    assert "immediately" not in guidance.lower()
    # The timeout never cancels the run.
    assert not run.completion_task.cancelled()
    assert not run.completion_task.done()
    await registry.drain_children()


@pytest.mark.asyncio
async def test_wait_pins_run_identity_across_a_replacement_race() -> None:
    registry, record, pinned_run = _running_record()

    waiter = asyncio.create_task(
        registry.wait_for_agent("agent-1", "run-1", timeout=0.05)
    )
    await asyncio.sleep(0)
    assert registry._wait_leases[("agent-1", "run-1")] == 1

    # A replacement swaps in a newer run before the timeout fires.
    newer_run = RunRecord(
        run_id="run-2",
        agent_id="agent-1",
        identity="coder",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
    )
    record.run_history.append(pinned_run)
    record.current_run = newer_run
    record.latest_run_id = "run-2"
    # The replacement run makes progress; that progress must never leak into
    # the pinned run's disposition.
    record.runtime.agent_loop.messages.append(
        LLMMessage(role=Role.assistant, content="newer run progress")
    )

    disposition = await waiter

    # The still_running result reports the run the caller targeted, never the
    # newer run's progress.
    assert disposition.status is WaitStatus.STILL_RUNNING
    assert disposition.run_id == "run-1"
    assert disposition.run_status is RunStatus.RUNNING
    # A displaced pinned run reports zero turns, not the newer run's count.
    assert disposition.turns_used == 0
    assert not newer_run.completion_task.cancelled()
    await registry.drain_children()


@pytest.mark.asyncio
async def test_wait_against_a_released_handle_raises_unknown_run() -> None:
    # §3.1 wait × Released: a wait on a released handle surfaces the typed
    # unknown-run error (the release fencing is unchanged from pre-WP3).
    registry, _record, _run = _running_record()
    assert await registry.release_agent("agent-1") is ReleaseAgentOutcome.RELEASED

    with pytest.raises(UnknownAgentError, match="Unknown run run-1"):
        await registry.wait_for_agent("agent-1", "run-1", timeout=0.01)

    await registry.drain_children()


@pytest.mark.asyncio
async def test_wait_explicit_none_is_indefinite() -> None:
    registry, _record, run = _running_record()

    waiter = asyncio.create_task(
        registry.wait_for_agent("agent-1", "run-1", timeout=None)
    )
    await asyncio.sleep(0)
    assert not waiter.done()
    # The indefinite wait survives well past any bounded timeout.
    await asyncio.sleep(0.05)
    assert not waiter.done()

    result = TaskResult(
        response="done",
        turns_used=1,
        completed=True,
        agent_id="agent-1",
        run_id="run-1",
    )
    run.result = result
    run.status = RunStatus.COMPLETED
    run.completion_task.set_result(None)

    disposition = await asyncio.wait_for(waiter, 1)
    assert disposition.status is WaitStatus.COMPLETED
    assert disposition.result == result
    await registry.drain_children()


def test_wait_args_default_timeout_is_1500_seconds() -> None:
    args = WaitForAgentArgs(agent_id="agent-1")
    assert args.timeout == 1500
    assert args.timeout == DEFAULT_WAIT_TIMEOUT_SECONDS
    description = WaitForAgentArgs.model_fields["timeout"].description or ""
    assert "1500" in description


def test_wait_args_explicit_none_means_indefinite() -> None:
    assert WaitForAgentArgs(agent_id="agent-1", timeout=None).timeout is None


def test_wait_args_allows_larger_than_default_timeouts() -> None:
    assert WaitForAgentArgs(agent_id="agent-1", timeout=99999).timeout == 99999


@pytest.mark.parametrize("timeout", [0, -1])
def test_wait_args_rejects_non_positive_timeouts(timeout: float) -> None:
    with pytest.raises(ValidationError):
        WaitForAgentArgs(agent_id="agent-1", timeout=timeout)


@pytest.mark.asyncio
async def test_wait_lease_protects_result_from_count_expiry() -> None:
    registry = _retention_registry()
    key = ("agent-1", "run-1")
    target = RunRecord(
        run_id="run-1",
        agent_id="agent-1",
        identity="coder",
        status=RunStatus.RUNNING,
        completion_task=asyncio.get_running_loop().create_future(),
    )
    record = _idle_record(key[0], clock=_ManualClock())
    record.runtime.agent_loop.messages = MessageList()
    record.current_run = target
    record.latest_run_id = "run-1"
    record.state = _AgentState.RUNNING
    registry._agent_records[key[0]] = record
    registry._result_store[key] = _stored_result(*key)
    for index in range(33):
        stored = _stored_result(f"agent-{index + 1:02}", "run")
        registry._result_store[(stored.agent_id, stored.run_id)] = stored

    waiter = asyncio.create_task(registry.wait_for_agent(*key))
    await asyncio.sleep(0)
    registry._expire_results_locked()
    assert key not in registry._expired_results

    target.result = _stored_result(*key).result
    target.status = RunStatus.COMPLETED
    target.completion_task.set_result(None)
    disposition = await waiter
    assert disposition.status is WaitStatus.COMPLETED
    assert disposition.result == target.result
    await registry.drain_children()


@pytest.mark.asyncio
async def test_wait_evicted_agent_with_retained_result_succeeds() -> None:
    registry = _retention_registry()
    stored = _stored_result("agent-1", "run-1")
    registry._result_store[(stored.agent_id, stored.run_id)] = stored
    registry._evicted_agents[stored.agent_id] = _AgentTombstone(
        summary=AgentSummary(
            agent_id=stored.agent_id,
            identity="coder",
            availability=AgentAvailability.EVICTED,
            current_run_id=stored.run_id,
            current_run_status=RunStatus.COMPLETED,
            name=derive_agent_name(stored.agent_id, "coder", None, None, None),
        ),
        child_session_id=f"session-{stored.agent_id}",
        parent_identity=("root", 0),
    )

    disposition = await registry.wait_for_agent("agent-1", "run-1")

    assert disposition.status is WaitStatus.COMPLETED
    assert disposition.result == stored.result
    await registry.drain_children()


@pytest.mark.asyncio
async def test_wait_cancelled_run_returns_partial_result() -> None:
    registry, _record, run = _running_record()
    partial = TaskResult(
        response="partial",
        turns_used=1,
        completed=False,
        stop_reason=RunStopReason.USER_CANCELLED,
        agent_id="agent-1",
        run_id="run-1",
    )
    run.result = partial
    run.status = RunStatus.CANCELLED
    run.completion_task.set_result(None)

    disposition = await registry.wait_for_agent("agent-1", "run-1")

    assert disposition.status is WaitStatus.COMPLETED
    assert disposition.result == partial
    assert disposition.result is not None
    assert not disposition.result.completed
    await registry.drain_children()


@pytest.mark.asyncio
async def test_wait_tool_serializes_still_running_as_a_normal_result() -> None:
    manager = FakeSubagentManager([
        AgentSummary(
            agent_id="agent-1",
            identity="coder",
            availability=AgentAvailability.RUNNING,
            current_run_id="run-1",
            current_run_status=RunStatus.RUNNING,
            name=derive_agent_name("agent-1", "coder", None, None, None),
        )
    ])
    tool = WaitForAgent(
        config_getter=lambda: WaitForAgentConfig(), state=BaseToolState()
    )

    result = await collect_result(
        tool.run(
            WaitForAgentArgs(agent_id="agent-1", run_id="run-1", timeout=0.01),
            InvokeContext(tool_call_id="wait", subagent_manager=manager),
        )
    )

    assert result.status is WaitStatus.STILL_RUNNING
    serialized = result.model_dump(mode="json")
    assert serialized["status"] == "still_running"
    assert serialized["agent_id"] == "agent-1"
    assert serialized["run_id"] == "run-1"
    assert serialized["run_status"] == "running"
    assert "guidance" in serialized
    assert serialized["result"] is None
    # The run was never cancelled: the completion event stays unset.
    assert not manager.completed[("agent-1", "run-1")].is_set()


# ---------------------------------------------------------------------------
# Legacy nameless payload acceptance (client-side, design §5)
# ---------------------------------------------------------------------------


def _projection() -> ClientProjection:
    return ClientProjection(
        PublicSessionState(
            event_id=0,
            session=PublicSession(
                id="session-1", status=IdleSessionStatus(), created_at=1, updated_at=1
            ),
            history=[],
            turns=[],
            active_callbacks=[],
        )
    )


def test_agents_update_accepts_legacy_nameless_payload() -> None:
    projection = _projection()
    params = {
        "eventId": 1,
        "sessionId": "session-1",
        "emittedAt": 2,
        "agents": [
            {"agentId": "agent-1", "availability": "idle"},
            {"agentId": "agent-2", "availability": "idle"},
        ],
    }

    event = projection.consume(Notification(method="agents/update", params=params))

    assert isinstance(event, AgentsUpdate)
    assert [agent.name for agent in event.agents] == ["", ""]
    assert [agent.agent_id for agent in event.agents] == ["agent-1", "agent-2"]


def test_agents_update_preserves_name_when_present() -> None:
    projection = _projection()
    params = {
        "eventId": 1,
        "sessionId": "session-1",
        "emittedAt": 2,
        "agents": [
            {"agentId": "agent-1", "name": "coder-base-1", "availability": "idle"}
        ],
    }

    event = projection.consume(Notification(method="agents/update", params=params))

    assert isinstance(event, AgentsUpdate)
    assert event.agents[0].name == "coder-base-1"
    # The server payload is not mutated by the client-side normalization.
    assert "name" in params["agents"][0]
