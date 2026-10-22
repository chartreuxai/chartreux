"""Launch behavior tests for subagents."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._projection import project_launch_purposes
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import (
    RunRecord,
    SessionRuntime,
    SessionRuntimeRegistry,
    _AgentState,
)
from chartreux.app_server._turns import TurnController
from chartreux.app_server.models import (
    PublicEffectEntry,
    PublicMessageEntry,
    SubagentEffectDetail,
)
from chartreux.app_server.protocol import AppServerResponseError, ProtocolErrorCode
from chartreux.app_server.server import AppServer
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.identity_launch import (
    InvalidChildIdentityError,
    InvalidLaunchPurposeError,
)
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.config.chartreux_schema import OrchestrationConfig
from chartreux.core.dispatch.route_models import (
    PackageTier as RoutePackageTier,
    PurposeRoute,
    ReplacementCapability,
    RouteTarget,
)
from chartreux.core.message_list import MessageList
from chartreux.core.model_catalog.loader import CatalogSnapshot
from chartreux.core.recovery_models import PackageTier
from chartreux.core.session.session_loader import SessionLoader
from chartreux.core.session.session_logger import SessionLogger
from chartreux.core.session_types import ChildSessionLink, LaunchMetadataV2
from chartreux.core.subagents import (
    CancelOutcome,
    LaunchConfig,
    LaunchConfigError,
    RunStatus,
    RunStopReason,
    TaskArgs,
    TaskResult,
    UnknownAgentError,
    derive_agent_name,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission, ToolPermissionError
from chartreux.utils.tool_presentation import ToolEffectKind
from tests.app_server._subagent_test_support import (
    BlockingBackend,
    GatedSequenceBackend,
    _background_result,
    _config,
    _consume,
    _dynamic_config,
    _dynamic_registry,
    _foreground_registry,
    _foreground_result,
    _GatedWakeup,
    _idle_record,
    _ManualClock,
    _persistent_dynamic_registry,
    _read_child,
    _real_reused_background,
    _reaper_registry,
    _retention_registry,
    _task_call,
    _wait_result,
)
from tests.conftest import (
    build_test_agent_loop,
    build_test_vibe_config,
    set_agent_config,
    single_model_catalog,
)
from tests.mock.utils import mock_llm_chunk
from tests.stubs.app_server import (
    attach_test_app_server_session,
    create_test_app_server_session,
    legacy_backend,
    start_test_app_server,
)
from tests.stubs.fake_backend import FakeBackend


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["completion", "release"])
async def test_running_subagent_cap_excludes_idle_and_frees_capacity(
    monkeypatch: pytest.MonkeyPatch, finish: str
) -> None:
    backends = [BlockingBackend()]

    def create_backend(**_kwargs):
        return backends[-1]

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    config = _config()
    config.subagents.max_running_subagents = 1
    parent = build_test_agent_loop(
        config=config,
        backend=FakeBackend([
            [mock_llm_chunk(content="Notification received")] for _ in range(3)
        ]),
    )
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    registry.bind_root(registry._build_child_runtime(parent))
    ctx = InvokeContext(tool_call_id="cap", session_id=parent.session_id)
    # The first agent is reused below, so its launch opts into retention
    # explicitly instead of relying on the ephemeral-by-default global.
    args = TaskArgs(task="work", identity="coder", background=True, retain=True)
    create_child = AsyncMock(wraps=registry._runtime_factory.create_child)
    monkeypatch.setattr(registry._runtime_factory, "create_child", create_child)
    try:
        first = await _background_result(registry, args, ctx)
        assert first.agent_id is not None and first.run_id is not None
        await asyncio.wait_for(backends[0].started.wait(), timeout=5)
        for background in (True, False):
            with pytest.raises(
                RuntimeError,
                match=r"Subagent launch rejected: 1 agents already running \(cap 1\); "
                "wait for completion or release agents",
            ):
                await asyncio.wait_for(
                    _background_result(
                        registry,
                        args.model_copy(update={"background": background}),
                        ctx,
                    ),
                    timeout=1,
                )
        assert create_child.await_count == 1  # Rejected before child allocation.

        backends[0].release.set()
        await registry.wait_for_agent(first.agent_id, first.run_id)
        assert registry._agent_records[first.agent_id].state is _AgentState.IDLE
        backends.append(BlockingBackend())
        second = await _background_result(registry, args, ctx)
        assert second.agent_id is not None and second.run_id is not None
        await asyncio.wait_for(backends[1].started.wait(), timeout=5)
        # An idle retained agent cannot bypass the active-work cap by being reused.
        with pytest.raises(RuntimeError, match="cap 1"):
            await _background_result(
                registry, args.model_copy(update={"agent_id": first.agent_id}), ctx
            )
        assert create_child.await_count == 2
        if finish == "release":
            await registry.release_agent(second.agent_id)
        else:
            backends[1].release.set()
            await registry.wait_for_agent(second.agent_id, second.run_id)
        third = await _background_result(
            registry, args.model_copy(update={"agent_id": first.agent_id}), ctx
        )
        assert third.agent_id == first.agent_id
    finally:
        for backend in backends:
            backend.release.set()
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_foreground_subagent_consumes_cap_until_completion(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    backend = BlockingBackend()
    registry, parent = await _foreground_registry(monkeypatch, backend)
    config = parent.config.model_copy(deep=True)
    config.subagents.max_running_subagents = 1
    set_agent_config(parent, config)
    launch = asyncio.create_task(_foreground_result(registry, parent, "foreground"))
    try:
        await asyncio.wait_for(backend.started.wait(), timeout=5)
        foreground_id = next(iter(registry._children))
        with pytest.raises(ValueError, match="Foreground agents.*only background"):
            await registry.cancel_run(
                foreground_id,
                reason=RunStopReason.ORCHESTRATOR_CANCELLED,
                requester_session_id=parent.session_id,
            )
        assert not backend.stopped.is_set()
        with pytest.raises(RuntimeError, match="cap 1"):
            await _background_result(
                registry,
                TaskArgs(task="more", identity="coder", background=True),
                InvokeContext(tool_call_id="more", session_id=parent.session_id),
            )
        if cancel:
            launch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await launch
        else:
            backend.release.set()
            assert (await launch).completed
        backend.release.set()
        await asyncio.gather(*registry._teardown_tasks)
        assert (await _foreground_result(registry, parent, "next")).completed
        assert not registry._active_work_slots
    finally:
        backend.release.set()
        await asyncio.gather(launch, return_exceptions=True)
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_subagent_cap_reserves_pending_creation_and_rolls_back_failure(
    monkeypatch: pytest.MonkeyPatch, cancel: bool
) -> None:
    config = _config()
    config.subagents.max_running_subagents = 1
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    registry.bind_root(registry._build_child_runtime(parent))
    entered = asyncio.Event()
    release = asyncio.Event()

    async def fail_creation(*_args):
        entered.set()
        await release.wait()
        raise RuntimeError("creation failed")

    monkeypatch.setattr(registry._runtime_factory, "create_child", fail_creation)
    args = TaskArgs(task="work", identity="coder", background=True)
    ctx = InvokeContext(tool_call_id="cap", session_id=parent.session_id)
    launch = asyncio.create_task(_background_result(registry, args, ctx))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(RuntimeError, match="cap 1"):
            await asyncio.wait_for(_background_result(registry, args, ctx), timeout=1)
        if cancel:
            launch.cancel()
            with pytest.raises(asyncio.CancelledError):
                await launch
        else:
            release.set()
            with pytest.raises(RuntimeError, match="creation failed"):
                await launch
        assert not registry._active_work_slots
        release.set()
        with pytest.raises(RuntimeError, match="creation failed"):
            await _background_result(registry, args, ctx)
        assert not registry._active_work_slots
    finally:
        release.set()
        await asyncio.gather(launch, return_exceptions=True)
        await registry.close()
        await parent.aclose()


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_subagent_cap_constructor_rejects_invalid_overrides(value: Any) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SessionRuntimeRegistry(
            AsyncMock(), AsyncMock(), lambda _: 0, max_running_subagents=value
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("override", [None, 16])
async def test_subagent_cap_root_config_replacement_and_override_precedence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, override: int | None
) -> None:
    from chartreux.core.config.chartreux_schema import ChartreuxConfigSchema
    from chartreux.core.config.layers.overrides import OverridesLayer
    from chartreux.core.config.layers.user import UserConfigLayer
    from chartreux.core.config.orchestrator import ConfigOrchestrator
    from tests.stubs.fake_mcp_registry import FakeMCPRegistry

    backend = BlockingBackend()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    path = tmp_path / "config.toml"
    path.write_text("[subagents]\nmax_running_subagents = 2\n")
    user = UserConfigLayer(path=path)
    orchestrator = await ConfigOrchestrator.create(
        schema=ChartreuxConfigSchema,
        layers=[OverridesLayer(data=_config().model_dump(exclude_unset=True)), user],
        default_layer_resolver=lambda: user,
    )
    parent = AgentLoop(
        config_orchestrator=orchestrator,
        backend=FakeBackend(),
        mcp_registry=FakeMCPRegistry(),
    )
    registry = SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _: 0, max_running_subagents=override
    )
    registry.bind_root(registry._build_child_runtime(parent))
    args = TaskArgs(task="work", identity="coder", background=True)
    ctx = InvokeContext(tool_call_id="config-cap", session_id=parent.session_id)
    try:
        first = await _background_result(registry, args, ctx)
        assert first.agent_id is not None and first.run_id is not None
        await asyncio.wait_for(backend.started.wait(), timeout=5)
        child = registry._agent_records[first.agent_id].runtime.agent_loop
        child_config = child.config.model_copy(deep=True)
        child_config.subagents.max_running_subagents = 1
        set_agent_config(child, child_config)
        # A child cap of one does not prevent a second root-admitted run.
        second = await _background_result(registry, args, ctx)
        assert second.agent_id is not None and second.run_id is not None
        path.write_text("[subagents]\nmax_running_subagents = 1\n")
        await orchestrator.reload()
        assert len(registry._active_work_slots) == 2
        assert not backend.stopped.is_set()
        if override is None:
            with pytest.raises(RuntimeError, match="cap 1"):
                await _background_result(registry, args, ctx)
            path.write_text("[subagents]\nmax_running_subagents = 3\n")
            await orchestrator.reload()
        # Explicit 16 remains distinct from the unset constructor sentinel.
        third = await _background_result(registry, args, ctx)
        assert third.agent_id is not None
        assert len(registry._active_work_slots) == 3
    finally:
        backend.release.set()
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_bare_registry_subagent_cap_defaults_to_sixteen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    entered = asyncio.Event()
    release = asyncio.Event()
    count = 0

    async def admitted(self, args, ctx, slot, **kwargs):
        nonlocal count
        count += 1
        if count == 16:
            entered.set()
        await release.wait()
        yield TaskResult(completed=True, response="done", turns_used=1)

    monkeypatch.setattr(SessionRuntimeRegistry, "_run_admitted", admitted)
    args = TaskArgs(task="work", identity="coder", background=True)
    ctx = InvokeContext(tool_call_id="bare", session_id="bare")
    launches = [
        asyncio.create_task(_background_result(registry, args, ctx)) for _ in range(16)
    ]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        with pytest.raises(RuntimeError, match=r"16 agents already running \(cap 16\)"):
            await _background_result(registry, args, ctx)
    finally:
        release.set()
        await asyncio.gather(*launches)
    assert not registry._active_work_slots


@pytest.mark.asyncio
async def test_task_creates_independently_readable_child_session(monkeypatch) -> None:
    parent_backend = FakeBackend([
        [mock_llm_chunk(content="", tool_calls=[_task_call(background=False)])],
        [mock_llm_chunk(content="Parent completed")],
    ])
    child_backend = FakeBackend([mock_llm_chunk(content="Child completed")])
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(), backend=parent_backend, enable_streaming=True
    )
    session = await create_test_app_server_session(parent)

    try:
        _ = [event async for event in session.act("Delegate this")]
        effect = next(
            entry
            for entry in session.history
            if isinstance(entry, PublicEffectEntry)
            and entry.detail.kind is ToolEffectKind.SUBAGENT
        )
        assert isinstance(effect.detail, SubagentEffectDetail)
        child_session_id = effect.detail.child_session_id
        assert child_session_id is not None
        child_state = await _read_child(session, child_session_id)
        assert child_state.session.parent_session_id == session.session_id
        assert any(
            isinstance(entry, PublicMessageEntry) and entry.text == "Child completed"
            for entry in child_state.history or []
        )
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_reused_launch_failure_restores_idle_bookkeeping_and_rearms_reaper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    wakeup = _GatedWakeup()
    registry, root = _reaper_registry(clock, ttl=10, wakeup=wakeup)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock(side_effect=RuntimeError("projection failed"))
    record = _idle_record("agent-1", clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.runtime.agent_loop.session_id = "child"
    record.runtime.turns._event_sink = None
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = record.runtime

    def fail_start(*_args: object) -> tuple[str, object]:
        raise RuntimeError("start failed")

    monkeypatch.setattr(SessionRuntimeRegistry, "_start_child_turn", fail_start)
    with pytest.raises(RuntimeError, match="projection failed"):
        await _background_result(
            registry,
            TaskArgs(
                task="retry", identity="coder", agent_id="agent-1", background=True
            ),
            InvokeContext(tool_call_id="retry", session_id="root"),
        )

    assert record.state is _AgentState.IDLE
    assert record.idle_since == 0
    assert registry._eligible_idle_locked() == [record]
    assert registry._reaper_task is not None
    await registry.drain_children()


@pytest.mark.asyncio
async def test_child_creation_cancellation_at_publication_lock_closes_orphan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, root = _reaper_registry(_ManualClock(), generation=1)
    created = MagicMock()
    created.session_id = "unpublished-child"
    created_ready = asyncio.Event()
    lease_released = False
    runtime = MagicMock(spec=SessionRuntime)
    runtime.agent_loop = created
    runtime._closed = False
    runtime._close_task = None

    async def close() -> None:
        nonlocal lease_released
        lease_released = True
        runtime._closed = True

    runtime.close = close

    class Factory:
        async def create_child(self, *_args: object) -> MagicMock:
            created_ready.set()
            return created

    registry._runtime_factory = Factory()  # type: ignore[assignment]
    monkeypatch.setattr(registry, "_resolve_launch_candidate", MagicMock())
    monkeypatch.setattr(registry, "_build_child_runtime", lambda _child: runtime)
    await registry._ensure_child_lock.acquire()
    launch = asyncio.create_task(
        registry._create_registered_child(
            root,
            TaskArgs(task="new", identity="coder", background=True),
            InvokeContext(tool_call_id="new", session_id="root"),
        )
    )
    await created_ready.wait()
    await asyncio.sleep(0)
    launch.cancel()

    with pytest.raises(asyncio.CancelledError):
        await launch
    registry._ensure_child_lock.release()
    await registry.drain_children()

    assert lease_released
    assert registry._children == {}
    assert registry._pending_child_closes == {}
    assert registry._teardown_tasks == set()


@pytest.mark.asyncio
async def test_child_creation_generation_change_discards_orphan_without_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _ManualClock()
    registry, root = _reaper_registry(clock, generation=1)
    created = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()

    class Factory:
        async def create_child(self, *_args: object) -> MagicMock:
            entered.set()
            await release.wait()
            return created

    registry._runtime_factory = Factory()  # type: ignore[assignment]
    candidate = MagicMock()
    resolve = MagicMock(return_value=candidate)
    monkeypatch.setattr(registry, "_resolve_launch_candidate", resolve)
    discard = AsyncMock()
    monkeypatch.setattr(registry, "_discard_child", discard)
    create = asyncio.create_task(
        registry._create_registered_child(
            root,
            TaskArgs(task="new", identity="coder", background=True),
            InvokeContext(tool_call_id="new", session_id="root"),
        )
    )
    await entered.wait()
    resolve.assert_called_once_with(
        root, TaskArgs(task="new", identity="coder", background=True)
    )
    root.agent_loop._session_generation = 2
    registry.begin_root_generation()
    release.set()

    with pytest.raises(RuntimeError, match="admission changed"):
        await create
    discard.assert_awaited_once_with(created)
    assert registry._children == {}
    assert registry._child_links == {}


@pytest.mark.asyncio
async def test_background_launch_normalizes_and_caps_task_summary(monkeypatch) -> None:
    task_summary = "  " + "summary\n" * 100
    child_backend = GatedSequenceBackend([mock_llm_chunk(content="done")])
    started, release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([
            [
                mock_llm_chunk(
                    content="", tool_calls=[_task_call(task_summary=task_summary)]
                )
            ],
            [mock_llm_chunk(content="parent complete")],
        ]),
        enable_streaming=True,
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)

    try:
        await _consume(session.act("launch"))
        await asyncio.wait_for(started.wait(), timeout=1)
        registry = legacy_backend(server).children
        record = registry._agent_records["agent-1"]
        expected = " ".join(" ".join(task_summary.split())[:240].split())[:240]
        assert record.initial_task_summary == expected[:240]
        assert record.current_run is not None
        assert record.current_run.task_summary == expected[:240]
        release.set()
        await registry.wait_for_agent("agent-1", record.current_run.run_id)
    finally:
        release.set()
        await session.close()


@pytest.mark.asyncio
async def test_post_commit_launch_cancellation_keeps_the_monitor_and_exact_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling the acknowledgement cannot roll back a committed child run."""
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child done")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(), backend=FakeBackend(), enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    parent_runtime = registry._root
    assert parent_runtime is not None
    parent_runtime.turns._projector = MagicMock()
    parent_runtime.turns.link_subagent = AsyncMock()
    update_entered = asyncio.Event()
    update_release = asyncio.Event()
    calls = 0
    original_emit = registry._emit_agents_update

    async def gated_emit(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            update_entered.set()
            await update_release.wait()
        await cast(Any, original_emit)(*args)

    monkeypatch.setattr(registry, "_emit_agents_update", gated_emit)
    launch = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
    )
    try:
        await update_entered.wait()
        record = registry._agent_records["agent-1"]
        run = record.current_run
        assert run is not None and isinstance(run.completion_task, asyncio.Task)
        monitor = run.completion_task
        await child_started.wait()

        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch
        assert record.current_run is run
        assert record.latest_run_id == run.run_id
        assert record.state is _AgentState.RUNNING
        assert not monitor.done()

        child_release.set()
        await monitor
        result = await registry.get_agent_result(record.agent_id, run.run_id)
        assert result is not None
        assert result.completed is True
        assert result.run_id == run.run_id
    finally:
        update_release.set()
        child_release.set()
        await session.close()


@pytest.mark.asyncio
async def test_post_commit_reused_launch_cancellation_keeps_the_monitor_and_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child done")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    update_entered = asyncio.Event()
    update_release = asyncio.Event()
    original_emit = registry._emit_agents_update

    async def gated_emit(*values: object) -> None:
        if not update_entered.is_set():
            update_entered.set()
            await update_release.wait()
        await cast(Any, original_emit)(*values)

    monkeypatch.setattr(registry, "_emit_agents_update", gated_emit)
    launch = asyncio.create_task(_background_result(registry, args, context))
    try:
        await update_entered.wait()
        run = record.current_run
        assert run is not None and isinstance(run.completion_task, asyncio.Task)
        await child_started.wait()
        launch.cancel()
        with pytest.raises(asyncio.CancelledError):
            await launch
        assert record.current_run is run
        assert record.state is _AgentState.RUNNING
        assert not run.completion_task.done()

        child_release.set()
        update_release.set()
        await run.completion_task
        result = await registry.get_agent_result(record.agent_id, run.run_id)
        assert result is not None and result.completed
    finally:
        update_release.set()
        child_release.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_post_commit_update_failure_does_not_cancel_busy_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child_backend = GatedSequenceBackend([[mock_llm_chunk(content="child done")]])
    child_started, child_release = child_backend.add_gate()
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: child_backend
    )
    parent = build_test_agent_loop(
        config=_config(), backend=FakeBackend(), enable_streaming=True
    )
    client = start_test_app_server(parent)
    run_peer = client._run_peer
    assert run_peer is not None
    server = cast(AppServer, cast(Any, run_peer).__self__)
    session = await attach_test_app_server_session(client)
    registry = legacy_backend(server).children
    parent_runtime = registry._root
    assert parent_runtime is not None
    parent_runtime.turns._projector = MagicMock()
    parent_runtime.turns.link_subagent = AsyncMock()
    real_emit = registry._emit_agents_update
    emit_calls = 0

    async def fail_first_emit(*args: object) -> None:
        nonlocal emit_calls
        emit_calls += 1
        if emit_calls == 1:
            raise RuntimeError("initial update failed")
        await cast(Any, real_emit)(*args)

    monkeypatch.setattr(registry, "_emit_agents_update", fail_first_emit)
    try:
        acknowledgement = await _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
        assert acknowledgement.agent_id == "agent-1"
        assert acknowledgement.run_id is not None
        assert acknowledgement.status == "launched"
        record = registry._agent_records[acknowledgement.agent_id]
        expected_name = derive_agent_name(
            record.agent_id,
            record.identity,
            record.identity_display_label,
            record.base_model,
            record.effective_model,
        )
        assert acknowledgement.name == expected_name
        assert acknowledgement.response == (
            f"Background agent {expected_name} "
            f"({acknowledgement.agent_id}, run {acknowledgement.run_id}) "
            "launched and running; use check_agents to monitor, "
            "get_agent_result or wait_for_agent to retrieve the result. "
            "This agent is not retained and closes when the run completes; "
            "the result stays retrievable via get_agent_result. "
            "Launch with retain=true to keep a resident agent."
        )
        assert acknowledgement.turns_used == 0
        assert acknowledgement.completed is True
        run = record.current_run
        assert run is not None and run.run_id == acknowledgement.run_id
        assert isinstance(run.completion_task, asyncio.Task)
        monitor = run.completion_task
        await child_started.wait()
        assert record.state is _AgentState.RUNNING
        assert not monitor.done()

        child_release.set()
        await monitor
        result = await registry.get_agent_result(
            record.agent_id, acknowledgement.run_id
        )
        assert result is not None
        assert result.completed is True
        assert result.status is None
    finally:
        child_release.set()
        await session.close()


@pytest.mark.asyncio
async def test_cancel_in_aborted_launch_rollback_window_is_not_accepted() -> None:
    registry, parent, record, args, context = await _real_reused_background()
    record.parent_session_id = parent.session_id
    try:
        turn_id, action = registry._start_child_turn(
            record.runtime, args, context, record.session_id
        )
        run = RunRecord(
            run_id="attempt",
            agent_id=record.agent_id,
            identity=record.identity,
            status=RunStatus.RUNNING,
            completion_task=asyncio.get_running_loop().create_future(),
            initial_turn_id=turn_id,
        )
        record.current_run = run
        record.state = _AgentState.RUNNING
        # Reproduce the rollback window: real prepared start aborted, but the
        # registry has not restored its prior current_run yet.
        action.abort()
        assert record.runtime.turns.operation_pending_turn_id(turn_id) == turn_id
        result = await registry.cancel_run(
            record.agent_id,
            reason=RunStopReason.USER_CANCELLED,
            requester_session_id=parent.session_id,
        )
        assert result.outcome is CancelOutcome.ALREADY_FINISHING
        assert result.stop_reason is None and run.requested_stop_reason is None
    finally:
        record.current_run = None
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_reused_projection_linkage_failure_never_prepares_a_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock(side_effect=RuntimeError("link failed"))
    start = MagicMock(wraps=record.runtime.turns.start)
    monkeypatch.setattr(record.runtime.turns, "start", start)
    try:
        with pytest.raises(RuntimeError, match="link failed"):
            await _background_result(registry, args, context)
        start.assert_not_called()
        assert record.state is _AgentState.IDLE
        assert record.idle_since == 0
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.timeout(30)
async def test_reused_start_failure_aborts_real_prepared_turn_and_allows_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # AgentRuntimeFactory creates a fresh child backend; it does not reuse the
    # parent's injected FakeBackend, so keep the retry path offline explicitly.
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="Child completed")]),
    )
    registry, parent, record, args, context = await _real_reused_background()
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    original_start = record.runtime.turns.start
    failing_start = MagicMock(side_effect=RuntimeError("start failed"))

    def prepare_then_fail(*values: Any, **kwargs: Any) -> tuple[Any, Any]:
        response, action = original_start(*values, **kwargs)
        action._run = failing_start  # type: ignore[attr-defined]
        return response, action

    monkeypatch.setattr(record.runtime.turns, "start", prepare_then_fail)
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            await _background_result(registry, args, context)
        failing_start.assert_called_once()
        assert record.runtime.execution.active is None
        assert record.runtime.turns._pending_start is None
        assert record.runtime.turns.active_turn is None
        assert record.runtime.turns._active_task is None
        assert not registry._monitor_tasks
        assert registry._agent_summaries()[0].run_elapsed_seconds is None
        monkeypatch.setattr(record.runtime.turns, "start", original_start)
        result = await _background_result(registry, args, context)
        assert result.run_id is not None
        run = record.current_run
        assert run is not None
        try:
            await asyncio.wait_for(asyncio.shield(run.completion_task), timeout=20)
        except TimeoutError as exc:
            pending_tasks = [
                f"{task.get_name()}: {task.get_coro()!r}"
                for task in asyncio.all_tasks()
                if not task.done() and task is not asyncio.current_task()
            ]
            init_thread = record.runtime.agent_loop._deferred_init_thread
            thread_status = (
                "not started"
                if init_thread is None
                else f"alive={init_thread.is_alive()}"
            )
            raise AssertionError(
                "Retry completion timed out after 20 seconds.\n"
                f"Pending asyncio tasks: {pending_tasks!r}\n"
                f"Child deferred-init thread: {thread_status}"
            ) from exc
    finally:
        await registry.drain_children()
        await root.turns.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_run", [False, True])
async def test_prelaunch_registry_lock_cancellation_aborts_real_prepared_turn(
    monkeypatch: pytest.MonkeyPatch, cancel_run: bool
) -> None:
    registry, parent, record, args, context = await _real_reused_background()
    record.parent_session_id = parent.session_id
    root = registry._root
    assert root is not None
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    prepared = asyncio.Event()
    final_lock_entered = asyncio.Event()
    release_final_lock = asyncio.Event()
    original_start_child_turn = SessionRuntimeRegistry._start_child_turn

    def signal_prepared(*values: Any) -> tuple[str, Any]:
        result = original_start_child_turn(*values)
        prepared.set()
        return result

    class GatedRegistryLock:
        def __init__(self, lock: asyncio.Lock) -> None:
            self.lock = lock
            self.entries = 0

        async def __aenter__(self) -> None:
            await self.lock.acquire()
            self.entries += 1
            if self.entries == 3:
                final_lock_entered.set()
                self.lock.release()
                await release_final_lock.wait()
                await self.lock.acquire()

        async def __aexit__(self, *exc: object) -> None:
            self.lock.release()

    monkeypatch.setattr(SessionRuntimeRegistry, "_start_child_turn", signal_prepared)
    registry._registry_lock = GatedRegistryLock(registry._registry_lock)  # type: ignore[assignment]
    launch = asyncio.create_task(_background_result(registry, args, context))
    try:
        await final_lock_entered.wait()
        assert prepared.is_set()
        if cancel_run:
            run = record.current_run
            assert run is not None
            waiter = asyncio.create_task(_wait_result(registry, record.agent_id))
            await asyncio.sleep(0)
            assert registry._wait_leases[(record.agent_id, run.run_id)] == 1
            outcome = await registry.cancel_run(
                record.agent_id,
                reason=RunStopReason.USER_CANCELLED,
                requester_session_id=parent.session_id,
            )
            assert outcome.outcome is CancelOutcome.STOP_REQUESTED
            release_final_lock.set()
            await asyncio.wait_for(launch, 2)
            result = await asyncio.wait_for(waiter, 2)
            assert result.stop_reason is RunStopReason.USER_CANCELLED
            assert run.status is RunStatus.CANCELLED and not result.completed
            assert len(record.runtime.turns.completed_turns) == 1
        else:
            launch.cancel()
            release_final_lock.set()
            with pytest.raises(asyncio.CancelledError):
                await launch
            assert not record.runtime.turns.completed_turns
        assert record.runtime.execution.active is None
        assert record.runtime.turns._pending_start is None
        assert record.runtime.turns.active_turn is None
        assert record.runtime.turns._active_task is None
        assert not registry._monitor_tasks
    finally:
        release_final_lock.set()
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "boundary",
    [
        "factory",
        "persistence",
        "metadata",
        "projection",
        "ensure_lock",
        "registry_lock",
    ],
)
@pytest.mark.parametrize("identity_change", ["session_id", "generation"])
async def test_background_creation_fences_live_identity_at_each_admission_boundary(
    boundary: str, identity_change: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every awaited creation boundary rejects a changed live root identity."""
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    root.turns._projector = MagicMock()
    entered = asyncio.Event()
    release = asyncio.Event()
    original_factory = registry._runtime_factory.create_child

    async def gate(awaitable):
        result = await awaitable
        entered.set()
        await release.wait()
        return result

    if boundary == "factory":

        async def create_child(
            parent: AgentLoop,
            agent_name: str,
            *,
            session_id: str | None = None,
            session_dir: Path | None = None,
        ) -> AgentLoop:
            return await gate(
                original_factory(
                    parent, agent_name, session_id=session_id, session_dir=session_dir
                )
            )

        monkeypatch.setattr(registry._runtime_factory, "create_child", create_child)
    elif boundary == "persistence":
        original = AgentLoop.persist_empty_session

        async def persist(child: AgentLoop) -> None:
            await gate(original(child))

        monkeypatch.setattr(AgentLoop, "persist_empty_session", persist)
    elif boundary == "metadata":
        original = AgentLoop.record_child_session

        async def record(
            parent: AgentLoop, child: AgentLoop, tool_call_id: str, agent_name: str
        ) -> None:
            await gate(original(parent, child, tool_call_id, agent_name))

        monkeypatch.setattr(AgentLoop, "record_child_session", record)
    elif boundary == "projection":
        original = TurnController.link_subagent

        async def link(
            turns: TurnController, tool_call_id: str, child_session_id: str
        ) -> None:
            await gate(original(turns, tool_call_id, child_session_id))

        monkeypatch.setattr(TurnController, "link_subagent", link)
    else:
        lock_name = (
            "_ensure_child_lock" if boundary == "ensure_lock" else "_registry_lock"
        )
        lock = getattr(registry, lock_name)

        class GatedLock:
            async def __aenter__(self) -> None:
                await lock.acquire()
                entered.set()
                await release.wait()

            async def __aexit__(self, *exc: object) -> None:
                lock.release()

        setattr(registry, lock_name, GatedLock())

    launch = asyncio.create_task(
        _background_result(
            registry,
            TaskArgs(task="work", identity="coder", background=True),
            InvokeContext(tool_call_id="launch", session_id=parent.session_id),
        )
    )
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        if identity_change == "session_id":
            parent.session_id = f"{parent.session_id}-changed"
        else:
            parent._session_generation += 1
        release.set()
        with pytest.raises(RuntimeError, match="admission changed"):
            await launch
        assert registry._agent_records == {}
        assert registry._children == {}
        assert registry._child_links == {}
        assert registry._monitor_tasks == set()
    finally:
        release.set()
        if not launch.done():
            launch.cancel()
            await asyncio.gather(launch, return_exceptions=True)
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_real_child_creation_uses_global_idle_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, _context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        # Policy reversal: a per-agent idle TTL no longer reaches the record.
        assert record.idle_ttl_seconds is None
        global_ttl = registry._retention_policy.idle_ttl_seconds
        assert record.effective_idle_ttl(global_ttl) == global_ttl
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("admission", ["disabled", "never"])
async def test_direct_runner_enforces_effective_task_admission(
    monkeypatch: pytest.MonkeyPatch, admission: str
) -> None:
    registry, parent, _record, _context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    create_child = AsyncMock()
    monkeypatch.setattr(registry._runtime_factory, "create_child", create_child)
    if admission == "disabled":
        denied_config = parent.tool_manager._config.model_copy(
            update={"disabled_tools": ["task"]}
        )
    else:
        tools = dict(parent.tool_manager._config.tools)
        tools["task"] = {"permission": ToolPermission.NEVER.value}
        denied_config = parent.tool_manager._config.model_copy(update={"tools": tools})
    monkeypatch.setattr(parent.tool_manager, "_config_getter", lambda: denied_config)
    # The mocked policy replacement must also advance the accepted authority signal.
    accepted_revision = object()
    monkeypatch.setattr(
        parent.tool_manager, "_accepted_token_getter", lambda: accepted_revision
    )
    try:
        with pytest.raises(ToolPermissionError, match=f"Task tool .*{admission}"):
            await _background_result(
                registry,
                TaskArgs(task="blocked", identity="user:dynamic", background=False),
                InvokeContext(
                    tool_call_id=f"direct-{admission}", session_id=parent.session_id
                ),
            )
        create_child.assert_not_awaited()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_dynamic_launch_uses_resolver_and_rejects_retained_handle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, parent, record, context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        with pytest.raises(ValueError, match="agent_id"):
            await _background_result(
                registry,
                TaskArgs(task="no reuse", agent_id=record.agent_id, background=False),
                context,
            )
        result = await _background_result(
            registry,
            TaskArgs(
                task="foreground",
                identity="user:dynamic",
                background=False,
                config=cast(LaunchConfig, {"model": "large", "thinking": "high"}),
            ),
            InvokeContext(
                tool_call_id="dynamic-foreground", session_id=parent.session_id
            ),
        )
        assert result.agent_id is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retained_launch_consumes_the_top_level_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained launch goes through the same canonical model resolution as
    a fresh launch: the top-level model field selects the reused child's
    model (the legacy nested config.model is the promoted equivalent).
    """
    registry, parent, record, context, _first, backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        child = record.runtime.agent_loop
        assert child.committed_model is not None
        assert child.committed_model.base_model == "small"
        result = await _background_result(
            registry,
            TaskArgs(
                task="top-level model",
                agent_id=record.agent_id,
                model="other",
                background=True,
            ),
            context,
        )
        assert result.run_id is not None
        await registry.wait_for_agent(record.agent_id, result.run_id)
        assert child.config.active_model == "other"
        assert child.committed_model is not None
        assert child.committed_model.base_model == "other"
        assert child.committed_model.provider == "two"
        assert backends[-1][0] == "two"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retained_launch_validates_the_purpose_seat_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained launch accepts only a valid purpose/seat binding: an
    unknown purpose, a multi-seat purpose without a seat, and an unknown
    seat all fail loudly instead of being accepted unvalidated.
    """
    registry, parent, record, context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        for args in (
            TaskArgs(
                task="unknown purpose",
                agent_id=record.agent_id,
                purpose="no-such-purpose",
                background=True,
            ),
            TaskArgs(
                task="missing seat",
                agent_id=record.agent_id,
                purpose="review.deep",
                background=True,
            ),
            TaskArgs(
                task="unknown seat",
                agent_id=record.agent_id,
                purpose="review.deep",
                seat="chair",
                background=True,
            ),
        ):
            with pytest.raises(InvalidLaunchPurposeError):
                await _background_result(registry, args, context)
        # The retained instance is untouched: still idle on its committed
        # model, with no new run recorded.
        child = record.runtime.agent_loop
        assert child.committed_model is not None
        assert child.committed_model.base_model == "small"
        assert record.state is _AgentState.IDLE
        assert record.current_run is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retained_launch_installs_its_own_purpose_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retask's own purpose label installs on the reused loop — never
    copied from the previous run: a routed retask carries its purpose, and
    a bare continuation carries none. The retained identity/model stay
    pinned (the route's own defaults are validated, never adopted).
    """
    registry, parent, record, context, _first, _backends = await _dynamic_registry(
        monkeypatch
    )
    try:
        child = record.runtime.agent_loop
        assert child.launch_purpose is None  # the first launch was direct
        routed = await _background_result(
            registry,
            TaskArgs(
                task="routed retask",
                agent_id=record.agent_id,
                purpose="research",
                background=True,
            ),
            context,
        )
        assert routed.run_id is not None
        await registry.wait_for_agent(record.agent_id, routed.run_id)
        assert child.launch_purpose == "research"
        assert project_launch_purposes(child) == ["research"]
        assert child._bound_identity == "user:dynamic"
        assert child.committed_model is not None
        assert child.committed_model.base_model == "small"
        bare = await _background_result(
            registry,
            TaskArgs(
                task="bare continuation", agent_id=record.agent_id, background=True
            ),
            context,
        )
        assert bare.run_id is not None
        await registry.wait_for_agent(record.agent_id, bare.run_id)
        assert child.launch_purpose is None
        assert project_launch_purposes(child) == []
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retained_launch_normalizes_the_model_shorthand_like_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 delta D2 (Ab3 residual): the retained path promoted the raw
    top-level model field into the retained semantic config, so a shorthand
    model ('@main:user:dynamic') reached model resolution as one opaque
    expression and raised InvalidLaunchModelError while the fresh path
    resolved it. The retained model selection must parse/normalize the
    shorthand exactly like fresh resolution: the model component selects the
    model, and the identity component matches the retained record's
    committed identity.
    """
    backends: list[tuple[str, FakeBackend]] = []

    def create_backend(*, provider, **_kwargs):
        backend = FakeBackend([mock_llm_chunk(content=provider.name)])
        backends.append((provider.name, backend))
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(config=_dynamic_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="dynamic", session_id=parent.session_id)
    try:
        first = await _background_result(
            registry,
            TaskArgs(
                task="first",
                identity="user:dynamic",
                background=True,
                retain=True,
                config=LaunchConfig(
                    model="small",
                    enabled_tools=["read_file"],
                    instructions="Retained role instruction.",
                ),
            ),
            context,
        )
        assert first.agent_id is not None and first.run_id is not None
        await registry.wait_for_agent(first.agent_id, first.run_id)
        record = registry._agent_records[first.agent_id]
        child = record.runtime.agent_loop
        assert child.committed_model is not None
        assert child.committed_model.base_model == "small"

        # The fresh control resolves the shorthand through canonical
        # resolution: the model component selects the model, the identity
        # component selects the identity.
        fresh = await _background_result(
            registry,
            TaskArgs(
                task="shorthand fresh",
                model="@main:user:dynamic",
                background=True,
                retain=True,
            ),
            context,
        )
        assert fresh.agent_id is not None and fresh.run_id is not None
        await registry.wait_for_agent(fresh.agent_id, fresh.run_id)

        # The retained re-task normalizes the same shorthand: the model
        # component selects the reused child's model; the identity component
        # matches the retained record's committed identity.
        retained = await _background_result(
            registry,
            TaskArgs(
                task="shorthand retask",
                agent_id=record.agent_id,
                model="@main:user:dynamic",
                background=True,
            ),
            context,
        )
        assert retained.run_id is not None
        await registry.wait_for_agent(record.agent_id, retained.run_id)
        assert child.config.active_model == "@main"
        assert child.committed_model is not None
        assert child.committed_model.base_model == "glm-5-3"
        assert child._bound_identity == "user:dynamic"
        assert backends[-1][0] == "mistral"
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_retained_launch_enforces_purpose_diversity_like_fresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round-2 delta D2 (Ab3 residual): the retained path validated a
    purpose/seat binding with the route's distinct-model requirement
    disabled, so on a single-model catalog a retained review.deep launch
    succeeded where the fresh launch fails InvalidLaunchPurposeError. The
    retained purpose validation must enforce the same canonical-model
    diversity rule a fresh launch enforces (design §9).
    """
    backends: list[tuple[str, FakeBackend]] = []

    def create_backend(*, provider, **_kwargs):
        backend = FakeBackend([mock_llm_chunk(content=provider.name)])
        backends.append((provider.name, backend))
        return backend

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    config = build_test_vibe_config(
        active_model="@main",
        include_project_context=False,
        include_prompt_detail=False,
        enabled_tools=["task", "read_file"],
        tools={"task": {"permission": "always"}, "read_file": {"permission": "always"}},
    ).attach_catalog_snapshot(CatalogSnapshot(single_model_catalog(), "test"))
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="dynamic", session_id=parent.session_id)
    try:
        first = await _background_result(
            registry,
            TaskArgs(
                task="first",
                identity="user:dynamic",
                background=True,
                retain=True,
                config=LaunchConfig(
                    model="@main",
                    enabled_tools=["read_file"],
                    instructions="Retained role instruction.",
                ),
            ),
            context,
        )
        assert first.agent_id is not None and first.run_id is not None
        await registry.wait_for_agent(first.agent_id, first.run_id)
        record = registry._agent_records[first.agent_id]

        # The fresh control fails: the shipped review.deep route requires 3
        # distinct canonical models and the single-model catalog resolves 1.
        with pytest.raises(InvalidLaunchPurposeError):
            await _background_result(
                registry,
                TaskArgs(
                    task="fresh deep review",
                    purpose="review.deep",
                    seat="analytical",
                    background=True,
                ),
                context,
            )
        # The retained re-task fails the SAME way: the purpose/seat binding
        # is validated through the same canonical route grammar, diversity
        # included — never accepted unvalidated.
        with pytest.raises(InvalidLaunchPurposeError):
            await _background_result(
                registry,
                TaskArgs(
                    task="retained deep review",
                    agent_id=record.agent_id,
                    purpose="review.deep",
                    seat="analytical",
                    background=True,
                ),
                context,
            )
        # The retained instance is untouched: still idle on its committed
        # model, with no new run recorded.
        child = record.runtime.agent_loop
        assert child.committed_model is not None
        assert child.committed_model.base_model == "solo-model"
        assert record.state is _AgentState.IDLE
        assert record.current_run is None
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_foreground_turn_start_failure_rolls_back_persisted_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        registry,
        parent,
        _record,
        _context,
        _first,
        _backends,
    ) = await _persistent_dynamic_registry(tmp_path, monkeypatch)
    children_before = set(registry._children)
    metadata = parent.session_logger.session_metadata
    assert metadata is not None
    links_before = list(metadata.child_sessions)
    child_dirs: list[Path] = []
    original_create_child = registry._runtime_factory.create_child

    async def capture_created_child(*args, **kwargs):
        child = await original_create_child(*args, **kwargs)
        assert child.session_logger.session_dir is not None
        child_dirs.append(child.session_logger.session_dir)
        return child

    monkeypatch.setattr(
        registry._runtime_factory, "create_child", capture_created_child
    )
    monkeypatch.setattr(
        SessionRuntimeRegistry,
        "_start_child_turn",
        MagicMock(side_effect=RuntimeError("foreground turn start failed")),
    )
    try:
        with pytest.raises(RuntimeError, match="foreground turn start failed"):
            await _background_result(
                registry,
                TaskArgs(task="fail", identity="user:dynamic", background=False),
                InvokeContext(
                    tool_call_id="foreground-failure", session_id=parent.session_id
                ),
            )
        assert set(registry._children) == children_before
        assert metadata.child_sessions == links_before
        assert len(child_dirs) == 1
        assert not child_dirs[0].exists()
        with pytest.raises(ValueError, match="Session metadata not found"):
            SessionLoader.load_metadata(child_dirs[0])
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_concurrent_agent_id_issuance_reserves_unique_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    await parent.persist_empty_session()
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    entered = 0
    both_persisting = asyncio.Event()
    release = asyncio.Event()

    async def gated_persist(_field: str, _value: object) -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_persisting.set()
        await release.wait()

    monkeypatch.setattr(parent.session_logger, "_persist_metadata_field", gated_persist)
    try:
        issued = asyncio.gather(
            registry._issue_agent_id(root), registry._issue_agent_id(root)
        )
        await asyncio.wait_for(both_persisting.wait(), timeout=1)
        release.set()
        assert await issued == ["agent-1", "agent-2"]
    finally:
        release.set()
        await parent.aclose()


@pytest.mark.asyncio
async def test_resume_preserves_sparse_released_agent_id_high_water_mark(
    tmp_path: Path,
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    await parent.persist_empty_session()
    session_id = parent.session_id
    registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    try:
        assert await registry._issue_agent_id(root) == "agent-1"
        # Simulate sparse history after both issued handles have been released.
        registry._next_agent_seq = 4
        assert await registry._issue_agent_id(root) == "agent-5"
        assert registry._agent_records == {}
    finally:
        await parent.aclose()

    async def resume_registry() -> tuple[
        SessionRuntimeRegistry, AgentLoop, InvokeContext
    ]:
        resumed_parent = build_test_agent_loop(
            config=_config(logging), backend=FakeBackend()
        )
        await AgentRuntimeFactory().resume_root(resumed_parent, session_id)
        resumed_registry = SessionRuntimeRegistry(AsyncMock(), AsyncMock(), lambda _: 0)
        resumed_registry.bind_root(
            resumed_registry._build_child_runtime(resumed_parent)
        )
        return (
            resumed_registry,
            resumed_parent,
            InvokeContext(tool_call_id="resumed", session_id=resumed_parent.session_id),
        )

    resumed_registry, resumed_parent, resumed_context = await resume_registry()
    try:
        resumed_root = resumed_registry._root
        assert resumed_root is not None
        assert await resumed_registry._issue_agent_id(resumed_root) == "agent-6"
        for historical_agent_id in ("agent-1", "agent-5"):
            with pytest.raises(UnknownAgentError):
                await _background_result(
                    resumed_registry,
                    TaskArgs(
                        task="historical", agent_id=historical_agent_id, background=True
                    ),
                    resumed_context,
                )
    finally:
        await resumed_parent.aclose()

    resumed_registry, resumed_parent, _resumed_context = await resume_registry()
    try:
        resumed_root = resumed_registry._root
        assert resumed_root is not None
        assert await resumed_registry._issue_agent_id(resumed_root) == "agent-7"
    finally:
        await resumed_parent.aclose()


@pytest.mark.asyncio
async def test_resume_reconstructs_accumulated_launch_state_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        registry,
        parent,
        record,
        context,
        _first,
        _backends,
    ) = await _persistent_dynamic_registry(tmp_path, monkeypatch)
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        retask = await _background_result(
            registry,
            TaskArgs(
                task="raise thinking",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(thinking="high"),
            ),
            context,
        )
        assert retask.run_id is not None
        await registry.wait_for_agent(record.agent_id, retask.run_id)

        persisted = SessionLoader.load_metadata(child_dir).launch_config
        assert persisted is not None
        assert persisted.overrides.model == "small"
        assert persisted.overrides.thinking == "high"
        assert persisted.overrides.enabled_tools == ["read_file"]
        assert persisted.persona.instructions == "Retained role instruction."

        # A normal turn save must retain the committed dedicated envelope.
        await child.session_logger.save_interaction(
            child.messages, child.stats, child.config, child.tool_manager
        )
        assert SessionLoader.load_metadata(child_dir).launch_config == persisted

        child_id = child.session_id
        await registry.drain_children()
        resumed = await AgentRuntimeFactory().resume_child(
            parent, "user:dynamic", child_id, child_dir
        )
        try:
            await resumed.wait_until_ready()
            assert resumed.config.active_model == "small"
            assert resumed.config.get_active_model().thinking == "high"
            assert set(resumed.tool_manager.available_tools) == {"read_file"}
            assert "Retained role instruction." in str(resumed.messages[0].content)
        finally:
            await resumed.aclose()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_resume_restores_the_committed_purpose_tier_and_capability_from_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted launch envelope carries the resolved purpose/tier/
    capability, and a disk-resumed child restores them instead of
    re-deriving defaults (before the envelope carried them, a resumed
    mechanical-edit child came back substantive and purpose-less).
    """
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    routes = OrchestrationConfig(
        routes={
            "mechanical-edit": PurposeRoute(
                targets=(
                    RouteTarget(
                        model="small",
                        identity="user:dynamic",
                        tier=RoutePackageTier.MECHANICAL,
                        replacement_capability=ReplacementCapability.MECHANICAL,
                    ),
                )
            )
        }
    )
    config = _dynamic_config().model_copy(
        update={"session_logging": logging, "orchestration": routes}
    )

    def create_backend(*, provider, **_kwargs):
        return FakeBackend([mock_llm_chunk(content=provider.name)])

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(
        config=config,
        backend=FakeBackend([mock_llm_chunk(content="Notification received")]),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="dynamic", session_id=parent.session_id)
    try:
        first = await _background_result(
            registry,
            TaskArgs(
                task="mechanical package",
                purpose="mechanical-edit",
                background=True,
                retain=True,
            ),
            context,
        )
        assert first.agent_id is not None and first.run_id is not None
        await registry.wait_for_agent(first.agent_id, first.run_id)
        record = registry._agent_records[first.agent_id]
        child = record.runtime.agent_loop
        child_dir = child.session_logger.session_dir
        assert child_dir is not None
        # The launch resolved through the mechanical-tier route.
        assert child.launch_purpose == "mechanical-edit"
        assert child.launch_package_tier is PackageTier.MECHANICAL
        assert child.launch_replacement_capability is ReplacementCapability.MECHANICAL
        # The persisted envelope carries the committed purpose/tier/capability.
        persisted = SessionLoader.load_metadata(child_dir).launch_config
        assert isinstance(persisted, LaunchMetadataV2)
        assert persisted.purpose == "mechanical-edit"
        assert persisted.package_tier is PackageTier.MECHANICAL
        assert persisted.replacement_capability is ReplacementCapability.MECHANICAL

        child_id = child.session_id
        await registry.drain_children()
        resumed = await AgentRuntimeFactory().resume_child(
            parent, "user:dynamic", child_id, child_dir
        )
        try:
            await resumed.wait_until_ready()
            # The resumed child keeps the committed purpose/tier/capability.
            assert resumed.launch_purpose == "mechanical-edit"
            assert resumed.launch_package_tier is PackageTier.MECHANICAL
            assert (
                resumed.launch_replacement_capability
                is ReplacementCapability.MECHANICAL
            )
        finally:
            await resumed.aclose()
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (
            # The retired pre-identity persona format (keyed on
            # system_prompt_id) still parses, but resume rejects it with
            # the actionable "start a new session" failure — there is no
            # migration for the old persisted format.
            lambda metadata: metadata["launch_config"].update(
                persona={"system_prompt_id": "gone", "instructions": None}
            ),
            "launch_config.persona",
        ),
        (
            lambda metadata: metadata["launch_config"].update(version=99),
            "launch_config",
        ),
    ],
)
async def test_ensure_child_fails_closed_for_invalid_persisted_envelopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, field: str
) -> None:
    (
        registry,
        parent,
        record,
        _context,
        _first,
        _backends,
    ) = await _persistent_dynamic_registry(tmp_path, monkeypatch)
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    parent_dir = parent.session_logger.session_dir
    assert child_dir is not None and parent_dir is not None
    child_id = child.session_id
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        mutation(raw)
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        await registry.drain_children()

        # ensure_child is the real session-access boundary, not resume_child directly.
        metadata = parent.session_logger.session_metadata
        assert metadata is not None
        metadata.child_sessions = [
            ChildSessionLink(
                session_id=child_id,
                tool_call_id="persisted",
                agent="user:dynamic",
                relative_path=str(child_dir.relative_to(parent_dir)),
            )
        ]
        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == field
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_persisted_child_launch_error_is_reported_as_protocol_diagnostic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    logging = parent.config.session_logging
    await registry.drain_children()
    await registry.close()
    await parent.aclose()

    base_config = _dynamic_config()
    resumed_config = base_config.model_copy(
        update={
            "active_model": "large",
            "allowed_models": ["large", "other"],
            "session_logging": logging,
        }
    )
    resumed_loop = build_test_agent_loop(config=resumed_config, backend=FakeBackend())
    with pytest.raises(AppServerResponseError) as exc_info:
        await attach_test_app_server_session(
            start_test_app_server(resumed_loop), resume_session_id=parent.session_id
        )
    assert exc_info.value.error.code is ProtocolErrorCode.INVALID_PARAMS


@pytest.mark.asyncio
async def test_legacy_no_config_child_resume_uses_identity_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent (legacy) envelope is distinct from a malformed present one:
    resume admits it with identity defaults — no committed overrides, the
    session's default model.
    """
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        raw.pop("launch_config")
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        child_id = child.session_id
        await registry.drain_children()

        assert await registry.ensure_child(child_id)
        resumed = registry._children[child_id].agent_loop
        assert resumed.launch_overrides is not None
        assert resumed.launch_overrides.model_fields_set == set()
        assert resumed.config.active_model == "small"
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rechecks_tightened_parent_model_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        child_id = child.session_id
        object.__setattr__(parent.config, "allowed_models", ["other"])
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "config.model"
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda raw: raw["launch_config"].update(unexpected=True), "launch_config"),
        (lambda raw: raw["launch_config"].update(overrides=[]), "launch_config"),
        (lambda raw: raw.update(launch_config="not-an-envelope"), "launch_config"),
    ],
)
async def test_ensure_child_rejects_present_malformed_envelope_variants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation, field: str
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        raw = json.loads((child_dir / "meta.json").read_text(encoding="utf-8"))
        mutation(raw)
        (child_dir / "meta.json").write_text(json.dumps(raw), encoding="utf-8")
        child_id = child.session_id
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == field
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rejects_corrupt_present_envelope_as_typed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    try:
        (child_dir / "meta.json").write_text("{corrupt", encoding="utf-8")
        child_id = child.session_id
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "launch_config"
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rejects_mismatched_identity_without_worker_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link naming a different identity rejects instead of substituting
    defaults: the retired profile name "worker" bridges onto the coder
    identity, which does not match the child's committed launch identity.
    """
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    parent_dir = parent.session_logger.session_dir
    assert child_dir is not None and parent_dir is not None
    try:
        child_id = child.session_id
        metadata = parent.session_logger.session_metadata
        assert metadata is not None
        metadata.child_sessions = [
            ChildSessionLink(
                session_id=child_id,
                tool_call_id="persisted",
                agent="worker",
                relative_path=str(child_dir.relative_to(parent_dir)),
            )
        ]
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "launch_config.persona"
        assert "does not match its link" in str(raised.value)
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_ensure_child_rejects_link_naming_a_different_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A link naming an unrelated identity ("explore") rejects: the stored
    launch identity is re-read from the persisted envelope, never re-derived
    from the link.
    """
    registry, parent, record, _context, _first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    parent_dir = parent.session_logger.session_dir
    assert child_dir is not None and parent_dir is not None
    try:
        child_id = child.session_id
        metadata = parent.session_logger.session_metadata
        assert metadata is not None
        metadata.child_sessions = [
            ChildSessionLink(
                session_id=child_id,
                tool_call_id="persisted",
                agent="explore",
                relative_path=str(child_dir.relative_to(parent_dir)),
            )
        ]
        await registry.drain_children()

        with pytest.raises(LaunchConfigError) as raised:
            await registry.ensure_child(child_id)
        assert raised.value.field == "launch_config.persona"
        assert "does not match its link" in str(raised.value)
        assert registry._children == {}
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_preacceptance_retask_failure_keeps_prior_launch_envelope_and_resources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, context, first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    old_manager, old_config = child.tool_manager, child.config
    old_envelope = SessionLoader.load_metadata(child_dir).launch_config
    assert old_envelope is not None

    def fail_before_acceptance(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("injected preacceptance failure")

    monkeypatch.setattr(
        SessionRuntimeRegistry, "_start_child_turn", fail_before_acceptance
    )
    try:
        with pytest.raises(RuntimeError, match="injected preacceptance failure"):
            await _background_result(
                registry,
                TaskArgs(
                    task="failed before acceptance",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(model="large"),
                ),
                context,
            )
        assert record.state is _AgentState.IDLE
        assert record.current_run is None and record.latest_run_id == first.run_id
        assert child.config is old_config and child.tool_manager is old_manager
        assert SessionLoader.load_metadata(child_dir).launch_config == old_envelope
    finally:
        await registry.close()
        await parent.aclose()


@pytest.mark.asyncio
async def test_postacceptance_launch_envelope_write_failure_keeps_accepted_retask_dirty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, parent, record, context, first, _ = await _persistent_dynamic_registry(
        tmp_path, monkeypatch
    )
    child = record.runtime.agent_loop
    child_dir = child.session_logger.session_dir
    assert child_dir is not None
    old_envelope = SessionLoader.load_metadata(child_dir).launch_config
    assert old_envelope is not None
    persist_metadata = SessionLogger.persist_metadata

    async def fail_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected metadata failure after acceptance")

    monkeypatch.setattr(SessionLogger, "persist_metadata", fail_write)
    try:
        with pytest.raises(RuntimeError, match="injected metadata failure"):
            await _background_result(
                registry,
                TaskArgs(
                    task="accepted but dirty",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(model="large"),
                ),
                context,
            )
        accepted_run_id = record.latest_run_id
        assert accepted_run_id is not None and accepted_run_id != first.run_id
        assert child.config.active_model == "large"
        assert child.session_logger.session_metadata is not None
        in_memory = child.session_logger.session_metadata.launch_config
        assert in_memory is not None and in_memory.overrides.model == "large"
        assert SessionLoader.load_metadata(child_dir).launch_config == old_envelope

        # The next ordinary save repairs the dirty accepted envelope.
        monkeypatch.setattr(SessionLogger, "persist_metadata", persist_metadata)
        await child.session_logger.save_interaction(
            child.messages, child.stats, child.config, child.tool_manager
        )
        persisted = SessionLoader.load_metadata(child_dir).launch_config
        assert persisted is not None and persisted.overrides.model == "large"
        await registry.wait_for_agent(record.agent_id, accepted_run_id)
    finally:
        await registry.close()
        await parent.aclose()


def test_cross_provider_dedicated_compaction_model_is_catalog_resolved() -> None:
    config = _dynamic_config().model_copy(update={"compaction_model": "other"})
    assert config.get_compaction_model().provider == "two"


@pytest.mark.asyncio
async def test_ensure_child_joins_blocked_foreground_teardown_before_rematerializing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(logging), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    close_started = asyncio.Event()
    close_release = asyncio.Event()
    original_close = SessionRuntime.close

    async def blocked_close(runtime: SessionRuntime) -> None:
        if runtime is not root:
            close_started.set()
            await close_release.wait()
        await original_close(runtime)

    monkeypatch.setattr(SessionRuntime, "close", blocked_close)
    try:
        result = await _foreground_result(registry, parent)
        child_session_id = next(iter(registry._stored_children))
        assert result.completed
        await asyncio.wait_for(close_started.wait(), timeout=1)

        rematerialize = asyncio.create_task(registry.ensure_child(child_session_id))
        await asyncio.sleep(0)
        assert not rematerialize.done()
        close_release.set()
        assert await asyncio.wait_for(rematerialize, timeout=1)
        child = registry._children[child_session_id]
        assert not child._closed
        assert (
            registry.public_state(child_session_id, history_limit=10).session.id
            == child_session_id
        )
    finally:
        close_release.set()
        await registry.close()
        await parent.aclose()


def test_legacy_fan_out_is_rejected_with_task_guidance() -> None:
    with pytest.raises(ValueError, match="Roles are single presets"):
        TaskArgs.model_validate({"task": "panel", "fan_out": True})

    assert "fan_out" not in TaskArgs.model_json_schema()["properties"]


@pytest.mark.asyncio
async def test_identity_keyed_launch_resolves_canonically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """11c cutover: a structured identity launch resolves through the
    canonical grammar — the child binds the researcher identity with its
    read-only posture, the record carries (identity, revision), and the
    model is the session's committed binding, not a profile role preset.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(
        config=_config(enabled_tools=["task", "read_file", "edit", "bash"]),
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    ctx = InvokeContext(tool_call_id="identity-launch", session_id=parent.session_id)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(
                task="research the codebase",
                identity="researcher",
                background=True,
                retain=True,
            ),
            ctx,
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.identity == "researcher"
        assert record.identity_revision is not None
        child = record.runtime.agent_loop
        assert child._bound_identity == "researcher"
        assert child.launch_identity == "researcher"
        # The researcher posture: non-mutating personal tool set.
        assert "edit" not in child.tool_manager.available_tools
        assert "read_file" in child.tool_manager.available_tools
        # The session default is the committed session binding.
        assert child.committed_model is not None
        assert parent.committed_model is not None
        assert child.committed_model.base_model == parent.committed_model.base_model
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_default_launch_binds_the_coder_identity_with_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A launch naming no target binds the default coder identity: the
    record carries the canonical identity and its content revision.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    ctx = InvokeContext(tool_call_id="default-launch", session_id=parent.session_id)
    try:
        launch = await _background_result(
            registry, TaskArgs(task="plain work", background=True), ctx
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        assert record.identity == "coder"
        assert record.identity_revision is not None
        run = record.current_run or record.run_history[-1]
        assert run.identity == "coder"
        assert run.identity_revision == record.identity_revision
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("background", [True, False])
async def test_identity_denylist_applies_after_final_resolution(
    monkeypatch: pytest.MonkeyPatch, background: bool
) -> None:
    """The launch admission denylist keys on the RESOLVED identity — a
    structured, shorthand, or default selection is checked against its final
    identity (the same surface the task tool enforces) before any child
    resource is constructed.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    tools = dict(parent.tool_manager._config.tools)
    tools["task"] = {**tools["task"], "denylist": ["research*"]}
    denied_config = parent.tool_manager._config.model_copy(update={"tools": tools})
    monkeypatch.setattr(parent.tool_manager, "_config_getter", lambda: denied_config)
    accepted_revision = object()
    monkeypatch.setattr(
        parent.tool_manager, "_accepted_token_getter", lambda: accepted_revision
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    ctx = InvokeContext(tool_call_id="denied-launch", session_id=parent.session_id)
    try:
        for args in (
            TaskArgs(task="research", identity="researcher", background=background),
            TaskArgs(task="research", model="@scout:researcher", background=background),
        ):
            with pytest.raises(
                ToolPermissionError, match="Task denied for launch identity: researcher"
            ):
                await _background_result(registry, args, ctx)
        assert registry._children == {}
        assert registry._agent_records == {}
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("denylist", "identity"),
    [
        (["coder"], "builtin:coder"),
        (["builtin:coder"], "coder"),
        (["user:noisy"], "noisy"),
        (["noisy"], "user:noisy"),
    ],
    ids=[
        "bare-builtin-pattern-blocks-builtin-qualified",
        "builtin-qualified-pattern-blocks-bare-builtin",
        "user-qualified-pattern-blocks-bare-custom",
        "bare-custom-pattern-blocks-user-qualified",
    ],
)
async def test_identity_denylist_matches_every_equivalent_spelling(
    monkeypatch: pytest.MonkeyPatch, denylist: list[str], identity: str
) -> None:
    """The launch admission denylist is SYMMETRIC across spellings (finding
    1): a pattern written in any equivalent spelling blocks the launch
    identity in every spelling — denying "coder" blocks "builtin:coder",
    denying "builtin:coder" blocks "coder", denying "user:noisy" blocks the
    bare custom id "noisy", and denying "noisy" blocks "user:noisy".
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    tools = dict(parent.tool_manager._config.tools)
    tools["task"] = {**tools["task"], "denylist": denylist}
    denied_config = parent.tool_manager._config.model_copy(update={"tools": tools})
    monkeypatch.setattr(parent.tool_manager, "_config_getter", lambda: denied_config)
    accepted_revision = object()
    monkeypatch.setattr(
        parent.tool_manager, "_accepted_token_getter", lambda: accepted_revision
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    ctx = InvokeContext(tool_call_id="denied-launch", session_id=parent.session_id)
    try:
        with pytest.raises(
            ToolPermissionError, match="Task denied for launch identity"
        ):
            await _background_result(
                registry, TaskArgs(task="work", identity=identity, background=True), ctx
            )
        assert registry._children == {}
        assert registry._agent_records == {}
    finally:
        await registry.drain_children()
        await parent.aclose()


@pytest.mark.asyncio
async def test_direct_runner_string_launches_resolve_canonically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct runner's string path resolves through the canonical
    identity grammar: create_child('coder') binds coder and commits the
    session's model binding, while a legacy profile name ('worker') fails
    loudly with identity guidance — the retired alias bridges nothing.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, "coder")
        await child.wait_until_ready()
        assert child._bound_identity == "coder"
        assert child.committed_model is not None
        assert parent.committed_model is not None
        assert child.committed_model.base_model == parent.committed_model.base_model
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


@pytest.mark.asyncio
async def test_direct_runner_string_rejects_a_legacy_profile_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct runner's string path carries no legacy profile bridge
    (finding 3): a legacy 'worker' spelling is not an identity and fails
    loudly with identity successor guidance.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    try:
        with pytest.raises(
            InvalidChildIdentityError, match="Unknown identity 'worker'"
        ):
            await AgentRuntimeFactory().create_child(parent, "worker")
    finally:
        await parent.aclose()
