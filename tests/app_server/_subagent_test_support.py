"""Shared test-support surface for the subagents suites (W10, stages 19a and 19b).

The definitions here are the shared module-level helper surface of
``tests/app_server/test_subagents.py``, relocated AST-verbatim in the 19a
phase-(i) helper-relocation commit: the fake agent backends
(``BlockingBackend``, ``ImmediateFailureBackend``, ``GatedFailureBackend``,
``GatedSequenceBackend``, ``RetaskBackend``), the task-call builders
(``_task_call``, ``_todo_call``), the config and registry factories
(``_config``, ``_retention_registry``, ``_reaper_registry``,
``_dynamic_config``, ``_dynamic_registry``, ``_persistent_dynamic_registry``,
``_foreground_registry``, ``_zero_retention_registry``), the result and
retention record builders (``_background_result``, ``_wait_result``,
``_stored_result``, ``_evicted_tombstone``, ``_idle_record``,
``_foreground_result``, ``_real_reused_background``), the clock and wakeup
doubles (``_ManualClock``, ``_GatedWakeup``), the reaper arm helper
(``_rearm``), the event-stream helpers (``_consume``,
``_wait_for_turn_completion``). No test bodies, fixtures, or marks live here; the
test population stays in ``test_subagents.py`` until the phase-(ii)
concern moves. Importers: ``test_subagents.py`` itself plus the seven seam
files (``test_wait_steering_app``, ``test_subagent_management_surface``,
``test_subagent_ownership_barriers``, ``test_agent_context_capture``,
``test_wait_steering``, ``test_subagent_parent_auto_compact``,
``test_subagent_lifecycle``). The 19b phase-(i) helper-relocation commit
adds the shared surface of ``test_subagent_lifecycle.py``: the record
builders ``_record``, ``_run`` and ``_tombstone`` plus the root-bound
registry factory ``_root_registry``, relocated AST-verbatim; that file's
test population stays put until the 19b phase-(ii) concern moves.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._model import validate_wire
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import (
    AgentRecord,
    RunRecord,
    SessionRuntimeRegistry,
    StoredRunResult,
    _AgentState,
    _AgentTombstone,
)
from chartreux.app_server.events import TurnCompleted
from chartreux.app_server.models import PublicSessionState, TextContentBlock
from chartreux.app_server.protocol import (
    SessionReadParams,
    SessionReadResponse,
    TurnStartParams,
)
from chartreux.app_server.session import AppServerSession
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import SessionLoggingConfig
from chartreux.core.llm_models import FunctionCall, LLMChunk, ToolCall
from chartreux.core.message_list import MessageList
from chartreux.core.subagents import (
    AgentAvailability,
    AgentSummary,
    LaunchConfig,
    RunStatus,
    TaskArgs,
    TaskResult,
    WaitStatus,
    derive_agent_name,
)
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.models import ToolPermission
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


async def _read_child(
    session: AppServerSession, child_session_id: str
) -> PublicSessionState:
    client = session._connection.current
    assert client is not None
    response = validate_wire(
        SessionReadResponse,
        await client.request(
            "session/read", SessionReadParams(session_id=child_session_id)
        ),
    )
    return response.state


def _task_call(
    identity: str = "coder",
    *,
    background: bool = True,
    agent_id: str | None = None,
    retain: bool | None = None,
    task_summary: str | None = None,
    tool_call_id: str = "task-1",
) -> ToolCall:
    arguments: dict[str, object] = {
        "task": "Inspect the project",
        # Package 16a: the task call names the launch identity directly
        # (the legacy profile-spelling bridge is retired with the alias).
        "identity": identity,
        "background": background,
    }
    if agent_id is not None:
        arguments["agent_id"] = agent_id
    if retain is not None:
        arguments["retain"] = retain
    if task_summary is not None:
        arguments["task_summary"] = task_summary
    return ToolCall(
        id=tool_call_id,
        index=0,
        function=FunctionCall(name="task", arguments=json.dumps(arguments)),
    )


def _config(
    session_logging: SessionLoggingConfig | None = None,
    *,
    enabled_tools: list[str] | None = None,
):
    tool_names = enabled_tools or ["task"]
    return build_test_vibe_config(
        enabled_tools=tool_names,
        tools={
            name: {"permission": ToolPermission.ALWAYS.value} for name in tool_names
        },
        session_logging=session_logging or SessionLoggingConfig(enabled=False),
    )


class BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.stopped = asyncio.Event()

    async def complete(self, **_kwargs):
        self.started.set()
        try:
            await self.release.wait()
            return mock_llm_chunk(content="Child completed")
        finally:
            self.stopped.set()

    async def complete_streaming(self, **kwargs):
        yield await self.complete(**kwargs)


class ImmediateFailureBackend(FakeBackend):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception

    async def complete(self, **_kwargs):
        raise self.exception

    async def complete_streaming(self, **_kwargs) -> AsyncGenerator[LLMChunk, None]:
        raise self.exception
        yield mock_llm_chunk(content="")


class GatedFailureBackend(FakeBackend):
    def __init__(self, exception: Exception) -> None:
        super().__init__()
        self.exception = exception
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, **_kwargs):
        self.started.set()
        await self.release.wait()
        raise self.exception

    async def complete_streaming(self, **_kwargs) -> AsyncGenerator[LLMChunk, None]:
        self.started.set()
        await self.release.wait()
        raise self.exception
        yield mock_llm_chunk(content="")


class GatedSequenceBackend(FakeBackend):
    """Return one response per gate, keeping every selected child run observable."""

    def __init__(self, chunks) -> None:
        super().__init__(chunks)
        self.started: list[asyncio.Event] = []
        self.releases: list[asyncio.Event] = []

    def add_gate(self) -> tuple[asyncio.Event, asyncio.Event]:
        started = asyncio.Event()
        release = asyncio.Event()
        self.started.append(started)
        self.releases.append(release)
        return started, release

    async def complete(self, **kwargs):
        gate_index = len(self.requests_messages)
        if gate_index < len(self.started):
            self.started[gate_index].set()
            await self.releases[gate_index].wait()
        return await super().complete(**kwargs)


async def _background_result(
    registry: SessionRuntimeRegistry, args: TaskArgs, ctx: InvokeContext
) -> TaskResult:
    return cast(TaskResult, [result async for result in registry.run(args, ctx)][-1])


class RetaskBackend(BlockingBackend):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.replacement_started = asyncio.Event()
        self.calls = 0

    async def complete(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            try:
                await self.release.wait()
                return mock_llm_chunk(content="old result")
            finally:
                self.cleanup_started.set()
                await self.cleanup_release.wait()
                self.stopped.set()
        assert self.stopped.is_set(), "Replacement overlapped old cleanup"
        self.replacement_started.set()
        await self.release.wait()
        return await FakeBackend.complete(self, **kwargs)


def _todo_call(call_id: str) -> ToolCall:
    return ToolCall(
        id=call_id,
        index=0,
        function=FunctionCall(name="todo", arguments='{"action":"read"}'),
    )


async def _consume(events) -> None:
    _ = [event async for event in events]


async def _wait_for_turn_completion(session: AppServerSession) -> None:
    async def wait() -> None:
        async for event in session.events():
            if isinstance(event, TurnCompleted):
                return

    await asyncio.wait_for(wait(), timeout=5)


class _ManualClock:
    def __init__(self, value: float = 0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _retention_registry(clock: _ManualClock | None = None) -> SessionRuntimeRegistry:
    return SessionRuntimeRegistry(
        AsyncMock(), AsyncMock(), lambda _session_id: 0, clock=clock or _ManualClock()
    )


async def _wait_result(
    registry: SessionRuntimeRegistry,
    agent_id: str,
    run_id: str | None = None,
    *,
    timeout: float | None = None,
) -> TaskResult:
    """Await a registry wait and unwrap its completed TaskResult (design §6)."""
    disposition = await registry.wait_for_agent(agent_id, run_id, timeout=timeout)
    assert disposition.status is WaitStatus.COMPLETED
    assert disposition.result is not None
    return disposition.result


def _stored_result(
    agent_id: str, run_id: str, *, completed_at: float = 0, generation: int = 0
) -> StoredRunResult:
    return StoredRunResult(
        agent_id=agent_id,
        run_id=run_id,
        result=TaskResult(
            response=f"result for {run_id}",
            turns_used=1,
            completed=True,
            agent_id=agent_id,
            run_id=run_id,
        ),
        completed_at=completed_at,
        root_generation=generation,
    )


def _evicted_tombstone(
    agent_id: str, *, run_id: str | None, status: RunStatus | None
) -> _AgentTombstone:
    """A minimal evicted tombstone for result/lease/transcript fixture setup."""
    return _AgentTombstone(
        summary=AgentSummary(
            agent_id=agent_id,
            identity="coder",
            availability=AgentAvailability.EVICTED,
            current_run_id=run_id,
            current_run_status=status,
            name=derive_agent_name(agent_id, "coder", None, None, None),
        ),
        child_session_id=f"session-{agent_id}",
        parent_identity=("root", 0),
    )


def _idle_record(
    agent_id: str,
    *,
    clock: _ManualClock,
    summary: str = "initial",
    idle_ttl_seconds: int | None = None,
) -> AgentRecord:
    runtime = MagicMock()
    runtime.close = AsyncMock()
    # A direct launch carries no purpose label (the projection reads the
    # loop-owned record); a bare MagicMock would auto-create a truthy one.
    runtime.agent_loop.launch_purpose = None
    # The P14 advisory budget-fit log formats these with a float format
    # spec at admission; a bare MagicMock attribute would raise TypeError.
    runtime.agent_loop.runtime_policy.run_max_price = None
    runtime.agent_loop.runtime_policy.run_max_session_tokens = None
    return AgentRecord(
        agent_id=agent_id,
        identity="coder",
        session_id=f"session-{agent_id}",
        runtime=runtime,
        root_generation=0,
        idle_ttl_seconds=idle_ttl_seconds,
        initial_task_summary=summary,
        state=_AgentState.IDLE,
        idle_since=clock(),
        last_task_summary=summary,
    )


class _GatedWakeup:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.arrivals: asyncio.Queue[float] = asyncio.Queue()
        self.called = asyncio.Event()
        self.gates: list[asyncio.Event] = []

    async def __call__(self, delay: float) -> None:
        self.calls.append(delay)
        self.arrivals.put_nowait(delay)
        gate = asyncio.Event()
        self.gates.append(gate)
        self.called.set()
        await gate.wait()


def _reaper_registry(
    clock: _ManualClock,
    *,
    ttl: float = 0,
    cap: int = 0,
    wakeup: _GatedWakeup | None = None,
    generation: int = 0,
) -> tuple[SessionRuntimeRegistry, MagicMock]:
    registry = SessionRuntimeRegistry(
        AsyncMock(),
        AsyncMock(),
        lambda _session_id: 0,
        notify_agents=AsyncMock(),
        clock=clock,
        wakeup=wakeup or _GatedWakeup(),
    )
    root = MagicMock()
    root.agent_loop.config.subagents.max_running_subagents = 16
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = generation
    root.agent_loop.config.subagents.idle_ttl_seconds = ttl
    root.agent_loop.config.subagents.max_idle_agents = cap
    root.agent_loop.config.subagents.retain_agents = False
    root.agent_loop.config.subagents.tombstone_ttl_seconds = 900
    root.agent_loop.config.subagents.max_tombstones = 32
    registry.bind_root(root)
    return registry, root


async def _rearm(registry: SessionRuntimeRegistry) -> None:
    async with registry._registry_lock:
        registry._rearm_reaper_locked()


async def _real_reused_background() -> tuple[
    SessionRuntimeRegistry, Any, AgentRecord, TaskArgs, InvokeContext
]:
    """Build a retained child with real turn controllers for launch rollback tests."""
    parent = build_test_agent_loop(
        config=_config(),
        backend=FakeBackend([mock_llm_chunk(content="Notification received")]),
    )
    child = await AgentRuntimeFactory().create_child(parent, "coder")
    registry = _retention_registry(_ManualClock())
    root = registry._build_child_runtime(parent)
    registry.bind_root(root)
    runtime = registry._build_child_runtime(child)
    record = AgentRecord(
        agent_id="agent-1",
        identity="coder",
        session_id=child.session_id,
        runtime=runtime,
        root_generation=parent._session_generation,
        initial_task_summary="initial",
        state=_AgentState.IDLE,
        idle_since=0,
        last_task_summary="initial",
    )
    registry._agent_records[record.agent_id] = record
    registry._children[record.session_id] = runtime
    return (
        registry,
        parent,
        record,
        # The fabricated record is reused and must stay resident across the
        # reused runs, so the reuse opts into retention explicitly.
        TaskArgs(
            task="retry",
            identity="coder",
            agent_id=record.agent_id,
            background=True,
            retain=True,
        ),
        InvokeContext(tool_call_id="retry", session_id=parent.session_id),
    )


def _dynamic_config():
    return build_test_vibe_config(**{
        "active_model": "small",
        "include_project_context": False,
        "include_prompt_detail": False,
        "enabled_tools": ["task", "read_file"],
        "tools": {
            "task": {"permission": "always"},
            "read_file": {"permission": "always"},
        },
        "providers": [
            {"name": "one", "api_base": "https://one.test", "backend": "generic"},
            {"name": "two", "api_base": "https://two.test", "backend": "generic"},
        ],
        "models": [
            {
                "name": "small",
                "alias": "small",
                "provider": "one",
                "thinking": "off",
                "supported_thinking_levels": ["off", "high"],
            },
            {
                "name": "large",
                "alias": "large",
                "provider": "one",
                "thinking": "high",
                "supported_thinking_levels": ["off", "high"],
            },
            {
                "name": "other",
                "alias": "other",
                "provider": "two",
                "thinking": "high",
                "supported_thinking_levels": ["off", "high"],
            },
        ],
    })


async def _dynamic_registry(monkeypatch: pytest.MonkeyPatch):
    """Real retained-agent machinery with inspectable fake provider backends."""
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
    first = await _background_result(
        registry,
        TaskArgs(
            task="first",
            # Package 16a: the custom profile is the custom identity whose
            # prompt resource the app_server conftest installs.
            identity="user:dynamic",
            background=True,
            # Identity retention no longer rides the record; the helper opts
            # in so the child stays resident across the reuse checks.
            retain=True,
            config=LaunchConfig(
                model="small",
                thinking="off",
                enabled_tools=["read_file"],
                instructions="Retained role instruction.",
            ),
        ),
        context,
    )
    assert first.agent_id is not None and first.run_id is not None
    await registry.wait_for_agent(first.agent_id, first.run_id)
    return (
        registry,
        parent,
        registry._agent_records[first.agent_id],
        context,
        first,
        backends,
    )


async def _persistent_dynamic_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the real retained-agent path with a real child-session directory."""
    logging = SessionLoggingConfig(
        enabled=True, save_dir=str(tmp_path / "sessions"), session_prefix="session"
    )

    def create_backend(*, provider, **_kwargs):
        return FakeBackend([mock_llm_chunk(content=provider.name)])

    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", create_backend
    )
    parent = build_test_agent_loop(
        config=_dynamic_config().model_copy(update={"session_logging": logging}),
        backend=FakeBackend([mock_llm_chunk(content="Notification received")]),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="dynamic", session_id=parent.session_id)
    first = await _background_result(
        registry,
        TaskArgs(
            task="first",
            # Package 16a: the custom profile is the custom identity whose
            # prompt resource the app_server conftest installs.
            identity="user:dynamic",
            background=True,
            # Identity retention no longer rides the record; the helper opts
            # in so the child stays resident across the reuse checks.
            retain=True,
            config=LaunchConfig(
                model="small",
                thinking="off",
                enabled_tools=["read_file"],
                instructions="Retained role instruction.",
            ),
        ),
        context,
    )
    assert first.agent_id is not None and first.run_id is not None
    await registry.wait_for_agent(first.agent_id, first.run_id)
    return registry, parent, registry._agent_records[first.agent_id], context, first, []


async def _foreground_registry(monkeypatch: pytest.MonkeyPatch, backend: FakeBackend):
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    parent = build_test_agent_loop(config=_config(), backend=FakeBackend())
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    return registry, parent


async def _foreground_result(
    registry: SessionRuntimeRegistry,
    parent: AgentLoop,
    tool_call_id: str = "foreground",
) -> TaskResult:
    return cast(
        TaskResult,
        [
            item
            async for item in registry.run(
                TaskArgs(task="foreground", identity="coder", background=False),
                InvokeContext(tool_call_id=tool_call_id, session_id=parent.session_id),
            )
        ][-1],
    )


async def _zero_retention_registry(
    monkeypatch: pytest.MonkeyPatch, backend: FakeBackend, *, zero_limits: bool = True
) -> tuple[SessionRuntimeRegistry, AgentLoop, list[str]]:
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend", lambda **_: backend
    )
    config = _config()
    if zero_limits:
        config = config.model_copy(
            update={
                "subagents": config.subagents.model_copy(
                    update={"idle_ttl_seconds": 0, "max_idle_agents": 0}
                )
            }
        )
    parent = build_test_agent_loop(config=config, backend=FakeBackend())
    registry = _retention_registry()
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
    return registry, parent, notifications


def _record(
    agent_id: str,
    *,
    clock: _ManualClock,
    state: _AgentState = _AgentState.IDLE,
    retained: bool = False,
) -> AgentRecord:
    """An ``_idle_record`` whose mock runtime exposes an iterable message list."""
    record = _idle_record(agent_id, clock=clock)
    record.runtime.agent_loop.messages = MessageList()
    record.state = state
    record.retained = retained
    return record


def _run(
    record: AgentRecord,
    run_id: str,
    *,
    started_at: float,
    completed_at: float | None = None,
    status: RunStatus = RunStatus.COMPLETED,
    done: bool = True,
) -> RunRecord:
    """A run record for ``record``; ``done`` controls whether completion settled."""
    future = asyncio.get_running_loop().create_future()
    if done:
        future.set_result(None)
    run = RunRecord(
        run_id,
        record.agent_id,
        status,
        future,
        record.identity,
        started_at=started_at,
        completed_at=completed_at,
    )
    return run


def _tombstone(
    agent_id: str,
    *,
    reason: str,
    closed_at: float,
    expires_at: float | None,
    parent_identity: tuple[str, int] = ("root", 0),
    availability: AgentAvailability | None = None,
    run_id: str = "run-1",
) -> _AgentTombstone:
    availability = availability or (
        AgentAvailability.CLOSED if reason == "closed" else AgentAvailability.EVICTED
    )
    return _AgentTombstone(
        summary=AgentSummary(
            agent_id=agent_id,
            identity="coder",
            availability=availability,
            current_run_id=run_id,
            current_run_status=RunStatus.COMPLETED,
        ),
        child_session_id=f"session-{agent_id}",
        parent_identity=parent_identity,
        reason=reason,
        closed_at=closed_at,
        expires_at=expires_at,
    )


def _root_registry(
    clock: _ManualClock, *, generation: int = 0
) -> tuple[SessionRuntimeRegistry, MagicMock]:
    """A registry with a MagicMock root bound for admission-style calls."""
    registry = _retention_registry(clock)
    root = MagicMock()
    root.agent_loop.config.subagents.max_running_subagents = 16
    root.agent_loop.session_id = "root"
    root.agent_loop._session_generation = generation
    root.turns._projector = None
    registry._root = root
    registry._generation_identity = ("root", generation)
    return registry, root
