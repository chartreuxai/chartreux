"""Policy transactions through the real registry, runtime factory and wire."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from chartreux.agents import AgentSafety
from chartreux.app_server._execution import SessionExecutionKind
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server.models import SessionTextContentBlock, TurnUserInputEntry
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    PolicyReadResponse,
    PolicyToolReplacement,
    ProtocolErrorCode,
    SessionOptions,
    TurnEnqueueParams,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import AgentLoopStateError
from chartreux.core.agents.models import AgentProfile
from chartreux.core.config import MCPHttp
from chartreux.core.config.orchestrator import ConfigOrchestrator
from chartreux.core.session_types import SessionMetadata
from chartreux.core.tools.base import ToolPermissionError
from chartreux.core.tools.mcp.authorization import (
    MCPAuthorizationRef,
    MCPAuthorizationRequired,
    MCPAuthorizationSnapshot,
)
from chartreux.core.tools.remote import RemoteTool
from tests.app_server.backend_contract.conftest import connect_backend_contract_host
from tests.core.agent_loop.test_policy_replacement import make_loop, replace
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


def queued(session_id: str) -> TurnEnqueueParams:
    return TurnEnqueueParams(
        session_id=session_id,
        entries=[TurnUserInputEntry(content=[SessionTextContentBlock(text="queued")])],
    )


async def ignore(*args: Any, **kwargs: Any) -> None:
    pass


NEUTRAL_TEST_AGENT = AgentProfile(
    name="neutral-test-agent",
    display_name="Neutral Test Agent",
    description="Unrestricted test subagent",
    safety=AgentSafety.NEUTRAL,
)


def register_neutral_test_agent(loop: AgentLoop) -> None:
    loop.agent_manager._discovered[NEUTRAL_TEST_AGENT.name] = NEUTRAL_TEST_AGENT


@asynccontextmanager
async def tree(
    tmp_path: Path, *, reassert: bool = False
) -> AsyncIterator[tuple[SessionRuntimeRegistry, list[SessionRuntime]]]:
    root = await make_loop(tmp_path)
    registry = SessionRuntimeRegistry(ignore, ignore, lambda _: 0)
    runtimes = [registry._build_child_runtime(root)]
    registry.bind_root(runtimes[0])
    try:
        for parent, agent in [
            (root, NEUTRAL_TEST_AGENT.name),
            (None, NEUTRAL_TEST_AGENT.name),
            (root, NEUTRAL_TEST_AGENT.name),
        ]:
            parent = parent or runtimes[1].agent_loop
            register_neutral_test_agent(parent)
            child = await AgentRuntimeFactory().create_child(parent, agent)
            runtime = registry._build_child_runtime(child)
            registry._children[child.session_id] = runtime
            runtimes.append(runtime)
            await child.wait_until_ready()
            if len(runtimes) == 2 and reassert:
                await replace(child, ["A"])
        yield registry, runtimes
    finally:
        await registry.close_children()
        await runtimes[0].close()


def state(runtime: SessionRuntime) -> tuple[object, ...]:
    loop = runtime.agent_loop
    return (
        loop.config,
        loop.config_orchestrator.restrictions,
        loop.config_orchestrator.accepted_token,
        loop.tool_manager,
        loop.runtime_policy.inherited_restrictions,
    )


async def update(
    registry: SessionRuntimeRegistry, root: SessionRuntime, patterns: list[str]
) -> str:
    return await registry.replace_policy(
        session_id=root.agent_loop.session_id,
        source="user-toml",
        tools={"read_file": {"denylist": patterns}},
        expected_revision=registry.policy_revision,
    )


@pytest.mark.parametrize("reassert", [False, True])
async def test_transitive_refresh_preserves_own_assertions_and_mode(
    tmp_path: Path, reassert: bool
) -> None:
    async with tree(tmp_path, reassert=reassert) as (registry, runtimes):
        old = [r.agent_loop.tool_manager.get("read_file") for r in runtimes]
        modes = [
            r.agent_loop.runtime_policy.inherited_mode_restrictions for r in runtimes
        ]
        scopes = [
            r.agent_loop.runtime_policy.inherited_plan_write_scopes for r in runtimes
        ]
        source_ids = [r.agent_loop.config_orchestrator.policy_owner for r in runtimes]
        for runtime in runtimes:
            assert not hasattr(runtime.agent_loop, "_permission_store")
        for patterns in (["B"], []):
            revision = await update(registry, runtimes[0], patterns)
            assert revision == registry.policy_revision
            for index, runtime in enumerate(runtimes):
                loop = runtime.agent_loop
                denied = set(loop.tool_manager.get_tool_config("read_file").denylist)
                assert ("A" in denied) == (reassert and index in {1, 2})
                assert ("B" in denied) == bool(patterns)
                assert "other" in denied
                assert not hasattr(loop, "_permission_store")
                assert loop.runtime_policy.inherited_mode_restrictions == modes[index]
                assert loop.runtime_policy.inherited_plan_write_scopes == scopes[index]
                assert loop.config_orchestrator.policy_owner == source_ids[index]
                with pytest.raises(ToolPermissionError, match="retired"):
                    _ = old[index].config
                # Refreshing ordinary runtime state must not resurrect copied policy.
                await loop.reload_with_initial_messages()
                assert (
                    set(loop.tool_manager.get_tool_config("read_file").denylist)
                    == denied
                )
        assert '"A"' in (tmp_path / "settings.toml").read_text()
        # A later child captures the newly accepted ancestor ceiling.
        later = await AgentRuntimeFactory().create_child(
            runtimes[0].agent_loop, "worker"
        )
        try:
            await later.wait_until_ready()
            assert "A" not in later.tool_manager.get_tool_config("read_file").denylist
        finally:
            await later.aclose()


@pytest.mark.parametrize(
    "busy",
    ["turn", "shell", "lifecycle", "held", "queued", "promotion", "construction"],
)
async def test_busy_descendant_rejected_without_policy_mutation(
    tmp_path: Path, busy: str
) -> None:
    async with tree(tmp_path) as (registry, runtimes):
        child = runtimes[2]
        before = [state(r) for r in runtimes]
        execution = None
        task = None
        if busy in {"turn", "shell", "lifecycle"}:
            execution = child.execution.begin(SessionExecutionKind(busy), "busy")
        elif busy == "held":
            child.agent_loop._take_session("test")
        elif busy == "queued":
            child.turns.enqueue(queued(child.agent_loop.session_id))
        elif busy == "promotion":

            async def pending() -> None:
                await asyncio.Event().wait()

            task = asyncio.create_task(pending())
            child.turns._queue_tasks.add(task)
        else:
            await registry._ensure_child_lock.acquire()
        try:
            with pytest.raises((RuntimeError, AgentLoopStateError)):
                await update(registry, runtimes[0], [])
            assert [state(r) for r in runtimes] == before
            assert runtimes[0].execution.active is None
        finally:
            if execution is not None:
                child.execution.finish(execution)
            if busy == "held":
                child.agent_loop._release_session("test")
            if busy == "construction":
                registry._ensure_child_lock.release()
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                child.turns._queue_tasks.discard(task)


@pytest.mark.parametrize(
    "failure", ["prepare", "stale", "held", "cancel", "layers", "session"]
)
async def test_preparation_failure_preserves_entire_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    async with tree(tmp_path) as (registry, runtimes):
        before = [state(r) for r in runtimes]
        original = AgentLoop._prepare_policy_refresh

        async def fail(loop: AgentLoop, **kwargs: Any) -> Any:
            prepared = await original(loop, **kwargs)
            if loop is runtimes[-1].agent_loop:
                if failure == "prepare":
                    raise RuntimeError("last descendant failed")
                if failure == "cancel":
                    raise asyncio.CancelledError
                if failure == "stale":
                    loop.config_orchestrator._accepted_token = uuid4()
                elif failure == "layers":
                    prepared.orchestrator.remove_layer(-1)
                elif failure == "session":
                    loop._session_generation += 1
                else:
                    loop._take_session("late-holder")
            return prepared

        monkeypatch.setattr(AgentLoop, "_prepare_policy_refresh", fail)
        try:
            with pytest.raises((
                RuntimeError,
                ValueError,
                AgentLoopStateError,
                asyncio.CancelledError,
            )):
                await update(registry, runtimes[0], [])
            # Stale revision injection itself is the only permitted difference.
            for runtime, previous in zip(runtimes, before, strict=True):
                current = state(runtime)
                assert current[:2] == previous[:2]
                assert current[3:] == previous[3:]
                assert runtime.execution.active is None
            assert not registry._policy_reserved
        finally:
            if failure == "held":
                runtimes[-1].agent_loop._release_session("late-holder")


async def test_tree_reserved_before_first_await(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with tree(tmp_path) as (registry, runtimes):
        entered, release = asyncio.Event(), asyncio.Event()
        original = ConfigOrchestrator._stage_policy_replacement

        async def gated(orchestrator: Any, **kwargs: Any) -> Any:
            entered.set()
            await release.wait()
            return await original(orchestrator, **kwargs)

        monkeypatch.setattr(ConfigOrchestrator, "_stage_policy_replacement", gated)
        task = asyncio.create_task(update(registry, runtimes[0], []))
        try:
            await asyncio.wait_for(entered.wait(), 2)
            for runtime in runtimes:
                assert runtime.execution.active is not None
                assert runtime.execution.active.kind == SessionExecutionKind.LIFECYCLE
                with pytest.raises((RuntimeError, AgentLoopStateError)):
                    runtime.turns.enqueue(queued(runtime.agent_loop.session_id))
                assert not runtime.turns.has_queued_turns
            with pytest.raises((RuntimeError, AgentLoopStateError)):
                await registry.ensure_child("missing")
            with pytest.raises((RuntimeError, AgentLoopStateError)):
                await registry.close_children()
            with pytest.raises((RuntimeError, AgentLoopStateError)):
                await update(registry, runtimes[0], [])
        finally:
            release.set()
            await task
        assert all(r.execution.active is None for r in runtimes)


@pytest.mark.parametrize("fail_last", [False, True])
@pytest.mark.parametrize("expired", ["descriptors", "authorization"])
async def test_policy_staging_does_not_touch_expired_mcp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_last: bool, expired: str
) -> None:
    discover = AsyncMock(return_value=[RemoteTool(name="search")])
    monkeypatch.setattr("chartreux.core.tools.mcp.registry.list_tools_http", discover)
    async with tree(tmp_path) as (registry, runtimes):
        providers, sinks, before_registry, before_specs = [], [], [], []
        fields = (
            "_cache",
            "_memory_records",
            "_cache_keys_by_alias",
            "_servers_by_alias",
            "_needs_auth",
            "_descriptor_revisions",
            "_failed",
            "_force_refresh",
            "_authorization_refs",
        )
        for index, runtime in enumerate(runtimes):
            loop = runtime.agent_loop
            server = MCPHttp(
                name="fake", transport="streamable-http", url="https://fake.invalid/mcp"
            )
            await loop.config_orchestrator.set_field(
                "/mcp_servers", [server.model_dump()]
            )
            provider, sink = AsyncMock(), AsyncMock()
            provider.resolve.return_value = MCPAuthorizationSnapshot(
                headers={},
                connection_revision="fake-connection",
                descriptor_revision="fake-descriptors",
            )
            mcp = loop.mcp_registry
            assert mcp is not None
            mcp.configure_authorization(
                provider,
                {
                    "fake": MCPAuthorizationRef(
                        server_name="fake",
                        server_fingerprint="fake-fingerprint",
                        kind="none",
                        descriptor_revision="fake-descriptors",
                    )
                },
                required_sink=sink,
                descriptor_cache_root=tmp_path / "cache" / str(index),
                descriptor_cache_ttl_s=0,
            )
            await loop.tool_manager.reconfigure_mcp_async()
            assert "fake_search" in loop.tool_manager.available_tools
            loop.tool_manager.get("fake_search")
            if expired == "authorization":
                provider.resolve.return_value = MCPAuthorizationRequired(
                    reason="expired", descriptor_revision="fake-expired"
                )
            provider.reset_mock()
            providers.append(provider)
            sinks.append(sink)
            before_registry.append(deepcopy(tuple(getattr(mcp, f) for f in fields)))
            before_specs.append(loop.tool_manager.available_tool_specs())
        discover.reset_mock()
        before = [state(r) for r in runtimes]
        cache_before = {
            p: p.read_bytes() for p in (tmp_path / "cache").rglob("*") if p.is_file()
        }
        original = AgentLoop._prepare_policy_refresh

        async def prepare(loop: AgentLoop, **kwargs: Any) -> Any:
            candidate = await original(loop, **kwargs)
            assert "fake_search" in candidate.tool_manager.available_tools
            assert "B" in candidate.tool_manager.get_tool_config("fake_search").denylist
            if fail_last and loop is runtimes[-1].agent_loop:
                raise RuntimeError("last descendant failed")
            return candidate

        monkeypatch.setattr(AgentLoop, "_prepare_policy_refresh", prepare)

        async def apply() -> None:
            await registry.replace_policy(
                session_id=runtimes[0].agent_loop.session_id,
                source="user-toml",
                tools={"fake_search": {"denylist": ["B"]}},
                expected_revision=registry.policy_revision,
            )

        if fail_last:
            with pytest.raises(RuntimeError, match="last descendant failed"):
                await apply()
            assert [state(r) for r in runtimes] == before
        else:
            await apply()
        for index, runtime in enumerate(runtimes):
            loop = runtime.agent_loop
            assert (
                tuple(getattr(loop.mcp_registry, f) for f in fields)
                == before_registry[index]
            )
            assert loop.tool_manager.available_tool_specs() == before_specs[index]
            providers[index].resolve.assert_not_awaited()
            providers[index].reject.assert_not_awaited()
            sinks[index].assert_not_called()
            assert runtime.execution.active is None
        assert {
            p: p.read_bytes() for p in (tmp_path / "cache").rglob("*") if p.is_file()
        } == cache_before
        discover.assert_not_awaited()


@pytest.mark.parametrize("same_session", [False, True])
async def test_public_serialized_session_only_policy(
    tmp_path: Path,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    same_session: bool,
) -> None:
    monkeypatch.setattr(AgentLoop, "backend_factory", lambda *_: FakeBackend())
    registries: list[SessionRuntimeRegistry] = []
    bind_root = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: SessionRuntime) -> None:
        bind_root(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    path = config_dir / "config.toml"
    before = path.read_bytes()
    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    try:
        registry = registries[0]
        assert registry._root is not None
        parent = registry._root.agent_loop
        descendants: list[AgentLoop] = []
        for agent in ("worker", "worker"):
            child = await AgentRuntimeFactory().create_child(parent, agent)
            registry._children[child.session_id] = registry._build_child_runtime(child)
            descendants.append(child)
            await child.wait_until_ready()
            parent = child
        resource = session.resources.config
        policy = await resource.read_policy()
        assert "user-toml" in policy.sources
        result = await resource.replace_policy(
            source="user-toml",
            expected_revision=policy.revision,
            tools={
                "read_file": PolicyToolReplacement(
                    denylist=["A"], sensitive_patterns=["*.private"]
                )
            },
            user_initiated=True,
        )
        assert result.revision != policy.revision
        assert (await resource.read_policy()).revision == result.revision
        for child in descendants:
            assert "A" in child.tool_manager.get_tool_config("read_file").denylist
            assert (
                "*.private"
                in child.tool_manager.get_tool_config("read_file").sensitive_patterns
            )
        with pytest.raises(AppServerResponseError):
            await resource.replace_policy(
                source="user-toml",
                expected_revision=policy.revision,
                tools={},
                user_initiated=True,
            )
        with pytest.raises(ValueError):
            await resource.replace_policy(
                source="user-toml",
                expected_revision=result.revision,
                tools={},
                user_initiated=False,
            )
        for invalid in (
            {"scope": "user"},
            {"userInitiated": False},
            {"tools": {"read_file": {"denylist": [1]}}},
        ):
            params: dict[str, Any] = {
                "sessionId": session.session_id,
                "source": "user-toml",
                "expectedRevision": result.revision,
                "scope": "session",
                "userInitiated": True,
                "tools": {},
            }
            params.update(invalid)
            with pytest.raises(AppServerResponseError):
                await connection.client.request("config/policy/replace", params)
        await resource.replace_policy(
            source="user-toml",
            expected_revision=result.revision,
            tools={},
            user_initiated=True,
        )
        for child in descendants:
            assert "A" not in child.tool_manager.get_tool_config("read_file").denylist
            assert (
                "*.private"
                not in child.tool_manager.get_tool_config(
                    "read_file"
                ).sensitive_patterns
            )
        assert path.read_bytes() == before
        await registry.close_children()
        root = registry._root
        assert root is not None
        loop = root.agent_loop
        old_policy = PolicyReadResponse.model_validate(
            await connection.client.request(
                "config/policy/read", {"sessionId": loop.session_id}
            )
        )
        token = loop.config_orchestrator.accepted_token
        new_id = loop.session_id if same_session else "rebound-policy-session"
        loop.rebind_to_session(
            new_id,
            tmp_path / new_id,
            [],
            session_metadata=SessionMetadata(
                session_id=new_id,
                start_time="2026-01-01T00:00:00",
                end_time=None,
                git_commit=None,
                git_branch=None,
                environment={"working_directory": str(tmp_path)},
                username="fixture",
                config={},
            ),
        )
        assert loop.config_orchestrator.accepted_token is token
        rebound = state(root)
        params = {
            "sessionId": new_id,
            "source": "user-toml",
            "expectedRevision": old_policy.revision,
            "scope": "session",
            "userInitiated": True,
            "tools": {"read_file": {"denylist": ["stale"]}},
        }
        with pytest.raises(AppServerResponseError) as error:
            await connection.client.request("config/policy/replace", params)
        assert error.value.error.code == ProtocolErrorCode.CONFLICT
        assert state(root) == rebound
        assert root.execution.active is None
        fresh = PolicyReadResponse.model_validate(
            await connection.client.request("config/policy/read", {"sessionId": new_id})
        )
        assert fresh.revision != old_policy.revision
        params["expectedRevision"] = fresh.revision
        await connection.client.request("config/policy/replace", params)
        assert "stale" in loop.tool_manager.get_tool_config("read_file").denylist
    finally:
        await session.close()
        await connection.host.close()
