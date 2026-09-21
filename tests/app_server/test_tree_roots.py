"""Root authority replacement through the serialized public resource."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from chartreux.app_server._execution import SessionExecutionKind
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._sessions import SessionRuntime, SessionRuntimeRegistry
from chartreux.app_server.protocol import (
    AppServerResponseError,
    ClientCapabilities,
    SessionOptions,
)
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import ToolExecutionResponse
from chartreux.core.config.layer import ConfigLayer
from chartreux.core.subagents import TaskArgs
from chartreux.core.tools.base import InvokeContext, ToolPermission, ToolPermissionError
from chartreux.core.tools.builtins.read_file import ReadFileResult
from tests.app_server.backend_contract.conftest import connect_backend_contract_host
from tests.app_server.test_tree_policy import queued, state
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize(
    "outcome",
    [
        "success",
        "busy",
        "queue",
        "failure",
        "cancel",
        "malformed",
        "stale",
        "validation",
        "race",
        "missing-parent",
        "invalidated-parent",
    ],
)
async def test_public_root_tree(
    tmp_path: Path, config_dir: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(AgentLoop, "backend_factory", lambda *_: FakeBackend())
    registries: list[SessionRuntimeRegistry] = []
    bind = SessionRuntimeRegistry.bind_root

    def capture(registry: SessionRuntimeRegistry, runtime: SessionRuntime) -> None:
        bind(registry, runtime)
        registries.append(registry)

    monkeypatch.setattr(SessionRuntimeRegistry, "bind_root", capture)
    path = config_dir / "config.toml"
    disk = path.read_bytes()
    connection = await connect_backend_contract_host(
        session_options=SessionOptions(), capabilities=ClientCapabilities()
    )
    session = await connection.host.open_session()
    registry = registries[0]
    assert registry._root is not None
    runtimes = [registry._root]
    try:
        for agent in ("worker", "worker"):
            parent = runtimes[-1]
            child = await AgentRuntimeFactory().create_child(parent.agent_loop, agent)
            runtime = registry._build_child_runtime(child)
            registry._children[child.session_id] = runtime
            registry._child_links[child.session_id] = (parent, "synthetic-link")
            runtimes.append(runtime)
            await child.wait_until_ready()
        # Deliberately reverse insertion order: preparation must still be top down.
        registry._children = dict(reversed(list(registry._children.items())))
        resource = session.resources.config
        policy = await resource.read_roots()
        for invalid in (
            {"roots": {"private-root-marker": []}},
            {"scope": "user"},
            {"userInitiated": False},
            {"sessionId": runtimes[-1].agent_loop.session_id},
        ):
            params: dict[str, Any] = {
                "sessionId": session.session_id,
                "expectedRevision": policy.revision,
                "roots": [],
                "userInitiated": True,
            }
            params.update(invalid)
            with pytest.raises(AppServerResponseError) as error:
                await connection.client.request("policy/roots/replace", params)
            assert "private-root-marker" not in str(error.value)
        assert (await resource.read_roots()).revision == policy.revision
        target = tmp_path / "related" / "file.txt"
        target.parent.mkdir()
        target.write_text("safe related-root fixture\n")
        assert not runtimes[0].agent_loop.tool_manager.workspace.allows(target)
        before = [state(r) for r in runtimes]
        old = [r.agent_loop.tool_manager.get("read_file") for r in runtimes]
        inherited = [r.agent_loop._inherited_workspace for r in runtimes]
        execution = None
        if outcome == "busy":
            execution = runtimes[-1].execution.begin(SessionExecutionKind.SHELL, "busy")
        if outcome == "queue":
            runtimes[-1].turns.enqueue(queued(runtimes[-1].agent_loop.session_id))
        original = AgentLoop._prepare_policy_refresh
        entered = asyncio.Event()
        release = asyncio.Event()
        paused = outcome in {"race", "missing-parent", "invalidated-parent"}
        workspaces = [r.agent_loop.tool_manager.workspace for r in runtimes]

        async def fail(loop: AgentLoop, **kwargs: Any) -> Any:
            candidate = await original(loop, **kwargs)
            if paused and loop is runtimes[1].agent_loop:
                entered.set()
                await release.wait()
                if outcome == "race":
                    raise RuntimeError("private-root-marker")
            if loop is runtimes[-1].agent_loop:
                if outcome == "failure":
                    raise RuntimeError("private-root-marker")
                if outcome == "cancel":
                    entered.set()
                    await asyncio.Event().wait()
            return candidate

        monkeypatch.setattr(AgentLoop, "_prepare_policy_refresh", fail)
        validate = AgentLoop._validate_policy_replacement

        def reject_last(loop: AgentLoop, prepared: Any) -> None:
            validate(loop, prepared)
            if loop is runtimes[-1].agent_loop:
                raise ValueError("private-root-marker")

        if outcome == "validation":
            monkeypatch.setattr(AgentLoop, "_validate_policy_replacement", reject_last)
        roots = (
            [str(target.parent)] if outcome != "malformed" else ["private-root-marker"]
        )
        if paused:
            task = asyncio.create_task(
                resource.replace_roots(
                    roots=roots, expected_revision=policy.revision, user_initiated=True
                )
            )
            links = dict(registry._child_links)
            children = dict(registry._children)
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert registry._policy_reserved
                for runtime in runtimes:
                    assert runtime.execution.active is not None
                    assert (
                        runtime.execution.active.kind == SessionExecutionKind.LIFECYCLE
                    )
                    with pytest.raises(
                        RuntimeError, match="Session tree configuration change"
                    ):
                        runtime.turns.enqueue(queued(runtime.agent_loop.session_id))
                    with pytest.raises(RuntimeError):
                        runtime.execution.begin(
                            SessionExecutionKind.TURN, "racing-turn"
                        )
                    assert not runtime.turns.has_queued_turns
                # Exercise the real subagent entry point, not just its guard helper.
                with pytest.raises(RuntimeError):
                    async for _ in registry.run(
                        TaskArgs(task="must not run", agent="plan"),
                        InvokeContext(
                            tool_call_id="racing-child",
                            session_id=runtimes[-1].agent_loop.session_id,
                        ),
                    ):
                        pytest.fail("Child creation escaped the tree reservation")
                assert registry._children == children
                assert registry._creating_children == 0
                if outcome == "missing-parent":
                    registry._child_links.pop(runtimes[-1].agent_loop.session_id)
                elif outcome == "invalidated-parent":
                    # Still traversable, but different from the captured tree.
                    registry._child_links[runtimes[1].agent_loop.session_id] = (
                        runtimes[0],
                        "changed-link",
                    )
            finally:
                release.set()
                try:
                    with pytest.raises(AppServerResponseError):
                        await asyncio.wait_for(task, 5)
                finally:
                    registry._child_links = links
        elif outcome == "cancel":
            # Cancellation at the owning operation boundary, not transport request
            # abandonment (which is deliberately a separate lifecycle contract).
            task = asyncio.create_task(
                registry.replace_roots(
                    session_id=runtimes[0].agent_loop.session_id,
                    roots=roots,
                    expected_revision=policy.revision,
                )
            )
            await asyncio.wait_for(entered.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif outcome != "success":
            try:
                with pytest.raises(AppServerResponseError) as error:
                    await resource.replace_roots(
                        roots=roots,
                        expected_revision="stale"
                        if outcome == "stale"
                        else policy.revision,
                        user_initiated=True,
                    )
                assert "private-root-marker" not in str(error.value)
            finally:
                if execution is not None:
                    runtimes[-1].execution.finish(execution)
        else:
            readers = []
            for replacement in (roots, []):
                live_layers = {
                    id(layer)
                    for runtime in runtimes
                    for layer in runtime.agent_loop.config_orchestrator.layers
                }
                accept_cache = ConfigLayer._accept_loaded_state

                def reject_cache_adoption(
                    layer: Any,
                    staged: Any,
                    *,
                    live_layers: set[int] = live_layers,
                    accept_cache: Any = accept_cache,
                ) -> None:
                    if id(layer) in live_layers:
                        raise RuntimeError(
                            "Policy publication must not adopt source caches"
                        )
                    accept_cache(layer, staged)

                with monkeypatch.context() as publication:
                    publication.setattr(
                        ConfigLayer, "_accept_loaded_state", reject_cache_adoption
                    )
                    result = await resource.replace_roots(
                        roots=replacement,
                        expected_revision=policy.revision,
                        user_initiated=True,
                    )
                assert result.revision != policy.revision
                policy = await resource.read_roots()
                assert policy.roots == replacement
                assert policy.project == str(runtimes[0].agent_loop.cwd)
                for index, runtime in enumerate(runtimes):
                    loop = runtime.agent_loop
                    assert loop.tool_manager.workspace.allows(target) == bool(
                        replacement
                    )
                    tool = loop.tool_manager.get("read_file")
                    decision = tool.resolve_permission(
                        tool.validate_arguments({"file_path": str(target)})
                    )
                    assert decision is not None
                    assert (decision.permission != ToolPermission.NEVER) == bool(
                        replacement
                    )
                    # Invoke only after the normal loop authorization decision;
                    # bare invoke checks retirement, not all permission ceilings.
                    authorization = await loop._should_execute_tool(
                        tool, tool.validate_arguments({"file_path": str(target)})
                    )
                    if replacement:
                        assert authorization.verdict == ToolExecutionResponse.EXECUTE
                        results = [
                            item async for item in tool.invoke(file_path=str(target))
                        ]
                        assert len(results) == 1
                        assert isinstance(results[0], ReadFileResult)
                        assert "safe related-root fixture" in results[0].content
                        readers.append(tool)
                    else:
                        assert authorization.verdict == ToolExecutionResponse.SKIP
                        assert authorization.approval_type == ToolPermission.NEVER
                        with pytest.raises(ToolPermissionError, match="retired"):
                            async for _ in readers[index].invoke(file_path=str(target)):
                                pytest.fail(
                                    "Previously successful reader remained live"
                                )
                    with pytest.raises(ToolPermissionError, match="retired"):
                        async for _ in old[index].invoke(file_path=str(target)):
                            pytest.fail("Retired tool yielded a result")
                    if index:
                        assert (
                            loop._inherited_workspace
                            == runtimes[index - 1].agent_loop.tool_manager.workspace
                        )
                    with pytest.raises(ToolPermissionError, match="retired"):
                        _ = old[index].config
                    # Ordinary changes and mode switches must retain the overlay.
                    errors = await loop.config_orchestrator.set_field(
                        "/active_model", "", reason="ordinary", target_layer="overrides"
                    )
                    assert not errors
                    await loop.reload_with_initial_messages()
                    assert loop.tool_manager.workspace.allows(target) == bool(
                        replacement
                    )
                policy = await resource.read_roots()
                old = [r.agent_loop.tool_manager.get("read_file") for r in runtimes]
        if outcome != "success":
            assert [state(r) for r in runtimes] == before
            assert [r.agent_loop._inherited_workspace for r in runtimes] == inherited
            assert [r.agent_loop.tool_manager.workspace for r in runtimes] == workspaces
            assert (await resource.read_roots()).revision == policy.revision
            assert all(r.execution.active is None for r in runtimes)
            for runtime, previous in zip(runtimes, before, strict=True):
                assert runtime.agent_loop.config is previous[0]
                assert runtime.agent_loop.tool_manager is previous[3]
            for tool in old:
                _ = tool.config
            assert not registry._policy_reserved
        assert path.read_bytes() == disk
    finally:
        await session.close()
        await connection.host.close()
