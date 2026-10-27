"""Actual factory inheritance and child permission enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._handler import CoreRequestHandler
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server._subagent_launch import ChildLaunchAdmission
from chartreux.app_server.protocol import ProtocolErrorCode, SessionForkParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop.errors import AgentLoopStateError
from chartreux.core.agents.launch import resolve_launch
from chartreux.core.agents.manager import AgentManager
from chartreux.core.agents.registry import apply_launch_overrides
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.subagents import (
    InvalidLaunchToolError,
    LaunchConfig,
    TaskArgs,
    UnsupportedChildForkError,
)
from chartreux.core.tools.base import BaseToolState, InvokeContext, ToolError
from chartreux.core.tools.builtins.bash import BashArgs
from chartreux.core.tools.builtins.edit import EditArgs
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.builtins.task import Task, TaskToolConfig
from chartreux.core.tools.classification import (
    TOOL_CLASSIFICATIONS,
    ToolClass,
    reclassify_tool,
)
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.models import ToolPermission
from tests.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.app_server._subagent_test_support import (
    _background_result,
    _retention_registry,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import collect_result, mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


async def test_resume_captures_current_parent_not_stored_child_policy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    orchestrator = await make_orchestrator(path)
    assert not await orchestrator.set_field(
        "/session_logging",
        {"enabled": True, "save_dir": str(tmp_path / "sessions")},
        target_layer=OverridesLayer.NAME,
    )
    parent = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    try:
        await parent.persist_empty_session()
        child = await factory.create_child(parent, "coder")
        await child.wait_until_ready()
        await child.persist_empty_session()
        directory = child.session_logger.session_dir
        assert directory is not None
        child_id = child.session_id
        await child.aclose()
        path.write_text('[tools.read_file]\npermission = "never"\n')
        await orchestrator.reload()
        resumed = await factory.resume_child(parent, "coder", child_id, directory)
        loops.append(resumed)
        await resumed.wait_until_ready()
        cached = resumed.tool_manager.get("read_file")
        assert cached.config.permission == ToolPermission.NEVER
        path.write_text('[tools.read_file]\npermission = "always"\n')
        await orchestrator.reload()
        await resumed.config_orchestrator.reload()
        assert resumed.tool_manager.get("read_file") is cached
        assert cached.config.permission == ToolPermission.NEVER
        with pytest.raises(UnsupportedChildForkError) as exc_info:
            await factory.fork(resumed, None)
        assert exc_info.value.field == "session"
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_public_fork_route_rejects_child_before_factory(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, "coder")
        await child.wait_until_ready()
        handler = MagicMock(spec=CoreRequestHandler)
        handler._agent_loop = child
        handler._require_attached = MagicMock()
        handler._runtime_factory = MagicMock()
        handler._runtime_factory.fork = AsyncMock()

        with pytest.raises(RequestFailure) as exc_info:
            await CoreRequestHandler._session_fork(
                handler, SessionForkParams(source_session_id=child.session_id)
            )

        assert exc_info.value.code is ProtocolErrorCode.INVALID_PARAMS
        assert "Child sessions" in str(exc_info.value)
        handler._runtime_factory.fork.assert_not_awaited()
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


@pytest.mark.parametrize("failure", [FileNotFoundError, ValueError])
async def test_missing_or_invalid_parent_authority_aborts_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: type[Exception]
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.read_file]\npermission = "never"\n')
    orchestrator = await make_orchestrator(path)
    parent = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    accepted = orchestrator.restrictions

    def unavailable(_self: object) -> object:
        raise failure("source unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(type(orchestrator), "restrictions", property(unavailable))
            with pytest.raises(failure, match="source unavailable"):
                await AgentRuntimeFactory().create_child(parent, "coder")
        assert orchestrator.restrictions is accepted
    finally:
        await parent.aclose()


WP3_AGENT_NAME = "wp3-agent"
"""The custom identity prompt installed by this directory's autouse fixture."""


async def test_child_scratch_access_survives_rebind_close_and_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    orchestrator = await make_orchestrator(path)
    assert not await orchestrator.set_field(
        "/session_logging",
        {"enabled": True, "save_dir": str(tmp_path / "sessions")},
        target_layer=OverridesLayer.NAME,
    )
    parent = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    scratch = parent.scratchpad_dir
    assert scratch is not None
    try:
        await parent.persist_empty_session()
        generation = parent._session_generation
        await parent._reset_session()
        assert parent._session_generation == generation + 1
        assert parent.scratchpad_dir == scratch and scratch.is_dir()
        # Construction, rebind and resume must not create a child-owned directory.
        import chartreux.core.agent_loop._loop as loop_module

        def forbidden_init(_session_id: str):
            pytest.fail("Child attempted to create a scratchpad")

        monkeypatch.setattr(loop_module, "init_scratchpad", forbidden_init)
        child = await factory.create_child(parent, WP3_AGENT_NAME)
        loops.append(child)
        await child.wait_until_ready()
        await child.persist_empty_session()
        directory = child.session_logger.session_dir
        metadata = child.session_logger.session_metadata
        assert directory is not None and metadata is not None
        child_id = child.session_id
        cached = child.tool_manager.get("read_file")
        args = ReadFileArgs(file_path=str(scratch / "notes.txt"))
        decision = cached.resolve_permission(args)
        assert decision is not None and decision.permission == ToolPermission.ALWAYS
        assert child.scratchpad_dir is None
        with pytest.raises(ValueError, match="cannot own"):
            child.rebind_to_session(
                child_id,
                directory,
                [],
                session_metadata=metadata,
                prepared_scratchpad=scratch,
            )
        child.rebind_to_session(child_id, directory, [], session_metadata=metadata)
        assert child.scratchpad_dir is None and scratch.is_dir()
        decision = cached.resolve_permission(args)
        assert decision is not None and decision.permission == ToolPermission.ALWAYS
        await child.aclose()
        assert scratch.is_dir()
        resumed = await factory.resume_child(
            parent, WP3_AGENT_NAME, child_id, directory
        )
        loops.append(resumed)
        await resumed.wait_until_ready()
        assert resumed.scratchpad_dir is None
        decision = resumed.tool_manager.get("read_file").resolve_permission(args)
        assert decision is not None and decision.permission == ToolPermission.ALWAYS
        await resumed.aclose()
        assert scratch.is_dir()
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()
    assert not scratch.exists()


async def test_parent_disabled_tool_remains_unavailable_despite_child_enablement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        await parent.config_orchestrator.set_field("/disabled_tools", ["read_file"])
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT_NAME)
        apply_launch_overrides(
            child.config_orchestrator, {"enabled_tools": ["read_file"]}
        )

        assert "read_file" not in child.tool_manager.available_tools
        with pytest.raises(Exception, match="Unknown or disabled tool"):
            child.tool_manager.get("read_file")
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_parent_argument_deny_wins_over_child_broad_allowlist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text('[tools.bash]\ndenylist = ["printf"]\n')
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT_NAME)
        apply_launch_overrides(
            child.config_orchestrator, {"tools": {"bash": {"permission": "always"}}}
        )

        context = child.tool_manager.get("bash").resolve_permission(
            BashArgs(command="printf child-attempt")
        )
        assert context is not None and context.permission == ToolPermission.NEVER
        assert context.reason and "denylist" in context.reason
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_identity_posture_is_never_loosenable_but_parent_never_is_not(
    tmp_path: Path,
) -> None:
    """Documented v0.5.7 policy reversal: launch-level tool overrides can
    never loosen the child identity posture.

    This test rewrites the pre-0.5.7 pin ``test_profile_never_is_loosenable_
    but_parent_never_is_not``, which proved launch-level overrides widen
    against identity-posture restrictions. Under the per-axis authority model
    (design §8) the identity-level restriction is the identity posture,
    enforced inside ToolManager BELOW the config layer; overrides may only
    narrow. The reversal is implemented and reviewed as one change with the
    enforcement points (ToolManager posture sets, the ``_parent_allows_tool``
    delegation chain-walk, the ``_validate_tools`` second ceiling, and the
    admission ceiling from the parent's delegation set).

    Coverage preserved from the reversed test, with the loosening case now
    denied: narrowing stays allowed; loosening the identity posture is
    denied (both at admission and below the config layer); an explicit
    parent (granted-axis) denial still cannot be loosened.
    """
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    loosened = None
    denied = None
    try:
        # Identity posture (reviewer child, personal non-mutating): a launch
        # override cannot enable a mutating tool — enforced below the config
        # layer, so neither an allowlist nor a per-tool "always" patch helps.
        loosened = await AgentRuntimeFactory().create_child(parent, "reviewer")
        apply_launch_overrides(
            loosened.config_orchestrator,
            {"enabled_tools": ["edit"], "tools": {"edit": {"permission": "always"}}},
        )
        assert "edit" not in loosened.tool_manager.available_tools
        with pytest.raises(Exception, match="Unknown or disabled tool"):
            loosened.tool_manager.get("edit")
        # The same loosening attempt is rejected at launch admission by the
        # second ceiling in _validate_tools (the parent's delegation set
        # admits edit; the child identity posture does not).
        with pytest.raises(InvalidLaunchToolError, match="identity posture"):
            resolve_launch(
                identity="reviewer",
                config=LaunchConfig(enabled_tools=["edit"]),
                parent_orchestrator=parent.config_orchestrator,
                tool_inventory={
                    name: object() for name in parent.tool_manager.registered_tools
                },
                authorized_tool_names=frozenset(parent.tool_manager.delegable_tools),
            )
        # Narrowing stays allowed: disabling an allowed tool still works.
        apply_launch_overrides(
            loosened.config_orchestrator,
            {"tools": {"read_file": {"permission": "never"}}},
        )
        assert (
            loosened.tool_manager.get_tool_config("read_file").permission
            == ToolPermission.NEVER
        )

        # Parent (granted-axis) denial still cannot be loosened.
        await parent.config_orchestrator.set_field(
            "/tools/read_file/permission", "never"
        )
        denied = await AgentRuntimeFactory().create_child(parent, "coder")
        apply_launch_overrides(
            denied.config_orchestrator,
            {"tools": {"read_file": {"permission": "always"}}},
        )
        assert (
            denied.tool_manager.get_tool_config("read_file").permission
            == ToolPermission.NEVER
        )
    finally:
        if denied is not None:
            await denied.aclose()
        if loosened is not None:
            await loosened.aclose()
        await parent.aclose()


async def test_root_researcher_cannot_obtain_a_writing_coder_through_the_task_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task-tool path: a root researcher with GRANTED write tools still
    cannot obtain a writing coder — the delegation axis (non-mutating)
    caps the child, and admission rejects explicit write-tool overrides.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(agent={"identity": "researcher"}),
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
    try:
        # The researcher root keeps the control-plane task tool personally
        # and the v0.5.9 carve-out admits the mutating tools on its personal
        # axis...
        assert "task" in parent.tool_manager.available_tools
        assert "edit" in parent.tool_manager.available_tools
        result = await _background_result(
            registry,
            TaskArgs(task="write code", identity="coder", background=True),
            context,
        )
        assert result.agent_id is not None
        assert len(registry._children) == 1
        child = next(iter(registry._children.values()))
        # The coder child of the researcher cannot write: the chain-walk
        # consults the parent's DELEGATION set, not its personal inventory.
        assert "edit" not in child.agent_loop.tool_manager.available_tools
        assert "bash" not in child.agent_loop.tool_manager.available_tools
        with pytest.raises(Exception, match="Unknown or disabled tool"):
            child.agent_loop.tool_manager.get("edit")
        assert "read_file" in child.agent_loop.tool_manager.available_tools
        # Admission: an explicit write-tool override is rejected against the
        # parent's delegation ceiling (the first ceiling in _validate_tools).
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            await _background_result(
                registry,
                TaskArgs(
                    task="write code",
                    identity="coder",
                    background=True,
                    config=LaunchConfig(enabled_tools=["edit"]),
                ),
                context,
            )
    finally:
        await registry.drain_children()
        await parent.aclose()


async def test_root_researcher_cannot_obtain_a_writing_coder_via_direct_launch(
    tmp_path: Path,
) -> None:
    """Direct-launch path: the factory ceiling is the parent's delegation
    set, and the constructed child still cannot write.
    """
    path = tmp_path / "settings.toml"
    path.write_text('[agent]\nidentity = "researcher"\n')
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    child = None
    try:
        # The v0.5.9 carve-out admits edit on the researcher's personal
        # axis; the delegation axis stays non-mutating.
        assert "edit" in parent.tool_manager.available_tools
        assert "edit" not in parent.tool_manager.delegable_tools
        child = await AgentRuntimeFactory().create_child(parent, "coder")
        assert "edit" not in child.tool_manager.available_tools
        with pytest.raises(Exception, match="Unknown or disabled tool"):
            child.tool_manager.get("edit")
        assert "read_file" in child.tool_manager.available_tools
        # The task-tool admission path uses the same delegation ceiling, and
        # the resolved child carries the bridged identity (worker -> coder).
        admission = ChildLaunchAdmission(MagicMock())
        candidate = admission.resolve_candidate(
            root, TaskArgs(task="write code", identity="coder")
        )
        assert candidate.orchestrator.config.agent.identity == "coder"
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            admission.resolve_candidate(
                root,
                TaskArgs(
                    task="write code",
                    identity="coder",
                    config=LaunchConfig(enabled_tools=["edit"]),
                ),
            )
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_root_orchestrator_grants_writes_personally_and_delegates_writes(
    tmp_path: Path,
) -> None:
    """The orchestrator's personal carve-out admits the mutating builtin set
    while its delegation set is the full granted ceiling: it persists its own
    scratchpad/deliverable writes and delegates repo-source writes to a coder
    child.
    """
    path = tmp_path / "settings.toml"
    path.write_text('[agent]\nidentity = "orchestrator"\n')
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        # Personal axis: the v0.5.9 carve-out admits the mutating tools...
        assert "edit" in parent.tool_manager.available_tools
        assert parent.tool_manager.get("edit") is not None
        # ...and the control-plane task tool stays personal, so the
        # orchestrator can delegate at all.
        assert "task" in parent.tool_manager.available_tools
        # Delegation axis: the full granted ceiling, writes included.
        assert "edit" in parent.tool_manager.delegable_tools
        child = await AgentRuntimeFactory().create_child(parent, "coder")
        assert "edit" in child.tool_manager.available_tools
        assert "bash" in child.tool_manager.available_tools
        # Delegated execution resolves permission through the parent's policy
        # (the parent's personal posture excludes the tool it lends).
        target = tmp_path / "notes.txt"
        target.write_text("original\n", encoding="utf-8")
        context = child.tool_manager.get("edit").resolve_permission(
            EditArgs(file_path=str(target), old_string="original", new_string="changed")
        )
        assert context is None or context.permission != ToolPermission.NEVER
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_depth_limit_remains_one_for_authority_widening(tmp_path: Path) -> None:
    """Depth limit remains one: a child cannot launch, so the authority
    guarantees are exercised at the root identity — a child cannot widen
    authority by delegating.
    """
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, "coder")
        task_tool = Task(config_getter=lambda: TaskToolConfig(), state=BaseToolState())
        nested_context = InvokeContext(
            tool_call_id="nested",
            is_subagent=True,
            agent_manager=cast(AgentManager, object()),
        )
        with pytest.raises(ToolError, match="depth limit|cannot be spawned"):
            await collect_result(
                task_tool.run(
                    TaskArgs(task="nested", identity="coder", background=True),
                    nested_context,
                )
            )
        with pytest.raises(UnsupportedChildForkError):
            await AgentRuntimeFactory().fork(child, None)
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_retask_launch_permission_replaces_prior_never(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT_NAME)
        apply_launch_overrides(
            child.config_orchestrator, {"tools": {"read_file": {"permission": "never"}}}
        )
        assert (
            child.tool_manager.get_tool_config("read_file").permission
            == ToolPermission.NEVER
        )

        apply_launch_overrides(
            child.config_orchestrator,
            {"tools": {"read_file": {"permission": "always"}}},
        )
        assert (
            child.tool_manager.get_tool_config("read_file").permission
            == ToolPermission.ALWAYS
        )
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_workspace_and_sensitive_guards_survive_launch_refresh_and_cached_access_while_parent_policy_tightens(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    related = tmp_path / "related"
    project.mkdir()
    related.mkdir()
    path = tmp_path / "settings.toml"
    path.write_text(
        '[tools.read_file]\nsensitive_patterns = ["*.private"]\n'
        "[authorized_roots_by_project]\n"
        f"{json.dumps(str(project))} = [{json.dumps(str(related))}]\n"
    )
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=project,
        backend=FakeBackend(),
    )
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT_NAME)
        apply_launch_overrides(
            child.config_orchestrator,
            {"tools": {"read_file": {"permission": "always"}}},
        )
        cached = child.tool_manager.get("read_file")
        sensitive = cached.resolve_permission(
            ReadFileArgs(file_path=str(related / "secret.private"))
        )
        assert sensitive is not None and sensitive.permission == ToolPermission.NEVER
        ordinary = cached.resolve_permission(
            ReadFileArgs(file_path=str(related / "ordinary.txt"))
        )
        assert ordinary is not None and ordinary.permission != ToolPermission.NEVER

        prepared_authority_revision = child.tool_manager.authority_revision
        path.write_text('[tools.read_file]\nsensitive_patterns = ["*.private"]\n')
        await parent.reload_with_initial_messages(reload_config=True)
        assert prepared_authority_revision != child.tool_manager.authority_revision
        assert child.tool_manager.authority_revision == parent._authority_revision
        tightened = cached.resolve_permission(
            ReadFileArgs(file_path=str(related / "ordinary.txt"))
        )
        assert tightened is not None and tightened.permission == ToolPermission.NEVER
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


async def test_resume_admits_stored_write_override_under_orchestrator_parent(
    tmp_path: Path,
) -> None:
    """3b lifecycle pin: the resume path's admission ceiling is the parent's
    DELEGATION set, not its personal inventory (design §8).

    A child launched under an orchestrator parent with a stored write-tool
    override (admitted at launch: the delegation ceiling includes writes)
    must resume without rejection. Post-grant the orchestrator also holds
    the write tools personally (the v0.5.9 carve-out), so the axes agree
    there; the delegation-ceiling discrimination re-pins against the
    researcher, whose personal grant does not widen its non-mutating
    delegation set. The child's own identity posture still applies below
    the config layer inside its ToolManager.
    """
    path = tmp_path / "settings.toml"
    path.write_text('[agent]\nidentity = "orchestrator"\n')
    researcher_path = tmp_path / "researcher-settings.toml"
    researcher_path.write_text('[agent]\nidentity = "researcher"\n')
    orchestrator = await make_orchestrator(path)
    assert not await orchestrator.set_field(
        "/session_logging",
        {"enabled": True, "save_dir": str(tmp_path / "sessions")},
        target_layer=OverridesLayer.NAME,
    )
    parent = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    try:
        # The v0.5.9 carve-out admits the write tools on the parent's
        # personal axis; its delegation set is the full granted ceiling.
        assert "edit" in parent.tool_manager.available_tools
        assert "edit" in parent.tool_manager.delegable_tools
        # Post-grant the axes agree for the orchestrator, so the 3b ceiling
        # discrimination re-pins against the researcher: the carve-out gives
        # it edit PERSONALLY while its delegation set stays non-mutating.
        # The miswired personal-inventory axis would now ADMIT the override
        # (fail open); the delegation ceiling rejects it.
        researcher = AgentLoop(
            config_orchestrator=await make_orchestrator(researcher_path),
            cwd=tmp_path,
            backend=FakeBackend(),
        )
        loops.append(researcher)
        assert "edit" in researcher.tool_manager.available_tools
        assert "edit" not in researcher.tool_manager.delegable_tools
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            resolve_launch(
                identity="coder",
                config=LaunchConfig(enabled_tools=["edit"]),
                parent_orchestrator=researcher.config_orchestrator,
                tool_inventory={
                    name: object() for name in researcher.tool_manager.registered_tools
                },
                authorized_tool_names=frozenset(
                    researcher.tool_manager.delegable_tools
                ),
            )
        # Launch with the write override: admitted through the delegation
        # ceiling, and persisted as the child's stored launch state.
        candidate = resolve_launch(
            identity="coder",
            config=LaunchConfig(enabled_tools=["edit"]),
            parent_orchestrator=parent.config_orchestrator,
            tool_inventory={
                name: object() for name in parent.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(parent.tool_manager.delegable_tools),
        )
        child = await factory.create_child(parent, candidate)
        await child.wait_until_ready()
        await child.persist_empty_session()
        directory = child.session_logger.session_dir
        assert directory is not None
        child_id = child.session_id
        await child.aclose()
        # Resume: the stored write override is re-validated against the
        # parent's DELEGATION ceiling and admitted; the resumed child keeps
        # the write — delegation rides the ceiling, not the parent's
        # personal inventory (which also holds the write post-grant).
        resumed = await factory.resume_child(parent, "coder", child_id, directory)
        loops.append(resumed)
        await resumed.wait_until_ready()
        assert "edit" in resumed.tool_manager.available_tools
        assert resumed.tool_manager.get("edit") is not None
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_retask_uses_the_parent_delegation_ceiling_not_personal_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3b lifecycle pin: the retask path's admission ceiling is the parent's
    delegation set, never its personal inventory.

    Post-grant the orchestrator holds the write tools personally (the
    v0.5.9 carve-out), so the pre-grant personal-vs-delegation gap for edit
    no longer discriminates there: the retask override stays admitted
    through the delegation ceiling. The distinction that still exists is
    the reviewer — bash is admitted personally while its delegation set
    stays non-mutating — so a retask bash override is rejected against the
    delegation ceiling even though the parent's personal inventory contains
    bash.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(agent={"identity": "orchestrator"}),
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
    try:
        # The v0.5.9 carve-out admits edit on the parent's personal axis;
        # the delegation set keeps the full granted ceiling.
        assert "edit" in parent.tool_manager.available_tools
        assert "edit" in parent.tool_manager.delegable_tools
        launch = await _background_result(
            registry,
            TaskArgs(
                task="write code",
                identity="coder",
                background=True,
                retain=True,
                config=LaunchConfig(enabled_tools=["edit"]),
            ),
            context,
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        first_run = record.current_run
        assert first_run is not None
        await first_run.completion_task
        child = record.runtime
        assert "edit" in child.agent_loop.tool_manager.available_tools
        # Retask the same agent with the same write override: admitted again
        # through the parent's delegation ceiling — the admission consults
        # the delegated set, never the personal inventory.
        retask = await _background_result(
            registry,
            TaskArgs(
                task="write more code",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(enabled_tools=["edit"]),
            ),
            InvokeContext(tool_call_id="task-2", session_id=parent.session_id),
        )
        assert retask.agent_id == record.agent_id
        assert retask.run_id is not None and retask.run_id != first_run.run_id
        second_run = record.current_run
        assert second_run is not None
        await second_run.completion_task
        assert "edit" in child.agent_loop.tool_manager.available_tools
    finally:
        await registry.drain_children()
        await parent.aclose()

    # The delegation-ceiling distinction that still exists post-grant: the
    # reviewer holds bash personally (its carve-out) while its delegation
    # set stays non-mutating. A cleanly launched coder child retasked with a
    # bash override is rejected — the ceiling is the delegation set, not the
    # personal inventory that contains bash.
    reviewer = build_test_agent_loop(
        config=build_test_vibe_config(agent={"identity": "reviewer"}),
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(reviewer)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="task-1", session_id=reviewer.session_id)
    try:
        assert "bash" in reviewer.tool_manager.available_tools
        assert "bash" not in reviewer.tool_manager.delegable_tools
        clean = await _background_result(
            registry,
            TaskArgs(task="read code", identity="coder", background=True, retain=True),
            context,
        )
        assert clean.agent_id is not None
        record = registry._agent_records[clean.agent_id]
        first_run = record.current_run
        assert first_run is not None
        await first_run.completion_task
        child = record.runtime
        assert "bash" not in child.agent_loop.tool_manager.available_tools
        # The retask's bash override exceeds the reviewer's delegation set.
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            await _background_result(
                registry,
                TaskArgs(
                    task="run checks",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(enabled_tools=["bash"]),
                ),
                InvokeContext(tool_call_id="task-2", session_id=reviewer.session_id),
            )
        assert "bash" not in child.agent_loop.tool_manager.available_tools
    finally:
        await registry.drain_children()
        await reviewer.aclose()


async def test_retask_rejects_write_override_beyond_the_parent_delegation_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3b lifecycle pin: a retask cannot widen authority past the parent's
    delegation ceiling — a researcher parent's retasked child may not gain a
    write tool the parent's delegation set excludes.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(agent={"identity": "researcher"}),
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
    try:
        assert "edit" not in parent.tool_manager.delegable_tools
        launch = await _background_result(
            registry,
            TaskArgs(task="read code", identity="coder", background=True, retain=True),
            context,
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        first_run = record.current_run
        assert first_run is not None
        await first_run.completion_task
        child = record.runtime
        assert "edit" not in child.agent_loop.tool_manager.available_tools
        # The retask's write override exceeds the parent's delegation set.
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            await _background_result(
                registry,
                TaskArgs(
                    task="write code",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(enabled_tools=["edit"]),
                ),
                InvokeContext(tool_call_id="task-2", session_id=parent.session_id),
            )
        assert "edit" not in child.agent_loop.tool_manager.available_tools
    finally:
        await registry.drain_children()
        await parent.aclose()


async def test_retask_rejects_override_loosening_the_child_identity_posture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3b lifecycle pin: the identity-posture second ceiling applies on the
    retask path — a retasked reviewer child cannot be loosened into a writer
    by a launch override.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(), backend=FakeBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="review", identity="reviewer", background=True, retain=True),
            context,
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        first_run = record.current_run
        assert first_run is not None
        await first_run.completion_task
        child = record.runtime
        assert "edit" not in child.agent_loop.tool_manager.available_tools
        # The parent's delegation set admits edit (coder default), but the
        # reviewer child's identity posture is the second ceiling.
        with pytest.raises(InvalidLaunchToolError, match="identity posture"):
            await _background_result(
                registry,
                TaskArgs(
                    task="write code",
                    agent_id=record.agent_id,
                    background=True,
                    config=LaunchConfig(enabled_tools=["edit"]),
                ),
                InvokeContext(tool_call_id="task-2", session_id=parent.session_id),
            )
        assert "edit" not in child.agent_loop.tool_manager.available_tools
    finally:
        await registry.drain_children()
        await parent.aclose()


async def test_retask_publish_fails_closed_when_parent_ceiling_narrowed_in_place(
    tmp_path: Path,
) -> None:
    """3b lifecycle pin: the retask publish re-derives the admission ceilings
    from the LIVE parent authority.

    An in-place parent config edit narrows the parent's delegation set
    WITHOUT bumping the loop's authority revision, so the reservation guards
    pass; the publish-time re-check must fail closed instead of installing a
    beyond-ceiling child.
    """
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    try:
        # Launch a child whose stored override enables a write tool; the
        # parent (coder default) admits it through its delegation ceiling.
        launch_candidate = resolve_launch(
            identity="coder",
            config=LaunchConfig(enabled_tools=["edit"]),
            parent_orchestrator=parent.config_orchestrator,
            tool_inventory={
                name: object() for name in parent.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(parent.tool_manager.delegable_tools),
        )
        child = await factory.create_child(parent, launch_candidate)
        loops.append(child)
        await child.wait_until_ready()
        child_runtime = registry._build_child_runtime(child)
        # Resolve the retask candidate while the parent's delegation set
        # still admits the write override.
        admission = ChildLaunchAdmission(MagicMock())
        candidate = admission.resolve_candidate(
            root,
            TaskArgs(
                task="write more",
                agent_id="agent-1",
                background=True,
                config=LaunchConfig(enabled_tools=["edit"]),
            ),
            runtime=child_runtime,
            identity="coder",
        )
        prepared = await child.prepare_launch_reconfiguration(
            candidate,
            expected_session_generation=child._session_generation,
            expected_parent_authority_revision=parent._authority_revision,
        )
        # Narrow the parent IN PLACE: no loop authority-revision bump, so the
        # reservation guards pass — only the live publish-time re-check can
        # catch the moved delegation ceiling.
        revision_before = parent._authority_revision
        await parent.config_orchestrator.set_field("/disabled_tools", ["edit"])
        assert parent._authority_revision == revision_before
        assert "edit" not in parent.tool_manager.delegable_tools
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            child.publish_launch_reconfiguration(prepared)
        # The failed publish leaves the child's runtime untouched.
        assert child.tool_manager is prepared.previous_tool_manager
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_retask_publish_fails_closed_when_parent_authority_unavailable(
    tmp_path: Path,
) -> None:
    """3b lifecycle pin: a parent chain gap during retask fails closed — the
    publish-time re-validation resolves the parent's live delegation ceiling
    and aborts the retask when the parent authority is unavailable, never
    opening a widened child.
    """
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    try:
        launch_candidate = resolve_launch(
            identity="coder",
            config=LaunchConfig(enabled_tools=["edit"]),
            parent_orchestrator=parent.config_orchestrator,
            tool_inventory={
                name: object() for name in parent.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(parent.tool_manager.delegable_tools),
        )
        child = await factory.create_child(parent, launch_candidate)
        loops.append(child)
        await child.wait_until_ready()
        child_runtime = registry._build_child_runtime(child)
        admission = ChildLaunchAdmission(MagicMock())
        candidate = admission.resolve_candidate(
            root,
            TaskArgs(
                task="write more",
                agent_id="agent-1",
                background=True,
                config=LaunchConfig(enabled_tools=["edit"]),
            ),
            runtime=child_runtime,
            identity="coder",
        )
        prepared = await child.prepare_launch_reconfiguration(
            candidate,
            expected_session_generation=child._session_generation,
            expected_parent_authority_revision=parent._authority_revision,
        )

        def unavailable() -> ToolManager:
            raise RuntimeError("parent authority unavailable")

        original_getter = child._parent_authority_getter
        child._parent_authority_getter = unavailable
        try:
            with pytest.raises(
                AgentLoopStateError, match="Parent authority unavailable"
            ):
                child.publish_launch_reconfiguration(prepared)
        finally:
            child._parent_authority_getter = original_getter
        # The failed publish leaves the child's runtime untouched.
        assert child.tool_manager is prepared.previous_tool_manager
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_reload_freezes_identity_posture_and_still_narrows_the_granted_axis(
    tmp_path: Path,
) -> None:
    """S2 re-pin (design §3, decision log item 3): identity is frozen per
    conversation. A mid-conversation reload that changes ``[agent] identity``
    is a PENDING selection for new conversations — it never re-postures an
    existing manager or child. The granted axis (enabled/disabled/permission)
    stays live: its narrowing still propagates through the reload, and the
    pre-reload authority/token caches cannot serve stale decisions on either
    axis.
    """
    path = tmp_path / "settings.toml"
    path.write_text('[agent]\nidentity = "coder"\n')
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    try:
        old_manager = parent.tool_manager
        assert parent._bound_identity == "coder"
        assert "edit" in old_manager.available_tools
        child = await factory.create_child(parent, "coder")
        loops.append(child)
        await child.wait_until_ready()
        assert child._bound_identity == "coder"
        # Warm the child's authority caches BEFORE the reload: a cached
        # instance, a cached config, and the model-facing spec list.
        warm_instance = child.tool_manager.get("edit")
        assert warm_instance is not None
        assert "edit" in {
            spec.name for spec in child.tool_manager.available_tool_specs()
        }
        # A reload that changes the identity selection is PENDING for this
        # conversation: the live config reports the new selection, but the
        # bound identity — and with it the personal and delegation posture
        # of every existing manager — is unchanged.
        path.write_text('[agent]\nidentity = "researcher"\n')
        await parent.reload_with_initial_messages(reload_config=True)
        assert parent.tool_manager is not old_manager
        assert parent.config.agent.identity == "researcher"  # pending selection
        assert parent._bound_identity == "coder"  # the conversation's binding
        assert "edit" in parent.tool_manager.available_tools
        assert "edit" in parent.tool_manager.delegable_tools
        # The pre-reload manager is retired: no warm cache survives it.
        with pytest.raises(Exception, match="retired"):
            old_manager.get("edit")
        # The existing child keeps its frozen coder posture: the write stays
        # available, in the specs, and the warm instance stays admitted.
        assert child._bound_identity == "coder"
        assert "edit" in child.tool_manager.available_tools
        assert "edit" in child.tool_manager.delegable_tools
        assert "edit" in {
            spec.name for spec in child.tool_manager.available_tool_specs()
        }
        assert child.tool_manager.get("edit") is warm_instance
        target = tmp_path / "notes.txt"
        target.write_text("original\n", encoding="utf-8")
        decision = warm_instance.resolve_permission(
            EditArgs(file_path=str(target), old_string="original", new_string="changed")
        )
        assert decision is None or decision.permission != ToolPermission.NEVER
        # The granted axis stays live: reloading with a disabled tool narrows
        # the parent AND the existing child in place, and the warm instance
        # can no longer resolve a permitted write.
        path.write_text(
            'disabled_tools = ["edit"]\n\n[agent]\nidentity = "researcher"\n'
        )
        await parent.reload_with_initial_messages(reload_config=True)
        assert parent._bound_identity == "coder"  # identity still frozen
        assert "edit" not in parent.tool_manager.available_tools
        assert "edit" not in parent.tool_manager.delegable_tools
        assert "edit" not in child.tool_manager.available_tools
        assert "edit" not in {
            spec.name for spec in child.tool_manager.available_tool_specs()
        }
        with pytest.raises(Exception, match="Unknown or disabled tool"):
            child.tool_manager.get("edit")
        decision = warm_instance.resolve_permission(
            EditArgs(file_path=str(target), old_string="original", new_string="changed")
        )
        assert decision is not None and decision.permission == ToolPermission.NEVER
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_reclassification_propagates_through_parent_reload(
    tmp_path: Path,
) -> None:
    """3b lifecycle pin: a reviewed reclassification invalidates the loop's
    cached authority decisions — the classification revision rides the
    authority token through the parent chain, so a warm child cache cannot
    serve a pre-reclassification decision after a config reload.
    """
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    original = TOOL_CLASSIFICATIONS["read_file"]
    try:
        child = await factory.create_child(parent, "reviewer")
        loops.append(child)
        await child.wait_until_ready()
        # The reviewer child (non-mutating posture) starts with read_file,
        # and its authority caches are warm.
        assert "read_file" in child.tool_manager.available_tools
        assert "read_file" in {
            spec.name for spec in child.tool_manager.available_tool_specs()
        }
        # Reviewed reclassification: read_file becomes mutating.
        reclassify_tool("read_file", ToolClass.MUTATING)
        # The warm cache cannot serve the stale decision: the token's
        # classification revision moved, so the child re-derives.
        assert "read_file" not in child.tool_manager.available_tools
        assert "read_file" not in {
            spec.name for spec in child.tool_manager.available_tool_specs()
        }
        with pytest.raises(Exception, match="Unknown or disabled tool"):
            child.tool_manager.get("read_file")
        # The classification revision propagates through the loop's cached
        # authority decisions across a config reload: the reloaded parent
        # manager carries the new revision in its authority token, so the
        # child re-derives instead of reusing its pre-reload caches.
        await parent.reload_with_initial_messages(reload_config=True)
        assert "read_file" not in child.tool_manager.available_tools
        assert "read_file" not in {
            spec.name for spec in child.tool_manager.available_tool_specs()
        }
        # The parent's own unrestricted (coder) posture is unaffected by the
        # reclassification of a read tool.
        assert "read_file" in parent.tool_manager.available_tools
    finally:
        reclassify_tool("read_file", original)
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()


async def test_reclassification_propagates_through_retask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """3b lifecycle pin: the classification revision also propagates through
    the retask path — a retasked child's freshly prepared consumers reflect
    the reviewed reclassification, not a pre-reclassification cache.
    """
    monkeypatch.setattr(
        "chartreux.core.agent_loop._loop.create_backend",
        lambda **_: FakeBackend([mock_llm_chunk(content="done")]),
    )
    parent = build_test_agent_loop(
        config=build_test_vibe_config(), backend=FakeBackend()
    )
    registry = _retention_registry()
    root = registry._build_child_runtime(parent)
    root.turns._projector = MagicMock()
    root.turns.link_subagent = AsyncMock()
    registry.bind_root(root)
    context = InvokeContext(tool_call_id="task-1", session_id=parent.session_id)
    original = TOOL_CLASSIFICATIONS["read_file"]
    try:
        launch = await _background_result(
            registry,
            TaskArgs(task="review", identity="reviewer", background=True, retain=True),
            context,
        )
        assert launch.agent_id is not None
        record = registry._agent_records[launch.agent_id]
        first_run = record.current_run
        assert first_run is not None
        await first_run.completion_task
        child = record.runtime
        assert "read_file" in child.agent_loop.tool_manager.available_tools
        # Reviewed reclassification: read_file becomes mutating, so the
        # reviewer child's live authority drops it.
        reclassify_tool("read_file", ToolClass.MUTATING)
        assert "read_file" not in child.agent_loop.tool_manager.available_tools
        # The retask's narrowing override rebuilds the child's consumers;
        # the published child reflects BOTH the reclassification (read_file
        # gone from the reviewer posture) and the narrowing (grep gone).
        retask = await _background_result(
            registry,
            TaskArgs(
                task="review again",
                agent_id=record.agent_id,
                background=True,
                config=LaunchConfig(disabled_tools=["grep"]),
            ),
            InvokeContext(tool_call_id="task-2", session_id=parent.session_id),
        )
        assert retask.run_id is not None
        second_run = record.current_run
        assert second_run is not None
        await second_run.completion_task
        names = {
            spec.name for spec in child.agent_loop.tool_manager.available_tool_specs()
        }
        assert "read_file" not in names
        assert "grep" not in names
    finally:
        reclassify_tool("read_file", original)
        await registry.drain_children()
        await parent.aclose()


async def test_custom_identity_lifecycle_cannot_widen_delegation_beyond_granted(
    tmp_path: Path,
) -> None:
    """3b lifecycle pin (design decision 8): a custom identity carries an
    unrestricted PERSONAL posture, but its DELEGATION ceiling is still the
    granted set — an in-place grant narrowing caps the custom child live,
    and the resume path re-validates the stored override against the parent's
    live delegation ceiling so a custom child cannot resurrect a
    beyond-grant write on resume.
    """
    path = tmp_path / "settings.toml"
    path.write_text("")
    orchestrator = await make_orchestrator(path)
    assert not await orchestrator.set_field(
        "/session_logging",
        {"enabled": True, "save_dir": str(tmp_path / "sessions")},
        target_layer=OverridesLayer.NAME,
    )
    parent = AgentLoop(
        config_orchestrator=orchestrator, cwd=tmp_path, backend=FakeBackend()
    )
    factory = AgentRuntimeFactory()
    loops: list[AgentLoop] = []
    try:
        # The custom identity (a user prompt file) carries an unrestricted
        # personal posture, delegation defaulting to the full granted set.
        candidate = resolve_launch(
            identity=WP3_AGENT_NAME,
            config=LaunchConfig(enabled_tools=["edit"]),
            parent_orchestrator=parent.config_orchestrator,
            tool_inventory={
                name: object() for name in parent.tool_manager.registered_tools
            },
            authorized_tool_names=frozenset(parent.tool_manager.delegable_tools),
        )
        child = await factory.create_child(parent, candidate)
        await child.wait_until_ready()
        # Unrestricted personal posture within the granted ceiling...
        assert "edit" in child.tool_manager.available_tools
        # ...and the delegation set defaults to the full granted ceiling.
        assert "edit" in child.tool_manager.delegable_tools
        await child.persist_empty_session()
        directory = child.session_logger.session_dir
        assert directory is not None
        child_id = child.session_id
        await child.aclose()
        # The granted ceiling narrows IN PLACE (no loop revision bump): the
        # custom identity's unrestricted posture cannot widen past it — the
        # running child's live chain re-derives through the parent's new
        # delegation set.
        await parent.config_orchestrator.set_field("/disabled_tools", ["edit"])
        assert "edit" not in parent.tool_manager.delegable_tools
        assert "edit" not in child.tool_manager.available_tools
        assert "edit" not in child.tool_manager.delegable_tools
        # The resume path re-validates the stored override against the
        # parent's LIVE delegation ceiling: the custom child cannot widen
        # delegation beyond the (now narrower) granted set on resume.
        with pytest.raises(InvalidLaunchToolError, match="exceeds parent authority"):
            await factory.resume_child(parent, WP3_AGENT_NAME, child_id, directory)
    finally:
        for loop in reversed(loops):
            await loop.aclose()
        await parent.aclose()
