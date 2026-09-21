"""Actual factory inheritance; sensitive matching still has legacy ASK semantics."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.agents import AgentSafety, AgentType
from chartreux.app_server._dispatch import RequestFailure
from chartreux.app_server._handler import CoreRequestHandler
from chartreux.app_server._runtime import AgentRuntimeFactory
from chartreux.app_server.protocol import ProtocolErrorCode, SessionForkParams
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agents.models import AgentProfile
from chartreux.core.agents.registry import apply_launch_overrides
from chartreux.core.config.layers.overrides import OverridesLayer
from chartreux.core.subagents import UnsupportedChildForkError
from chartreux.core.tools.builtins.bash import BashArgs
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.models import ToolPermission
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
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
        child = await factory.create_child(parent, "worker")
        await child.wait_until_ready()
        await child.persist_empty_session()
        directory = child.session_logger.session_dir
        assert directory is not None
        child_id = child.session_id
        await child.aclose()
        path.write_text('[tools.read_file]\npermission = "never"\n')
        await orchestrator.reload()
        resumed = await factory.resume_child(parent, "worker", child_id, directory)
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
        child = await AgentRuntimeFactory().create_child(parent, "worker")
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
                await AgentRuntimeFactory().create_child(parent, "worker")
        assert orchestrator.restrictions is accepted
    finally:
        await parent.aclose()


WP3_AGENT = AgentProfile(
    name="wp3-agent",
    display_name="WP3 Agent",
    description="Unrestricted acceptance-test subagent",
    safety=AgentSafety.NEUTRAL,
    agent_type=AgentType.SUBAGENT,
)


def register_wp3_agent(
    parent: AgentLoop, *, overrides: dict[str, object] | None = None
) -> None:
    parent.agent_manager._discovered[WP3_AGENT.name] = AgentProfile(**{
        **WP3_AGENT.__dict__,
        "overrides": overrides or {},
    })


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
    register_wp3_agent(parent)
    child = None
    try:
        await parent.config_orchestrator.set_field("/disabled_tools", ["read_file"])
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT.name)
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
    register_wp3_agent(parent)
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT.name)
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


async def test_profile_never_is_loosenable_but_parent_never_is_not(
    tmp_path: Path,
) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    register_wp3_agent(
        parent, overrides={"tools": {"read_file": {"permission": "never"}}}
    )
    loosened = None
    denied = None
    try:
        loosened = await AgentRuntimeFactory().create_child(parent, WP3_AGENT.name)
        apply_launch_overrides(
            loosened.config_orchestrator,
            {"tools": {"read_file": {"permission": "always"}}},
        )
        assert (
            loosened.tool_manager.get_tool_config("read_file").permission
            == ToolPermission.ALWAYS
        )

        await parent.config_orchestrator.set_field(
            "/tools/read_file/permission", "never"
        )
        denied = await AgentRuntimeFactory().create_child(parent, WP3_AGENT.name)
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


async def test_retask_launch_permission_replaces_prior_never(tmp_path: Path) -> None:
    path = tmp_path / "settings.toml"
    path.write_text("")
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=tmp_path,
        backend=FakeBackend(),
    )
    register_wp3_agent(parent)
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT.name)
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
    register_wp3_agent(parent)
    child = None
    try:
        child = await AgentRuntimeFactory().create_child(parent, WP3_AGENT.name)
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
