"""Accepted user roots reach real tools, without granting child config authority."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.events import ToolResultEvent
from chartreux.core.tools.base import ToolPermission
from chartreux.core.tools.manager import NoSuchToolError
from chartreux.core.tools.utils import resolve_file_tool_permission
from chartreux.core.workspace import Workspace
from tests.core.agent_loop.test_accepted_source_snapshot import make_orchestrator
from tests.core.agent_loop.test_chartreux_policy_denials import _call, _collect
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend

pytestmark = pytest.mark.asyncio


def settings(path: Path, project: Path, roots: list[Path]) -> None:
    path.write_text(
        '[tools.read_file]\npermission = "always"\ndenylist = ["*blocked.txt"]\n'
        '[tools.write_file]\npermission = "always"\n'
        "[authorized_roots_by_project]\n"
        f"{json.dumps(str(project))} = {json.dumps(list(map(str, roots)))}\n"
    )


def permission(loop: AgentLoop, target: Path) -> ToolPermission | None:
    tool = loop.tool_manager.get("read_file")
    result = tool.resolve_permission(
        tool.validate_arguments({"file_path": str(target)})
    )
    return result.permission if result is not None else None


async def test_user_file_to_real_read_write_and_denial(tmp_path: Path) -> None:
    project, related, outside = (
        tmp_path / name for name in ("project", "related", "outside")
    )
    for directory in (project, related, outside):
        directory.mkdir()
    target = related / "input.txt"
    target.write_text("accepted root content")
    path = tmp_path / "settings.toml"
    settings(path, project, [related])
    backend = FakeBackend([
        [
            mock_llm_chunk(
                tool_calls=[_call("read_file", {"file_path": str(target)}, "read")]
            )
        ],
        [
            mock_llm_chunk(
                tool_calls=[
                    _call(
                        "write_file",
                        {
                            "file_path": str(related / "output.txt"),
                            "content": "written",
                        },
                        "write",
                    )
                ]
            )
        ],
        [mock_llm_chunk(content="done")],
    ])
    loop = AgentLoop(
        config_orchestrator=await make_orchestrator(path), cwd=project, backend=backend
    )
    try:
        # Mutating a public effective model cannot add or subtract authority.
        loop.config.authorized_roots_by_project.clear()
        loop.config.authorized_roots_by_project[str(project)] = [str(outside)]
        assert permission(loop, target) != ToolPermission.NEVER
        assert permission(loop, outside / "file.txt") == ToolPermission.NEVER
        assert permission(loop, related / "blocked.txt") == ToolPermission.NEVER
        events = await _collect(loop)
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(results) == 2
        assert all(event.result is not None and not event.skipped for event in results)
        assert results[0].result is not None
        assert "accepted root content" in results[0].result.model_dump_json()
        assert (related / "output.txt").read_text() == "written"
    finally:
        await loop.aclose()


async def test_direct_child_requires_actual_narrowed_parent_ceiling(
    tmp_path: Path,
) -> None:
    project, related = tmp_path / "project", tmp_path / "related"
    project.mkdir()
    related.mkdir()
    path = tmp_path / "settings.toml"
    settings(path, project, [related])
    parent = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=project,
        backend=FakeBackend(),
        inherited_workspace=Workspace.for_session(project),
    )
    child = None
    try:
        assert permission(parent, related / "file.txt") == ToolPermission.NEVER
        with pytest.raises(ValueError, match="requires inherited_workspace"):
            AgentLoop(
                config_orchestrator=parent.config_orchestrator.copy(),
                cwd=project,
                backend=FakeBackend(),
                is_subagent=True,
            )
        child = AgentLoop(
            config_orchestrator=parent.config_orchestrator.copy(),
            cwd=project,
            backend=FakeBackend(),
            is_subagent=True,
            inherited_workspace=parent.tool_manager.workspace,
        )
        assert permission(child, related / "file.txt") == ToolPermission.NEVER
    finally:
        if child is not None:
            await child.aclose()
        await parent.aclose()


@pytest.mark.parametrize("local_scope", [False, True])
async def test_inherited_plan_allowances_are_not_unioned(
    tmp_path: Path, local_scope: bool
) -> None:
    first, second = tmp_path / "first.md", tmp_path / "second.md"
    for target in (first, second):
        result = resolve_file_tool_permission(
            str(target),
            tool_name="write_file",
            allowlist=["*"],
            denylist=[],
            config_permission=ToolPermission.ALWAYS,
            sensitive_patterns=[],
            workspace=Workspace.for_session(tmp_path / "project"),
            inherited_plan_write_scopes=(
                ((first, None),) if local_scope else ((first, None), (second, None))
            ),
            plan_file_write_scope=second if local_scope else None,
        )
        assert result is not None and result.permission == ToolPermission.NEVER


async def test_cached_tool_observes_accepted_root_removal(tmp_path: Path) -> None:
    project, related = tmp_path / "project", tmp_path / "related"
    project.mkdir()
    related.mkdir()
    path = tmp_path / "settings.toml"
    settings(path, project, [related])
    loop = AgentLoop(
        config_orchestrator=await make_orchestrator(path),
        cwd=project,
        backend=FakeBackend(),
    )
    try:
        tool = loop.tool_manager.get("read_file")
        args = tool.validate_arguments({"file_path": str(related / "file.txt")})
        assert permission(loop, related / "file.txt") != ToolPermission.NEVER
        settings(path, project, [])
        # An accepted source update cannot leave a cached workspace grant behind,
        # even before a runtime reload replaces the manager.
        await loop.config_orchestrator.reload()
        result = tool.resolve_permission(args)
        assert result is not None and result.permission == ToolPermission.NEVER
        old_manager = loop.tool_manager
        await loop.reload_with_initial_messages(reload_config=True)
        with pytest.raises(NoSuchToolError, match="retired"):
            old_manager.get("read_file")
        assert permission(loop, related / "file.txt") == ToolPermission.NEVER
    finally:
        await loop.aclose()
