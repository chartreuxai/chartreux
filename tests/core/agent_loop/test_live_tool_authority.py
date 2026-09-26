"""Manager authority lifetimes; direct run is deliberately not an authorization API."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from chartreux.core.agent_loop._loop import ToolExecutionResponse
from chartreux.core.tools.base import InvokeContext, ToolPermissionError
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.builtins.write_file import WriteFileArgs
from chartreux.core.tools.io_port import ToolIOPort
from chartreux.core.tools.manager import NoSuchToolError, ToolManager
from chartreux.core.tools.models import ToolPermission
from tests.core.agent_loop.test_policy_replacement import make_loop, replace

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("evict_cache", [False, True])
async def test_retained_resolution_and_prepared_invocation_are_retired(
    tmp_path: Path, evict_cache: bool
) -> None:
    loop = await make_loop(tmp_path)
    old_manager = loop.tool_manager
    tool = old_manager.get("write_file")
    args = WriteFileArgs(file_path=str(tmp_path / "output.txt"), content="fixture")
    resolver = tool.resolve_permission
    tool_io = MagicMock(spec=ToolIOPort)
    tool_io.supports_write = True
    tool_io.write_text = AsyncMock()
    invocation = tool.invoke(
        ctx=InvokeContext(tool_call_id="retained", tool_io=tool_io), **args.model_dump()
    )
    if evict_cache:
        old_manager.reset_all()
    try:
        await replace(loop, [str(tmp_path / "blocked.txt")])
        with pytest.raises(ToolPermissionError, match="retired"):
            await anext(invocation)
        result = resolver(args)
        assert result is not None and result.permission == ToolPermission.NEVER
        decision = await loop._should_execute_tool(tool, args)
        assert decision.verdict == ToolExecutionResponse.SKIP
        tool_io.write_text.assert_not_awaited()
        assert not (tmp_path / "output.txt").exists()
        assert not old_manager._instances
        with pytest.raises(NoSuchToolError, match="retired"):
            old_manager.get("write_file")
        with pytest.raises(ToolPermissionError, match="retired"):
            _ = tool.config
        current = loop.tool_manager.get("read_file")
        context = current.resolve_permission(
            ReadFileArgs(file_path=str(tmp_path / "blocked.txt"))
        )
        assert context is not None and context.permission == ToolPermission.NEVER
    finally:
        await invocation.aclose()
        await loop.aclose()


async def test_inherited_plan_refresh_retires_retained_instance(tmp_path: Path) -> None:
    loop = await make_loop(tmp_path)
    old = loop.tool_manager.get("write_file")
    target = tmp_path / "ordinary.txt"
    args = WriteFileArgs(file_path=str(target), content="fixture")
    try:
        context = old.resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.ALWAYS
        # Simulate the private local adoption step, not a public tree mutation API.
        loop._inherited_plan_write_scopes = ((tmp_path / "plan.md", None),)
        await loop.reload_with_initial_messages()
        context = old.resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.NEVER
        current = loop.tool_manager.get("write_file")
        context = current.resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.NEVER
        assert context.reason and "Parent Plan" in context.reason
        context = current.resolve_permission(
            WriteFileArgs(file_path=str(tmp_path / "plan.md"), content="fixture")
        )
        assert context is not None and context.permission == ToolPermission.ALWAYS
    finally:
        await loop.aclose()


@pytest.mark.parametrize("failure", ["prepare", "stale"])
async def test_failed_preparation_keeps_retained_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    loop = await make_loop(tmp_path)
    tool = loop.tool_manager.get("write_file")
    args = WriteFileArgs(file_path=str(tmp_path / "output.txt"), content="fixture")
    config = tool.config.model_dump()
    try:
        with monkeypatch.context() as patch:
            if failure == "prepare":

                def reject(self: ToolManager, name: str) -> None:
                    raise RuntimeError("synthetic prepare failure")

                patch.setattr(ToolManager, "get_tool_config", reject)
            with pytest.raises((RuntimeError, ValueError)):
                await loop._prepare_policy_replacement(
                    source="user-toml",
                    tools={},
                    expected_token=object()
                    if failure == "stale"
                    else loop.config_orchestrator.accepted_token,
                )
        assert tool.config.model_dump() == config
        context = tool.resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.ALWAYS
        assert loop.tool_manager.get("write_file") is tool
        # Invocation validation/execution remains intact after a failed prepare.
        result = await anext(tool.invoke(**args.model_dump()))
        assert result is not None
        assert (tmp_path / "output.txt").read_text() == "fixture"
    finally:
        await loop.aclose()


async def test_removed_root_does_not_survive_in_retained_workspace(
    tmp_path: Path,
) -> None:
    loop = await make_loop(tmp_path)
    extra = tmp_path.parent / "extra-root"
    try:
        path = tmp_path / "settings.toml"
        path.write_text(
            "[authorized_roots_by_project]\n"
            f"{json.dumps(str(tmp_path))} = [{json.dumps(str(extra))}]\n"
        )
        await loop.reload_with_initial_messages(reload_config=True)
        old = loop.tool_manager.get("read_file")
        args = ReadFileArgs(file_path=str(extra / "file.txt"))
        context = old.resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.ALWAYS
        path.write_text("")
        await loop.reload_with_initial_messages(reload_config=True)
        context = old.resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.NEVER
        context = loop.tool_manager.get("read_file").resolve_permission(args)
        assert context is not None and context.permission == ToolPermission.NEVER
    finally:
        await loop.aclose()


async def test_disabled_tool_cannot_use_retained_resolver(tmp_path: Path) -> None:
    loop = await make_loop(tmp_path)
    tool = loop.tool_manager.get("read_file")
    try:
        await loop.config_orchestrator.set_field("/disabled_tools", ["read_file"])
        context = tool.resolve_permission(
            ReadFileArgs(file_path=str(tmp_path / "file.txt"))
        )
        assert context is not None and context.permission == ToolPermission.NEVER
    finally:
        await loop.aclose()


async def test_live_source_deny_overrides_configuration_and_survives_replacement(
    tmp_path: Path,
) -> None:
    loop = await make_loop(tmp_path)
    tool = loop.tool_manager.get("read_file")
    target = str(tmp_path / "blocked.txt")
    try:
        await loop.config_orchestrator.set_field(
            "/tools/read_file/permission", "always"
        )
        await loop.config_orchestrator.set_field(
            "/tools/read_file/denylist", ["other", target]
        )
        context = tool.resolve_permission(ReadFileArgs(file_path=target))
        assert context is not None and context.permission == ToolPermission.NEVER
        assert (
            await loop._should_execute_tool(tool, ReadFileArgs(file_path=target))
        ).verdict == ToolExecutionResponse.SKIP
        await replace(loop, [])
        context = tool.resolve_permission(
            ReadFileArgs(file_path=str(tmp_path / "other.txt"))
        )
        assert context is not None and context.permission == ToolPermission.NEVER
    finally:
        await loop.aclose()
