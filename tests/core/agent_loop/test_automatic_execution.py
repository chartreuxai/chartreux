"""Automatic execution retains guards, not routine approval callbacks."""

from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock

import pytest

from chartreux.app_server.events import AppServerEvent
from chartreux.app_server.models import CallbackDetail, CallbackOutput
from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import MCPHttp
import chartreux.core.events as event_module
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools import permissions
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.manager import ToolManager
from chartreux.core.tools.remote import MCPToolResult
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


def test_public_contracts_have_no_routine_approval_variant() -> None:
    for name in ("ApprovalRequestEvent", "ApprovalResponse"):
        assert not hasattr(event_module, name)
    event_types = get_args(AppServerEvent.__value__)
    assert event_types
    assert all("approval" not in event.__name__.lower() for event in event_types)
    for callback_type, discriminator in (
        (CallbackDetail, "kind"),
        (CallbackOutput, "type"),
    ):
        variants = get_args(callback_type) or (callback_type,)
        assert {
            variant.model_fields[discriminator].default for variant in variants
        } == {"user_input"}


@pytest.mark.asyncio
async def test_actual_file_shell_and_fake_mcp_continue_after_denial_without_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = AsyncMock(
        return_value=MCPToolResult(
            server="fixture", tool="fake_tool", text="MCP fixture"
        )
    )
    monkeypatch.setattr("chartreux.core.tools.mcp.tools.call_tool_http", remote)
    calls = [
        ("read_file", {"file_path": str(tmp_path.parent / "outside.txt")}),
        ("write_file", {"file_path": "fixture.txt", "content": "before"}),
        (
            "edit",
            {"file_path": "fixture.txt", "old_string": "before", "new_string": "after"},
        ),
        ("read_file", {"file_path": "fixture.txt"}),
        ("bash", {"command": "printf shell-fixture"}),
        ("fixture_fake_tool", {}),
    ]
    backend = FakeBackend(
        [
            [
                mock_llm_chunk(
                    tool_calls=[
                        ToolCall(
                            id=str(i),
                            index=0,
                            function=FunctionCall(
                                name=name, arguments=json.dumps(args)
                            ),
                        )
                    ]
                )
            ]
            for i, (name, args) in enumerate(calls)
        ]
        + [[mock_llm_chunk(content="Finished")]]
    )
    loop = build_test_agent_loop(
        config=build_test_vibe_config(
            tools={name: {"permission": "ask"} for name, _ in calls},
            mcp_servers=[
                MCPHttp(
                    name="fixture",
                    transport="streamable-http",
                    url="https://fixture.invalid/mcp",
                )
            ],
        ),
        cwd=tmp_path,
        backend=backend,
    )
    assert not hasattr(loop, "_permission_store")
    assert not hasattr(loop.runtime_policy, "permission_store")
    assert not hasattr(loop.child_runtime_policy, "permission_store")
    remembered_store = AsyncMock(side_effect=AssertionError("remembered grant read"))
    monkeypatch.setattr(loop, "_permission_store", remembered_store, raising=False)
    monkeypatch.setattr(
        loop.tool_manager, "_permission_getter", remembered_store, raising=False
    )
    assert not hasattr(loop._request_broker, "request_approval")
    try:
        async with asyncio.timeout(15):
            events = [event async for event in loop.act("Exercise fixture tools")]
        assert not any("approval" in type(event).__name__.lower() for event in events)
        results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(results) == len(calls)
        assert results[0].skip_reason
        assert all(
            not result.error and not result.skip_reason for result in results[1:]
        )
        assert (tmp_path / "fixture.txt").read_text() == "after"
        assert "shell-fixture" in str(results[4].result)
        remote.assert_awaited_once()
        assert not remembered_store.mock_calls
    finally:
        await loop.aclose()


@pytest.mark.asyncio
async def test_retired_grant_apis_cannot_write_policy() -> None:
    assert "permission_store" not in inspect.signature(AgentLoop).parameters
    assert "permission_store" not in inspect.signature(InvokeContext).parameters
    assert "permission_getter" not in inspect.signature(ToolManager).parameters
    assert not hasattr(permissions, "PermissionStore")
    loop = build_test_agent_loop(config=build_test_vibe_config())
    before = loop.config.model_dump()
    try:
        for name in ("approve_always", "approve_invocation", "set_tool_permission"):
            assert not hasattr(loop, name)
        assert loop.config.model_dump() == before
    finally:
        await loop.aclose()
