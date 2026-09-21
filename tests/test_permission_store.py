from __future__ import annotations

import inspect

import pytest

from chartreux.core.agent_loop import AgentLoop
import chartreux.core.events as event_module
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools import permissions
from chartreux.core.tools.base import InvokeContext
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class TestRetiredPermissionStore:
    def test_grant_state_and_injection_points_are_removed(self):
        assert not hasattr(permissions, "PermissionStore")
        assert "permission_store" not in inspect.signature(AgentLoop).parameters
        assert "permission_store" not in inspect.signature(InvokeContext).parameters
        assert "permission_getter" not in inspect.signature(ToolManager).parameters

    @pytest.mark.asyncio
    async def test_automatic_execution_has_no_grant_state_or_approval_callback(self):
        tool_call = ToolCall(
            id="todo",
            index=0,
            function=FunctionCall(name="todo", arguments='{"action":"read"}'),
        )
        loop = build_test_agent_loop(
            config=build_test_vibe_config(tools={"todo": {"permission": "ask"}}),
            backend=FakeBackend([
                [mock_llm_chunk(tool_calls=[tool_call])],
                [mock_llm_chunk(content="Done")],
            ]),
        )
        try:
            assert not hasattr(loop, "_permission_store")
            events = [event async for event in loop.act("run the fixture")]
            results = [event for event in events if isinstance(event, ToolResultEvent)]
            assert len(results) == 1
            assert not results[0].skipped and not results[0].error
            assert not hasattr(event_module, "ApprovalRequestEvent")
            assert not any(
                "approval" in type(event).__name__.lower() for event in events
            )
        finally:
            await loop.aclose()

    @pytest.mark.asyncio
    async def test_explicit_denial_skips_without_approval_callback(self, tmp_path):
        target = tmp_path / "fixture.txt"
        target.write_text("unchanged")
        tool_call = ToolCall(
            id="read",
            index=0,
            function=FunctionCall(
                name="read_file", arguments=f'{{"file_path":"{target}"}}'
            ),
        )
        loop = build_test_agent_loop(
            config=build_test_vibe_config(tools={"read_file": {"permission": "never"}}),
            cwd=tmp_path,
            backend=FakeBackend([
                [mock_llm_chunk(tool_calls=[tool_call])],
                [mock_llm_chunk(content="Continued")],
            ]),
        )
        try:
            events = [event async for event in loop.act("read the fixture")]
            result = next(
                event for event in events if isinstance(event, ToolResultEvent)
            )
            assert result.skipped and result.skip_reason
            assert result.result is None
            assert not hasattr(event_module, "ApprovalRequestEvent")
            assert not any(
                "approval" in type(event).__name__.lower() for event in events
            )
            assert target.read_text() == "unchanged"
        finally:
            await loop.aclose()
