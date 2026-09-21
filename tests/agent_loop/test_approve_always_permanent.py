from __future__ import annotations

import inspect

import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.agent_loop._loop import ToolExecutionResponse
import chartreux.core.events as event_module
from chartreux.core.events import ToolResultEvent
from chartreux.core.llm_models import FunctionCall, ToolCall
from chartreux.core.tools import permissions
from chartreux.core.tools.base import InvokeContext, ToolPermission
from chartreux.core.tools.builtins.read_file import ReadFileArgs
from chartreux.core.tools.manager import ToolManager
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class TestRetiredApprovalSurface:
    @pytest.mark.asyncio
    async def test_approval_mutators_and_grant_state_are_absent(self):
        assert not hasattr(permissions, "PermissionStore")
        assert "permission_store" not in inspect.signature(AgentLoop).parameters
        assert "permission_store" not in inspect.signature(InvokeContext).parameters
        assert "permission_getter" not in inspect.signature(ToolManager).parameters
        agent = build_test_agent_loop()
        before = agent.config.model_dump()
        try:
            for name in ("approve_always", "approve_invocation", "set_tool_permission"):
                assert not hasattr(agent, name)
            assert not hasattr(agent, "_permission_store")
            assert agent.config.model_dump() == before
        finally:
            await agent.aclose()

    @pytest.mark.asyncio
    async def test_ask_policy_executes_without_approval_callback(self):
        agent = build_test_agent_loop(
            config=build_test_vibe_config(tools={"todo": {"permission": "ask"}}),
            backend=FakeBackend([
                [
                    mock_llm_chunk(
                        tool_calls=[
                            ToolCall(
                                id="todo",
                                index=0,
                                function=FunctionCall(
                                    name="todo", arguments='{"action":"read"}'
                                ),
                            )
                        ]
                    )
                ],
                [mock_llm_chunk(content="Done")],
            ]),
        )
        try:
            events = [event async for event in agent.act("read the todo")]
            assert not hasattr(event_module, "ApprovalRequestEvent")
            assert not any(
                "approval" in type(event).__name__.lower() for event in events
            )
            result = next(
                event for event in events if isinstance(event, ToolResultEvent)
            )
            assert not result.skipped and not result.error
        finally:
            await agent.aclose()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [{"permission": "never"}, {"permission": "ask", "denylist": ["*fixture.txt"]}],
    )
    async def test_explicit_denial_skips_without_approval_callback(
        self, tmp_path, config
    ):
        target = tmp_path / "fixture.txt"
        target.write_text("unchanged")
        agent = build_test_agent_loop(
            config=build_test_vibe_config(tools={"read_file": config}),
            cwd=tmp_path,
            backend=FakeBackend(),
        )
        try:
            tool = agent.tool_manager.get("read_file")
            decision = await agent._should_execute_tool(
                tool, ReadFileArgs(file_path=str(target))
            )
            assert decision.verdict == ToolExecutionResponse.SKIP
            assert decision.approval_type == ToolPermission.NEVER
            assert decision.feedback
            assert not hasattr(agent._request_broker, "request_approval")
            assert target.read_text() == "unchanged"
        finally:
            await agent.aclose()
