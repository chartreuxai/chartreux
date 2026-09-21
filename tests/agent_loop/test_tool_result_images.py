from __future__ import annotations

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field
import pytest

from chartreux.core.agent_loop import AgentLoop
from chartreux.core.config import ModelConfig
from chartreux.core.events import ToolResultEvent, ToolStreamEvent
from chartreux.core.hooks.manager import HooksManager
from chartreux.core.hooks.models import HookTextReplacement, PostToolInvocation
from chartreux.core.llm_models import (
    FileImageSource,
    FunctionCall,
    ImageAttachment,
    Role,
    ToolCall,
)
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    CancellableToolResult,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from tests.conftest import build_test_agent_loop, build_test_vibe_config
from tests.mock.utils import mock_llm_chunk
from tests.stubs.fake_backend import FakeBackend


class ImageToolArgs(BaseModel):
    mode: str = "success"


class ImageToolResult(CancellableToolResult):
    message: str = "image result"
    attachment: ImageAttachment = Field(exclude=True)


class ImageTool(
    BaseTool[ImageToolArgs, ImageToolResult, BaseToolConfig, BaseToolState]
):
    @classmethod
    def get_name(cls) -> str:
        return "todo"

    async def run(
        self, args: ImageToolArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | ImageToolResult, None]:
        if args.mode == "failure":
            raise ToolError("image tool failed")
        yield ImageToolResult(
            message="image result",
            cancelled=args.mode == "cancelled",
            attachment=ImageAttachment(
                source=FileImageSource(path=Path("result.png")),
                alias="result.png",
                mime_type="image/png",
            ),
        )

    def get_result_images(
        self, result: ImageToolResult
    ) -> list[ImageAttachment] | None:
        return [result.attachment]


class PlainToolResult(BaseModel):
    message: str = "plain result"


class PlainTool(
    BaseTool[ImageToolArgs, PlainToolResult, BaseToolConfig, BaseToolState]
):
    @classmethod
    def get_name(cls) -> str:
        return "todo"

    async def run(
        self, args: ImageToolArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | PlainToolResult, None]:
        yield PlainToolResult()


def _tool_call(mode: str = "success") -> ToolCall:
    return ToolCall(
        id="image-call",
        index=0,
        function=FunctionCall(name="todo", arguments=f'{{"mode": "{mode}"}}'),
    )


def _build_agent(
    tool: type[BaseTool],
    *,
    mode: str = "success",
    permission: ToolPermission = ToolPermission.ALWAYS,
) -> AgentLoop:
    agent = build_test_agent_loop(
        config=build_test_vibe_config(
            active_model="vision",
            models=[
                ModelConfig(
                    name="vision",
                    provider="mistral/default",
                    alias="vision",
                    supports_images=True,
                )
            ],
            enabled_tools=["todo"],
            tools={"todo": {"permission": permission.value}},
        ),
        backend=FakeBackend([
            [mock_llm_chunk(content="Calling tool.", tool_calls=[_tool_call(mode)])],
            [mock_llm_chunk(content="Done.")],
        ]),
    )
    agent.tool_manager._all_tools["todo"] = tool
    agent.tool_manager._tool_variants_by_name["todo"] = [tool]
    agent.tool_manager._custom_tool_variants_by_name["todo"] = [False]
    agent.tool_manager._instances.pop("todo", None)
    return agent


async def _act(agent: AgentLoop) -> list[ToolResultEvent]:
    events: list[ToolResultEvent] = []
    async for event in agent.act("show me an image"):
        if isinstance(event, ToolResultEvent):
            events.append(event)
    return events


def _tool_message(agent: AgentLoop):
    return next(message for message in agent.messages if message.role == Role.tool)


@pytest.mark.asyncio
async def test_tool_result_images_are_durable_and_excluded_from_output() -> None:
    agent = _build_agent(ImageTool)

    events = await _act(agent)

    message = _tool_message(agent)
    assert message.images is not None
    assert message.images[0].alias == "result.png"
    assert message.tool_result is not None
    assert message.tool_result.output == {"cancelled": False, "message": "image result"}
    assert events[0].result is not None
    assert "attachment" not in events[0].result.model_dump(mode="json")
    assert "attachment" not in message.tool_result.output
    assert (
        len([message for message in agent.messages if message.role == Role.user]) == 1
    )


class _TextReplacementHooks:
    def __init__(self) -> None:
        self.tool_output: dict[str, object] | None = None

    def reset_retry_count(self) -> None:
        return

    async def run(
        self, invocation: object
    ) -> AsyncGenerator[HookTextReplacement, None]:
        if isinstance(invocation, PostToolInvocation):
            self.tool_output = invocation.tool_output
            yield HookTextReplacement(text="replacement text")


@pytest.mark.asyncio
async def test_text_replacement_retains_tool_result_images() -> None:
    agent = _build_agent(ImageTool)
    hooks = _TextReplacementHooks()
    agent._hooks_manager = cast(HooksManager, hooks)

    await _act(agent)

    message = _tool_message(agent)
    assert message.content == "replacement text"
    assert message.images is not None
    assert message.images[0].alias == "result.png"
    assert hooks.tool_output is not None
    assert "attachment" not in hooks.tool_output


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "permission"),
    [
        ("failure", ToolPermission.ALWAYS),
        ("success", ToolPermission.NEVER),
        ("cancelled", ToolPermission.ALWAYS),
    ],
)
async def test_failed_skipped_and_cancelled_tools_do_not_attach_images(
    mode: str, permission: ToolPermission
) -> None:
    agent = _build_agent(ImageTool, mode=mode, permission=permission)

    await _act(agent)

    assert _tool_message(agent).images is None


@pytest.mark.asyncio
async def test_tools_without_result_images_hook_are_unchanged() -> None:
    agent = _build_agent(PlainTool)

    await _act(agent)

    message = _tool_message(agent)
    assert message.images is None
    assert message.tool_result is not None
    assert message.tool_result.output == {"message": "plain result"}
