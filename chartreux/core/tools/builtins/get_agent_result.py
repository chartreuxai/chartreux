from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import (
    AgentEvictedError,
    AgentResultExpiredError,
    TaskResult,
    UnknownAgentError,
)
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.utils.tool_presentation import ToolEffectKind


class GetAgentResultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The stable agent handle returned by task")
    run_id: str | None = Field(
        default=None,
        description="The specific run to retrieve; defaults to the agent's latest run",
    )


class GetAgentResultResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TaskResult | None = Field(
        description="The completed task result, or None while the agent is running; unknown agents and runs raise a typed tool error"
    )
    message: str = Field(description="Status of the requested agent run")


class GetAgentResultConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class GetAgentResult(
    BaseTool[
        GetAgentResultArgs, GetAgentResultResult, GetAgentResultConfig, BaseToolState
    ],
    ToolUIData[GetAgentResultArgs, GetAgentResultResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        agent_id = args.agent_id if isinstance(args, GetAgentResultArgs) else "agent"
        return ToolCallDisplay(
            summary=f"Getting result for agent {agent_id}",
            verb="Getting",
            message=f"result for agent {agent_id}",
            settled_verb="Retrieved",
            settled_message=f"result for agent {agent_id}",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        result = event.result
        if isinstance(result, GetAgentResultResult):
            if result.result is None:
                return ToolResultDisplay(
                    success=True, verb="Pending", message=result.message
                )
            return ToolResultDisplay(
                success=True, verb="Completed", message="agent result"
            )
        return ToolResultDisplay(success=True, verb="Completed", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Getting agent result"

    async def run(
        self, args: GetAgentResultArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[GetAgentResultResult, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError("get_agent_result requires a subagent manager in context")
        try:
            result = await ctx.subagent_manager.get_agent_result(
                args.agent_id, args.run_id
            )
        except AgentEvictedError as exc:
            raise ToolError(
                "Agent was evicted and cannot run again; its result remains retrievable "
                "while retained."
            ) from exc
        except AgentResultExpiredError as exc:
            raise ToolError("Agent result expired and is no longer available.") from exc
        except UnknownAgentError as exc:
            raise ToolError(
                "No such agent or run. Check the agent_id and run_id with check_agents."
            ) from exc
        if result is None:
            yield GetAgentResultResult(
                result=None,
                message="Agent is still running; use wait_for_agent or try again later.",
            )
            return
        yield GetAgentResultResult(result=result, message="Agent result is available")
