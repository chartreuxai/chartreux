from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import ReleaseAgentOutcome, UnknownAgentError
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


class ReleaseAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The stable agent handle to release")


class ReleaseAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The released agent handle")
    message: str = Field(description="Status of the release operation")


class ReleaseAgentConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class ReleaseAgent(
    BaseTool[ReleaseAgentArgs, ReleaseAgentResult, ReleaseAgentConfig, BaseToolState],
    ToolUIData[ReleaseAgentArgs, ReleaseAgentResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        agent_id = args.agent_id if isinstance(args, ReleaseAgentArgs) else "agent"
        return ToolCallDisplay(
            summary=f"Releasing agent {agent_id}",
            verb="Releasing",
            message=f"agent {agent_id}",
            settled_verb="Released",
            settled_message=f"agent {agent_id}",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        if isinstance(event.result, ReleaseAgentResult):
            return ToolResultDisplay(
                success=True, verb="Released", message=event.result.agent_id
            )
        return ToolResultDisplay(success=True, verb="Released", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Releasing agent"

    async def run(
        self, args: ReleaseAgentArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ReleaseAgentResult, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError("release_agent requires a subagent manager in context")
        try:
            outcome = await ctx.subagent_manager.release_agent(args.agent_id)
        except UnknownAgentError as exc:
            raise ToolError(
                "No such agent. Check the agent_id with check_agents."
            ) from exc
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        message = (
            "Evicted agent tombstone removed"
            if outcome is ReleaseAgentOutcome.EVICTED
            else "Agent released"
        )
        yield ReleaseAgentResult(agent_id=args.agent_id, message=message)
