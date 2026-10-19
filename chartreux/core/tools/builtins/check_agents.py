from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import AgentSummary
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


class CheckAgentsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CheckAgentsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reuse_guidance: str = Field(
        default="Continue a retained agent instance with task(agent_id=..., background=true, task=...).",
        description="How to reuse the instance handles listed in agents.",
    )

    resume_guidance: str = Field(
        default=(
            "A suspended agent failed recoverably and keeps its identity, "
            "conversation, tool state, and budgets: resume it explicitly with "
            "resume_agent(agent_id, expected_run_id=latest_run_id). When "
            "recovery_eligibility is reconcile_required, verify the unknown "
            "tool effects and call reconcile_result first. Resume is always "
            "explicit; replacement (task with replace_run=true) and "
            "release_agent remain available while suspended."
        ),
        description="How to resume a suspended agent (WP2).",
    )

    agents: list[AgentSummary] = Field(
        default_factory=list,
        description=(
            "The bounded agent inventory and its status: resident records "
            "(running, finalizing, idle, suspended, restoring) plus "
            "closed/evicted terminal metadata within its metadata window. "
            "Each entry's name is display-only; agent_id is the only handle. "
            "ttl_remaining_seconds is advisory and does not guarantee the "
            "agent will remain available."
        ),
    )


class CheckAgentsConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class CheckAgents(
    BaseTool[CheckAgentsArgs, CheckAgentsResult, CheckAgentsConfig, BaseToolState],
    ToolUIData[CheckAgentsArgs, CheckAgentsResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        return ToolCallDisplay(
            summary="Checking subagents",
            verb="Checking",
            message="subagents",
            settled_verb="Checked",
            settled_message="subagents",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        result = event.result
        if isinstance(result, CheckAgentsResult):
            count = len(result.agents)
            message = f"{count} agent{'s' if count != 1 else ''}"
            return ToolResultDisplay(success=True, verb="Completed", message=message)
        return ToolResultDisplay(success=True, verb="Completed", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Checking subagents"

    async def run(
        self, args: CheckAgentsArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[CheckAgentsResult, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError("check_agents requires a subagent manager in context")
        agents = await ctx.subagent_manager.check_agents()
        yield CheckAgentsResult(agents=agents)
