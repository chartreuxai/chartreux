from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import ResumeAgentResult, ResumeOutcome
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


class ResumeAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        description="Stable background agent handle (not a foreground session)"
    )
    expected_run_id: str = Field(
        description="The suspended failed run to resume from (latest_run_id)"
    )
    idempotency_key: str | None = Field(
        default=None,
        description=(
            "Single-flight key for retries; omitted defaults to "
            "'<agent_id>:<expected_run_id>'"
        ),
    )


class ResumeAgentConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class ResumeAgent(
    BaseTool[ResumeAgentArgs, ResumeAgentResult, ResumeAgentConfig, BaseToolState],
    ToolUIData[ResumeAgentArgs, ResumeAgentResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        target = (
            event.args.agent_id if isinstance(event.args, ResumeAgentArgs) else "agent"
        )
        return ToolCallDisplay(
            summary=f"Resuming {target}",
            verb="Resuming",
            message=target,
            settled_verb="Resume",
            settled_message=target,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        result = event.result
        if isinstance(result, ResumeAgentResult):
            admitted = result.outcome in {
                ResumeOutcome.SCHEDULED,
                ResumeOutcome.ALREADY_SCHEDULED,
                ResumeOutcome.ALREADY_RESUMED,
            }
            message = f"run {result.run_id}"
            if result.successor_run_id is not None:
                message += f" -> {result.successor_run_id}"
            return ToolResultDisplay(
                success=admitted,
                verb=result.outcome.value.replace("_", " ").capitalize(),
                message=message,
            )
        return ToolResultDisplay(success=True, verb="Resume", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Resuming agent"

    async def run(
        self, args: ResumeAgentArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ResumeAgentResult, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError(
                "resume_agent requires a subagent manager in context; background agents only"
            )
        if not ctx.session_id:
            raise ToolError("resume_agent requires requester session_id in context")
        if ctx.is_subagent:
            raise ToolError(
                "resume_agent is only available to the parent orchestrator for background agents"
            )
        try:
            result = await ctx.subagent_manager.resume_agent(
                args.agent_id,
                args.expected_run_id,
                args.idempotency_key or f"{args.agent_id}:{args.expected_run_id}",
                requester_session_id=ctx.session_id,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        yield result
