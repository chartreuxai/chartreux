from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import CancelOutcome, CancelResult, RunStopReason
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


class CancelAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        description="Stable background agent handle (not a foreground session)"
    )
    run_id: str | None = Field(
        default=None, description="Pinned run to stop; omitted selects the active run"
    )


class CancelAgentConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class CancelAgent(
    BaseTool[CancelAgentArgs, CancelResult, CancelAgentConfig, BaseToolState],
    ToolUIData[CancelAgentArgs, CancelResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        target = (
            event.args.agent_id if isinstance(event.args, CancelAgentArgs) else "agent"
        )
        return ToolCallDisplay(
            summary=f"Requesting stop for {target}",
            verb="Requesting stop",
            message=target,
            settled_verb="Stop request",
            settled_message=target,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        result = event.result
        if isinstance(result, CancelResult):
            return ToolResultDisplay(
                success=result.outcome is not CancelOutcome.FORBIDDEN,
                verb=result.outcome.value.replace("_", " ").capitalize(),
                message=f"run {result.run_id}"
                + (
                    f" ({result.stop_reason.value})"
                    if result.stop_reason is not None
                    else ""
                ),
            )
        return ToolResultDisplay(success=True, verb="Stop request", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Requesting agent stop"

    async def run(
        self, args: CancelAgentArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[CancelResult, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError(
                "cancel_agent requires a subagent manager in context; background agents only"
            )
        if not ctx.session_id:
            raise ToolError("cancel_agent requires requester session_id in context")
        if ctx.is_subagent:
            raise ToolError(
                "cancel_agent is only available to the parent orchestrator for background agents"
            )
        try:
            result = await ctx.subagent_manager.cancel_run(
                args.agent_id,
                args.run_id,
                reason=RunStopReason.ORCHESTRATOR_CANCELLED,
                requester_session_id=ctx.session_id,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        yield result
