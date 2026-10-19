from __future__ import annotations

from collections.abc import AsyncGenerator

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import (
    EffectVerification,
    ReconcileOutcome,
    ReconcileResult as ReconcileResultModel,
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


class ReconcileResultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        description="Stable background agent handle (not a foreground session)"
    )
    expected_run_id: str = Field(
        description="The suspended failed run the evidence is pinned to"
    )
    transcript_boundary: str = Field(
        description=(
            "The failed run's transcript boundary (the failure envelope's "
            "progress boundary) the evidence is pinned to"
        )
    )
    verified_effects: list[EffectVerification] = Field(
        description=(
            "One entry per unknown tool call: tool_call_id with disposition "
            "'settled' or 'unrepeatable_damage' and an optional bounded note. "
            "An empty list is the complete zero-entry accounting for a run "
            "that left no tool call outstanding (round-2 V4)"
        )
    )


class ReconcileResultConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class ReconcileResult(
    BaseTool[
        ReconcileResultArgs, ReconcileResultModel, ReconcileResultConfig, BaseToolState
    ],
    ToolUIData[ReconcileResultArgs, ReconcileResultModel],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        target = (
            event.args.agent_id
            if isinstance(event.args, ReconcileResultArgs)
            else "agent"
        )
        return ToolCallDisplay(
            summary=f"Reconciling effects for {target}",
            verb="Reconciling",
            message=target,
            settled_verb="Reconciled",
            settled_message=target,
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        result = event.result
        if isinstance(result, ReconcileResultModel):
            applied = result.outcome in {
                ReconcileOutcome.CLEARED,
                ReconcileOutcome.PROHIBITED,
            }
            message = f"run {result.run_id}"
            if result.eligibility is not None:
                message += f" ({result.eligibility.value})"
            return ToolResultDisplay(
                success=applied,
                verb=result.outcome.value.replace("_", " ").capitalize(),
                message=message,
            )
        return ToolResultDisplay(success=True, verb="Reconciled", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Reconciling agent effects"

    async def run(
        self, args: ReconcileResultArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ReconcileResultModel, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError(
                "reconcile_result requires a subagent manager in context; background agents only"
            )
        if not ctx.session_id:
            raise ToolError("reconcile_result requires requester session_id in context")
        if ctx.is_subagent:
            raise ToolError(
                "reconcile_result is only available to the parent orchestrator for background agents"
            )
        try:
            result = await ctx.subagent_manager.reconcile_result(
                args.agent_id,
                args.expected_run_id,
                args.transcript_boundary,
                args.verified_effects,
                requester_session_id=ctx.session_id,
            )
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        yield result
