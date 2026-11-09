from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Literal

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

#: P2d fetch representation: the marking recorded for a delivered result.
ResultDeliveryMode = Literal["digest", "full"]


class GetAgentResultArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The stable agent handle returned by task")
    run_id: str | None = Field(
        default=None,
        description="The specific run to retrieve; defaults to the agent's latest run",
    )
    mode: ResultDeliveryMode = Field(
        default="digest",
        description=(
            "Delivery representation of this fetch: 'digest' (default) or 'full'. "
            "The marking is shared across the get and wait routes: a run already "
            "delivered in the same representation with no compaction boundary since "
            "is mechanically refused unless force=true."
        ),
    )
    force: bool = Field(
        default=False,
        description=(
            "Re-deliver even when this run was already delivered in the requested "
            "mode and no compaction boundary has landed since. Forced refetch of "
            "an expired result surfaces a result-expired error."
        ),
    )


class GetAgentResultResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TaskResult | None = Field(
        description="The completed task result, or None while the agent is running; unknown agents and runs raise a typed tool error"
    )
    message: str = Field(description="Status of the requested agent run")
    # P2d: the delivery marking payload. The session logger hoists it to the
    # per-message envelope key so it commits atomically with the transcript
    # write; absent when this fetch delivered nothing.
    delivery: dict[str, str] | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
        description=(
            "The delivery marking recorded for this fetch "
            "(conversation_id, agent_id, run_id, representation); persisted as the "
            "message's result_delivery envelope key with the transcript write."
        ),
    )


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
        manager = ctx.subagent_manager
        # P2d: the mode/force delivery path rides the registry's results
        # collaborator (duck-typed: core never imports app_server). Managers
        # without it (test doubles) keep the legacy raw fetch.
        results = getattr(manager, "_run_results", None)
        try:
            if results is not None:
                outcome = await results.get_result(
                    args.agent_id, args.run_id, mode=args.mode, force=args.force
                )
            else:
                outcome = await manager.get_agent_result(args.agent_id, args.run_id)
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
        if results is None:
            # Legacy manager without the delivery collaborator.
            if outcome is None:
                yield GetAgentResultResult(
                    result=None,
                    message="Agent is still running; use wait_for_agent or try again later.",
                )
                return
            yield GetAgentResultResult(
                result=outcome, message="Agent result is available"
            )
            return
        # P2d delivery states (the ResultDelivery shape is duck-typed here).
        status = getattr(outcome, "status", None)
        if status == "pending":
            yield GetAgentResultResult(
                result=None,
                message=(
                    "Agent is still running; a completion notification arrives "
                    "when the run finishes — fetch the result then with "
                    'mode="digest" (default) or mode="full", or block with '
                    "wait_for_agent (same modes)."
                ),
            )
            return
        if status in {"refused", "nothing_left"}:
            # Mechanical refusal / nothing-left: no payload is re-delivered.
            raise ToolError(getattr(outcome, "message", "Result already delivered."))
        marking = getattr(outcome, "marking", None)
        if isinstance(marking, dict) and ctx.session_id:
            marking = {**marking, "conversation_id": ctx.session_id}
        yield GetAgentResultResult(
            result=getattr(outcome, "result", None),
            message="Agent result is available",
            delivery=marking if isinstance(marking, dict) else None,
        )
