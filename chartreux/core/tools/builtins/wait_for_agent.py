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


class WaitForAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The stable agent handle returned by task")
    run_id: str | None = Field(
        default=None,
        description="The specific run to wait for; defaults to the agent's latest run",
    )
    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Maximum seconds to wait without cancelling the agent run",
    )


class WaitForAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    result: TaskResult = Field(description="The completed task result")


class WaitForAgentConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS


class WaitForAgent(
    BaseTool[WaitForAgentArgs, WaitForAgentResult, WaitForAgentConfig, BaseToolState],
    ToolUIData[WaitForAgentArgs, WaitForAgentResult],
):
    effect_kind = ToolEffectKind.TOOL

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        agent_id = args.agent_id if isinstance(args, WaitForAgentArgs) else "agent"
        return ToolCallDisplay(
            summary=f"Waiting for agent {agent_id}",
            verb="Waiting",
            message=f"for agent {agent_id}",
            settled_verb="Waited",
            settled_message=f"for agent {agent_id}",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        if isinstance(event.result, WaitForAgentResult):
            return ToolResultDisplay(
                success=True, verb="Completed", message="agent run"
            )
        if isinstance(event.error, str) and "timed out" in event.error.lower():
            return ToolResultDisplay(
                success=False, verb="Timeout", message="waiting for agent"
            )
        if event.error is not None:
            return ToolResultDisplay(success=False, verb="Failed", message=event.error)
        return ToolResultDisplay(success=True, verb="Completed", message="")

    @classmethod
    def get_status_text(cls) -> str:
        return "Waiting for agent"

    async def run(
        self, args: WaitForAgentArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[WaitForAgentResult, None]:
        if not ctx or not ctx.subagent_manager:
            raise ToolError("wait_for_agent requires a subagent manager in context")
        try:
            result = await ctx.subagent_manager.wait_for_agent(
                args.agent_id, args.run_id, timeout=args.timeout
            )
        except AgentEvictedError as exc:
            raise ToolError(
                "Agent was evicted and cannot be waited on; use get_agent_result "
                "while its result is retained."
            ) from exc
        except AgentResultExpiredError as exc:
            raise ToolError(
                "Agent result expired and can no longer be waited on."
            ) from exc
        except UnknownAgentError as exc:
            raise ToolError(
                "No such agent or run. Check the agent_id and run_id with check_agents."
            ) from exc
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        except TimeoutError as exc:
            raise ToolError(
                f"Timed out waiting for agent {args.agent_id}"
                + (f" run {args.run_id}" if args.run_id else "")
            ) from exc
        yield WaitForAgentResult(result=result)
