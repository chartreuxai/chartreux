from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from chartreux.core.events import ToolCallEvent, ToolResultEvent
from chartreux.core.subagents import (
    AgentEvictedError,
    AgentResultExpiredError,
    TaskResult,
    UnknownAgentError,
    WaitForAgentResult,
    WaitStatus,
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

DEFAULT_WAIT_TIMEOUT_SECONDS = 1500.0


class WaitForAgentArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(description="The stable agent handle returned by task")
    run_id: str | None = Field(
        default=None,
        description="The specific run to wait for; defaults to the agent's latest run",
    )
    timeout: float | None = Field(
        default=DEFAULT_WAIT_TIMEOUT_SECONDS,
        gt=0,
        description=(
            "Maximum seconds to wait without cancelling the agent run. "
            f"Defaults to {DEFAULT_WAIT_TIMEOUT_SECONDS:g}s; pass null explicitly "
            "to wait indefinitely. On timeout the wait returns a still_running "
            "result (a normal outcome, not a failure) whose guidance tells you "
            "to end your turn and re-invoke the wait in a later turn."
        ),
    )


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
        result = event.result
        if isinstance(result, WaitForAgentResult):
            if result.status is WaitStatus.STILL_RUNNING:
                # A timed-out wait is a normal checkpoint, not a failure.
                return ToolResultDisplay(
                    success=True,
                    verb="Waiting",
                    message=f"agent {result.agent_id} still running",
                )
            return ToolResultDisplay(
                success=True, verb="Completed", message="agent run"
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
            wait = ctx.subagent_manager.wait_for_agent(
                args.agent_id, args.run_id, timeout=args.timeout
            )
            if ctx.register_wait_task is None:
                result = await wait
            else:
                task = asyncio.create_task(wait)
                # The loop only ever cancels registered wait tasks, so the
                # disposition type is irrelevant to cancellation.
                ctx.register_wait_task(cast("asyncio.Task[TaskResult] | None", task))
                try:
                    result = await asyncio.shield(task)
                finally:
                    if not task.done() and not task.cancelling():
                        task.cancel()
                    joined = asyncio.gather(task, return_exceptions=True)
                    interrupted = False
                    while not joined.done():
                        try:
                            await asyncio.shield(joined)
                        except asyncio.CancelledError:
                            interrupted = True
                    ctx.register_wait_task(None)
                    if interrupted:
                        raise asyncio.CancelledError
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
        yield result
