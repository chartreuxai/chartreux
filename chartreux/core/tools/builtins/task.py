from __future__ import annotations

from collections.abc import AsyncGenerator
import fnmatch
from typing import cast

from pydantic import Field, JsonValue

from chartreux.core.agents.models import AgentType
from chartreux.core.events import ToolCallEvent, ToolResultEvent, ToolStreamEvent
from chartreux.core.subagents import TaskArgs, TaskResult
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
    ToolPermission,
)
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolCallDisplay, ToolResultDisplay, ToolUIData
from chartreux.model_display import format_model_display_name
from chartreux.utils.tool_presentation import ToolEffectKind


def _skipped_member_lines(result: TaskResult) -> list[str]:
    if result.members is None:
        return []
    return [
        f"Skipped {member.base_model}: "
        + (
            member.error.get("message", "member was rejected during preflight")
            if member.error is not None
            else "member was rejected during preflight"
        )
        for member in result.members
        if member.status == "skipped"
    ]


def _switch_notice_lines(metadata: dict[str, object] | None) -> list[str]:
    if not metadata:
        return []
    notices = metadata.get("switch_notices")
    if not isinstance(notices, list):
        return []
    lines: list[str] = []
    for notice in notices:
        if not isinstance(notice, dict):
            continue
        base = notice.get("base_model")
        old = notice.get("old_provider")
        new = notice.get("new_provider")
        reason = notice.get("reason")
        if all(isinstance(value, str) and value for value in (base, old, new, reason)):
            lines.append(f"Switched {base}: {old} → {new} ({reason})")
    return lines


def _result_display_names(result: TaskResult) -> list[str]:
    if result.members is not None:
        return [
            member.display_name
            for member in result.members
            if member.status != "skipped"
        ]
    metadata = result.metadata or {}
    return (
        [
            format_model_display_name(
                metadata.get("active_provider")
                if isinstance(metadata.get("active_provider"), str)
                else None,
                metadata.get("base_model")
                if isinstance(metadata.get("base_model"), str)
                else None,
            )
        ]
        if metadata.get("active_provider") or metadata.get("base_model")
        else []
    )


class TaskToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ASK
    allowlist: list[str] = Field(default=["worker"])


class Task(
    BaseTool[TaskArgs, TaskResult, TaskToolConfig, BaseToolState],
    ToolUIData[TaskArgs, TaskResult],
):
    effect_kind = ToolEffectKind.SUBAGENT

    @classmethod
    def get_call_display(cls, event: ToolCallEvent) -> ToolCallDisplay:
        args = event.args
        if isinstance(args, TaskArgs):
            background = " (background)" if args.background else ""
            message = f"{args.agent} agent{background}: {args.task}"
            return ToolCallDisplay(
                summary=f"Running {message}",
                verb="Running",
                message=message,
                settled_verb="Ran",
                settled_message=message,
            )
        return ToolCallDisplay(
            summary="Running subagent",
            verb="Running",
            message="subagent",
            settled_verb="Ran",
            settled_message="subagent",
        )

    @classmethod
    def get_result_display(cls, event: ToolResultEvent) -> ToolResultDisplay:
        result = event.result
        if isinstance(result, TaskResult):
            names = _result_display_names(result)
            switches = _switch_notice_lines(result.metadata)
            skipped = _skipped_member_lines(result)
            detail = "; ".join([*names, *switches, *skipped])
            if result.members is not None:
                members_running = any(
                    member.status == "running" for member in result.members
                )
                launched_count = len(result.members) - len(skipped)
                skipped_summary = f", {len(skipped)} skipped" if skipped else ""
                return ToolResultDisplay(
                    success=result.completed,
                    verb="Launched" if members_running else "Completed",
                    message=f"{launched_count} agents{skipped_summary}"
                    + (f": {detail}" if detail else ""),
                )
            if result.agent_id is not None:
                return ToolResultDisplay(
                    success=True,
                    verb="Launched",
                    message=(
                        f"agent {result.agent_id} (run {result.run_id})"
                        + (f": {detail}" if detail else "")
                    ),
                )
            turn_word = "turn" if result.turns_used == 1 else "turns"
            if not result.completed:
                return ToolResultDisplay(
                    success=False,
                    verb="Interrupted",
                    message=(
                        f"after {result.turns_used} {turn_word}"
                        + (f": {detail}" if detail else "")
                    ),
                )
            return ToolResultDisplay(
                success=True,
                verb="Completed",
                message=(
                    f"in {result.turns_used} {turn_word}"
                    + (f": {detail}" if detail else "")
                ),
            )
        return ToolResultDisplay(success=True, verb="Completed", message="")

    @classmethod
    def project_result(cls, result: TaskResult) -> dict[str, JsonValue] | None:
        names = _result_display_names(result)
        switches = _switch_notice_lines(result.metadata)
        skipped = _skipped_member_lines(result)
        if not names and not switches and not skipped:
            return None
        output: dict[str, JsonValue] = {}
        if names:
            output["models"] = cast(JsonValue, names)
        if switches:
            output["switches"] = cast(JsonValue, switches)
        if skipped:
            output["skipped"] = cast(JsonValue, skipped)
        return output

    @classmethod
    def get_status_text(cls) -> str:
        return "Running subagent"

    def resolve_permission(self, args: TaskArgs) -> PermissionContext | None:
        if args.agent_id is not None and "agent" not in args.model_fields_set:
            # The retained profile is resolved by the runner before resources.
            return None
        agent_name = args.agent

        for pattern in self.config.denylist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.NEVER)

        for pattern in self.config.allowlist:
            if fnmatch.fnmatch(agent_name, pattern):
                return PermissionContext(permission=ToolPermission.ALWAYS)

        return None

    async def run(
        self, args: TaskArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if not ctx or not ctx.agent_manager:
            raise ToolError("Task tool requires agent_manager in context")

        agent_manager = ctx.agent_manager
        if ctx.is_subagent:
            if args.background:
                raise ToolError("Background agents cannot be spawned from subagents")
            raise ToolError(
                "Agent depth limit of 1 reached. Complete the task in the current "
                "subagent."
            )
        if args.agent_id is not None and not args.background:
            raise ToolError("agent_id can only be used with background mode")
        if args.agent_id is not None and "agent" not in args.model_fields_set:
            if ctx.subagent_runner is None:
                raise ToolError("Task tool requires a subagent runner in context")
            async for event in ctx.subagent_runner.run(args, ctx):
                yield event
            return

        try:
            agent_profile = agent_manager.get_agent(args.agent)
        except ValueError as e:
            raise ToolError(f"Unknown agent: {args.agent}") from e

        if agent_profile.agent_type != AgentType.SUBAGENT:
            raise ToolError(
                f"Agent '{args.agent}' is a {agent_profile.agent_type.value} agent. "
                f"Only subagents can be used with the task tool. "
                f"This is a security constraint to prevent recursive spawning."
            )
        if any(
            fnmatch.fnmatch(agent_profile.name, pattern)
            for pattern in self.config.denylist
        ):
            raise ToolError(f"Task denied for agent profile: {agent_profile.name}")
        if ctx.subagent_runner is None:
            raise ToolError("Task tool requires a subagent runner in context")

        async for event in ctx.subagent_runner.run(args, ctx):
            yield event
