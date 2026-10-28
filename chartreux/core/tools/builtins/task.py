from __future__ import annotations

from collections.abc import AsyncGenerator
import fnmatch
from typing import cast

from pydantic import Field, JsonValue

from chartreux.core.agents.identity_launch import (
    DEFAULT_CHILD_IDENTITY,
    parse_model_shorthand,
)
from chartreux.core.events import ToolCallEvent, ToolResultEvent, ToolStreamEvent
from chartreux.core.subagents import LaunchOutcome, TaskArgs, TaskResult
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


def _override_diagnostics_line(metadata: dict[str, object] | None) -> str:
    """The bounded un-stick diagnostics line for the result display (C4).

    Rendered only for an explicit override_cooldown launch: the forced
    deployment, the cooldown exclusions the permit bypassed, and the start
    allowance remaining after the attempt consumes.
    """
    if not metadata:
        return ""
    diagnostics = metadata.get("cooldown_override")
    if not isinstance(diagnostics, dict):
        return ""
    provider = diagnostics.get("provider")
    wire_name = diagnostics.get("wire_name")
    bypassed = diagnostics.get("exclusions_bypassed")
    remaining = diagnostics.get("remaining_starts")
    parts = ["forced un-stick"]
    if isinstance(provider, str) and isinstance(wire_name, str):
        parts.append(f"{format_model_display_name(provider, wire_name)}")
    if isinstance(bypassed, list) and bypassed:
        parts.append(f"bypassed {', '.join(str(item) for item in bypassed)}")
    if isinstance(remaining, int):
        parts.append(f"{remaining} forced start(s) left in the episode")
    return "; ".join(parts)


def _result_display_names(result: TaskResult) -> list[str]:
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


def _requested_identity(args: TaskArgs) -> str | None:
    """The canonical launch identity this call names, if any (design §7).

    The structured ``identity`` field wins; a model shorthand carrying an
    identity component (``@fast:researcher``) names the identity too. The
    retired ``agent_type`` field is rejected by TaskArgs itself with the
    successor guidance, so it needs no handling here.
    """
    if args.identity is not None:
        return args.identity
    if args.model is not None:
        shorthand = parse_model_shorthand(args.model)
        if shorthand is not None and shorthand.identity is not None:
            return shorthand.identity
    return None


def _launch_target(args: TaskArgs) -> str:
    """The identity the task denylist and displays key on (design §12).

    Precedence: the canonical identity (structured or shorthand), then the
    default child identity.
    """
    requested = _requested_identity(args)
    if requested is not None:
        return requested
    return DEFAULT_CHILD_IDENTITY


def _retained_reuse_without_target(args: TaskArgs) -> bool:
    """A retained-instance continuation that names no launch target."""
    return args.agent_id is not None and _requested_identity(args) is None


def _validate_replacement_claim_args(args: TaskArgs) -> None:
    """WP5 claim-shape guards: a cross-model replacement claim
    (``config.predecessor_agent_id``) launches a NEW agent in the background
    and names its model and package explicitly; launch admission validates
    the rest at failure time.
    """
    if args.config is None or args.config.predecessor_agent_id is None:
        return
    if args.agent_id is not None:
        raise ToolError(
            "a replacement claim (config.predecessor_agent_id) launches "
            "a NEW agent: omit agent_id"
        )
    if not args.background:
        raise ToolError("a replacement claim requires background mode")
    if args.model is None and args.config.model is None:
        # The canonical top-level model field and the legacy nested
        # config.model are the same selection (the schema promotes the
        # nested form; a dual supply is a validation error).
        raise ToolError(
            "a replacement claim requires the replacement model "
            "(the top-level model field or config.model)"
        )
    if args.config.work_package_id is None:
        raise ToolError(
            "a replacement claim requires config.work_package_id (the "
            "package identity the per-package budget is keyed on)"
        )


class TaskToolConfig(BaseToolConfig):
    permission: ToolPermission = ToolPermission.ALWAYS
    allowlist: list[str] = Field(default=["coder"])


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
            message = f"{_launch_target(args)} agent{background}: {args.task}"
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
            if (
                result.launch_outcome is not None
                and result.launch_outcome is not LaunchOutcome.LAUNCHED
            ):
                return ToolResultDisplay(
                    success=False,
                    verb="Not launched",
                    message=f"agent {result.agent_id}: {result.launch_outcome.value}",
                )
            names = _result_display_names(result)
            switches = _switch_notice_lines(result.metadata)
            override = _override_diagnostics_line(result.metadata)
            detail = "; ".join([*names, *switches, *([override] if override else [])])
            if result.agent_id is not None:
                # WP5: a replacement launch acknowledgment shows the
                # validated decision's milestone notice (with its honest
                # cost implication).
                notice = (
                    f"; {result.replacement.notice}"
                    if result.replacement is not None and result.replacement.notice
                    else ""
                )
                return ToolResultDisplay(
                    success=True,
                    verb="Launched",
                    message=(
                        f"agent {result.agent_id} (run {result.run_id})"
                        + (f": {detail}" if detail else "")
                        + notice
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
        metadata = result.metadata or {}
        diagnostics = metadata.get("cooldown_override")
        output: dict[str, JsonValue] = {}
        if names:
            output["models"] = cast(JsonValue, names)
        if switches:
            output["switches"] = cast(JsonValue, switches)
        # C4: the bounded un-stick diagnostics persist with the projection.
        if isinstance(diagnostics, dict):
            output["cooldown_override"] = cast(JsonValue, diagnostics)
        return output or None

    @classmethod
    def get_status_text(cls) -> str:
        return "Running subagent"

    def resolve_permission(self, args: TaskArgs) -> PermissionContext | None:
        if _retained_reuse_without_target(args):
            # The retained identity is resolved by the runner before resources.
            return None
        target = _launch_target(args)

        for pattern in self.config.denylist:
            if fnmatch.fnmatch(target, pattern):
                return PermissionContext(permission=ToolPermission.NEVER)

        for pattern in self.config.allowlist:
            if fnmatch.fnmatch(target, pattern):
                return PermissionContext(permission=ToolPermission.ALWAYS)

        return None

    async def run(
        self, args: TaskArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[ToolStreamEvent | TaskResult, None]:
        if ctx is None:
            raise ToolError("Task tool requires an invoke context")
        if ctx.is_subagent:
            if args.background:
                raise ToolError("Background agents cannot be spawned from subagents")
            raise ToolError(
                "Agent depth limit of 1 reached. Complete the task in the current "
                "subagent."
            )
        if args.replace_run and (args.agent_id is None or not args.background):
            raise ToolError("replace_run requires agent_id and background mode")
        if args.agent_id is not None and not args.background:
            raise ToolError("agent_id can only be used with background mode")
        if args.retain is not None and not args.background:
            raise ToolError("retain can only be used with background mode")
        # WP5: a cross-model replacement claim (config.predecessor_agent_id)
        # launches a NEW agent in the background and names its model and
        # package explicitly; launch admission validates the rest at failure
        # time.
        _validate_replacement_claim_args(args)
        if _retained_reuse_without_target(args):
            if ctx.subagent_runner is None:
                raise ToolError("Task tool requires a subagent runner in context")
            async for event in ctx.subagent_runner.run(args, ctx):
                yield event
            return

        # The task denylist keys on the resolved launch identity (design §12).
        target = _launch_target(args)
        if any(fnmatch.fnmatch(target, pattern) for pattern in self.config.denylist):
            raise ToolError(f"Task denied for launch identity: {target}")
        if ctx.subagent_runner is None:
            raise ToolError("Task tool requires a subagent runner in context")

        async for event in ctx.subagent_runner.run(args, ctx):
            yield event
