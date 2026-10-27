from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast

from pydantic import JsonValue

from chartreux.core.background_jobs import BashStartArgs, StartResult
from chartreux.core.env import project_env_for_cwd
from chartreux.core.tools.base import (
    BaseTool,
    BaseToolConfig,
    BaseToolState,
    InvokeContext,
    ToolError,
)
from chartreux.core.tools.builtins._shell_permission_resolver import (
    ShellPermissionResolver,
    shell_command_checkout_no_overwrite_ignore,
    shell_command_mutation_roots,
    shell_command_write_paths,
)
from chartreux.core.tools.builtins.bash import (
    Bash,
    BashToolConfig,
    CancelledShellCommand,
)
from chartreux.core.tools.permissions import PermissionContext
from chartreux.core.tools.ui import ToolResultDisplay, ToolUIData


class FailedJobStart(ToolError):
    """ToolError from a background job whose admission failed.

    A failed admission may still have spawned the command (partial
    execution), so it carries the command's write-classified paths and its
    unknown-write-surface flag; the tool executor copies them onto the error
    result event, and the submission scan collects the footprint exactly as
    a successful start's would be.
    """

    def __init__(
        self,
        message: str,
        *,
        write_paths: list[str] | None = None,
        write_paths_unknown: bool = False,
        mutation_roots: list[str] | None = None,
        mutation_roots_unknown: bool = False,
    ) -> None:
        super().__init__(message)
        self.write_paths = list(write_paths or [])
        self.write_paths_unknown = write_paths_unknown
        self.mutation_roots = list(mutation_roots or [])
        self.mutation_roots_unknown = mutation_roots_unknown


class BashStart(
    BaseTool[BashStartArgs, StartResult, BashToolConfig, BaseToolState],
    ToolUIData[BashStartArgs, StartResult],
):
    @classmethod
    def path_sensitive_patterns(cls, config: BaseToolConfig) -> tuple[str, ...]:
        return Bash.path_sensitive_patterns(config)

    def resolve_permission(self, args: BashStartArgs) -> PermissionContext | None:
        return ShellPermissionResolver(
            self.config,
            self.cwd,
            self.workspace,
            self.path_authority,
            self.dangerous_command_authorities_getter,
            # The same trust-gated environment the background job's
            # subprocess receives; the egress rule consults it for injected
            # non-loopback proxies (v0.5.9 R1).
            injected_env=project_env_for_cwd(self.cwd),
        ).resolve_permission(args)

    async def run(
        self, args: BashStartArgs, ctx: InvokeContext | None = None
    ) -> AsyncGenerator[StartResult, None]:
        if ctx is None or ctx.background_jobs is None:
            raise ToolError("Background jobs unavailable in this context")
        # A2: the same executed-string rewriting as Bash.run(): a
        # whole-call-eligible exempt checkout executes with
        # --no-overwrite-ignore so an ignored-file collision fails closed.
        # The rewritten string is the executed command (execute_command=...
        # below); args.command stays original, so the stored job summary
        # keeps the original command string.
        command = shell_command_checkout_no_overwrite_ignore(
            args.command, workspace=self.workspace, authority=self.path_authority
        )
        if command is None:
            command = args.command
        # R7: surface the started command's write-classified paths (redirect
        # targets, write-option values, write-classified command operands)
        # on the result, so the submission secret scan covers a started
        # background job's write surface exactly as a foreground run's. The
        # footprint is computed before the admission attempt: a failed or
        # cancelled start may still have spawned the command, and its write
        # surface must reach the submission scan regardless of the outcome.
        # A1: the mutation-roots footprint (the resolved work-tree roots of
        # the command's checkout/switch invocations) is collected alongside
        # it — a mutation-scope footprint that never enters the submission
        # scan's path list.
        write_paths, write_paths_unknown = shell_command_write_paths(
            command,
            cwd=self.cwd,
            workspace=self.workspace,
            authority=self.path_authority,
        )
        mutation_roots, mutation_roots_unknown = shell_command_mutation_roots(
            command,
            cwd=self.cwd,
            workspace=self.workspace,
            authority=self.path_authority,
        )
        try:
            result = await ctx.background_jobs.start(
                args, cwd=self.cwd, execute_command=command
            )
        except asyncio.CancelledError as exc:
            # The registry finishes its pre-commit recovery before
            # re-raising; the executor's cancellation handler copies the
            # write footprint onto the cancelled result event.
            raise CancelledShellCommand(
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        except (ValueError, OSError) as exc:
            raise FailedJobStart(
                "Unable to start background job",
                write_paths=write_paths,
                write_paths_unknown=write_paths_unknown,
                mutation_roots=mutation_roots,
                mutation_roots_unknown=mutation_roots_unknown,
            ) from exc
        yield result.model_copy(
            update={
                "write_paths": list(write_paths),
                "write_paths_unknown": write_paths_unknown,
                "mutation_roots": list(mutation_roots),
                "mutation_roots_unknown": mutation_roots_unknown,
            }
        )

    @classmethod
    def format_result_display(cls, result: StartResult) -> ToolResultDisplay:
        return ToolResultDisplay(
            success=True, message=f"{result.job.job_id}: {result.job.state}"
        )

    @classmethod
    def project_result(cls, result: StartResult) -> JsonValue:
        return cast(JsonValue, result.model_dump(mode="json"))

    @classmethod
    def get_status_text(cls) -> str:
        return "Starting background job"
